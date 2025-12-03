"""
Model Configuration Extraction Script
Extracts Qwen2.5-0.5B and DeiT-Tiny configurations and saves to JSON
"""

import json
import torch
from pathlib import Path

def extract_qwen_config():
    """Extract Qwen2.5-0.5B configuration"""
    config = {
        "hidden_size": 896,
        "num_hidden_layers": 24,
        "num_attention_heads": 14,
        "num_key_value_heads": 2,
        "intermediate_size": 4864,
        "vocab_size": 151936,
        "max_position_embeddings": 32768,
        "rope_theta": 1000000.0,
        "rms_norm_eps": 1e-06,
        "tie_word_embeddings": True,
        "use_sliding_window": False,
        "sliding_window": None,
        "max_window_layers": 21,
        "head_dim": 64,
        "kv_head_dim": 128,
        "model_type": "qwen2"
    }
    return config

def extract_deit_config():
    """Extract DeiT-Tiny configuration"""
    config = {
        "image_size": 224,
        "patch_size": 16,
        "num_channels": 3,
        "hidden_size": 192,
        "num_hidden_layers": 12,
        "num_attention_heads": 3,
        "intermediate_size": 768,
        "hidden_dropout_prob": 0.0,
        "attention_probs_dropout_prob": 0.0,
        "layer_norm_eps": 1e-06,
        "num_patches": 196,
        "encoder_stride": 16
    }
    return config

def extract_quantization_config():
    """Define quantization settings"""
    config = {
        "memory_quantization": {
            "bits": 1.58,
            "method": "bitnet_158",
            "description": "1.58-bit quantization for episodic memory and final model"
        },
        "training_quantization": {
            "weight_bits": 4,
            "activation_bits": 4,
            "method": "symmetric",
            "description": "4-bit quantization for weights and activations during training"
        },
        "inference_quantization": {
            "model_bits": 1.58,
            "method": "bitnet_158",
            "description": "1.58-bit quantization for final deployed model"
        }
    }
    return config

def compute_model_dimensions():
    """
    Compute derived dimensions and parameters.
    
    NOTE: Reduced for compact model (<1GB target):
    - alignment_dim: 256 -> 128
    - memory_size: 512 -> 64
    - memory_target_layers: 24 -> 6
    - memory_num_heads: 14 -> 4
    """
    qwen_config = extract_qwen_config()
    deit_config = extract_deit_config()
    
    # Compute K_prefix dynamically
    num_patches = deit_config["num_patches"]
    k_prefix = max(8, min(64, (num_patches + 7) // 8))  # ceil(num_patches / 8), clamped [8, 64]
    
    dimensions = {
        "qwen_hidden_dim": qwen_config["hidden_size"],
        "deit_embed_dim": deit_config["hidden_size"],
        "vision_hidden_size": deit_config["hidden_size"],  # Alias for microvlm.py
        "language_hidden_size": qwen_config["hidden_size"],  # Alias for microvlm.py
        "num_patches": num_patches,
        "k_prefix": k_prefix,
        "adapter_projection_dim": qwen_config["hidden_size"],
        "alignment_dim": 128,  # Reduced from 256 for compact model
        "memory_size": 64,  # Reduced from 512 (K_mem)
        "memory_dim": qwen_config["hidden_size"],  # C_mem = qwen_hidden_dim
        "memory_target_layers": 6,  # Reduced from 24 (full layers)
        "memory_num_heads": 4,  # Reduced from 14 (full heads)
        "scope_hidden_dim": 256,
        "itm_hidden_dim": 256,  # Reduced from 512
        "fusion_layers": [9, 11],  # Reduced from [8, 9, 10, 11]
        "num_fusion_heads": 2  # Reduced from 4
    }
    
    return dimensions

def estimate_model_sizes():
    """
    Estimate model component sizes.
    
    NOTE: With 4-bit quantization enabled for Qwen (default), 
    the model fits under 1GB for edge deployment.
    """
    qwen_config = extract_qwen_config()
    deit_config = extract_deit_config()
    dims = compute_model_dimensions()
    
    # Parameter counts (in millions)
    qwen_params = 494  # Qwen2.5-0.5B actual parameter count
    deit_params = 5.7  # DeiT-Tiny parameter count
    
    # Adapter parameters (reduced dimensions)
    adapter_params = (dims["deit_embed_dim"] * dims["qwen_hidden_dim"] + 
                     dims["k_prefix"] * dims["qwen_hidden_dim"]) / 1e6
    
    # Memory parameters (reduced from 512 to 64 slots)
    memory_params = (dims["memory_size"] * dims["memory_dim"]) / 1e6
    
    # W_M projection parameters (reduced to 6 layers x 4 heads)
    head_dim = qwen_config["head_dim"]
    wm_params = (dims["memory_dim"] * dims["memory_target_layers"] * 
                 dims["memory_num_heads"] * head_dim * 2) / 1e6
    
    # ScopeNet parameters (reduced hidden_dim)
    scope_params = ((dims["qwen_hidden_dim"] + dims["scope_hidden_dim"]) * 
                   dims["scope_hidden_dim"] + dims["scope_hidden_dim"]) / 1e6
    
    # FIBER fusion parameters (reduced to 2 layers, 2 heads)
    num_fusion_layers = len(dims["fusion_layers"])
    fusion_params = (num_fusion_layers * dims["num_fusion_heads"] * 
                     dims["vision_hidden_size"] * 3) / 1e6  # Q, K, V projections
    
    # Alignment projections (reduced alignment_dim)
    align_params = (dims["vision_hidden_size"] * dims["alignment_dim"] + 
                    dims["language_hidden_size"] * dims["alignment_dim"]) / 1e6
    
    # ITM head (reduced hidden_dim)
    itm_params = ((dims["vision_hidden_size"] + dims["language_hidden_size"]) * 
                  dims["itm_hidden_dim"] + dims["itm_hidden_dim"] * 2) / 1e6
    
    # Total overhead (non-backbone parameters)
    overhead_params = adapter_params + memory_params + wm_params + scope_params + fusion_params + align_params + itm_params
    
    sizes = {
        # Original sizes (FP32)
        "qwen_original_mb": qwen_params * 4,  # ~1976 MB
        "deit_original_mb": deit_params * 4,  # ~23 MB
        
        # 4-bit quantized sizes (DEFAULT)
        "qwen_4bit_mb": qwen_params * 0.5,    # ~247 MB
        "deit_4bit_mb": deit_params * 0.5,    # ~3 MB
        
        # 1.58-bit quantized sizes (optional)
        "qwen_158bit_mb": qwen_params * 0.2,  # ~99 MB
        
        # Component sizes
        "adapter_mb": adapter_params * 4,
        "memory_original_mb": memory_params * 4,
        "memory_158bit_mb": memory_params * 0.2,
        "wm_projection_mb": wm_params * 4,
        "scopenet_mb": scope_params * 4,
        "fusion_mb": fusion_params * 4,
        "alignment_mb": align_params * 4,
        "itm_head_mb": itm_params * 4,
        "overhead_total_mb": overhead_params * 4,
        
        # Total model sizes
        "total_fp32_mb": (qwen_params + deit_params + overhead_params) * 4,
        "total_4bit_qwen_mb": qwen_params * 0.5 + deit_params * 2 + overhead_params * 4,
        "total_fully_quantized_mb": qwen_params * 0.2 + deit_params * 0.5 + overhead_params * 4,
    }
    
    return sizes

def save_config():
    """Save all configurations to JSON file"""
    config = {
        "qwen2_5_config": extract_qwen_config(),
        "deit_tiny_config": extract_deit_config(),
        "quantization_config": extract_quantization_config(),
        "model_dimensions": compute_model_dimensions(),
        "estimated_sizes": estimate_model_sizes(),
        "metadata": {
            "qwen_model": "Qwen/Qwen2.5-0.5B",
            "deit_model": "facebook/deit-tiny-patch16-224",
            "target_total_size_mb": 500,
            "description": "MicroVLM-V compact model configuration with 4-bit quantization for <1GB deployment",
            "quantization_default": "4-bit Qwen, FP16 DeiT, FP32 overhead"
        }
    }
    
    output_path = Path(__file__).parent.parent / "configs" / "model_config.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, "w") as f:
        json.dump(config, f, indent=2)
    
    print(f"Configuration saved to {output_path}")
    print(f"\n{'='*60}")
    print("MODEL SIZE SUMMARY (Compact Configuration)")
    print("="*60)
    
    sizes = config["estimated_sizes"]
    dims = config["model_dimensions"]
    
    print("\n📦 BACKBONE MODELS (immutable):")
    print(f"  Qwen2.5-0.5B (FP32):    {sizes['qwen_original_mb']:.1f} MB")
    print(f"  Qwen2.5-0.5B (4-bit):   {sizes['qwen_4bit_mb']:.1f} MB  ← DEFAULT")
    print(f"  Qwen2.5-0.5B (1.58-bit):{sizes['qwen_158bit_mb']:.1f} MB")
    print(f"  DeiT-Tiny (FP32):       {sizes['deit_original_mb']:.1f} MB")
    print(f"  DeiT-Tiny (FP16):       {sizes['deit_original_mb']/2:.1f} MB  ← DEFAULT")
    
    print("\n🔧 REDUCED COMPONENTS:")
    print(f"  Adapter (k_prefix={dims['k_prefix']}):        {sizes['adapter_mb']:.2f} MB")
    print(f"  Memory (size={dims['memory_size']}):           {sizes['memory_original_mb']:.4f} MB")
    print(f"  W_M Projection (6L×4H):  {sizes['wm_projection_mb']:.2f} MB")
    print(f"  ScopeNet (hidden=256):   {sizes['scopenet_mb']:.2f} MB")
    print(f"  FIBER Fusion (2 layers): {sizes['fusion_mb']:.2f} MB")
    print(f"  Alignment (dim={dims['alignment_dim']}):      {sizes['alignment_mb']:.2f} MB")
    print(f"  ITM Head (hidden={dims['itm_hidden_dim']}):    {sizes['itm_head_mb']:.2f} MB")
    print(f"  ─────────────────────────────────")
    print(f"  Overhead Total:          {sizes['overhead_total_mb']:.2f} MB")
    
    print("\n📊 TOTAL MODEL SIZE:")
    print(f"  FP32 (unquantized):      {sizes['total_fp32_mb']:.1f} MB")
    print(f"  4-bit Qwen (DEFAULT):    {sizes['total_4bit_qwen_mb']:.1f} MB  ✓ Under 1GB!")
    print(f"  Fully quantized:         {sizes['total_fully_quantized_mb']:.1f} MB")
    
    print("\n" + "="*60)
    
    return config

if __name__ == "__main__":
    save_config()
