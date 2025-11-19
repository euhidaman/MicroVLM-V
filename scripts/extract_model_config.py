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
    """Compute derived dimensions and parameters"""
    qwen_config = extract_qwen_config()
    deit_config = extract_deit_config()
    
    # Compute K_prefix dynamically
    num_patches = deit_config["num_patches"]
    k_prefix = max(8, min(64, (num_patches + 7) // 8))  # ceil(num_patches / 8), clamped [8, 64]
    
    dimensions = {
        "qwen_hidden_dim": qwen_config["hidden_size"],
        "deit_embed_dim": deit_config["hidden_size"],
        "num_patches": num_patches,
        "k_prefix": k_prefix,
        "adapter_projection_dim": qwen_config["hidden_size"],
        "memory_size": 512,  # K_mem
        "memory_dim": qwen_config["hidden_size"],  # C_mem = qwen_hidden_dim
        "scope_hidden_dim": 256
    }
    
    return dimensions

def estimate_model_sizes():
    """Estimate model component sizes"""
    qwen_config = extract_qwen_config()
    deit_config = extract_deit_config()
    dims = compute_model_dimensions()
    
    # Parameter counts (in millions)
    qwen_params = 494  # Qwen2.5-0.5B actual parameter count
    deit_params = 5.7  # DeiT-Tiny parameter count
    
    # Adapter parameters
    adapter_params = (dims["deit_embed_dim"] * dims["qwen_hidden_dim"] + 
                     dims["k_prefix"] * dims["qwen_hidden_dim"]) / 1e6
    
    # Memory parameters
    memory_params = (dims["memory_size"] * dims["memory_dim"]) / 1e6
    
    # W_M projection parameters (memory to KV space)
    num_layers = qwen_config["num_hidden_layers"]
    num_heads = qwen_config["num_attention_heads"]
    head_dim = qwen_config["head_dim"]
    wm_params = (dims["memory_dim"] * num_layers * num_heads * head_dim * 2) / 1e6
    
    # ScopeNet parameters
    scope_params = ((dims["qwen_hidden_dim"] + dims["scope_hidden_dim"]) * 
                   dims["scope_hidden_dim"] + dims["scope_hidden_dim"]) / 1e6
    
    sizes = {
        "qwen_original_mb": qwen_params * 4,  # FP32
        "qwen_4bit_mb": qwen_params * 0.5,
        "qwen_158bit_mb": qwen_params * 0.2,
        "deit_original_mb": deit_params * 4,
        "deit_4bit_mb": deit_params * 0.5,
        "adapter_mb": adapter_params * 4,
        "memory_original_mb": memory_params * 4,
        "memory_158bit_mb": memory_params * 0.2,
        "wm_projection_mb": wm_params * 4,
        "scopenet_mb": scope_params * 4,
        "total_unquantized_mb": (qwen_params + deit_params + adapter_params + 
                                memory_params + wm_params + scope_params) * 4,
        "total_quantized_mb": (qwen_params * 0.2 + deit_params * 0.5 + 
                              adapter_params * 4 + memory_params * 0.2 + 
                              wm_params * 4 + scope_params * 4)
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
            "description": "MicroVLM-V model configuration with 1.58-bit quantization"
        }
    }
    
    output_path = Path(__file__).parent.parent / "configs" / "model_config.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, "w") as f:
        json.dump(config, f, indent=2)
    
    print(f"Configuration saved to {output_path}")
    print(f"\nModel Size Summary:")
    sizes = config["estimated_sizes"]
    print(f"  Qwen2.5-0.5B (original): {sizes['qwen_original_mb']:.1f} MB")
    print(f"  Qwen2.5-0.5B (4-bit): {sizes['qwen_4bit_mb']:.1f} MB")
    print(f"  Qwen2.5-0.5B (1.58-bit): {sizes['qwen_158bit_mb']:.1f} MB")
    print(f"  DeiT-Tiny (original): {sizes['deit_original_mb']:.1f} MB")
    print(f"  DeiT-Tiny (4-bit): {sizes['deit_4bit_mb']:.1f} MB")
    print(f"  Episodic Memory (original): {sizes['memory_original_mb']:.1f} MB")
    print(f"  Episodic Memory (1.58-bit): {sizes['memory_158bit_mb']:.1f} MB")
    print(f"  Total (unquantized): {sizes['total_unquantized_mb']:.1f} MB")
    print(f"  Total (quantized): {sizes['total_quantized_mb']:.1f} MB")
    
    return config

if __name__ == "__main__":
    save_config()
