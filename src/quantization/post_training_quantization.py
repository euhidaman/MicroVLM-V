"""
Post-Training Quantization Module

This module handles FINAL quantization AFTER best model selection is complete.

CRITICAL RULES:
1. Never call during training
2. Only call after best checkpoint is identified
3. Generates multiple bit-width variants from the SAME best model
4. Each variant is independent (no feedback into training)
5. Base models (Qwen + DiET) remain 4-bit throughout

Quantization Flow:
    Best Model (full-precision trainable) → 4-bit variant
                                           → 3-bit variant
                                           → 1.58-bit variant

All variants are evaluation/deployment artifacts only.
"""

import torch
import torch.nn as nn
from pathlib import Path
import json
from datetime import datetime
from typing import Dict, Optional, List
import copy

from .quantize_4bit import (
    quantize_4bit_symmetric,
    QuantizedLinear4bit,
    estimate_model_size_4bit
)
from .quantize_158bit import (
    quantize_weights_158bit,
    QuantizedLinear158,
    estimate_model_size_158bit
)
from .quantized_episodic_memory import get_memory_quantization_stats


def quantize_3bit_symmetric(tensor, bits=3):
    """
    Symmetric 3-bit quantization

    Args:
        tensor: input tensor
        bits: number of bits (default 3)

    Returns:
        quantized: quantized tensor
        scale: scaling factor
    """
    qmax = 2 ** (bits - 1) - 1  # 3 for 3-bit
    max_val = tensor.abs().max()

    scale = max_val / qmax
    scale = scale.clamp(min=1e-8)

    quantized = torch.round(tensor / scale).clamp(-qmax - 1, qmax)

    return quantized, scale


class QuantizedLinear3bit(nn.Module):
    """
    3-bit quantized linear layer
    """

    def __init__(self, in_features, out_features, bias=True):
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features

        self.register_buffer('quantized_weight',
                           torch.zeros(out_features, in_features, dtype=torch.int8))
        self.register_buffer('weight_scale', torch.ones(1))

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter('bias', None)

    def quantize_and_pack(self, weight):
        """Quantize and pack weights"""
        quantized, scale = quantize_3bit_symmetric(weight, bits=3)

        self.quantized_weight.data = quantized.to(torch.int8)
        self.weight_scale.data = scale.view(1)

    def forward(self, x):
        """Forward with quantized weights"""
        weight = self.quantized_weight.float() * self.weight_scale
        output = torch.nn.functional.linear(x, weight, self.bias)
        return output


def estimate_model_size_3bit(model):
    """
    Estimate model size with 3-bit quantization

    Args:
        model: model to estimate

    Returns:
        size_mb: estimated size in MB
    """
    total_bits = 0

    for param in model.parameters():
        param_elements = param.numel()

        if param.dim() >= 2:  # Weight matrices
            total_bits += param_elements * 3
            total_bits += 32  # Scale factor
        else:  # Biases
            total_bits += param_elements * 32

    size_bytes = total_bits / 8
    size_mb = size_bytes / (1024 * 1024)

    return size_mb


def apply_quantization_to_trainable_components(
    model,
    bit_width: int,
    skip_frozen: bool = True
) -> nn.Module:
    """
    Apply quantization to trainable components only (not base models).

    Base models (Qwen + DiET) are already loaded in 4-bit and should not be re-quantized.

    Args:
        model: Model with trainable components
        bit_width: Target bit width (4, 3, or 1.58)
        skip_frozen: If True, skip frozen parameters (base models)

    Returns:
        quantized_model: Model with quantized trainable components
    """
    # Create a deep copy to avoid modifying the original
    quantized_model = copy.deepcopy(model)

    # Select quantization layer type
    if bit_width == 4:
        QuantLayer = QuantizedLinear4bit
    elif bit_width == 3:
        QuantLayer = QuantizedLinear3bit
    elif bit_width == 1.58:
        QuantLayer = QuantizedLinear158
    else:
        raise ValueError(f"Unsupported bit width: {bit_width}. Use 4, 3, or 1.58")

    # Track which components are quantized
    quantized_count = 0
    skipped_count = 0

    for name, module in list(quantized_model.named_modules()):
        if isinstance(module, nn.Linear):
            # Check if this module is frozen (base model component)
            is_frozen = not any(p.requires_grad for p in module.parameters())

            if skip_frozen and is_frozen:
                skipped_count += 1
                continue

            # Create quantized version
            quantized_layer = QuantLayer(
                module.in_features,
                module.out_features,
                bias=(module.bias is not None)
            )

            # Quantize weights
            quantized_layer.quantize_and_pack(module.weight.data.cpu())

            # Copy bias
            if module.bias is not None:
                quantized_layer.bias.data = module.bias.data.cpu().clone()

            # Replace module
            parent_name = '.'.join(name.split('.')[:-1])
            child_name = name.split('.')[-1]

            if parent_name:
                parent = dict(quantized_model.named_modules())[parent_name]
                setattr(parent, child_name, quantized_layer)
            else:
                setattr(quantized_model, child_name, quantized_layer)

            quantized_count += 1

    print(f"   Quantized {quantized_count} trainable linear layers to {bit_width}-bit")
    if skip_frozen:
        print(f"   Skipped {skipped_count} frozen layers (base models remain 4-bit)")

    return quantized_model


def generate_quantized_variants(
    best_checkpoint_path: str,
    output_dir: str,
    config,
    model_factory_fn,
    bit_widths: Optional[List[int]] = None
) -> Dict[str, Dict]:
    """
    Generate multiple quantized variants from the best checkpoint.

    CRITICAL: This is called ONCE at the END of training, after best model is selected.

    Args:
        best_checkpoint_path: Path to best Stage-2 checkpoint
        output_dir: Directory to save quantized variants
        config: Training configuration
        model_factory_fn: Function to create model instance
        bit_widths: List of target bit widths to generate (default: [4, 3, 1.58])

    Returns:
        variants_info: Dict mapping bit_width -> {path, stats, metrics}
    """
    if bit_widths is None:
        bit_widths = [4, 3, 1.58]

    print("\n" + "="*80)
    print("🔧 POST-TRAINING QUANTIZATION")
    print("="*80)
    print(f"Best checkpoint: {best_checkpoint_path}")
    print(f"Generating variants: {bit_widths}-bit")
    print("="*80 + "\n")

    # Load best checkpoint
    print("Loading best checkpoint...")
    checkpoint = torch.load(best_checkpoint_path, map_location='cpu')

    # Create output directory
    output_path = Path(output_dir) / "quantized_variants"
    output_path.mkdir(parents=True, exist_ok=True)

    # Store variant information
    variants_info = {}

    # Generate full-precision reference
    print("\n📦 Saving full-precision reference...")
    full_precision_path = output_path / "best-model-full-precision.pt"
    torch.save(checkpoint, full_precision_path)

    # Calculate stats for full precision
    model_full = model_factory_fn(config)
    model_full.load_state_dict(checkpoint['model_state_dict'])

    total_params = sum(p.numel() for p in model_full.parameters())
    trainable_params = sum(p.numel() for p in model_full.parameters() if p.requires_grad)

    variants_info['full_precision'] = {
        'path': str(full_precision_path),
        'bit_width': 32,
        'total_params': total_params,
        'trainable_params': trainable_params,
        'estimated_size_mb': (total_params * 4) / (1024 * 1024),  # 4 bytes per float32
        'step': checkpoint.get('global_step', 0)
    }

    print(f"   ✓ Saved to: {full_precision_path}")
    print(f"   Total params: {total_params:,}")
    print(f"   Trainable params: {trainable_params:,}")

    # Generate each quantized variant
    for bit_width in bit_widths:
        print(f"\n📦 Generating {bit_width}-bit variant...")

        # Create fresh model instance
        model = model_factory_fn(config)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()

        # Apply quantization to trainable components only
        if bit_width in [4, 3, 1.58]:
            quantized_model = apply_quantization_to_trainable_components(
                model,
                bit_width=bit_width,
                skip_frozen=True
            )
        else:
            raise ValueError(f"Unsupported bit width: {bit_width}")

        # Estimate size
        estimated_size = 0.0  # Initialize to prevent unassigned variable
        if bit_width == 4:
            estimated_size = estimate_model_size_4bit(quantized_model)
        elif bit_width == 3:
            estimated_size = estimate_model_size_3bit(quantized_model)
        elif bit_width == 1.58:
            estimated_size = estimate_model_size_158bit(quantized_model)

        # Save variant
        variant_path = output_path / f"best-model-{bit_width}bit.pt"
        variant_checkpoint = {
            'model_state_dict': quantized_model.state_dict(),
            'quantization_bit_width': bit_width,
            'original_checkpoint': best_checkpoint_path,
            'config': checkpoint.get('config'),
            'global_step': checkpoint.get('global_step'),
            'best_metrics': checkpoint.get('best_metrics'),
            'quantization_timestamp': datetime.now().isoformat()
        }

        torch.save(variant_checkpoint, variant_path)

        # Store variant info
        variants_info[f'{bit_width}bit'] = {
            'path': str(variant_path),
            'bit_width': bit_width,
            'total_params': total_params,
            'trainable_params': trainable_params,
            'estimated_size_mb': estimated_size,
            'step': checkpoint.get('global_step', 0)
        }

        print(f"   ✓ Saved to: {variant_path}")
        print(f"   Estimated size: {estimated_size:.2f} MB")

        # Clean up
        del quantized_model
        del model
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

    # Save variants manifest
    manifest_path = output_path / "variants_manifest.json"
    manifest = {
        'best_checkpoint': best_checkpoint_path,
        'generation_timestamp': datetime.now().isoformat(),
        'variants': variants_info
    }

    with open(manifest_path, 'w') as f:
        json.dump(manifest, f, indent=2)

    print(f"\n✅ Manifest saved: {manifest_path}")
    print("\n" + "="*80)
    print("✨ QUANTIZATION COMPLETE")
    print("="*80 + "\n")

    return variants_info


def create_statistics_json(
    model,
    variant_info: Dict,
    checkpoint_name: str = "best"
) -> Dict:
    """
    Create statistics.json for Hugging Face upload.

    Args:
        model: Model instance
        variant_info: Variant information dict
        checkpoint_name: Name of checkpoint

    Returns:
        statistics: Complete statistics dictionary
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen_params = total_params - trainable_params

    # Component breakdown
    vision_params = 0
    language_params = 0
    adapter_params = 0
    memory_params = 0

    for name, module in model.named_modules():
        params = sum(p.numel() for p in module.parameters())
        if 'vision' in name.lower():
            vision_params += params
        elif 'language' in name.lower() or 'qwen' in name.lower():
            language_params += params
        elif 'adapter' in name.lower():
            adapter_params += params
        elif 'memory' in name.lower():
            memory_params += params

    statistics = {
        'checkpoint_name': checkpoint_name,
        'quantization_bit_width': variant_info.get('bit_width', 32),
        'total_parameters': total_params,
        'trainable_parameters': trainable_params,
        'frozen_parameters': frozen_params,
        'component_breakdown': {
            'vision_encoder': vision_params,
            'language_model': language_params,
            'multimodal_adapter': adapter_params,
            'episodic_memory': memory_params
        },
        'quantization_status': {
            'base_models_4bit': True,  # Qwen + DiET
            'trainable_components': f"{variant_info.get('bit_width', 32)}-bit"
        },
        'estimated_model_size_mb': variant_info.get('estimated_size_mb', 0),
        'upload_timestamp': datetime.now().isoformat()
    }

    return statistics


def publish_quantized_variants_to_hf(
    variants_info: Dict,
    config,
    repo_base_name: str = "MicroVLM-V-Stage2-Best"
):
    """
    Publish all quantized variants to Hugging Face.

    Args:
        variants_info: Dictionary of variant information
        config: Training configuration
        repo_base_name: Base name for HF repositories
    """
    print("\n" + "="*80)
    print("🤗 PUBLISHING TO HUGGING FACE")
    print("="*80 + "\n")

    try:
        from huggingface_hub import HfApi, create_repo
        import os

        # Get HF token
        hf_token = os.environ.get('HF_TOKEN') or os.environ.get('HUGGINGFACE_TOKEN')
        if not hf_token:
            print("⚠️  No HF token found, skipping upload")
            return

        username = getattr(config, 'hf_username', 'euhidaman')
        api = HfApi()

        # Publish each variant
        for variant_name, variant_data in variants_info.items():
            bit_width = variant_data.get('bit_width', 32)
            variant_path = Path(variant_data['path'])

            # Create repo for this variant
            if bit_width == 32:
                repo_name = f"{repo_base_name}-FP32"
            else:
                repo_name = f"{repo_base_name}-{bit_width}bit"

            repo_id = f"{username}/{repo_name}"

            print(f"\n📤 Uploading {variant_name} variant...")
            print(f"   Repository: {repo_id}")

            try:
                # Create repository
                create_repo(repo_id=repo_id, token=hf_token, exist_ok=True, repo_type="model", private=False)

                # Upload model file
                api.upload_file(
                    path_or_fileobj=str(variant_path),
                    path_in_repo="model.pt",
                    repo_id=repo_id,
                    token=hf_token,
                    commit_message=f"Upload best {variant_name} model"
                )
                print(f"   ✅ Model uploaded")

                # Create and upload README
                readme_content = f"""# {repo_name}

## Model Information
- **Bit Width**: {bit_width}
- **Total Parameters**: {variant_data['total_params']:,}
- **Trainable Parameters**: {variant_data['trainable_params']:,}
- **Estimated Size**: {variant_data['estimated_size_mb']:.2f} MB
- **Training Step**: {variant_data['step']}

## Quantization Details
- **Base Models (Qwen + DiET)**: 4-bit (loaded at initialization)
- **Trainable Components**: {bit_width}-bit (quantized post-training)

## Usage
```python
import torch
from microvlm import create_microvlm

# Load model
checkpoint = torch.load('model.pt')
model = create_microvlm(config)
model.load_state_dict(checkpoint['model_state_dict'])
```

## Notes
- This model is the BEST performing checkpoint from Stage-2 training
- Quantization was applied AFTER training completion
- Ready for Stage-3 training or deployment
"""

                readme_path = variant_path.parent / f"README_{variant_name}.md"
                with open(readme_path, 'w') as f:
                    f.write(readme_content)

                api.upload_file(
                    path_or_fileobj=str(readme_path),
                    path_in_repo="README.md",
                    repo_id=repo_id,
                    token=hf_token,
                    commit_message="Add model card"
                )
                print(f"   ✅ README uploaded")

                print(f"   🔗 https://huggingface.co/{repo_id}")

            except Exception as e:
                print(f"   ⚠️  Upload failed: {e}")

    except Exception as e:
        print(f"⚠️  HuggingFace publishing failed: {e}")

    print("\n" + "="*80)
    print("✨ PUBLISHING COMPLETE")
    print("="*80 + "\n")

