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
import torch.nn.functional as F
from pathlib import Path
import json
from datetime import datetime
from typing import Dict, Optional, List, Tuple, Any
import copy
import numpy as np

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


# ============================================================================
# COMPREHENSIVE QUANTIZATION EVALUATION MODULE
# ============================================================================

class QuantizationEvaluator:
    """
    Comprehensive evaluator for quantized model variants.
    Analyzes episodic memory, attention, vision/language encoders, and fusion.
    """

    def __init__(self, wandb_logger=None, device='cuda'):
        """
        Args:
            wandb_logger: WandB logger instance for metric logging
            device: Device to run evaluation on
        """
        self.wandb_logger = wandb_logger
        self.device = device if torch.cuda.is_available() else 'cpu'

        # Storage for comparative analysis
        self.variant_metrics = {}
        self.memory_heatmaps = {}
        self.attention_maps = {}

    def evaluate_all_variants(
        self,
        variants_info: Dict[str, Dict],
        model_factory_fn,
        config,
        eval_dataloader,
        num_eval_batches: int = 50
    ) -> Dict[str, Dict]:
        """
        Run comprehensive evaluation on all quantized variants.

        Args:
            variants_info: Dict mapping variant_name -> {path, bit_width, ...}
            model_factory_fn: Function to create model instance
            config: Training configuration
            eval_dataloader: DataLoader for evaluation data
            num_eval_batches: Number of batches to evaluate

        Returns:
            all_metrics: Dict mapping variant_name -> metrics
        """
        print("\n" + "="*80)
        print("🔬 COMPREHENSIVE QUANTIZATION EVALUATION")
        print("="*80 + "\n")

        all_metrics = {}

        for variant_name, variant_data in variants_info.items():
            print(f"\n{'='*60}")
            print(f"📊 Evaluating: {variant_name} ({variant_data['bit_width']}-bit)")
            print(f"{'='*60}")

            # Load model
            model = self._load_variant_model(
                variant_data['path'],
                model_factory_fn,
                config
            )
            model = model.to(self.device)
            model.eval()

            # Run evaluation
            metrics = self._evaluate_single_variant(
                model=model,
                variant_name=variant_name,
                bit_width=variant_data['bit_width'],
                eval_dataloader=eval_dataloader,
                num_batches=num_eval_batches
            )

            all_metrics[variant_name] = metrics
            self.variant_metrics[variant_name] = metrics

            # Log to wandb
            if self.wandb_logger:
                self._log_variant_metrics(variant_name, metrics)

            # Cleanup
            del model
            torch.cuda.empty_cache() if torch.cuda.is_available() else None

        # Generate comparative analysis
        self._generate_comparative_analysis(all_metrics)

        print("\n" + "="*80)
        print("✨ EVALUATION COMPLETE")
        print("="*80 + "\n")

        return all_metrics

    def _load_variant_model(self, checkpoint_path: str, model_factory_fn, config):
        """Load a model variant from checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        model = model_factory_fn(config)
        model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        return model

    def _evaluate_single_variant(
        self,
        model: nn.Module,
        variant_name: str,
        bit_width: int,
        eval_dataloader,
        num_batches: int
    ) -> Dict[str, Any]:
        """
        Comprehensive evaluation of a single model variant.
        """
        metrics = {
            'bit_width': bit_width,
            'variant_name': variant_name,
            'episodic_memory': {},
            'attention': {},
            'vision_encoder': {},
            'language_encoder': {},
            'fusion': {},
            'losses': {}
        }

        # Initialize accumulators
        memory_slot_activations = []
        memory_update_magnitudes = []
        attention_entropies = []
        attention_sparsities = []
        vision_feature_norms = []
        text_feature_norms = []
        alignment_similarities = []
        itc_losses = []
        itm_losses = []
        total_losses = []

        # Hook storage for attention capture
        attention_outputs = {}

        def capture_attention_hook(name):
            def hook(module, input, output):
                if isinstance(output, tuple):
                    attention_outputs[name] = output[1] if len(output) > 1 else output[0]
                else:
                    attention_outputs[name] = output
            return hook

        # Register hooks for attention layers
        hooks = []
        for name, module in model.named_modules():
            if 'attention' in name.lower() or 'attn' in name.lower():
                hooks.append(module.register_forward_hook(capture_attention_hook(name)))

        with torch.no_grad():
            for batch_idx, batch in enumerate(eval_dataloader):
                if batch_idx >= num_batches:
                    break

                # Move batch to device
                images = batch['image'].to(self.device) if 'image' in batch else batch[0].to(self.device)

                # Handle different batch formats
                if isinstance(batch, dict):
                    input_ids = batch.get('input_ids', batch.get('caption_ids')).to(self.device)
                    attention_mask = batch.get('attention_mask', torch.ones_like(input_ids)).to(self.device)
                else:
                    input_ids = batch[1].to(self.device)
                    attention_mask = batch[2].to(self.device) if len(batch) > 2 else torch.ones_like(input_ids)

                try:
                    # Forward pass
                    outputs = model(
                        images=images,
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        return_dict=True
                    )

                    # === 1. Episodic Memory Analysis ===
                    if hasattr(model, 'episodic_memory') or hasattr(model, 'module'):
                        base_model = model.module if hasattr(model, 'module') else model
                        if hasattr(base_model, 'episodic_memory'):
                            mem_metrics = self._analyze_episodic_memory(
                                base_model.episodic_memory,
                                batch_idx
                            )
                            memory_slot_activations.append(mem_metrics['slot_activations'])
                            memory_update_magnitudes.append(mem_metrics['update_magnitude'])

                    # === 2. Attention Analysis ===
                    for attn_name, attn_weights in attention_outputs.items():
                        if attn_weights is not None and isinstance(attn_weights, torch.Tensor):
                            entropy, sparsity = self._compute_attention_metrics(attn_weights)
                            attention_entropies.append(entropy)
                            attention_sparsities.append(sparsity)

                    # === 3. Vision Encoder Analysis ===
                    if hasattr(outputs, 'vision_features') or 'vision_features' in outputs:
                        vision_feats = outputs.get('vision_features', outputs.vision_features)
                        if vision_feats is not None:
                            vision_feature_norms.append(vision_feats.norm(dim=-1).mean().item())

                    # === 4. Language Encoder Analysis ===
                    if hasattr(outputs, 'text_features') or 'text_features' in outputs:
                        text_feats = outputs.get('text_features', outputs.text_features)
                        if text_feats is not None:
                            text_feature_norms.append(text_feats.norm(dim=-1).mean().item())

                    # === 5. Fusion & Alignment Analysis ===
                    if hasattr(outputs, 'alignment_sim') or 'alignment_sim' in outputs:
                        sim = outputs.get('alignment_sim', getattr(outputs, 'alignment_sim', None))
                        if sim is not None:
                            alignment_similarities.append(sim.mean().item() if torch.is_tensor(sim) else sim)

                    # === 6. Loss Analysis ===
                    if hasattr(outputs, 'loss') or 'loss' in outputs:
                        loss = outputs.get('loss', getattr(outputs, 'loss', None))
                        if loss is not None:
                            total_losses.append(loss.item() if torch.is_tensor(loss) else loss)

                    if hasattr(outputs, 'itc_loss') or 'itc_loss' in outputs:
                        itc = outputs.get('itc_loss', getattr(outputs, 'itc_loss', None))
                        if itc is not None:
                            itc_losses.append(itc.item() if torch.is_tensor(itc) else itc)

                    if hasattr(outputs, 'itm_loss') or 'itm_loss' in outputs:
                        itm = outputs.get('itm_loss', getattr(outputs, 'itm_loss', None))
                        if itm is not None:
                            itm_losses.append(itm.item() if torch.is_tensor(itm) else itm)

                except Exception as e:
                    print(f"   ⚠️ Batch {batch_idx} evaluation error: {e}")
                    continue

                attention_outputs.clear()

                if batch_idx % 10 == 0:
                    print(f"   Processed batch {batch_idx}/{num_batches}")

        # Remove hooks
        for hook in hooks:
            hook.remove()

        # === Aggregate Metrics ===

        # Episodic Memory
        if memory_slot_activations:
            slot_acts = np.array(memory_slot_activations)
            metrics['episodic_memory'] = {
                'slot_utilization_mean': float(np.mean(slot_acts > 0.1)),
                'slot_utilization_std': float(np.std(np.mean(slot_acts > 0.1, axis=1))),
                'update_magnitude_mean': float(np.mean(memory_update_magnitudes)) if memory_update_magnitudes else 0,
                'update_magnitude_std': float(np.std(memory_update_magnitudes)) if memory_update_magnitudes else 0,
                'dead_slots_ratio': float(np.mean(np.all(slot_acts < 0.01, axis=0))),
                'saturation_ratio': float(np.mean(np.all(slot_acts > 0.9, axis=0))),
                'heatmap': slot_acts.tolist()  # Time x Slot matrix
            }
            self.memory_heatmaps[variant_name] = slot_acts

        # Attention
        if attention_entropies:
            metrics['attention'] = {
                'entropy_mean': float(np.mean(attention_entropies)),
                'entropy_std': float(np.std(attention_entropies)),
                'sparsity_mean': float(np.mean(attention_sparsities)),
                'sparsity_std': float(np.std(attention_sparsities)),
                'stability': 1.0 - float(np.std(attention_entropies) / (np.mean(attention_entropies) + 1e-8))
            }

        # Vision Encoder
        if vision_feature_norms:
            metrics['vision_encoder'] = {
                'feature_norm_mean': float(np.mean(vision_feature_norms)),
                'feature_norm_std': float(np.std(vision_feature_norms)),
                'activation_stability': 1.0 - float(np.std(vision_feature_norms) / (np.mean(vision_feature_norms) + 1e-8))
            }

        # Language Encoder
        if text_feature_norms:
            metrics['language_encoder'] = {
                'feature_norm_mean': float(np.mean(text_feature_norms)),
                'feature_norm_std': float(np.std(text_feature_norms)),
                'embedding_stability': 1.0 - float(np.std(text_feature_norms) / (np.mean(text_feature_norms) + 1e-8))
            }

        # Fusion & Alignment
        if alignment_similarities:
            metrics['fusion'] = {
                'alignment_similarity_mean': float(np.mean(alignment_similarities)),
                'alignment_similarity_std': float(np.std(alignment_similarities)),
                'alignment_drift': float(np.std(alignment_similarities))
            }

        # Losses
        metrics['losses'] = {
            'total_loss_mean': float(np.mean(total_losses)) if total_losses else None,
            'total_loss_std': float(np.std(total_losses)) if total_losses else None,
            'itc_loss_mean': float(np.mean(itc_losses)) if itc_losses else None,
            'itm_loss_mean': float(np.mean(itm_losses)) if itm_losses else None,
        }

        return metrics

    def _analyze_episodic_memory(self, episodic_memory, step: int) -> Dict:
        """Analyze episodic memory state."""
        metrics = {
            'slot_activations': np.zeros(64),  # Default size
            'update_magnitude': 0.0,
            'write_frequency': 0.0,
            'read_frequency': 0.0
        }

        try:
            # Get memory state
            if hasattr(episodic_memory, 'memory_mean'):
                memory_mean = episodic_memory.memory_mean.detach().cpu().numpy()
                metrics['slot_activations'] = np.abs(memory_mean).mean(axis=-1)
                metrics['update_magnitude'] = float(np.std(memory_mean))

            # Check for quantized memory slots
            if hasattr(episodic_memory, 'quantized_memory_slots'):
                qms = episodic_memory.quantized_memory_slots
                if hasattr(qms, 'dequantize_memory'):
                    mem_mean, mem_logvar = qms.dequantize_memory()
                    metrics['slot_activations'] = mem_mean.abs().mean(dim=-1).cpu().numpy()
                    metrics['update_magnitude'] = float(mem_mean.std().item())
        except Exception as e:
            pass  # Use default values

        return metrics

    def _compute_attention_metrics(self, attention_weights: torch.Tensor) -> Tuple[float, float]:
        """Compute attention entropy and sparsity."""
        try:
            # Handle different attention shapes
            if attention_weights.dim() > 2:
                attn = attention_weights.mean(dim=tuple(range(attention_weights.dim() - 2)))
            else:
                attn = attention_weights

            # Normalize to probabilities
            attn = F.softmax(attn.float(), dim=-1)

            # Entropy: -sum(p * log(p))
            entropy = -(attn * (attn + 1e-10).log()).sum(dim=-1).mean().item()

            # Sparsity: ratio of near-zero values
            sparsity = (attn < 0.01).float().mean().item()

            return entropy, sparsity
        except:
            return 0.0, 0.0

    def _log_variant_metrics(self, variant_name: str, metrics: Dict):
        """Log metrics to wandb with proper prefixing."""
        if not self.wandb_logger:
            return

        bit_width = metrics['bit_width']
        prefix = f"quant_eval/{bit_width}bit"

        flat_metrics = {}

        # Episodic Memory
        if metrics['episodic_memory']:
            for k, v in metrics['episodic_memory'].items():
                if k != 'heatmap' and v is not None:
                    flat_metrics[f"{prefix}/memory/{k}"] = v

        # Attention
        if metrics['attention']:
            for k, v in metrics['attention'].items():
                if v is not None:
                    flat_metrics[f"{prefix}/attention/{k}"] = v

        # Vision Encoder
        if metrics['vision_encoder']:
            for k, v in metrics['vision_encoder'].items():
                if v is not None:
                    flat_metrics[f"{prefix}/vision/{k}"] = v

        # Language Encoder
        if metrics['language_encoder']:
            for k, v in metrics['language_encoder'].items():
                if v is not None:
                    flat_metrics[f"{prefix}/language/{k}"] = v

        # Fusion
        if metrics['fusion']:
            for k, v in metrics['fusion'].items():
                if v is not None:
                    flat_metrics[f"{prefix}/fusion/{k}"] = v

        # Losses
        if metrics['losses']:
            for k, v in metrics['losses'].items():
                if v is not None:
                    flat_metrics[f"{prefix}/loss/{k}"] = v

        # Log all metrics
        try:
            self.wandb_logger.log_metrics(flat_metrics, step=0)

            # Log memory heatmap as image if available
            if variant_name in self.memory_heatmaps:
                import wandb
                heatmap = self.memory_heatmaps[variant_name]
                wandb.log({
                    f"{prefix}/memory_heatmap": wandb.Image(
                        self._create_heatmap_image(heatmap),
                        caption=f"Memory Slot Activations ({bit_width}-bit)"
                    )
                })
        except Exception as e:
            print(f"   ⚠️ WandB logging error: {e}")

    def _create_heatmap_image(self, heatmap: np.ndarray) -> np.ndarray:
        """Create a heatmap image from numpy array."""
        try:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt

            fig, ax = plt.subplots(figsize=(12, 8))
            im = ax.imshow(heatmap, aspect='auto', cmap='hot', interpolation='nearest')
            ax.set_xlabel('Memory Slot')
            ax.set_ylabel('Time Step')
            ax.set_title('Episodic Memory Slot Activations Over Time')
            plt.colorbar(im, ax=ax, label='Activation')

            # Convert to numpy array
            fig.canvas.draw()
            img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
            img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
            plt.close(fig)

            return img
        except Exception as e:
            # Return blank image if matplotlib fails
            return np.zeros((480, 640, 3), dtype=np.uint8)

    def _generate_comparative_analysis(self, all_metrics: Dict[str, Dict]):
        """Generate cross-variant comparative analysis and log to wandb."""
        if not self.wandb_logger:
            return

        print("\n📈 Generating Comparative Analysis...")

        try:
            import wandb

            # Create comparison tables
            comparison_data = []
            for variant_name, metrics in all_metrics.items():
                row = {
                    'Variant': variant_name,
                    'Bit Width': metrics['bit_width'],
                }

                # Add key metrics
                if metrics['episodic_memory']:
                    row['Memory Utilization'] = metrics['episodic_memory'].get('slot_utilization_mean', 0)
                    row['Dead Slots %'] = metrics['episodic_memory'].get('dead_slots_ratio', 0) * 100

                if metrics['attention']:
                    row['Attn Entropy'] = metrics['attention'].get('entropy_mean', 0)
                    row['Attn Sparsity'] = metrics['attention'].get('sparsity_mean', 0)

                if metrics['vision_encoder']:
                    row['Vision Norm'] = metrics['vision_encoder'].get('feature_norm_mean', 0)

                if metrics['language_encoder']:
                    row['Text Norm'] = metrics['language_encoder'].get('feature_norm_mean', 0)

                if metrics['fusion']:
                    row['Alignment Sim'] = metrics['fusion'].get('alignment_similarity_mean', 0)

                if metrics['losses']:
                    row['Total Loss'] = metrics['losses'].get('total_loss_mean', 0)

                comparison_data.append(row)

            # Log comparison table
            wandb.log({
                "quant_eval/comparison_table": wandb.Table(
                    columns=list(comparison_data[0].keys()) if comparison_data else [],
                    data=[list(row.values()) for row in comparison_data]
                )
            })

            # Create bar charts for key metrics
            bit_widths = [m['bit_width'] for m in all_metrics.values()]

            # Memory utilization comparison
            if all(m.get('episodic_memory', {}).get('slot_utilization_mean') is not None for m in all_metrics.values()):
                mem_utils = [m['episodic_memory']['slot_utilization_mean'] for m in all_metrics.values()]
                wandb.log({
                    "quant_eval/memory_utilization_comparison": wandb.plot.bar(
                        wandb.Table(data=[[str(b), u] for b, u in zip(bit_widths, mem_utils)],
                                   columns=["Bit Width", "Memory Utilization"]),
                        "Bit Width", "Memory Utilization",
                        title="Memory Utilization by Quantization Level"
                    )
                })

            # Loss comparison
            if all(m.get('losses', {}).get('total_loss_mean') is not None for m in all_metrics.values()):
                losses = [m['losses']['total_loss_mean'] for m in all_metrics.values()]
                wandb.log({
                    "quant_eval/loss_comparison": wandb.plot.bar(
                        wandb.Table(data=[[str(b), l] for b, l in zip(bit_widths, losses)],
                                   columns=["Bit Width", "Loss"]),
                        "Bit Width", "Loss",
                        title="Loss by Quantization Level"
                    )
                })

            print("   ✅ Comparative analysis logged to WandB")

        except Exception as e:
            print(f"   ⚠️ Comparative analysis error: {e}")


def evaluate_quantized_variants(
    variants_info: Dict[str, Dict],
    model_factory_fn,
    config,
    eval_dataloader,
    wandb_logger=None,
    num_eval_batches: int = 50
) -> Dict[str, Dict]:
    """
    Main entry point for quantization evaluation.

    Args:
        variants_info: Dict from generate_quantized_variants()
        model_factory_fn: Function to create model instance
        config: Training configuration
        eval_dataloader: DataLoader for evaluation
        wandb_logger: Optional WandB logger
        num_eval_batches: Number of batches to evaluate

    Returns:
        all_metrics: Comprehensive evaluation metrics for all variants
    """
    evaluator = QuantizationEvaluator(wandb_logger=wandb_logger)

    return evaluator.evaluate_all_variants(
        variants_info=variants_info,
        model_factory_fn=model_factory_fn,
        config=config,
        eval_dataloader=eval_dataloader,
        num_eval_batches=num_eval_batches
    )


# ============================================================================
# ORIGINAL QUANTIZATION FUNCTIONS (UNCHANGED)
# ============================================================================

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
