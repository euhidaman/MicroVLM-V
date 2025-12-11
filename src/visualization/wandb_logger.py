"""
Comprehensive WandB Logging and Visualization Module
Handles all metrics, visualizations, and logging for MicroVLM-V training
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import wandb
from typing import Dict, Optional, List, Tuple
import warnings


class WandBLogger:
    """
    Comprehensive logger for WandB with organized metrics and visualizations
    """
    
    def __init__(self, config, wandb_run=None):
        self.config = config
        self.wandb_run = wandb_run
        self.enabled = wandb_run is not None
        
        # Tracking for memory heatmap
        self.memory_states = []
        self.memory_addressing_history = []
        
    def log_training_metrics(self, outputs, optimizer, epoch, global_step):
        """
        Log core training metrics
        
        Args:
            outputs: model outputs dict
            optimizer: optimizer instance
            epoch: current epoch
            global_step: global training step
        """
        if not self.enabled:
            return
        
        metrics = {
            'train/epoch': epoch,
            'train/global_step': global_step,
            'train/learning_rate': optimizer.param_groups[0]['lr']
        }
        
        # Total loss
        if 'loss' in outputs:
            metrics['train/total_loss'] = outputs['loss'].item()
        
        # LM loss
        lm_loss = outputs.get('lm_loss')
        if lm_loss is not None:
            metrics['train/lm_loss'] = lm_loss.item()
            # Calculate perplexity
            metrics['train/perplexity'] = torch.exp(lm_loss).item()
        
        # Alignment loss
        if 'alignment_loss' in outputs:
            metrics['alignment/loss'] = outputs['alignment_loss'].item()
        
        # Fine-grained alignment loss (text-to-patch attention supervision)
        if 'fine_grained_loss' in outputs:
            metrics['alignment/fine_grained_loss'] = outputs['fine_grained_loss'].item()
        
        # Memory losses
        if 'memory_kl' in outputs:
            metrics['memory/kl_divergence'] = outputs['memory_kl'].item()
        
        if 'addressing_kl' in outputs:
            metrics['memory/addressing_kl'] = outputs['addressing_kl'].item()
        
        # Scope probabilities (if present)
        if 'scope_probs' in outputs:
            scope_probs = outputs['scope_probs']
            metrics['memory/scope_prob_mean'] = scope_probs.mean().item()
            metrics['memory/scope_prob_std'] = scope_probs.std().item()
        
        # ITC/ITM losses (FIBER mode)
        if 'itc_loss' in outputs:
            itc_loss = outputs['itc_loss']
            if itc_loss is not None:
                metrics['alignment/itc_loss'] = itc_loss.item() if hasattr(itc_loss, 'item') else itc_loss
        
        if 'itm_loss' in outputs:
            itm_loss = outputs['itm_loss']
            if itm_loss is not None:
                metrics['alignment/itm_loss'] = itm_loss.item() if hasattr(itm_loss, 'item') else itm_loss
        
        if 'token_loss' in outputs:
            token_loss = outputs['token_loss']
            if token_loss is not None:
                metrics['alignment/token_loss'] = token_loss.item() if hasattr(token_loss, 'item') else token_loss
        
        # Anti-collapse regularization losses
        if 'anti_collapse_loss' in outputs:
            anti_collapse_loss = outputs['anti_collapse_loss']
            if anti_collapse_loss is not None:
                val = anti_collapse_loss.item() if hasattr(anti_collapse_loss, 'item') else anti_collapse_loss
                metrics['regularization/anti_collapse_loss'] = val
        
        if 'attention_entropy_loss' in outputs:
            attn_entropy_loss = outputs['attention_entropy_loss']
            if attn_entropy_loss is not None:
                val = attn_entropy_loss.item() if hasattr(attn_entropy_loss, 'item') else attn_entropy_loss
                metrics['regularization/attention_entropy_loss'] = val
        
        self.wandb_run.log(metrics, step=global_step)
    
    def log_temperature_metrics(self, model, global_step):
        """Log temperature/logit_scale metrics for monitoring alignment stability"""
        if not self.enabled:
            return
        
        metrics = {}
        base_model = model.module if hasattr(model, 'module') else model
        
        # Find alignment loss module and log temperature
        if hasattr(base_model, 'alignment_loss'):
            alignment_loss = base_model.alignment_loss
            if hasattr(alignment_loss, 'logit_scale'):
                logit_scale = alignment_loss.logit_scale
                if hasattr(logit_scale, 'item'):
                    logit_scale_val = logit_scale.item()
                else:
                    logit_scale_val = float(logit_scale)
                temperature = 1.0 / logit_scale_val if logit_scale_val > 0 else 0.0
                metrics['alignment/logit_scale'] = logit_scale_val
                metrics['alignment/temperature'] = temperature
        
        # Find and log alpha values from fusion blocks
        if hasattr(base_model, 'vision_encoder') and hasattr(base_model.vision_encoder, 'fusion_blocks'):
            fusion_blocks = base_model.vision_encoder.fusion_blocks
            for layer_idx, block in fusion_blocks.items():
                if hasattr(block, 'i2t_attention') and hasattr(block.i2t_attention, 'alpha'):
                    alpha = block.i2t_attention.alpha
                    if hasattr(alpha, 'item'):
                        metrics[f'fiber/alpha_i2t_layer{layer_idx}'] = alpha.item()
        
        if metrics:
            self.wandb_run.log(metrics, step=global_step)
    
    def log_gradient_metrics(self, model, global_step):
        """
        Log gradient norms and flow for monitoring training health
        
        Args:
            model: the model
            global_step: global training step
        """
        if not self.enabled:
            return
        
        metrics = {}
        
        # Overall gradient norm
        total_norm = 0.0
        for p in model.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        total_norm = total_norm ** 0.5
        metrics['gradients/total_norm'] = total_norm
        
        # Component-wise gradient norms
        component_grads = {
            'vision_encoder': [],
            'multimodal_adapter': [],
            'language_model': [],
            'episodic_memory': [],
            'image_proj_for_alignment': [],
            'text_proj_for_alignment': []
        }
        
        for name, param in model.named_parameters():
            if param.grad is not None:
                grad_norm = param.grad.data.norm(2).item()
                
                # Categorize by component
                if 'vision_encoder' in name:
                    component_grads['vision_encoder'].append(grad_norm)
                elif 'multimodal_adapter' in name:
                    component_grads['multimodal_adapter'].append(grad_norm)
                elif 'language_model' in name:
                    component_grads['language_model'].append(grad_norm)
                elif 'episodic_memory' in name:
                    component_grads['episodic_memory'].append(grad_norm)
                elif 'text_proj_for_alignment' in name:
                    component_grads['text_proj_for_alignment'].append(grad_norm)
                elif 'image_proj_for_alignment' in name:
                    component_grads['image_proj_for_alignment'].append(grad_norm)
        
        # Log mean gradient norms per component
        for component, grads in component_grads.items():
            if grads:
                metrics[f'gradients/{component}_mean'] = np.mean(grads)
                metrics[f'gradients/{component}_max'] = np.max(grads)
        
        self.wandb_run.log(metrics, step=global_step)
    
    def log_vision_encoder_metrics(self, model, images, global_step, num_samples=4):
        """
        Log vision encoder visualizations and metrics
        
        Args:
            model: the model
            images: batch of images (B, 3, H, W)
            global_step: global training step
            num_samples: number of samples to visualize
        """
        if not self.enabled:
            return
        
        with torch.no_grad():
            # Take first few samples
            sample_images = images[:num_samples]
            
            # Extract features - handle both baseline and FIBER vision encoders
            vision_output = model.vision_encoder(sample_images)
            if isinstance(vision_output, tuple):
                # FIBER encoder returns (patch_tokens, layer_outputs, fusion_attention)
                patch_embeddings = vision_output[0]
            else:
                # Baseline encoder returns just patch embeddings
                patch_embeddings = vision_output
            
            cls_features = model.vision_encoder.get_cls_token(sample_images)  # (B, hidden)
            
            # Log image samples
            # Denormalize for visualization
            mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(images.device)
            std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(images.device)
            denorm_images = sample_images * std + mean
            denorm_images = torch.clamp(denorm_images, 0, 1)
            
            # Convert to wandb images
            image_list = []
            for i in range(num_samples):
                img_np = denorm_images[i].cpu().permute(1, 2, 0).numpy()
                image_list.append(wandb.Image(img_np, caption=f"Sample {i}"))
            
            self.wandb_run.log({
                'vision/input_samples': image_list
            }, step=global_step)
            
            # Patch embedding statistics
            metrics = {
                'vision/patch_embedding_mean': patch_embeddings.mean().item(),
                'vision/patch_embedding_std': patch_embeddings.std().item(),
                'vision/patch_embedding_min': patch_embeddings.min().item(),
                'vision/patch_embedding_max': patch_embeddings.max().item(),
                'vision/cls_feature_norm': torch.norm(cls_features, dim=-1).mean().item()
            }
            
            self.wandb_run.log(metrics, step=global_step)
            
            # Visualize patch embeddings using PCA
            self._visualize_patch_embeddings(patch_embeddings[0], global_step)
    
    def _visualize_patch_embeddings(self, patch_embeddings, global_step):
        """
        Visualize patch embeddings using dimensionality reduction
        
        Args:
            patch_embeddings: (num_patches, hidden_dim)
            global_step: global training step
        """
        try:
            from sklearn.decomposition import PCA
            
            # Apply PCA
            pca = PCA(n_components=3)
            embeddings_np = patch_embeddings.cpu().numpy()
            reduced = pca.fit_transform(embeddings_np)
            
            # Create scatter plot
            fig, ax = plt.subplots(figsize=(8, 8))
            scatter = ax.scatter(reduced[:, 0], reduced[:, 1], 
                                c=reduced[:, 2], cmap='viridis', s=100)
            ax.set_title('Patch Embeddings (PCA)')
            ax.set_xlabel('PC1')
            ax.set_ylabel('PC2')
            plt.colorbar(scatter, ax=ax, label='PC3')
            
            self.wandb_run.log({
                'vision/patch_embeddings_pca': wandb.Image(fig)
            }, step=global_step)
            
            plt.close(fig)
        except ImportError:
            warnings.warn("sklearn not available, skipping PCA visualization")
    
    def log_language_model_metrics(self, outputs, input_ids, global_step):
        """
        Log language model specific metrics
        
        Args:
            outputs: model outputs
            input_ids: input token IDs
            global_step: global training step
        """
        if not self.enabled:
            return
        
        metrics = {}
        
        # Logits statistics (skip if LM was not run, e.g., Stage 1 alignment-only)
        logits = outputs.get('logits')
        if logits is not None:
            metrics['language/logits_mean'] = logits.mean().item()
            metrics['language/logits_std'] = logits.std().item()
            metrics['language/logits_max'] = logits.max().item()
            
            # Token prediction confidence (max probability)
            probs = torch.softmax(logits, dim=-1)
            max_probs = probs.max(dim=-1)[0]
            metrics['language/prediction_confidence_mean'] = max_probs.mean().item()
            metrics['language/prediction_confidence_std'] = max_probs.std().item()
            
            # Vocabulary usage (unique predicted tokens)
            predicted_tokens = logits.argmax(dim=-1)
            unique_tokens = torch.unique(predicted_tokens).numel()
            metrics['language/unique_predicted_tokens'] = unique_tokens
        
        # Hidden states statistics (if available)
        hidden_states = outputs.get('hidden_states')
        if hidden_states is not None and len(hidden_states) > 0:
            last_hidden = hidden_states[-1]
            metrics['language/hidden_state_mean'] = last_hidden.mean().item()
            metrics['language/hidden_state_std'] = last_hidden.std().item()
            metrics['language/hidden_state_norm'] = torch.norm(last_hidden, dim=-1).mean().item()
        
        # Only log if we have metrics to log
        if metrics:
            self.wandb_run.log(metrics, step=global_step)
    
    def log_alignment_metrics(self, image_features, text_features, global_step, 
                             num_samples=4, alignment_loss_module=None):
        """
        Log multimodal alignment visualizations
        
        Args:
            image_features: (B, hidden_dim)
            text_features: (B, hidden_dim)
            global_step: global training step
            num_samples: number of samples for similarity matrix
            alignment_loss_module: optional reference to ContrastiveAlignmentLoss for temperature logging
        """
        if not self.enabled:
            return
        
        with torch.no_grad():
            # Sample for visualization
            img_feat_sample = image_features[:num_samples]
            text_feat_sample = text_features[:num_samples]
            
            # Normalize features
            img_feat_norm = torch.nn.functional.normalize(img_feat_sample, dim=-1)
            text_feat_norm = torch.nn.functional.normalize(text_feat_sample, dim=-1)
            
            # Compute similarity matrix
            similarity = torch.matmul(img_feat_norm, text_feat_norm.t())
            similarity_np = similarity.cpu().numpy()
            
            # Create heatmap
            fig, ax = plt.subplots(figsize=(8, 6))
            sns.heatmap(similarity_np, annot=True, fmt='.3f', cmap='RdYlGn',
                       center=0, vmin=-1, vmax=1, ax=ax,
                       xticklabels=[f'Text {i}' for i in range(num_samples)],
                       yticklabels=[f'Image {i}' for i in range(num_samples)])
            ax.set_title('Image-Text Similarity Matrix')
            
            self.wandb_run.log({
                'alignment/similarity_matrix': wandb.Image(fig)
            }, step=global_step)
            
            plt.close(fig)
            
            # Diagonal similarity (correct pairs)
            diagonal_sim = torch.diag(similarity).mean().item()
            off_diagonal_sim = (similarity.sum() - torch.diag(similarity).sum()) / (num_samples * (num_samples - 1))
            
            metrics = {
                'alignment/correct_pair_similarity': diagonal_sim,
                'alignment/incorrect_pair_similarity': off_diagonal_sim.item(),
                'alignment/similarity_gap': diagonal_sim - off_diagonal_sim.item(),
                'alignment/image_feature_norm': torch.norm(image_features, dim=-1).mean().item(),
                'alignment/text_feature_norm': torch.norm(text_features, dim=-1).mean().item(),
                'alignment/feature_norm_ratio': torch.norm(image_features, dim=-1).mean().item() / (torch.norm(text_features, dim=-1).mean().item() + 1e-8)
            }
            
            # Log learnable temperature if available
            if alignment_loss_module is not None and hasattr(alignment_loss_module, 'logit_scale'):
                logit_scale = alignment_loss_module.logit_scale.exp().item()
                metrics['alignment/temperature'] = 1.0 / logit_scale  # Convert logit_scale to temperature
                metrics['alignment/logit_scale'] = logit_scale
            
            self.wandb_run.log(metrics, step=global_step)
    
    def log_text_to_patch_attention_metrics(self, text_to_patch_attn, attention_mask, global_step):
        """
        Log quantitative metrics for text-to-patch attention quality.
        
        These metrics help diagnose whether text tokens are learning to attend to
        semantically relevant image regions.
        
        Args:
            text_to_patch_attn: (B, seq_len, num_patches) attention weights
            attention_mask: (B, seq_len) text attention mask
            global_step: training step
        """
        if not self.enabled or text_to_patch_attn is None:
            return
        
        with torch.no_grad():
            B, seq_len, num_patches = text_to_patch_attn.shape
            
            # 1. Attention entropy (lower = more focused)
            eps = 1e-8
            entropy = -(text_to_patch_attn * torch.log(text_to_patch_attn + eps)).sum(dim=-1)
            max_entropy = np.log(num_patches)
            
            if attention_mask is not None:
                # Average over valid tokens only
                valid_tokens = attention_mask.sum(dim=-1).clamp(min=1)
                mean_entropy = ((entropy * attention_mask).sum(dim=-1) / valid_tokens).mean().item()
            else:
                mean_entropy = entropy.mean().item()
            
            normalized_entropy = mean_entropy / max_entropy
            
            # 2. Attention sparsity (fraction of patches with attention > threshold)
            threshold = 1.0 / num_patches  # Uniform attention level
            sparse_ratio = (text_to_patch_attn > threshold * 2).float().mean().item()
            
            # 3. Top-k attention concentration (what fraction of attention is in top-k patches)
            k = min(10, num_patches // 4)  # Top 10 or 25% of patches
            topk_attn, _ = text_to_patch_attn.topk(k, dim=-1)
            topk_concentration = topk_attn.sum(dim=-1).mean().item()
            
            # 4. Attention diversity across tokens
            # Different tokens should attend to different patches
            mean_attn_per_patch = text_to_patch_attn.mean(dim=1)  # (B, num_patches)
            patch_usage_variance = mean_attn_per_patch.var(dim=-1).mean().item()
            
            # 5. Spatial coherence - do nearby patches get similar attention?
            # Reshape to spatial grid (14x14 for 196 patches)
            grid_size = int(np.sqrt(num_patches))
            if grid_size * grid_size == num_patches:
                attn_grid = text_to_patch_attn.mean(dim=1).view(B, grid_size, grid_size)
                # Compute horizontal and vertical gradients
                h_grad = (attn_grid[:, :, 1:] - attn_grid[:, :, :-1]).abs().mean().item()
                v_grad = (attn_grid[:, 1:, :] - attn_grid[:, :-1, :]).abs().mean().item()
                spatial_smoothness = 1.0 - (h_grad + v_grad) / 2  # Higher = smoother/more coherent
            else:
                spatial_smoothness = 0.0
            
            metrics = {
                'attention/text_to_patch_entropy': mean_entropy,
                'attention/text_to_patch_entropy_normalized': normalized_entropy,
                'attention/text_to_patch_sparsity': 1.0 - sparse_ratio,  # Higher = more sparse
                'attention/topk_concentration': topk_concentration,
                'attention/patch_usage_variance': patch_usage_variance,
                'attention/spatial_coherence': spatial_smoothness
            }
            
            self.wandb_run.log(metrics, step=global_step)
    
    def log_memory_heatmap(self, memory_state, addressing_weights, global_step, episodic_memory_module=None):
        """
        Create and log episodic memory temporal evolution heatmaps.
        Shows how memory slots evolve across training rather than single snapshots.
        IMPORTANT: Visualizes QUANTIZED values (4-bit) when quantization is enabled.

        Args:
            memory_state: tuple (mean, cov) from episodic memory
            addressing_weights: (episode_size, batch, memory_size)
            global_step: global training step
            episodic_memory_module: Optional episodic memory module for quantization stats
        """
        if not self.enabled:
            return
        
        with torch.no_grad():
            memory_mean, memory_cov = memory_state
            
            # Check if memory is quantized and use quantized values for visualization
            use_quantized = False
            if episodic_memory_module is not None and hasattr(episodic_memory_module, 'quantized_memory_slots'):
                use_quantized = True
                quant_memory = episodic_memory_module.quantized_memory_slots
                # Use QUANTIZED values (4-bit INT8) for visualization
                memory_mean_quantized = quant_memory.memory_mean_quantized.cpu().numpy()
                # Compute norms from quantized values (preserves quantization visibility)
                memory_norms = np.linalg.norm(memory_mean_quantized, axis=1).astype(np.float32)
            else:
                # Fallback to dequantized/full-precision values
                memory_mean_np = memory_mean[0].cpu().numpy()  # Take first batch
                memory_norms = np.linalg.norm(memory_mean_np, axis=1)

            # Average addressing weights over batch and episode
            addr_weights_np = addressing_weights.mean(dim=1).cpu().numpy()
            avg_addressing = addr_weights_np.mean(axis=0)

            # Store snapshot for temporal evolution (sample every 5000 steps for efficiency)
            if global_step % 5000 == 0 or len(self.memory_states) == 0:
                self.memory_states.append((global_step, memory_norms.copy()))
                self.memory_addressing_history.append((global_step, avg_addressing.copy()))

            # Log numerical metrics
            avg_addressing_nonzero = avg_addressing[avg_addressing > 1e-10]
            if len(avg_addressing_nonzero) > 0:
                avg_addressing_prob = avg_addressing_nonzero / np.sum(avg_addressing_nonzero)
                addressing_entropy = -np.sum(avg_addressing_prob * np.log(avg_addressing_prob))
            else:
                addressing_entropy = 0.0
            
            metrics = {
                'memory/mean_activation': np.mean(memory_norms),
                'memory/max_activation': np.max(memory_norms),
                'memory/min_activation': np.min(memory_norms),
                'memory/active_slots': np.sum(memory_norms > 0.1),
                'memory/addressing_entropy': addressing_entropy,
                'memory/addressing_sparsity': np.sum(avg_addressing < 0.01) / len(avg_addressing),
                'memory/is_quantized': 1.0 if use_quantized else 0.0
            }

            # Add quantization-specific metrics if available
            if use_quantized and episodic_memory_module is not None:
                from ..quantization.quantized_episodic_memory import get_memory_quantization_stats
                quant_stats = get_memory_quantization_stats(episodic_memory_module)
                metrics.update(quant_stats)

            self.wandb_run.log(metrics, step=global_step)
            
            # Generate temporal evolution heatmaps when we have multiple snapshots
            if len(self.memory_states) >= 2:
                self._visualize_memory_temporal_evolution(global_step)

    def _visualize_memory_temporal_evolution(self, global_step):
        """
        Create temporal evolution heatmaps showing how memory slots change across training.
        Replaces 4-panel static diagnostics with side-by-side progression view.

        Args:
            global_step: current training step
        """
        if len(self.memory_states) < 2:
            return
        
        try:
            # Extract snapshots
            steps = np.array([s for s, _ in self.memory_states])
            num_snapshots = len(steps)

            # Memory Slot Activation Evolution (Primary Visualization)
            memory_activations = np.array([norms for _, norms in self.memory_states])  # (num_snapshots, num_slots)
            num_slots = memory_activations.shape[1]

            # Create side-by-side heatmap figure
            fig = plt.figure(figsize=(20, 8))

            # 1. Main heatmap: Memory Slot Evolution (Y=slots, X=time)
            ax1 = plt.subplot(2, 1, 1)
            im1 = ax1.imshow(memory_activations.T, aspect='auto', cmap='RdYlBu_r',
                            interpolation='nearest')
            ax1.set_xlabel('Training Snapshot', fontsize=12)
            ax1.set_ylabel('Memory Slot ID', fontsize=12)
            ax1.set_title(f'Memory Slot Activation Evolution (Step {global_step})', fontsize=14, fontweight='bold')

            # Set x-axis to show actual steps
            tick_indices = np.linspace(0, num_snapshots-1, min(8, num_snapshots), dtype=int)
            ax1.set_xticks(tick_indices)
            ax1.set_xticklabels([f'{steps[i]:,}' for i in tick_indices], rotation=45)

            # Colorbar
            cbar1 = plt.colorbar(im1, ax=ax1)
            cbar1.set_label('L2 Norm Activation', fontsize=10)

            # 2. Addressing Weight Evolution (Secondary Visualization)
            if len(self.memory_addressing_history) >= 2:
                ax2 = plt.subplot(2, 1, 2)
                addressing_evolution = np.array([addr for _, addr in self.memory_addressing_history])  # (num_snapshots, num_slots)

                im2 = ax2.imshow(addressing_evolution.T, aspect='auto', cmap='YlOrRd',
                               interpolation='nearest')
                ax2.set_xlabel('Training Snapshot', fontsize=12)
                ax2.set_ylabel('Memory Slot ID', fontsize=12)
                ax2.set_title('Memory Addressing Weight Evolution', fontsize=14, fontweight='bold')

                ax2.set_xticks(tick_indices)
                ax2.set_xticklabels([f'{steps[i]:,}' for i in tick_indices], rotation=45)

                cbar2 = plt.colorbar(im2, ax=ax2)
                cbar2.set_label('Avg. Addressing Weight', fontsize=10)

            plt.tight_layout()

            # Log to WandB
            self.wandb_run.log({
                'memory/temporal_evolution_heatmap': wandb.Image(fig)
            }, step=global_step)

            plt.close(fig)

            # Additional temporal metrics
            # Track how many slots become active over time
            active_slots_over_time = (memory_activations > 0.1).sum(axis=1)
            final_active_slots = active_slots_over_time[-1]
            initial_active_slots = active_slots_over_time[0]

            # Track variance in slot usage (higher = more diverse usage)
            slot_variance = memory_activations.var(axis=0).mean()

            # Track temporal stability (lower = more stable memory)
            if num_snapshots >= 3:
                temporal_stability = np.mean([
                    np.linalg.norm(memory_activations[i+1] - memory_activations[i])
                    for i in range(num_snapshots-1)
                ])
            else:
                temporal_stability = 0.0

            self.wandb_run.log({
                'memory/active_slots_final': final_active_slots,
                'memory/active_slots_initial': initial_active_slots,
                'memory/slot_usage_variance': slot_variance,
                'memory/temporal_stability': temporal_stability
            }, step=global_step)

        except Exception as e:
            warnings.warn(f"Failed to create temporal evolution visualization: {e}")

    def log_cross_modal_attention(self, attention_weights, global_step,
                                  text_tokens=None, prefix="alignment"):
        """
        Log cross-modal attention heatmap
        
        Args:
            attention_weights: (seq_len, k_prefix) or (batch, seq_len, k_prefix)
            global_step: global training step
            text_tokens: optional text token strings
            prefix: metric prefix for WandB grouping (default: "alignment")
        """
        if not self.enabled:
            return
        
        # Handle batch dimension
        if attention_weights.dim() == 3:
            attention_weights = attention_weights[0]
        
        attn_np = attention_weights.cpu().detach().numpy()
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        im = ax.imshow(attn_np, cmap='viridis', aspect='auto')
        ax.set_xlabel('Image Prefix Tokens')
        ax.set_ylabel('Text Tokens')
        ax.set_title(f'Cross-Modal Attention (Step {global_step})')
        
        plt.colorbar(im, ax=ax)
        
        if text_tokens is not None:
            ax.set_yticks(range(len(text_tokens)))
            ax.set_yticklabels(text_tokens, fontsize=8)
        
        plt.tight_layout()
        
        self.wandb_run.log({
            f'{prefix}/cross_modal_attention': wandb.Image(fig)
        }, step=global_step)
        
        plt.close(fig)
        
        # Attention statistics
        metrics = {
            'alignment/attention_mean': attn_np.mean(),
            'alignment/attention_max': attn_np.max(),
            'alignment/attention_entropy': -np.sum(attn_np * np.log(attn_np + 1e-10), axis=-1).mean()
        }
        
        self.wandb_run.log(metrics, step=global_step)
    
    def log_model_weights_histogram(self, model, global_step):
        """
        Log histograms of model weights
        
        Args:
            model: the model
            global_step: global training step
        """
        if not self.enabled:
            return
        
        for name, param in model.named_parameters():
            if param.requires_grad:
                tensor = param.detach().cpu()
                weights = tensor.numpy()
                if not np.isfinite(weights).all():
                    # Log the count of invalid values for easier debugging
                    invalid = np.logical_not(np.isfinite(weights))
                    self.wandb_run.log({
                        f'weights/{name}_invalid_count': int(invalid.sum())
                    }, step=global_step)
                    weights = np.nan_to_num(weights, nan=0.0, posinf=0.0, neginf=0.0)
                self.wandb_run.log({
                    f'weights/{name}': wandb.Histogram(weights)
                }, step=global_step)
    
    def log_metrics(self, metrics: Dict[str, float], step: int, prefix: str = ""):
        """
        Generic method to log arbitrary metrics
        
        Args:
            metrics: dictionary of metric names to values
            step: global step
            prefix: optional prefix to add to all metric names
        """
        if not self.enabled:
            return
        
        if prefix:
            metrics = {f"{prefix}/{k}": v for k, v in metrics.items()}
        
        self.wandb_run.log(metrics, step=step)
    
    def log_image(self, image, name: str, step: int):
        """
        Log a single image to WandB
        
        Args:
            image: torch.Tensor (C, H, W) or (1, C, H, W) or PIL Image
            name: name for the image in WandB
            step: global step
        """
        if not self.enabled:
            return
        
        # Convert tensor to numpy if needed
        if torch.is_tensor(image):
            # Handle batch dimension
            if image.dim() == 4:
                image = image[0]  # Take first image in batch
            
            # Convert to HWC format for WandB
            if image.shape[0] in [1, 3]:  # CHW format
                image = image.permute(1, 2, 0)
            
            # Denormalize if needed (assume ImageNet normalization)
            image = image.cpu().numpy()
            if image.max() <= 1.0 and image.min() >= -1.0:
                # Denormalize ImageNet
                mean = np.array([0.485, 0.456, 0.406])
                std = np.array([0.229, 0.224, 0.225])
                image = image * std + mean
                image = np.clip(image, 0, 1)
        
        self.wandb_run.log({name: wandb.Image(image)}, step=step)
