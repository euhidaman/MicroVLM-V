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
    
    def log_memory_heatmap(self, memory_state, addressing_weights, global_step):
        """
        Create and log episodic memory heatmap visualization
        
        Args:
            memory_state: tuple (mean, cov) from episodic memory
            addressing_weights: (episode_size, batch, memory_size)
            global_step: global training step
        """
        if not self.enabled:
            return
        
        with torch.no_grad():
            memory_mean, memory_cov = memory_state
            
            # Memory mean heatmap (memory_size x code_size)
            memory_mean_np = memory_mean[0].cpu().numpy()  # Take first batch
            
            fig, axes = plt.subplots(2, 2, figsize=(16, 12))
            
            # 1. Memory state heatmap
            ax = axes[0, 0]
            im1 = ax.imshow(memory_mean_np, aspect='auto', cmap='viridis')
            ax.set_title(f'Memory State (Step {global_step})')
            ax.set_xlabel('Code Dimension')
            ax.set_ylabel('Memory Slots')
            plt.colorbar(im1, ax=ax)
            
            # 2. Memory activation (norm per slot)
            ax = axes[0, 1]
            memory_norms = np.linalg.norm(memory_mean_np, axis=1)
            ax.bar(range(len(memory_norms)), memory_norms)
            ax.set_title('Memory Slot Activation')
            ax.set_xlabel('Memory Slot')
            ax.set_ylabel('L2 Norm')
            ax.grid(True, alpha=0.3)
            
            # 3. Addressing weights heatmap
            ax = axes[1, 0]
            # Average over batch: (episode_size, memory_size)
            addr_weights_np = addressing_weights.mean(dim=1).cpu().numpy()
            im3 = ax.imshow(addr_weights_np, aspect='auto', cmap='hot')
            ax.set_title('Addressing Weights')
            ax.set_xlabel('Memory Slots')
            ax.set_ylabel('Episode Steps')
            plt.colorbar(im3, ax=ax)
            
            # 4. Memory usage distribution
            ax = axes[1, 1]
            avg_addressing = addr_weights_np.mean(axis=0)
            ax.bar(range(len(avg_addressing)), avg_addressing)
            ax.set_title('Average Memory Usage')
            ax.set_xlabel('Memory Slot')
            ax.set_ylabel('Average Addressing Weight')
            ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            self.wandb_run.log({
                'memory/state_heatmap': wandb.Image(fig)
            }, step=global_step)
            
            plt.close(fig)
            
            # Memory statistics with numerical stability
            # Compute entropy safely by filtering out near-zero values
            avg_addressing_nonzero = avg_addressing[avg_addressing > 1e-10]
            if len(avg_addressing_nonzero) > 0:
                # Normalize only the non-zero values
                avg_addressing_prob = avg_addressing_nonzero / np.sum(avg_addressing_nonzero)
                addressing_entropy = -np.sum(avg_addressing_prob * np.log(avg_addressing_prob))
            else:
                # If all values are zero, entropy is 0
                addressing_entropy = 0.0
            
            metrics = {
                'memory/mean_activation': np.mean(memory_norms),
                'memory/max_activation': np.max(memory_norms),
                'memory/min_activation': np.min(memory_norms),
                'memory/active_slots': np.sum(memory_norms > 0.1),  # Threshold for "active"
                'memory/addressing_entropy': addressing_entropy,
                'memory/addressing_sparsity': np.sum(avg_addressing < 0.01) / len(avg_addressing)
            }
            
            self.wandb_run.log(metrics, step=global_step)
            
            # Store for temporal visualization
            self.memory_states.append((global_step, memory_mean_np.copy()))
            self.memory_addressing_history.append((global_step, avg_addressing.copy()))
            
            # Create temporal evolution plot every N steps
            if len(self.memory_states) > 1 and global_step % 1000 == 0:
                self._visualize_memory_evolution(global_step)
    
    def _visualize_memory_evolution(self, global_step):
        """
        Visualize how memory evolves over training
        
        Args:
            global_step: global training step
        """
        if len(self.memory_states) < 2:
            return
        
        # Extract data
        steps = [s for s, _ in self.memory_states]
        
        # Plot memory slot activation over time
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Get activation norms for each checkpoint
        activations = []
        for _, mem_state in self.memory_states:
            norms = np.linalg.norm(mem_state, axis=1)
            activations.append(norms)
        
        activations = np.array(activations).T  # (memory_slots, time_steps)
        
        # Plot each memory slot
        for i in range(min(20, activations.shape[0])):  # Plot first 20 slots
            ax.plot(steps, activations[i], alpha=0.6, label=f'Slot {i}' if i < 5 else None)
        
        ax.set_xlabel('Training Step')
        ax.set_ylabel('Memory Slot Activation')
        ax.set_title('Memory Learning Evolution')
        ax.grid(True, alpha=0.3)
        if activations.shape[0] <= 5:
            ax.legend()
        
        self.wandb_run.log({
            'memory/evolution_over_time': wandb.Image(fig)
        }, step=global_step)
        
        plt.close(fig)
    
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
