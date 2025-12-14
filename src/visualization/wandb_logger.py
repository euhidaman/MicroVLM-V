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
        # ALWAYS PRINT - to verify this file version is loaded
        print(f"[WandBLogger.__init__] INITIALIZING - wandb_run={wandb_run is not None}, type={type(wandb_run)}")

        self.config = config
        self.wandb_run = wandb_run
        self.enabled = wandb_run is not None
        
        print(f"[WandBLogger.__init__] enabled={self.enabled}")

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
        # CRITICAL DEBUG: Print immediately to verify function is called
        if global_step % 50 == 0:
            print(f"[WandBLogger] log_training_metrics CALLED at step {global_step}, enabled={self.enabled}, wandb_run={type(self.wandb_run)}")

        if not self.enabled:
            if global_step % 100 == 0:  # Only print warning periodically
                print(f"[WandBLogger] WARNING: Logger not enabled (wandb_run={self.wandb_run})")
            return
        
        try:
            # CRITICAL DEBUG: Show what we're working with
            if global_step % 50 == 0:
                print(f"[WandBLogger] outputs type: {type(outputs)}, keys: {list(outputs.keys()) if isinstance(outputs, dict) else 'N/A'}")

            metrics = {
                'train/epoch': epoch,
                'train/global_step': global_step,
                'train/learning_rate': optimizer.param_groups[0]['lr']
            }

            # Total loss - log as both 'train/loss' and 'train/total_loss' for compatibility
            # Handle both dict keys and direct tensor values
            loss_value = None
            if 'loss' in outputs:
                loss_tensor = outputs['loss']
                # Ensure we can extract the value even if it's a quantized tensor
                if hasattr(loss_tensor, 'item'):
                    loss_value = loss_tensor.item()
                elif torch.is_tensor(loss_tensor):
                    loss_value = float(loss_tensor.detach().cpu())
                else:
                    loss_value = float(loss_tensor)

                metrics['train/loss'] = loss_value  # Primary metric for wandb charts
                metrics['train/total_loss'] = loss_value  # Alias for clarity

                # DEBUG
                if global_step % 50 == 0:
                    print(f"[WandBLogger] Extracted loss value: {loss_value}")
            else:
                # Try alternative loss keys
                for alt_key in ['total_loss', 'lm_loss']:
                    if alt_key in outputs and outputs[alt_key] is not None:
                        loss_tensor = outputs[alt_key]
                        if hasattr(loss_tensor, 'item'):
                            loss_value = loss_tensor.item()
                        elif torch.is_tensor(loss_tensor):
                            loss_value = float(loss_tensor.detach().cpu())
                        else:
                            loss_value = float(loss_tensor)
                        metrics['train/loss'] = loss_value
                        metrics['train/total_loss'] = loss_value
                        break

                if loss_value is None and global_step % 100 == 0:
                    print(f"[WandBLogger] WARNING: No valid loss found. Available keys: {list(outputs.keys())}")

            # LM loss
            lm_loss = outputs.get('lm_loss')
            if lm_loss is not None:
                try:
                    if hasattr(lm_loss, 'item'):
                        lm_loss_val = lm_loss.item()
                    elif torch.is_tensor(lm_loss):
                        lm_loss_val = float(lm_loss.detach().cpu())
                    else:
                        lm_loss_val = float(lm_loss)

                    metrics['train/lm_loss'] = lm_loss_val
                    # Calculate perplexity safely
                    try:
                        perplexity = torch.exp(torch.tensor(lm_loss_val)).item()
                        # Clamp perplexity to reasonable range
                        perplexity = min(perplexity, 1e6)
                        metrics['train/perplexity'] = perplexity
                    except:
                        pass  # Skip perplexity if calculation fails
                except Exception as e:
                    if global_step % 100 == 0:
                        print(f"[WandBLogger] WARNING: Failed to log lm_loss: {e}")

            # Alignment loss
            if 'alignment_loss' in outputs and outputs['alignment_loss'] is not None:
                try:
                    align_loss = outputs['alignment_loss']
                    if hasattr(align_loss, 'item'):
                        metrics['alignment/loss'] = align_loss.item()
                    elif torch.is_tensor(align_loss):
                        metrics['alignment/loss'] = float(align_loss.detach().cpu())
                    else:
                        metrics['alignment/loss'] = float(align_loss)
                except Exception as e:
                    if global_step % 100 == 0:
                        print(f"[WandBLogger] WARNING: Failed to log alignment_loss: {e}")

            # Fine-grained alignment loss (text-to-patch attention supervision)
            if 'fine_grained_loss' in outputs and outputs['fine_grained_loss'] is not None:
                try:
                    fg_loss = outputs['fine_grained_loss']
                    if hasattr(fg_loss, 'item'):
                        metrics['alignment/fine_grained_loss'] = fg_loss.item()
                    elif torch.is_tensor(fg_loss):
                        metrics['alignment/fine_grained_loss'] = float(fg_loss.detach().cpu())
                except Exception as e:
                    if global_step % 100 == 0:
                        print(f"[WandBLogger] WARNING: Failed to log fine_grained_loss: {e}")

            # Memory losses - handle both quantized and non-quantized memory
            if 'memory_kl' in outputs and outputs['memory_kl'] is not None:
                try:
                    mem_kl = outputs['memory_kl']
                    if hasattr(mem_kl, 'item'):
                        metrics['memory/kl_divergence'] = mem_kl.item()
                    elif torch.is_tensor(mem_kl):
                        metrics['memory/kl_divergence'] = float(mem_kl.detach().cpu())
                    else:
                        metrics['memory/kl_divergence'] = float(mem_kl)
                except Exception as e:
                    if global_step % 100 == 0:
                        print(f"[WandBLogger] WARNING: Failed to log memory_kl: {e}")

            if 'addressing_kl' in outputs and outputs['addressing_kl'] is not None:
                try:
                    addr_kl = outputs['addressing_kl']
                    if hasattr(addr_kl, 'item'):
                        metrics['memory/addressing_kl'] = addr_kl.item()
                    elif torch.is_tensor(addr_kl):
                        metrics['memory/addressing_kl'] = float(addr_kl.detach().cpu())
                    else:
                        metrics['memory/addressing_kl'] = float(addr_kl)
                except Exception as e:
                    if global_step % 100 == 0:
                        print(f"[WandBLogger] WARNING: Failed to log addressing_kl: {e}")

            # Scope probabilities (if present)
            if 'scope_probs' in outputs and outputs['scope_probs'] is not None:
                try:
                    scope_probs = outputs['scope_probs']
                    if torch.is_tensor(scope_probs) and scope_probs.numel() > 0:
                        metrics['memory/scope_prob_mean'] = float(scope_probs.mean().detach().cpu())
                        metrics['memory/scope_prob_std'] = float(scope_probs.std().detach().cpu())
                except Exception as e:
                    if global_step % 100 == 0:
                        print(f"[WandBLogger] WARNING: Failed to log scope_probs: {e}")

            # ITC/ITM losses (FIBER mode)
            if 'itc_loss' in outputs and outputs['itc_loss'] is not None:
                try:
                    itc_loss = outputs['itc_loss']
                    if hasattr(itc_loss, 'item'):
                        metrics['alignment/itc_loss'] = itc_loss.item()
                    elif torch.is_tensor(itc_loss):
                        metrics['alignment/itc_loss'] = float(itc_loss.detach().cpu())
                    else:
                        metrics['alignment/itc_loss'] = float(itc_loss)
                except Exception as e:
                    if global_step % 100 == 0:
                        print(f"[WandBLogger] WARNING: Failed to log itc_loss: {e}")

            if 'itm_loss' in outputs and outputs['itm_loss'] is not None:
                try:
                    itm_loss = outputs['itm_loss']
                    if hasattr(itm_loss, 'item'):
                        metrics['alignment/itm_loss'] = itm_loss.item()
                    elif torch.is_tensor(itm_loss):
                        metrics['alignment/itm_loss'] = float(itm_loss.detach().cpu())
                    else:
                        metrics['alignment/itm_loss'] = float(itm_loss)
                except Exception as e:
                    if global_step % 100 == 0:
                        print(f"[WandBLogger] WARNING: Failed to log itm_loss: {e}")

            if 'token_loss' in outputs and outputs['token_loss'] is not None:
                try:
                    token_loss = outputs['token_loss']
                    if hasattr(token_loss, 'item'):
                        metrics['alignment/token_loss'] = token_loss.item()
                    elif torch.is_tensor(token_loss):
                        metrics['alignment/token_loss'] = float(token_loss.detach().cpu())
                    else:
                        metrics['alignment/token_loss'] = float(token_loss)
                except Exception as e:
                    if global_step % 100 == 0:
                        print(f"[WandBLogger] WARNING: Failed to log token_loss: {e}")

            # Anti-collapse regularization losses
            if 'anti_collapse_loss' in outputs and outputs['anti_collapse_loss'] is not None:
                try:
                    anti_collapse_loss = outputs['anti_collapse_loss']
                    if hasattr(anti_collapse_loss, 'item'):
                        val = anti_collapse_loss.item()
                    elif torch.is_tensor(anti_collapse_loss):
                        val = float(anti_collapse_loss.detach().cpu())
                    else:
                        val = float(anti_collapse_loss)
                    metrics['regularization/anti_collapse_loss'] = val
                except Exception as e:
                    if global_step % 100 == 0:
                        print(f"[WandBLogger] WARNING: Failed to log anti_collapse_loss: {e}")

            if 'attention_entropy_loss' in outputs and outputs['attention_entropy_loss'] is not None:
                try:
                    attn_entropy_loss = outputs['attention_entropy_loss']
                    if hasattr(attn_entropy_loss, 'item'):
                        val = attn_entropy_loss.item()
                    elif torch.is_tensor(attn_entropy_loss):
                        val = float(attn_entropy_loss.detach().cpu())
                    else:
                        val = float(attn_entropy_loss)
                    metrics['regularization/attention_entropy_loss'] = val
                except Exception as e:
                    if global_step % 100 == 0:
                        print(f"[WandBLogger] WARNING: Failed to log attention_entropy_loss: {e}")

            # Log metrics to wandb with error handling
            try:
                # CRITICAL DEBUG: Always print before logging
                if global_step % 50 == 0:
                    print(f"[WandBLogger] About to log {len(metrics)} metrics to wandb at step {global_step}")
                    print(f"[WandBLogger]   Metrics keys: {list(metrics.keys())}")
                    if 'train/loss' in metrics:
                        print(f"[WandBLogger]   train/loss = {metrics['train/loss']:.4f}")

                # Force commit=True to ensure metrics are immediately sent to WandB
                self.wandb_run.log(metrics, step=global_step, commit=True)

                # Debug: Print confirmation every 50 steps now
                if global_step % 50 == 0:
                    print(f"[WandBLogger] ✓✓✓ SUCCESSFULLY logged {len(metrics)} metrics at step {global_step}")
                    if 'train/loss' in metrics:
                        print(f"[WandBLogger]   train/loss = {metrics['train/loss']:.4f}")
                    if 'memory/kl_divergence' in metrics:
                        print(f"[WandBLogger]   memory/kl_divergence = {metrics['memory/kl_divergence']:.4f}")
            except Exception as e:
                print(f"[WandBLogger] ❌ ERROR: Failed to log metrics to wandb at step {global_step}: {e}")
                print(f"[WandBLogger]   Attempted to log {len(metrics)} metrics")
                print(f"[WandBLogger]   wandb_run type: {type(self.wandb_run)}")
                import traceback
                traceback.print_exc()

        except Exception as e:
            print(f"[WandBLogger] ERROR in log_training_metrics at step {global_step}: {e}")
            print(f"[WandBLogger]   outputs keys: {list(outputs.keys()) if isinstance(outputs, dict) else 'not a dict'}")
            import traceback
            traceback.print_exc()

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
        Shows how memory slots evolve across training with progressive warm color gradients.
        IMPORTANT: Visualizes QUANTIZED values (4-bit) when quantization is enabled.

        Args:
            memory_state: tuple (mean, cov) from episodic memory
            addressing_weights: (episode_size, batch, memory_size)
            global_step: global training step
            episodic_memory_module: Optional episodic memory module for quantization stats
        """
        if not self.enabled:
            return
        
        try:
            with torch.no_grad():
                memory_mean, memory_cov = memory_state

                # Check if memory is quantized and use quantized values for visualization
                use_quantized = False
                memory_norms = None

                if episodic_memory_module is not None and hasattr(episodic_memory_module, 'quantized_memory_slots'):
                    try:
                        use_quantized = True
                        quant_memory = episodic_memory_module.quantized_memory_slots

                        # Try to dequantize for proper visualization
                        try:
                            dequant_mean, _ = quant_memory.dequantize_memory()
                            memory_norms = torch.norm(dequant_mean, dim=-1).detach().cpu().numpy().astype(np.float32)
                        except Exception as e:
                            # Fallback: Use packed quantized values directly
                            if hasattr(quant_memory, 'memory_mean_quantized'):
                                memory_mean_quantized = quant_memory.memory_mean_quantized.detach().cpu().numpy()
                                # Compute norms from packed values
                                memory_norms = np.linalg.norm(memory_mean_quantized, axis=1).astype(np.float32)
                            else:
                                raise e

                        if global_step % 100 == 0:
                            print(f"[WandBLogger] Using quantized memory for heatmap (use_quantized={use_quantized})")

                    except Exception as e:
                        if global_step % 100 == 0:
                            print(f"[WandBLogger] WARNING: Failed to extract quantized memory, using fallback: {e}")
                        use_quantized = False

                # Fallback to non-quantized memory
                if memory_norms is None:
                    if memory_mean.dim() == 3:
                        memory_mean_np = memory_mean[0].detach().cpu().numpy()  # Take first batch
                    else:
                        memory_mean_np = memory_mean.detach().cpu().numpy()
                    memory_norms = np.linalg.norm(memory_mean_np, axis=1).astype(np.float32)

                # Average addressing weights over batch and episode
                addr_weights_np = addressing_weights.mean(dim=1).detach().cpu().numpy()
                avg_addressing = addr_weights_np.mean(axis=0)

                # Store snapshot for temporal evolution (log every step for step-wise visibility)
                # Only store every N steps to avoid memory explosion but ensure visibility
                should_store = (global_step % 100 == 0) or len(self.memory_states) == 0
                if should_store:
                    self.memory_states.append((global_step, memory_norms.copy()))
                    self.memory_addressing_history.append((global_step, avg_addressing.copy()))

                    # Limit history size to prevent memory issues (keep last 500 snapshots)
                    if len(self.memory_states) > 500:
                        self.memory_states = self.memory_states[-500:]
                        self.memory_addressing_history = self.memory_addressing_history[-500:]

                # Log numerical metrics at every interval
                avg_addressing_nonzero = avg_addressing[avg_addressing > 1e-10]
                if len(avg_addressing_nonzero) > 0:
                    avg_addressing_prob = avg_addressing_nonzero / np.sum(avg_addressing_nonzero)
                    addressing_entropy = -np.sum(avg_addressing_prob * np.log(avg_addressing_prob + 1e-10))
                else:
                    addressing_entropy = 0.0

                metrics = {
                    'memory/mean_activation': float(np.mean(memory_norms)),
                    'memory/max_activation': float(np.max(memory_norms)),
                    'memory/min_activation': float(np.min(memory_norms)),
                    'memory/active_slots': int(np.sum(memory_norms > 0.1)),
                    'memory/addressing_entropy': float(addressing_entropy),
                    'memory/addressing_sparsity': float(np.sum(avg_addressing < 0.01) / len(avg_addressing)),
                    'memory/is_quantized': 1.0 if use_quantized else 0.0,
                    'memory/num_snapshots': len(self.memory_states),  # Track snapshot count
                }

                # Add quantization-specific metrics if available
                if use_quantized and episodic_memory_module is not None:
                    try:
                        from ..quantization.quantized_episodic_memory import get_memory_quantization_stats
                        quant_stats = get_memory_quantization_stats(episodic_memory_module)
                        # Ensure all values are serializable
                        for key, value in quant_stats.items():
                            if value is not None:
                                try:
                                    if hasattr(value, 'item'):
                                        metrics[key] = value.item()
                                    elif torch.is_tensor(value):
                                        metrics[key] = float(value.detach().cpu())
                                    else:
                                        metrics[key] = float(value)
                                except:
                                    pass  # Skip non-serializable metrics
                    except Exception as e:
                        if global_step % 100 == 0:
                            print(f"[WandBLogger] WARNING: Failed to get quantization stats: {e}")

                # Log metrics with error handling
                try:
                    self.wandb_run.log(metrics, step=global_step)
                    if global_step % 100 == 0:
                        print(f"[WandBLogger] ✓ Logged {len(metrics)} memory metrics at step {global_step}")
                except Exception as e:
                    print(f"[WandBLogger] ERROR: Failed to log memory metrics at step {global_step}: {e}")

                # Generate temporal evolution heatmaps when we have multiple snapshots
                # Update visualization every 500 steps for step-wise visibility
                if len(self.memory_states) >= 2 and (global_step % 500 == 0 or global_step % 100 == 0):
                    try:
                        self._visualize_memory_temporal_evolution(global_step)
                    except Exception as e:
                        if global_step % 100 == 0:
                            print(f"[WandBLogger] WARNING: Failed to create memory heatmap visualization: {e}")

        except Exception as e:
            print(f"[WandBLogger] ERROR in log_memory_heatmap at step {global_step}: {e}")
            import traceback
            traceback.print_exc()

    def _visualize_memory_temporal_evolution(self, global_step):
        """
        Create temporal evolution heatmaps showing how memory slots change across training.
        Uses progressive warm color gradients to reflect learning dynamics over time.

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

            # Create side-by-side heatmap figure with warm color schemes
            fig = plt.figure(figsize=(20, 10))

            # 1. Main heatmap: Memory Slot Evolution over Time (Y=slots, X=time)
            # Use progressive warm gradient (YlOrRd = Yellow-Orange-Red)
            ax1 = plt.subplot(2, 1, 1)
            im1 = ax1.imshow(memory_activations.T, aspect='auto', cmap='YlOrRd',
                            interpolation='bilinear', vmin=0)
            ax1.set_xlabel('Training Step', fontsize=13, fontweight='bold')
            ax1.set_ylabel('Memory Slot ID', fontsize=13, fontweight='bold')
            ax1.set_title(f'Episodic Memory Slot Activations Over Time (Step {global_step:,})',
                         fontsize=15, fontweight='bold', pad=15)

            # Set x-axis to show actual steps with better formatting
            tick_indices = np.linspace(0, num_snapshots-1, min(10, num_snapshots), dtype=int)
            ax1.set_xticks(tick_indices)
            ax1.set_xticklabels([f'{steps[i]:,}' for i in tick_indices], rotation=45, ha='right')

            # Enhanced colorbar with better labels
            cbar1 = plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)
            cbar1.set_label('Activation Strength (L2 Norm)', fontsize=11, fontweight='bold')
            cbar1.ax.tick_params(labelsize=10)

            # Add grid for better readability
            ax1.grid(True, alpha=0.2, linestyle='--', linewidth=0.5)

            # 2. Addressing Weight Evolution with warm gradient (Secondary Visualization)
            if len(self.memory_addressing_history) >= 2:
                ax2 = plt.subplot(2, 1, 2)
                addressing_evolution = np.array([addr for _, addr in self.memory_addressing_history])

                # Use warm gradient for addressing (Oranges colormap)
                im2 = ax2.imshow(addressing_evolution.T, aspect='auto', cmap='Oranges',
                               interpolation='bilinear', vmin=0)
                ax2.set_xlabel('Training Step', fontsize=13, fontweight='bold')
                ax2.set_ylabel('Memory Slot ID', fontsize=13, fontweight='bold')
                ax2.set_title('Memory Addressing Weight Evolution',
                             fontsize=15, fontweight='bold', pad=15)

                ax2.set_xticks(tick_indices)
                ax2.set_xticklabels([f'{steps[i]:,}' for i in tick_indices], rotation=45, ha='right')

                # Enhanced colorbar
                cbar2 = plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)
                cbar2.set_label('Addressing Weight', fontsize=11, fontweight='bold')
                cbar2.ax.tick_params(labelsize=10)

                # Add grid
                ax2.grid(True, alpha=0.2, linestyle='--', linewidth=0.5)

            plt.tight_layout()

            # Log to WandB with descriptive key
            self.wandb_run.log({
                'memory/temporal_evolution_heatmap': wandb.Image(fig),
                'memory/visualization_step': global_step,
            }, step=global_step)

            plt.close(fig)

            # Additional temporal metrics with step-wise tracking
            # Track how many slots become active over time
            active_slots_over_time = (memory_activations > 0.1).sum(axis=1)
            final_active_slots = int(active_slots_over_time[-1])
            initial_active_slots = int(active_slots_over_time[0])

            # Track growth in active slots
            active_slots_growth = final_active_slots - initial_active_slots

            # Track variance in slot usage (higher = more diverse usage)
            slot_variance = float(memory_activations.var(axis=0).mean())

            # Track mean activation strength
            mean_activation_strength = float(memory_activations.mean())
            max_activation_strength = float(memory_activations.max())

            # Track temporal stability (lower = more stable memory)
            if num_snapshots >= 3:
                temporal_changes = [
                    np.linalg.norm(memory_activations[i+1] - memory_activations[i])
                    for i in range(num_snapshots-1)
                ]
                temporal_stability = float(np.mean(temporal_changes))
                temporal_volatility = float(np.std(temporal_changes))
            else:
                temporal_stability = 0.0
                temporal_volatility = 0.0

            # Log comprehensive temporal metrics
            temporal_metrics = {
                'memory/active_slots_final': final_active_slots,
                'memory/active_slots_initial': initial_active_slots,
                'memory/active_slots_growth': active_slots_growth,
                'memory/slot_usage_variance': slot_variance,
                'memory/mean_activation_strength': mean_activation_strength,
                'memory/max_activation_strength': max_activation_strength,
                'memory/temporal_stability': temporal_stability,
                'memory/temporal_volatility': temporal_volatility,
                'memory/snapshots_collected': num_snapshots,
            }

            self.wandb_run.log(temporal_metrics, step=global_step)

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
        # DEBUG: Print every 100 steps
        if step % 100 == 0:
            print(f"[WandBLogger.log_metrics] step={step}, enabled={self.enabled}, num_metrics={len(metrics)}")

        if not self.enabled:
            if step % 100 == 0:
                print(f"[WandBLogger.log_metrics] RETURNING EARLY - not enabled")
            return
        
        if prefix:
            metrics = {f"{prefix}/{k}": v for k, v in metrics.items()}
        
        # Force commit=True to ensure immediate sync
        self.wandb_run.log(metrics, step=step, commit=True)

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

    def log_quantization_metrics(self, model, global_step, config=None):
        """
        Log comprehensive quantization metrics for 4-bit and 1.58-bit quantized components.
        Monitors quantization health during training.

        Args:
            model: the model (may be wrapped in DDP)
            global_step: current training step
            config: training config (optional, for quantization flags)
        """
        if not self.enabled:
            return

        try:
            base_model = model.module if hasattr(model, 'module') else model
            metrics = {}

            # Debug print for first call
            if global_step <= 100:
                print(f"[WandBLogger] log_quantization_metrics called at step {global_step}")
                if config:
                    print(f"[WandBLogger]   quantize_memory_158bit: {getattr(config, 'quantize_memory_158bit', False)}")
                    print(f"[WandBLogger]   quantize_language_4bit: {getattr(config, 'quantize_language_4bit', False)}")
                    print(f"[WandBLogger]   quantize_vision_4bit: {getattr(config, 'quantize_vision_4bit', False)}")

            # 1.58-bit Memory Quantization Metrics
            if config and getattr(config, 'quantize_memory_158bit', False):
                if hasattr(base_model, 'episodic_memory'):
                    try:
                        from ..quantization.quantized_episodic_memory import get_memory_quantization_stats
                        quant_stats = get_memory_quantization_stats(base_model.episodic_memory)

                        # Add prefix for organized wandb grouping
                        for key, value in quant_stats.items():
                            if value is not None:
                                try:
                                    # Ensure value is a scalar
                                    if hasattr(value, 'item'):
                                        metrics[f'quantization/memory_158bit/{key}'] = value.item()
                                    elif torch.is_tensor(value):
                                        metrics[f'quantization/memory_158bit/{key}'] = float(value.detach().cpu())
                                    else:
                                        metrics[f'quantization/memory_158bit/{key}'] = float(value)
                                except Exception as e:
                                    if global_step % 100 == 0:
                                        print(f"[WandBLogger] WARNING: Failed to log quantization metric {key}: {e}")

                        # Additional runtime metrics
                        if hasattr(base_model.episodic_memory, 'quantized_memory_slots'):
                            quant_mem = base_model.episodic_memory.quantized_memory_slots

                            # Quantization value distribution
                            if hasattr(quant_mem, 'memory_mean_quantized'):
                                try:
                                    quant_vals = quant_mem.memory_mean_quantized.detach().cpu().numpy().flatten()
                                    unique_vals = np.unique(quant_vals)

                                    metrics['quantization/memory_158bit/unique_values'] = len(unique_vals)
                                    metrics['quantization/memory_158bit/mean_value'] = float(np.mean(quant_vals))
                                    metrics['quantization/memory_158bit/std_value'] = float(np.std(quant_vals))

                                    # Check if quantization is collapsing (all values same)
                                    if len(unique_vals) <= 2:
                                        metrics['quantization/memory_158bit/collapsed'] = 1.0
                                        if global_step % 100 == 0:
                                            print(f"[WandBLogger] WARNING: Memory quantization may be collapsing (only {len(unique_vals)} unique values)")
                                    else:
                                        metrics['quantization/memory_158bit/collapsed'] = 0.0
                                except Exception as e:
                                    if global_step % 100 == 0:
                                        print(f"[WandBLogger] WARNING: Failed to compute memory quantization distribution: {e}")

                    except Exception as e:
                        if global_step % 100 == 0:
                            print(f"[WandBLogger] WARNING: Failed to log memory quantization metrics: {e}")
                            import traceback
                            traceback.print_exc()
                else:
                    if global_step % 500 == 0:
                        print(f"[WandBLogger] WARNING: quantize_memory_158bit=True but episodic_memory not found in model")

            # 4-bit Language Model Quantization Metrics
            if config and getattr(config, 'quantize_language_4bit', False):
                if hasattr(base_model, 'language_model'):
                    try:
                        # Check for BitsAndBytes 4-bit quantization
                        lm_model = base_model.language_model

                        # Count quantized parameters
                        num_4bit_params = 0
                        num_total_params = 0

                        for name, param in lm_model.named_parameters():
                            num_total_params += param.numel()
                            # BitsAndBytes marks quantized params with specific dtype
                            if hasattr(param, 'quant_state') or 'int4' in str(param.dtype).lower():
                                num_4bit_params += param.numel()

                        if num_total_params > 0:
                            metrics['quantization/language_4bit/quantized_params'] = num_4bit_params
                            metrics['quantization/language_4bit/total_params'] = num_total_params
                            metrics['quantization/language_4bit/quantized_ratio'] = num_4bit_params / num_total_params

                        # Memory footprint estimation
                        # 4-bit: 0.5 bytes per param, full precision: 2-4 bytes per param
                        memory_saved_mb = (num_4bit_params * 1.5) / 1e6  # Saved bytes if using 2-byte fp16
                        metrics['quantization/language_4bit/memory_saved_mb'] = memory_saved_mb

                    except Exception as e:
                        warnings.warn(f"Failed to log language quantization metrics: {e}")

            # 4-bit Vision Encoder Quantization Metrics (if enabled)
            if config and getattr(config, 'quantize_vision_4bit', False):
                if hasattr(base_model, 'vision_encoder'):
                    try:
                        vision_model = base_model.vision_encoder

                        # Count quantized parameters
                        num_4bit_params = 0
                        num_total_params = 0

                        for name, param in vision_model.named_parameters():
                            num_total_params += param.numel()
                            if hasattr(param, 'quant_state') or 'int4' in str(param.dtype).lower():
                                num_4bit_params += param.numel()

                        if num_total_params > 0:
                            metrics['quantization/vision_4bit/quantized_params'] = num_4bit_params
                            metrics['quantization/vision_4bit/total_params'] = num_total_params
                            metrics['quantization/vision_4bit/quantized_ratio'] = num_4bit_params / num_total_params

                        memory_saved_mb = (num_4bit_params * 1.5) / 1e6
                        metrics['quantization/vision_4bit/memory_saved_mb'] = memory_saved_mb

                    except Exception as e:
                        warnings.warn(f"Failed to log vision quantization metrics: {e}")

            # Overall quantization summary
            if metrics:
                total_quantized_params = (
                    metrics.get('quantization/memory_158bit/quantized_params', 0) +
                    metrics.get('quantization/language_4bit/quantized_params', 0) +
                    metrics.get('quantization/vision_4bit/quantized_params', 0)
                )

                total_params = sum(p.numel() for p in base_model.parameters())

                if total_params > 0:
                    metrics['quantization/overall/quantized_params'] = total_quantized_params
                    metrics['quantization/overall/total_params'] = total_params
                    metrics['quantization/overall/quantized_ratio'] = total_quantized_params / total_params

                # Estimate total model size with quantization
                estimated_size_mb = 0
                for name, param in base_model.named_parameters():
                    is_quantized = (
                        hasattr(param, 'quant_state') or
                        'int4' in str(param.dtype).lower() or
                        'quantized' in name.lower()
                    )

                    if is_quantized:
                        estimated_size_mb += param.numel() * 0.5 / 1e6  # 4-bit or 1.58-bit ~0.5 bytes
                    else:
                        estimated_size_mb += param.numel() * 2 / 1e6  # fp16: 2 bytes

                metrics['quantization/overall/estimated_model_size_mb'] = estimated_size_mb

                # Log all quantization metrics with error handling
                try:
                    self.wandb_run.log(metrics, step=global_step)
                    if global_step % 100 == 0:
                        print(f"[WandBLogger] ✓ Logged {len(metrics)} quantization metrics at step {global_step}")
                except Exception as e:
                    print(f"[WandBLogger] ERROR: Failed to log quantization metrics at step {global_step}: {e}")
                    import traceback
                    traceback.print_exc()

        except Exception as e:
            print(f"[WandBLogger] ERROR in log_quantization_metrics at step {global_step}: {e}")
            import traceback
            traceback.print_exc()

