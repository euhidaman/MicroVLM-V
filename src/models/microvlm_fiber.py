"""
MicroVLM with FIBER-style Fusion-in-Backbone Alignment

This module extends MicroVLM to support FIBER-style cross-modal fusion,
where text information is injected into the vision backbone at intermediate layers.

Key features:
1. Configurable alignment mode: 'baseline' or 'fiber'
2. FIBER mode uses FIBERVisionEncoder with cross-modal attention blocks
3. Combined ITC + ITM + token-level alignment losses
4. Compatible with existing training pipeline
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Tuple
from dataclasses import dataclass

from .vision_encoder import DeiTVisionEncoder
from .language_model import Qwen2LanguageModel
from .multimodal_adapter import MultimodalAdapter, ContrastiveAlignmentLoss, MultimodalFusion, FineGrainedAlignmentLoss
from .fiber_fusion import (
    FIBERVisionEncoder,
    FIBERFusionBlock,
    FIBERAlignmentLoss,
    CrossModalAttention
)
from .episodic_memory import EpisodicMemory, ScopeDetector
from ..quantization.quantized_episodic_memory import apply_158bit_quantization_to_memory


@dataclass
class FIBERConfig:
    """
    Configuration for FIBER-style alignment (reduced for compact model).
    
    ITC Queue settings (FIBER-style):
    - use_itc_queue: Enable/disable queue-based negative sampling
    - itc_queue_size: Number of cached features (reduced from FIBER's 4096)
    - itc_embed_dim: Dimension of ITC projection space
    
    Dimension Alignment:
    - fusion_dim: Shared dimension (384) for cross-modal attention
    - Vision (192) and Text (896) are projected to this shared space
    """
    enabled: bool = True
    fusion_layers: list = None  # Which vision layers get cross-modal fusion
    num_fusion_heads: int = 2  # Reduced from 4 for compact model
    fusion_dim: int = 384  # Shared dimension for cross-modal fusion (NEW)
    bidirectional: bool = False  # T2I fusion in vision encoder
    itc_weight: float = 1.0
    itm_weight: float = 0.5
    token_weight: float = 0.3
    temperature: float = 0.07
    # FIBER-style ITC Queue settings (lightweight for edge devices)
    use_itc_queue: bool = True  # Enable queue-based ITC
    itc_queue_size: int = 256   # Reduced from FIBER's 4096 for compact model
    itc_embed_dim: int = 128    # Compact ITC embedding dimension
    
    def __post_init__(self):
        if self.fusion_layers is None:
            self.fusion_layers = [9, 11]  # Reduced to 2 layers for compact model


class MicroVLM_FIBER(nn.Module):
    """
    MicroVLM with FIBER-style Fusion-in-Backbone Alignment
    
    Supports two alignment modes:
    - 'baseline': Original architecture (late fusion only)
    - 'fiber': FIBER-style fusion with cross-modal attention in vision backbone
    """
    
    def __init__(
        self,
        config: dict,
        vision_checkpoint: Optional[str] = None,
        language_checkpoint: Optional[str] = None,
        quantize_4bit: bool = False,
        quantize_memory_158bit: bool = False,
        training_config=None,
        alignment_mode: str = 'fiber',  # 'baseline' or 'fiber'
        fiber_config: Optional[FIBERConfig] = None
    ):
        super().__init__()
        
        self.config = config
        self.training_config = training_config
        self.quantize_4bit = quantize_4bit
        self.quantize_memory_158bit = quantize_memory_158bit
        self.alignment_mode = alignment_mode
        
        # FIBER configuration
        self.fiber_config = fiber_config or FIBERConfig()
        
        # Vision encoder: Choose based on alignment mode
        if alignment_mode == 'fiber' and self.fiber_config.enabled:
            print(f"🔬 Initializing FIBER Vision Encoder with fusion at layers {self.fiber_config.fusion_layers}")
            print(f"   Fusion dimension: {self.fiber_config.fusion_dim} (shared space for vision-text alignment)")
            self.vision_encoder = FIBERVisionEncoder(
                config=config,
                pretrained_path=vision_checkpoint,
                fusion_layers=self.fiber_config.fusion_layers,
                text_dim=config.get('language_hidden_size', 896),
                num_fusion_heads=self.fiber_config.num_fusion_heads,
                fusion_dim=self.fiber_config.fusion_dim
            )
        else:
            print("📷 Using baseline DeiT Vision Encoder")
            self.vision_encoder = DeiTVisionEncoder(
                config, pretrained_path=vision_checkpoint
            )
        
        # Multimodal adapter
        self.multimodal_adapter = MultimodalAdapter(config)
        
        # Multimodal fusion (late fusion)
        self.fusion = MultimodalFusion()
        
        # Language model: Qwen2.5-0.5B
        self.language_model = Qwen2LanguageModel(
            config,
            pretrained_path=language_checkpoint,
            quantize_4bit=quantize_4bit
        )
        
        # Episodic memory
        self.episodic_memory = EpisodicMemory(config)
        
        if quantize_memory_158bit:
            print("Applying 1.58-bit quantization to episodic memory...")
            apply_158bit_quantization_to_memory(self.episodic_memory)
        
        # Scope detector
        self.scope_detector = ScopeDetector(config)
        
        if quantize_memory_158bit:
            apply_158bit_quantization_to_memory(self.scope_detector)
        
        # Alignment dimension (reduced for compact model)
        self.alignment_dim = config.get('alignment_dim', 128)  # Reduced from 256
        
        # Image feature projection for alignment (simplified)
        self.image_proj_for_alignment = nn.Sequential(
            nn.Linear(config.get('vision_hidden_size', 192), self.alignment_dim),
            nn.LayerNorm(self.alignment_dim)
        )
        
        # Text feature projection for alignment (simplified)
        self.text_proj_for_alignment = nn.Sequential(
            nn.Linear(config.get('language_hidden_size', 896), self.alignment_dim),
            nn.LayerNorm(self.alignment_dim)
        )
        
        self._init_alignment_projections()
        
        # Choose alignment loss based on mode
        if alignment_mode == 'fiber' and self.fiber_config.enabled:
            # Get ITC configuration from fiber_config or use defaults
            use_itc_queue = getattr(self.fiber_config, 'use_itc_queue', True)
            itc_queue_size = getattr(self.fiber_config, 'itc_queue_size', 256)
            itc_embed_dim = getattr(self.fiber_config, 'itc_embed_dim', 128)
            
            print(f"🔗 Using FIBER Alignment Loss (ITC + ITM + Token)")
            print(f"   ITC Queue: {'enabled' if use_itc_queue else 'disabled'} (size={itc_queue_size})")
            print(f"   ITC embed dim: {itc_embed_dim}")
            
            self.alignment_loss = FIBERAlignmentLoss(
                temperature=self.fiber_config.temperature,
                learnable_temperature=True,
                label_smoothing=0.1,
                itc_weight=self.fiber_config.itc_weight,
                itm_weight=self.fiber_config.itm_weight,
                token_weight=self.fiber_config.token_weight,
                use_itc_queue=use_itc_queue,
                queue_size=itc_queue_size,
                itc_embed_dim=itc_embed_dim
            )
            # Initialize ITM head
            self.alignment_loss.set_itm_head(
                vision_dim=config.get('vision_hidden_size', 192),
                text_dim=config.get('language_hidden_size', 896)
            )
            # Initialize ITC projection heads (FIBER-style)
            self.alignment_loss.set_itc_heads(
                image_dim=self.alignment_dim,  # Use alignment_dim as input
                text_dim=self.alignment_dim
            )
        else:
            print("🔗 Using baseline Contrastive Alignment Loss")
            self.alignment_loss = ContrastiveAlignmentLoss(
                temperature=config.get('alignment_temperature', 0.07)
            )
        
        # Fine-grained alignment (kept for compatibility)
        self.fine_grained_alignment_loss = FineGrainedAlignmentLoss(
            temperature=0.1,
            entropy_weight=0.3,
            diversity_weight=0.2
        )
        
        # Debug counter
        self._alignment_log_counter = 0
        self._last_text_to_patch_attention = None
        self._last_fiber_attention = None
        
        # Memory state
        self.memory_state = None
        
        # Training flags
        self.use_memory = True
        self.use_alignment = True
    
    def _init_alignment_projections(self):
        """Initialize alignment projection layers"""
        for module in [self.image_proj_for_alignment, self.text_proj_for_alignment]:
            for m in module.modules():
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight, gain=0.1)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)
                elif isinstance(m, nn.LayerNorm):
                    nn.init.ones_(m.weight)
                    nn.init.zeros_(m.bias)
    
    def encode_image(
        self,
        images: torch.Tensor,
        text_embeddings: Optional[torch.Tensor] = None,
        text_mask: Optional[torch.Tensor] = None,
        return_patch_embeddings: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], Optional[dict]]:
        """
        Encode images with optional FIBER fusion.
        
        Args:
            images: (B, 3, H, W)
            text_embeddings: (B, seq_len, text_dim) for FIBER fusion
            text_mask: (B, seq_len) attention mask
            return_patch_embeddings: whether to return patch embeddings
        
        Returns:
            prefix_tokens: (B, k_prefix, qwen_dim)
            image_features: (B, alignment_dim)
            patch_embeddings_proj: (B, num_patches, qwen_dim) if return_patch_embeddings
            fiber_attention: dict with fusion attention weights
        """
        fiber_attention = None
        
        # Extract patch embeddings with optional FIBER fusion
        if self.alignment_mode == 'fiber' and self.fiber_config.enabled and isinstance(self.vision_encoder, FIBERVisionEncoder):
            patch_embeddings, _, fiber_attention = self.vision_encoder(
                images, text_embeddings, text_mask
            )
            self._last_fiber_attention = fiber_attention
        else:
            patch_embeddings = self.vision_encoder(images)
        
        # Get CLS token for image features
        if self.alignment_mode == 'fiber' and self.fiber_config.enabled and isinstance(self.vision_encoder, FIBERVisionEncoder):
            raw_image_features = self.vision_encoder.get_cls_token(
                images, text_embeddings, text_mask
            )
        else:
            raw_image_features = self.vision_encoder.get_cls_token(images)
        
        # Project to alignment space and normalize for stable similarity metrics
        image_features = self.image_proj_for_alignment(raw_image_features)
        image_features = F.normalize(image_features, p=2, dim=-1, eps=1e-6)
        
        # Project to language space via adapter
        if return_patch_embeddings:
            prefix_tokens, patch_embeddings_proj = self.multimodal_adapter(
                patch_embeddings, return_patch_embeddings=True
            )
            return prefix_tokens, image_features, patch_embeddings_proj, fiber_attention
        else:
            prefix_tokens = self.multimodal_adapter(patch_embeddings)
            return prefix_tokens, image_features, None, fiber_attention
    
    def encode_text(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode text to embeddings.
        
        Args:
            input_ids: (B, seq_len)
            attention_mask: (B, seq_len)
        
        Returns:
            text_embeddings: (B, seq_len, qwen_dim)
            text_features: (B, alignment_dim)
        """
        # Get embeddings
        text_embeddings = self.language_model.get_input_embeddings()(input_ids)
        
        # Mean pooling for text features
        if attention_mask is not None:
            mask_expanded = attention_mask.unsqueeze(-1).expand(text_embeddings.size())
            sum_embeddings = torch.sum(text_embeddings * mask_expanded, dim=1)
            sum_mask = torch.clamp(mask_expanded.sum(dim=1), min=1e-9)
            raw_text_features = sum_embeddings / sum_mask
        else:
            raw_text_features = text_embeddings.mean(dim=1)
        
        # Project to alignment space and normalize to keep cosine similarities bounded
        text_features = self.text_proj_for_alignment(raw_text_features)
        text_features = F.normalize(text_features, p=2, dim=-1, eps=1e-6)
        
        return text_embeddings, text_features
    
    def forward(
        self,
        images: Optional[torch.Tensor] = None,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        use_memory: Optional[bool] = None,
        use_alignment: Optional[bool] = None,
        episode_size: int = 1,
        reset_memory: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass with FIBER-style alignment.
        
        The key difference from baseline:
        - In FIBER mode, text embeddings are passed to vision encoder for fusion
        - This creates text-conditioned visual features at intermediate layers
        """
        use_memory = use_memory if use_memory is not None else self.use_memory
        use_alignment = use_alignment if use_alignment is not None else self.use_alignment
        
        if images is not None:
            batch_size = images.size(0)
        elif input_ids is not None:
            batch_size = input_ids.size(0)
        else:
            raise ValueError("Either images or input_ids must be provided")
        
        outputs = {}
        
        if reset_memory:
            self.memory_state = None
        
        # First encode text (needed for FIBER fusion in vision encoder)
        text_embeddings, text_features = self.encode_text(input_ids, attention_mask)
        
        # Then encode images with FIBER fusion
        prefix_tokens, image_features, patch_embeddings_proj, fiber_attention = None, None, None, None
        if images is not None:
            prefix_tokens, image_features, patch_embeddings_proj, fiber_attention = self.encode_image(
                images,
                text_embeddings=text_embeddings if self.alignment_mode == 'fiber' else None,
                text_mask=attention_mask if self.alignment_mode == 'fiber' else None,
                return_patch_embeddings=True
            )
            
            if fiber_attention:
                outputs['fiber_attention'] = fiber_attention
        
        # Compute alignment loss
        if use_alignment and image_features is not None:
            if self.alignment_mode == 'fiber' and isinstance(self.alignment_loss, FIBERAlignmentLoss):
                # FIBER alignment: ITC + ITM + token-level
                # Get CLS tokens for ITM
                if isinstance(self.vision_encoder, FIBERVisionEncoder):
                    vision_cls = self.vision_encoder.get_cls_token(
                        images, text_embeddings, attention_mask
                    )
                else:
                    vision_cls = self.vision_encoder.get_cls_token(images)
                
                # Text CLS (use pooled representation)
                if attention_mask is not None:
                    mask_expanded = attention_mask.unsqueeze(-1).expand(text_embeddings.size())
                    sum_embeddings = torch.sum(text_embeddings * mask_expanded, dim=1)
                    sum_mask = torch.clamp(mask_expanded.sum(dim=1), min=1e-9)
                    text_cls = sum_embeddings / sum_mask
                else:
                    text_cls = text_embeddings.mean(dim=1)
                
                # Compute FIBER losses
                alignment_outputs = self.alignment_loss(
                    image_features=image_features,
                    text_features=text_features,
                    vision_cls=vision_cls,
                    text_cls=text_cls,
                    patch_embeddings=patch_embeddings_proj,
                    text_embeddings=text_embeddings,
                    text_mask=attention_mask,
                    compute_itm=True,
                    compute_token=True
                )
                
                outputs['alignment_loss'] = alignment_outputs['itc_loss']
                outputs['itm_loss'] = alignment_outputs.get('itm_loss', torch.tensor(0.0))
                outputs['token_loss'] = alignment_outputs.get('token_loss', torch.tensor(0.0))
                
                # Pass through anti-collapse regularization losses for logging
                if 'anti_collapse_loss' in alignment_outputs:
                    outputs['anti_collapse_loss'] = alignment_outputs['anti_collapse_loss']
                if 'attention_entropy_loss' in alignment_outputs:
                    outputs['attention_entropy_loss'] = alignment_outputs['attention_entropy_loss']
                
                if 'token_attention' in alignment_outputs:
                    self._last_text_to_patch_attention = alignment_outputs['token_attention'].detach()
                    # Also pass to outputs for attention monitoring in training loop
                    outputs['token_attention'] = alignment_outputs['token_attention']
            else:
                # Baseline alignment: contrastive only
                alignment_loss = self.alignment_loss(image_features, text_features)
                outputs['alignment_loss'] = alignment_loss
                
                # Fine-grained loss
                if patch_embeddings_proj is not None:
                    fine_grained_loss, text_to_patch_attn = self.fine_grained_alignment_loss(
                        patch_embeddings_proj, text_embeddings, attention_mask
                    )
                    outputs['fine_grained_loss'] = fine_grained_loss
                    self._last_text_to_patch_attention = text_to_patch_attn.detach()
            
            # Alignment statistics for downstream monitoring
            alignment_stats = self._compute_alignment_stats(image_features, text_features)
            outputs['alignment_stats'] = alignment_stats

            # Logging
            self._alignment_log_counter += 1
            if self._alignment_log_counter % 100 == 1:
                self._log_alignment_metrics(outputs, alignment_stats)
        
        # Fuse modalities
        if prefix_tokens is not None:
            fused_embeddings, fused_mask = self.fusion(
                prefix_tokens, text_embeddings, attention_mask
            )
        else:
            fused_embeddings = text_embeddings
            fused_mask = attention_mask
        
        # Convert dtype for LM
        if hasattr(self.language_model, 'model') and self.language_model.model is not None:
            model_dtype = next(self.language_model.model.parameters()).dtype
            if fused_embeddings.dtype != model_dtype:
                fused_embeddings = fused_embeddings.to(model_dtype)
        
        # Memory processing
        if use_memory:
            fused_context = fused_embeddings.mean(dim=1)
            
            if episode_size > 1 and batch_size % episode_size == 0:
                episode_batch_size = batch_size // episode_size
                z_for_memory = fused_context.view(episode_size, episode_batch_size, -1)
            else:
                z_for_memory = fused_context.unsqueeze(0)
                episode_size = 1
            
            memory_state, dkl_M = self.episodic_memory.write(z_for_memory)
            outputs['memory_kl'] = dkl_M
            
            z_retrieved, dkl_w = self.episodic_memory.read(
                z_for_memory, memory_state, deterministic=False
            )
            outputs['addressing_kl'] = dkl_w.mean()
            
            scope_probs = self.scope_detector(fused_context)
            outputs['scope_probs'] = scope_probs
            
            kv_memory = self.episodic_memory.project_to_kv(z_retrieved)
            self.memory_state = memory_state
        else:
            kv_memory = None
        
        # Get training config attributes
        def _get_training_attr(name, default):
            if self.training_config is None:
                return default
            if isinstance(self.training_config, dict):
                return self.training_config.get(name, default)
            return getattr(self.training_config, name, default)
        
        skip_lm_loss = _get_training_attr('skip_lm_loss', False)
        lm_weight = _get_training_attr('lm_loss_weight', 1.0)
        
        # LM forward
        if skip_lm_loss or lm_weight == 0.0:
            outputs['lm_loss'] = None
            outputs['logits'] = None
            outputs['hidden_states'] = None
        else:
            adjusted_labels = labels
            if labels is not None and prefix_tokens is not None:
                batch_size = labels.size(0)
                prefix_len = prefix_tokens.size(1)
                prefix_labels = torch.full(
                    (batch_size, prefix_len), -100,
                    dtype=labels.dtype, device=labels.device
                )
                adjusted_labels = torch.cat([prefix_labels, labels], dim=1)
            
            lm_outputs = self.language_model(
                inputs_embeds=fused_embeddings,
                attention_mask=fused_mask,
                labels=adjusted_labels,
                output_hidden_states=True,
                return_dict=True
            )
            
            if lm_outputs.loss is not None and not (torch.isnan(lm_outputs.loss) or torch.isinf(lm_outputs.loss)):
                outputs['lm_loss'] = lm_outputs.loss
            else:
                outputs['lm_loss'] = None
            
            outputs['logits'] = lm_outputs.logits
            outputs['hidden_states'] = lm_outputs.hidden_states
        
        # Compute total loss
        alignment_weight = _get_training_attr('alignment_loss_weight', 1.0)
        memory_weight = _get_training_attr('memory_kl_weight', 0.01)
        addressing_weight = _get_training_attr('addressing_kl_weight', 0.001)
        
        device = fused_embeddings.device
        total_loss = torch.tensor(0.0, device=device, requires_grad=True)
        
        # LM loss
        if outputs.get('lm_loss') is not None:
            total_loss = total_loss + lm_weight * outputs['lm_loss']
        
        # Alignment losses
        if 'alignment_loss' in outputs:
            total_loss = total_loss + alignment_weight * outputs['alignment_loss']
        
        if 'itm_loss' in outputs and not torch.isnan(outputs['itm_loss']):
            itm_weight = _get_training_attr('itm_loss_weight', 0.5)
            total_loss = total_loss + itm_weight * outputs['itm_loss']
        
        if 'token_loss' in outputs and not torch.isnan(outputs['token_loss']):
            token_weight = _get_training_attr('token_loss_weight', 0.3)
            total_loss = total_loss + token_weight * outputs['token_loss']
        
        if 'fine_grained_loss' in outputs:
            fg_weight = _get_training_attr('fine_grained_loss_weight', 0.5)
            total_loss = total_loss + fg_weight * outputs['fine_grained_loss']
        
        # Memory losses
        if use_memory and 'memory_kl' in outputs:
            total_loss = total_loss + memory_weight * outputs['memory_kl']
        
        if use_memory and 'addressing_kl' in outputs:
            total_loss = total_loss + addressing_weight * outputs['addressing_kl']
        
        outputs['loss'] = total_loss
        
        return outputs
    
    def _compute_alignment_stats(self, image_features, text_features):
        """Compute reusable alignment statistics"""
        with torch.no_grad():
            img_norm = image_features.norm(dim=-1).mean().item()
            txt_norm = text_features.norm(dim=-1).mean().item()

            img_normalized = F.normalize(image_features, p=2, dim=-1)
            txt_normalized = F.normalize(text_features, p=2, dim=-1)
            correct_sim = (img_normalized * txt_normalized).sum(dim=-1).mean().item()

        return {
            'img_norm': img_norm,
            'txt_norm': txt_norm,
            'correct_sim': correct_sim
        }

    def _log_alignment_metrics(self, outputs, alignment_stats):
        """Log alignment metrics for debugging"""
        if not alignment_stats:
            return
        print(f"  📊 Alignment [{self.alignment_mode}]: img_norm={alignment_stats['img_norm']:.3f}, "
              f"txt_norm={alignment_stats['txt_norm']:.3f}, correct_sim={alignment_stats['correct_sim']:.4f}")
        
        if 'itm_loss' in outputs:
            print(f"     ITM loss: {outputs['itm_loss'].item():.4f}")
        if 'token_loss' in outputs:
            print(f"     Token loss: {outputs['token_loss'].item():.4f}")
    
    def get_text_to_patch_attention(self) -> Optional[torch.Tensor]:
        """Get the last computed text-to-patch attention"""
        return self._last_text_to_patch_attention
    
    def get_fiber_attention(self) -> Optional[dict]:
        """Get FIBER fusion attention weights"""
        return self._last_fiber_attention
    
    def get_pooling_attention(self) -> Optional[torch.Tensor]:
        """Get adapter pooling attention"""
        return self.multimodal_adapter.get_pooling_attention_weights()
    
    def freeze_vision_encoder(self):
        """Freeze vision encoder (but keep FIBER fusion trainable)"""
        for name, param in self.vision_encoder.named_parameters():
            # Keep fusion blocks trainable in FIBER mode
            if 'fusion_block' in name and self.alignment_mode == 'fiber':
                param.requires_grad = True
            else:
                param.requires_grad = False
        print(f"Vision encoder frozen (FIBER fusion blocks: {'trainable' if self.alignment_mode == 'fiber' else 'frozen'})")
    
    def freeze_language_model(self, unfreeze_last_n: int = 0):
        """Freeze language model"""
        if unfreeze_last_n > 0:
            self.language_model.freeze_layers(
                num_layers_to_freeze=self.language_model.num_layers - unfreeze_last_n
            )
        else:
            for param in self.language_model.parameters():
                param.requires_grad = False
        print(f"Language model frozen (last {unfreeze_last_n} layers unfrozen)")
    
    def get_trainable_params(self) -> int:
        """Get count of trainable parameters"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def save_checkpoint(self, path: str):
        """Save model checkpoint"""
        checkpoint = {
            'model_state_dict': self.state_dict(),
            'config': self.config,
            'memory_state': self.memory_state,
            'alignment_mode': self.alignment_mode,
            'fiber_config': self.fiber_config.__dict__ if self.fiber_config else None
        }
        torch.save(checkpoint, path)
        print(f"Checkpoint saved to {path}")
    
    def load_checkpoint(self, path: str):
        """Load model checkpoint"""
        checkpoint = torch.load(path, map_location='cpu')
        self.load_state_dict(checkpoint['model_state_dict'])
        self.memory_state = checkpoint.get('memory_state', None)
        print(f"Checkpoint loaded from {path}")


def create_microvlm_fiber(
    config: dict,
    vision_checkpoint: Optional[str] = None,
    language_checkpoint: Optional[str] = None,
    quantize_4bit: bool = False,
    quantize_memory_158bit: bool = False,
    training_config=None,
    alignment_mode: str = 'fiber',
    fiber_config=None
) -> MicroVLM_FIBER:
    """
    Factory function to create MicroVLM with FIBER alignment.
    
    Args:
        config: model configuration dictionary
        vision_checkpoint: path to DeiT checkpoint
        language_checkpoint: path to Qwen checkpoint
        quantize_4bit: whether to apply 4-bit quantization
        quantize_memory_158bit: whether to apply 1.58-bit memory quantization
        training_config: optional training config
        alignment_mode: 'baseline' or 'fiber'
        fiber_config: FIBER-specific configuration (dict or FIBERConfig)
    
    Returns:
        model: MicroVLM_FIBER instance
    """
    # Convert dict to FIBERConfig if needed
    if fiber_config is not None and isinstance(fiber_config, dict):
        fiber_cfg = FIBERConfig()
        # Map dict keys to FIBERConfig fields
        if 'fiber_fusion_layers' in fiber_config:
            fiber_cfg.fusion_layers = fiber_config['fiber_fusion_layers']
        if 'cross_attention_heads' in fiber_config:
            fiber_cfg.num_fusion_heads = fiber_config['cross_attention_heads']
        if 'itc_weight' in fiber_config:
            fiber_cfg.itc_weight = fiber_config['itc_weight']
        if 'itm_weight' in fiber_config:
            fiber_cfg.itm_weight = fiber_config['itm_weight']
        if 'use_bidirectional' in fiber_config:
            fiber_cfg.bidirectional = fiber_config['use_bidirectional']
        if 'temperature' in fiber_config:
            fiber_cfg.temperature = fiber_config['temperature']
        # ITC Queue settings
        if 'use_itc_queue' in fiber_config:
            fiber_cfg.use_itc_queue = fiber_config['use_itc_queue']
        if 'itc_queue_size' in fiber_config:
            fiber_cfg.itc_queue_size = fiber_config['itc_queue_size']
        if 'itc_embed_dim' in fiber_config:
            fiber_cfg.itc_embed_dim = fiber_config['itc_embed_dim']
        fiber_config = fiber_cfg
    elif fiber_config is None:
        fiber_config = FIBERConfig()
    
    model = MicroVLM_FIBER(
        config=config,
        vision_checkpoint=vision_checkpoint,
        language_checkpoint=language_checkpoint,
        quantize_4bit=quantize_4bit,
        quantize_memory_158bit=quantize_memory_158bit,
        training_config=training_config,
        alignment_mode=alignment_mode,
        fiber_config=fiber_config
    )
    return model
