"""
MicroVLM Main Model
Integrates vision encoder, multimodal adapter, language model, and episodic memory
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Tuple

from .vision_encoder import DeiTVisionEncoder
from .language_model import Qwen2LanguageModel
from .multimodal_adapter import MultimodalAdapter, ContrastiveAlignmentLoss, MultimodalFusion, FineGrainedAlignmentLoss
from .episodic_memory import EpisodicMemory, ScopeDetector
from ..quantization.quantized_episodic_memory import apply_158bit_quantization_to_memory


class MicroVLM(nn.Module):
    """
    MicroVLM: Tiny Vision-Language Model with Episodic Memory

    Components:
    - DeiT-Tiny vision encoder
    - Multimodal adapter with EVO-1 alignment
    - Qwen2.5-0.5B language model
    - Larimar episodic memory
    """

    def __init__(self, config, vision_checkpoint=None, language_checkpoint=None,
                 quantize_4bit=False, quantize_memory_158bit=False,
                 training_config=None):
        super().__init__()

        self.config = config
        self.training_config = training_config
        self.quantize_4bit = quantize_4bit
        self.quantize_memory_158bit = quantize_memory_158bit

        # Vision encoder: DeiT-Tiny with optional 4-bit quantization
        quantize_vision_4bit = getattr(training_config, 'quantize_vision_4bit', False) if training_config else False
        use_hf_deit = getattr(training_config, 'use_hf_deit', True) if training_config else True

        self.vision_encoder = DeiTVisionEncoder(
            config,
            pretrained_path=vision_checkpoint,
            quantize_4bit=quantize_vision_4bit,
            use_hf_model=use_hf_deit
        )

        # Multimodal adapter
        self.multimodal_adapter = MultimodalAdapter(config)

        # Multimodal fusion
        self.fusion = MultimodalFusion()

        # Language model: Qwen2.5-0.5B
        self.language_model = Qwen2LanguageModel(
            config,
            pretrained_path=language_checkpoint,
            quantize_4bit=quantize_4bit
        )

        # Episodic memory
        self.episodic_memory = EpisodicMemory(config)

        # Apply 1.58-bit quantization to episodic memory if requested
        if quantize_memory_158bit:
            print("Applying 1.58-bit quantization to episodic memory...")
            apply_158bit_quantization_to_memory(self.episodic_memory)
            print("Episodic memory quantization applied!")

        # Scope detector
        self.scope_detector = ScopeDetector(config)

        # Apply 1.58-bit quantization to scope detector if requested
        if quantize_memory_158bit:
            print("Applying 1.58-bit quantization to scope detector...")
            apply_158bit_quantization_to_memory(self.scope_detector)
            print("Scope detector quantization applied!")

        # Alignment embedding dimension (shared projection space)
        self.alignment_dim = config.get('alignment_dim', 256)
        
        # Image feature projection for alignment (DeiT 192-dim -> alignment_dim)
        # Uses MLP for better representation
        self.image_proj_for_alignment = nn.Sequential(
            nn.Linear(config.get('vision_hidden_size', 192), config.get('vision_hidden_size', 192) * 2),
            nn.GELU(),
            nn.Linear(config.get('vision_hidden_size', 192) * 2, self.alignment_dim),
            nn.LayerNorm(self.alignment_dim)
        )
        
        # Text feature projection for alignment (Qwen 896-dim -> alignment_dim)
        # Crucial: both modalities must project to the same space!
        self.text_proj_for_alignment = nn.Sequential(
            nn.Linear(config.get('language_hidden_size', 896), config.get('language_hidden_size', 896)),
            nn.GELU(),
            nn.Linear(config.get('language_hidden_size', 896), self.alignment_dim),
            nn.LayerNorm(self.alignment_dim)
        )
        
        # Initialize projection layers with small weights for stable training
        self._init_alignment_projections()

        # Alignment loss
        self.alignment_loss = ContrastiveAlignmentLoss(
            temperature=config.get('alignment_temperature', 0.07)
        )
        
        # Fine-grained text-to-patch alignment loss
        # This encourages text tokens to attend to semantically relevant image patches
        self.fine_grained_alignment_loss = FineGrainedAlignmentLoss(
            temperature=0.1,
            entropy_weight=0.3,  # Encourage focused attention
            diversity_weight=0.2  # Encourage diverse attention patterns
        )
        
        # Debug counter for logging feature norms
        self._alignment_log_counter = 0
        
        # Store last fine-grained attention for visualization
        self._last_text_to_patch_attention = None

        # Memory state (persistent across forward passes)
        self.memory_state = None

        # Training configuration
        self.use_memory = True
        self.use_alignment = True
    
    def _init_alignment_projections(self):
        """Initialize alignment projection layers with stable weights"""
        for module in [self.image_proj_for_alignment, self.text_proj_for_alignment]:
            for m in module.modules():
                if isinstance(m, nn.Linear):
                    # Xavier initialization for stable gradients
                    nn.init.xavier_uniform_(m.weight, gain=0.1)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)
                elif isinstance(m, nn.LayerNorm):
                    nn.init.ones_(m.weight)
                    nn.init.zeros_(m.bias)

    def encode_image(self, images, return_patch_embeddings=False):
        """
        Encode images to prefix tokens

        Args:
            images: (batch_size, 3, H, W) or list of PIL Images
            return_patch_embeddings: if True, also return projected patch embeddings

        Returns:
            prefix_tokens: (batch_size, k_prefix, qwen_dim)
            image_features: (batch_size, alignment_dim) for alignment - projected to shared space
            patch_embeddings_proj: (batch_size, num_patches, qwen_dim) if return_patch_embeddings
        """
        # Extract patch embeddings
        patch_embeddings = self.vision_encoder(
            images)  # (B, num_patches, deit_dim)

        # Get image-level features (CLS token)
        raw_image_features = self.vision_encoder.get_cls_token(
            images)  # (B, deit_dim=192)

        # Project image features to alignment space (NOT language space)
        image_features = self.image_proj_for_alignment(
            raw_image_features)  # (B, alignment_dim=256)

        # Project to language space for prefix tokens
        # Also get projected patch embeddings for fine-grained attention
        if return_patch_embeddings:
            prefix_tokens, patch_embeddings_proj = self.multimodal_adapter(
                patch_embeddings, return_patch_embeddings=True)  # (B, k_prefix, qwen_dim), (B, num_patches, qwen_dim)
            return prefix_tokens, image_features, patch_embeddings_proj
        else:
            prefix_tokens = self.multimodal_adapter(
                patch_embeddings)  # (B, k_prefix, qwen_dim)
            return prefix_tokens, image_features

    def encode_text(self, input_ids, attention_mask=None):
        """
        Encode text to embeddings

        Args:
            input_ids: (batch_size, seq_len)
            attention_mask: (batch_size, seq_len)

        Returns:
            text_embeddings: (batch_size, seq_len, qwen_dim)
            text_features: (batch_size, alignment_dim) for alignment - projected to shared space
        """
        # Get embeddings
        text_embeddings = self.language_model.get_input_embeddings()(input_ids)

        # Get text-level features (mean pooling over valid tokens)
        if attention_mask is not None:
            mask_expanded = attention_mask.unsqueeze(
                -1).expand(text_embeddings.size())
            sum_embeddings = torch.sum(text_embeddings * mask_expanded, dim=1)
            sum_mask = torch.clamp(mask_expanded.sum(dim=1), min=1e-9)
            raw_text_features = sum_embeddings / sum_mask  # (B, qwen_dim=896)
        else:
            raw_text_features = text_embeddings.mean(dim=1)  # (B, qwen_dim=896)
        
        # Project text features to alignment space (crucial for contrastive learning!)
        text_features = self.text_proj_for_alignment(raw_text_features)  # (B, alignment_dim=256)

        return text_embeddings, text_features

    def forward(self, images=None, input_ids=None, attention_mask=None, labels=None,
                use_memory=None, use_alignment=None, episode_size=1, reset_memory=False):
        """
        Forward pass

        Args:
            images: (batch_size, 3, H, W) or list of PIL Images
            input_ids: (batch_size, seq_len)
            attention_mask: (batch_size, seq_len)
            labels: (batch_size, seq_len) for language modeling loss
            use_memory: whether to use episodic memory
            use_alignment: whether to compute alignment loss
            episode_size: size of episode for memory
            reset_memory: whether to reset memory state

        Returns:
            outputs: dictionary containing losses and logits
        """
        use_memory = use_memory if use_memory is not None else self.use_memory
        use_alignment = use_alignment if use_alignment is not None else self.use_alignment

        # Determine batch size - prioritize images if both present
        if images is not None:
            batch_size = images.size(0)
        elif input_ids is not None:
            batch_size = input_ids.size(0)
        else:
            raise ValueError("Either images or input_ids must be provided")

        outputs = {}

        # Reset memory if requested
        if reset_memory:
            self.memory_state = None

        # Encode images (with patch embeddings for fine-grained alignment)
        prefix_tokens, image_features, patch_embeddings_proj = None, None, None
        if images is not None:
            prefix_tokens, image_features, patch_embeddings_proj = self.encode_image(
                images, return_patch_embeddings=True)

        # Encode text
        text_embeddings, text_features = self.encode_text(
            input_ids, attention_mask)

        # Compute alignment loss if both modalities present
        if use_alignment and image_features is not None:
            # Global contrastive alignment loss
            alignment_loss = self.alignment_loss(image_features, text_features)
            outputs['alignment_loss'] = alignment_loss
            
            # Fine-grained text-to-patch alignment loss
            # This provides explicit supervision for text-conditioned attention
            fine_grained_loss, text_to_patch_attn = self.fine_grained_alignment_loss(
                patch_embeddings_proj, text_embeddings, attention_mask
            )
            outputs['fine_grained_loss'] = fine_grained_loss
            
            # Store attention for visualization
            self._last_text_to_patch_attention = text_to_patch_attn.detach()
            
            # Log alignment metrics periodically for debugging
            self._alignment_log_counter += 1
            if self._alignment_log_counter % 100 == 1:  # Log every 100 steps
                with torch.no_grad():
                    # Pre-normalization norms (should be similar after projection)
                    img_norm = image_features.norm(dim=-1).mean().item()
                    txt_norm = text_features.norm(dim=-1).mean().item()
                    
                    # Post-normalization similarity
                    img_normalized = F.normalize(image_features, p=2, dim=-1)
                    txt_normalized = F.normalize(text_features, p=2, dim=-1)
                    
                    # Correct pair similarity (diagonal)
                    correct_sim = (img_normalized * txt_normalized).sum(dim=-1).mean().item()
                    
                    # Temperature
                    temp = self.alignment_loss.logit_scale.exp().item() if hasattr(self.alignment_loss, 'logit_scale') else 14.29
                    
                    # Attention entropy (lower = more focused)
                    attn_entropy = -(text_to_patch_attn * torch.log(text_to_patch_attn + 1e-8)).sum(dim=-1).mean().item()
                    max_entropy = torch.log(torch.tensor(float(patch_embeddings_proj.size(1)))).item()
                    
                    print(f"  📊 Alignment: img_norm={img_norm:.3f}, txt_norm={txt_norm:.3f}, "
                          f"correct_sim={correct_sim:.4f}, temp={temp:.2f}, loss={alignment_loss.item():.4f}")
                    print(f"  📊 Fine-grained: attn_entropy={attn_entropy:.3f}/{max_entropy:.3f}, "
                          f"fine_loss={fine_grained_loss.item():.4f}")

        # Fuse modalities
        if prefix_tokens is not None:
            fused_embeddings, fused_mask = self.fusion(
                prefix_tokens, text_embeddings, attention_mask
            )
        else:
            fused_embeddings = text_embeddings
            fused_mask = attention_mask

        # Convert to same dtype as language model if needed
        # Qwen is loaded in float16, so we need to match that
        if hasattr(self.language_model, 'model') and self.language_model.model is not None:
            model_dtype = next(self.language_model.model.parameters()).dtype
            if fused_embeddings.dtype != model_dtype:
                fused_embeddings = fused_embeddings.to(model_dtype)

        # Validate embeddings before forward pass
        if torch.isnan(fused_embeddings).any() or torch.isinf(fused_embeddings).any():
            import warnings
            warnings.warn(
                "NaN or Inf detected in fused embeddings before LM forward pass")
            # Clamp to prevent propagation
            fused_embeddings = torch.nan_to_num(
                fused_embeddings, nan=0.0, posinf=1e4, neginf=-1e4)

        # Episodic memory processing
        if use_memory:
            # Reshape for episode processing
            fused_context = fused_embeddings.mean(dim=1)  # (B, qwen_dim)

            # Prepare for memory: (episode_size, batch//episode_size, qwen_dim)
            if episode_size > 1 and batch_size % episode_size == 0:
                episode_batch_size = batch_size // episode_size
                z_for_memory = fused_context.view(
                    episode_size, episode_batch_size, -1
                )
            else:
                # Warn about episode_size mismatch
                if episode_size > 1 and batch_size % episode_size != 0:
                    import warnings
                    warnings.warn(
                        f"Batch size {batch_size} not divisible by episode_size {episode_size}. "
                        f"Using episode_size=1 instead."
                    )
                z_for_memory = fused_context.unsqueeze(0)
                episode_size = 1

            # Write to memory
            memory_state, dkl_M = self.episodic_memory.write(z_for_memory)
            outputs['memory_kl'] = dkl_M

            # Read from memory
            z_retrieved, dkl_w = self.episodic_memory.read(
                z_for_memory, memory_state, deterministic=False
            )
            outputs['addressing_kl'] = dkl_w.mean()

            # Scope detection
            scope_probs = self.scope_detector(fused_context)
            outputs['scope_probs'] = scope_probs

            # Project to KV space
            kv_memory = self.episodic_memory.project_to_kv(z_retrieved)

            # Store memory state for next iteration
            self.memory_state = memory_state
        else:
            kv_memory = None

        def _get_training_attr(name, default):
            if self.training_config is None:
                return default
            if isinstance(self.training_config, dict):
                return self.training_config.get(name, default)
            return getattr(self.training_config, name, default)

        # Check if we should skip LM loss (for Stage 1 alignment-only training)
        skip_lm_loss = _get_training_attr('skip_lm_loss', False)
        lm_weight = _get_training_attr('lm_loss_weight', 1.0)
        
        # Debug: Print once at the start of training
        if not hasattr(self, '_lm_skip_logged'):
            print(f"\n🔧 LM Loss Config: skip_lm_loss={skip_lm_loss}, lm_weight={lm_weight}")
            print(f"   training_config type: {type(self.training_config)}")
            if self.training_config is not None:
                print(f"   skip_lm_loss attr exists: {hasattr(self.training_config, 'skip_lm_loss')}")
            self._lm_skip_logged = True
        
        # Skip LM forward pass if lm_weight is 0 or skip_lm_loss is True
        # This saves significant compute when LM is frozen
        if skip_lm_loss or lm_weight == 0.0:
            outputs['lm_loss'] = None
            outputs['logits'] = None
            outputs['hidden_states'] = None
        else:
            # Language model forward pass
            # Adjust labels to match fused_embeddings length if we added prefix tokens
            adjusted_labels = labels
            if labels is not None and prefix_tokens is not None:
                # Pad labels with -100 (ignore_index) for prefix tokens
                batch_size = labels.size(0)
                prefix_len = prefix_tokens.size(1)
                prefix_labels = torch.full(
                    (batch_size, prefix_len),
                    -100,
                    dtype=labels.dtype,
                    device=labels.device
                )
                adjusted_labels = torch.cat([prefix_labels, labels], dim=1)

            lm_outputs = self.language_model(
                inputs_embeds=fused_embeddings,
                attention_mask=fused_mask,
                labels=adjusted_labels,
                output_hidden_states=True,
                return_dict=True
            )

            # Validate LM loss before assignment
            if lm_outputs.loss is not None and (torch.isnan(lm_outputs.loss) or torch.isinf(lm_outputs.loss)):
                # Debug: print what caused NaN
                if torch.isnan(fused_embeddings).any():
                    print(f"⚠️  NaN in fused_embeddings despite sanitization")
                if adjusted_labels is not None:
                    if torch.isnan(adjusted_labels.float()).any():
                        print(f"⚠️  NaN in adjusted_labels")
                    # Check label range
                    valid_labels = adjusted_labels[adjusted_labels != -100]
                    if len(valid_labels) > 0:
                        if valid_labels.min() < 0 or valid_labels.max() >= self.language_model.vocab_size:
                            print(
                                f"⚠️  Invalid label range: [{valid_labels.min()}, {valid_labels.max()}] (vocab: {self.language_model.vocab_size})")
                outputs['lm_loss'] = None
            else:
                outputs['lm_loss'] = lm_outputs.loss

            outputs['logits'] = lm_outputs.logits
            outputs['hidden_states'] = lm_outputs.hidden_states

        alignment_weight = _get_training_attr('alignment_loss_weight', 1.0)
        memory_weight = _get_training_attr('memory_kl_weight', 0.01)
        addressing_weight = _get_training_attr('addressing_kl_weight', 0.001)

        device = fused_embeddings.device
        total_loss = None

        # LM loss (primary component)
        if outputs.get('lm_loss') is not None and not torch.isnan(outputs['lm_loss']):
            total_loss = lm_weight * outputs['lm_loss']
        else:
            # If LM loss is invalid, create a zero tensor with gradient
            total_loss = torch.tensor(0.0, device=device, requires_grad=True)

        # Add alignment loss if valid
        if 'alignment_loss' in outputs:
            align_loss = outputs['alignment_loss']
            if not torch.isnan(align_loss) and not torch.isinf(align_loss):
                total_loss = total_loss + alignment_weight * align_loss
        
        # Add fine-grained alignment loss if valid
        # Weight: Use same as alignment for now, can be tuned separately
        fine_grained_weight = _get_training_attr('fine_grained_loss_weight', alignment_weight * 0.5)
        if 'fine_grained_loss' in outputs:
            fg_loss = outputs['fine_grained_loss']
            if not torch.isnan(fg_loss) and not torch.isinf(fg_loss):
                total_loss = total_loss + fine_grained_weight * fg_loss

        # Add memory losses if valid
        if use_memory and 'memory_kl' in outputs:
            mem_kl = outputs['memory_kl']
            if not torch.isnan(mem_kl) and not torch.isinf(mem_kl):
                total_loss = total_loss + memory_weight * mem_kl

        if use_memory and 'addressing_kl' in outputs:
            addr_kl = outputs['addressing_kl']
            if not torch.isnan(addr_kl) and not torch.isinf(addr_kl):
                total_loss = total_loss + addressing_weight * addr_kl

        outputs['loss'] = total_loss

        return outputs

    def generate(self, images, prompt, max_length=50, temperature=0.7, top_p=0.9):
        """
        Generate text from images and prompt

        Args:
            images: (batch_size, 3, H, W) or list of PIL Images
            prompt: text prompt (string or list of strings)
            max_length: maximum generation length
            temperature: sampling temperature
            top_p: nucleus sampling parameter

        Returns:
            generated_text: list of generated strings
        """
        self.eval()

        with torch.no_grad():
            # Tokenize prompt
            if isinstance(prompt, str):
                prompt = [prompt]

            tokenizer = self.language_model.tokenizer
            inputs = tokenizer(
                prompt,
                return_tensors='pt',
                padding=True,
                truncation=True
            ).to(images.device)

            # Encode image
            prefix_tokens, _ = self.encode_image(images)

            # Get text embeddings
            text_embeddings = self.language_model.get_input_embeddings()(inputs.input_ids)

            # Fuse
            fused_embeddings, fused_mask = self.fusion(
                prefix_tokens, text_embeddings, inputs.attention_mask
            )

            # Generate
            outputs = self.language_model.generate(
                inputs_embeds=fused_embeddings,
                attention_mask=fused_mask,
                max_length=max_length,
                temperature=temperature,
                top_p=top_p,
                do_sample=True
            )

            # Decode
            generated_text = tokenizer.batch_decode(
                outputs, skip_special_tokens=True)

        return generated_text

    def freeze_vision_encoder(self):
        """Freeze vision encoder parameters"""
        for param in self.vision_encoder.parameters():
            param.requires_grad = False
        print("Vision encoder frozen")

    def freeze_language_model(self, unfreeze_last_n=0):
        """Freeze language model, optionally unfreezing last n layers"""
        if unfreeze_last_n > 0:
            self.language_model.freeze_layers(
                num_layers_to_freeze=self.language_model.num_layers - unfreeze_last_n
            )
        else:
            for param in self.language_model.parameters():
                param.requires_grad = False
        print(
            f"Language model frozen (last {unfreeze_last_n} layers unfrozen)")

    def get_trainable_params(self):
        """Get count of trainable parameters"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def get_text_to_patch_attention(self):
        """
        Get the last computed text-to-patch attention weights for visualization.
        
        Returns:
            attention: (batch_size, seq_len, num_patches) or None
        """
        return self._last_text_to_patch_attention
    
    def get_pooling_attention(self):
        """
        Get the adapter's pooling attention showing which patches each prefix token attends to.
        
        Returns:
            attention: (batch_size, k_prefix, num_patches) or None
        """
        return self.multimodal_adapter.get_pooling_attention_weights()

    def save_checkpoint(self, path):
        """Save model checkpoint"""
        checkpoint = {
            'model_state_dict': self.state_dict(),
            'config': self.config,
            'memory_state': self.memory_state
        }
        torch.save(checkpoint, path)
        print(f"Checkpoint saved to {path}")

    def load_checkpoint(self, path):
        """Load model checkpoint"""
        checkpoint = torch.load(path, map_location='cpu')
        self.load_state_dict(checkpoint['model_state_dict'])
        self.memory_state = checkpoint.get('memory_state', None)
        print(f"Checkpoint loaded from {path}")


def create_microvlm(config, vision_checkpoint=None, language_checkpoint=None,
                    quantize_4bit=False, quantize_memory_158bit=False,
                    training_config=None):
    """
    Factory function to create MicroVLM model

    Args:
        config: model configuration dictionary
        vision_checkpoint: path to DeiT checkpoint
        language_checkpoint: path to Qwen checkpoint or HF model name
        quantize_4bit: whether to apply 4-bit quantization
        quantize_memory_158bit: whether to apply 1.58-bit memory quantization
        training_config: optional training config for loss weighting

    Returns:
        model: MicroVLM instance
    """
    model = MicroVLM(
        config,
        vision_checkpoint=vision_checkpoint,
        language_checkpoint=language_checkpoint,
        quantize_4bit=quantize_4bit,
        quantize_memory_158bit=quantize_memory_158bit,
        training_config=training_config
    )
    return model
