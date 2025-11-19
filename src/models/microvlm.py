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
from .multimodal_adapter import MultimodalAdapter, ContrastiveAlignmentLoss, MultimodalFusion
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
                 quantize_4bit=False, quantize_memory_158bit=False):
        super().__init__()
        
        self.config = config
        self.quantize_4bit = quantize_4bit
        self.quantize_memory_158bit = quantize_memory_158bit
        
        # Vision encoder: DeiT-Tiny
        self.vision_encoder = DeiTVisionEncoder(config, pretrained_path=vision_checkpoint)
        
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
        
        # Image feature projection for alignment (DeiT 192-dim -> Qwen 896-dim)
        self.image_proj_for_alignment = nn.Linear(
            config.get('vision_hidden_size', 192), 
            config.get('language_hidden_size', 896)
        )
        
        # Alignment loss
        self.alignment_loss = ContrastiveAlignmentLoss(
            temperature=config.get('alignment_temperature', 0.07)
        )
        
        # Memory state (persistent across forward passes)
        self.memory_state = None
        
        # Training configuration
        self.use_memory = True
        self.use_alignment = True
    
    def encode_image(self, images):
        """
        Encode images to prefix tokens
        
        Args:
            images: (batch_size, 3, H, W) or list of PIL Images
        
        Returns:
            prefix_tokens: (batch_size, k_prefix, qwen_dim)
            image_features: (batch_size, qwen_dim) for alignment - projected to same space as text
        """
        # Extract patch embeddings
        patch_embeddings = self.vision_encoder(images)  # (B, num_patches, deit_dim)
        
        # Get image-level features (CLS token)
        raw_image_features = self.vision_encoder.get_cls_token(images)  # (B, deit_dim=192)
        
        # Project image features to language space for alignment
        image_features = self.image_proj_for_alignment(raw_image_features)  # (B, qwen_dim=896)
        
        # Project to language space
        prefix_tokens = self.multimodal_adapter(patch_embeddings)  # (B, k_prefix, qwen_dim)
        
        return prefix_tokens, image_features
    
    def encode_text(self, input_ids, attention_mask=None):
        """
        Encode text to embeddings
        
        Args:
            input_ids: (batch_size, seq_len)
            attention_mask: (batch_size, seq_len)
        
        Returns:
            text_embeddings: (batch_size, seq_len, qwen_dim)
            text_features: (batch_size, qwen_dim) for alignment
        """
        # Get embeddings
        text_embeddings = self.language_model.get_input_embeddings()(input_ids)
        
        # Get text-level features (mean pooling)
        if attention_mask is not None:
            mask_expanded = attention_mask.unsqueeze(-1).expand(text_embeddings.size())
            sum_embeddings = torch.sum(text_embeddings * mask_expanded, dim=1)
            sum_mask = torch.clamp(mask_expanded.sum(dim=1), min=1e-9)
            text_features = sum_embeddings / sum_mask
        else:
            text_features = text_embeddings.mean(dim=1)
        
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
        
        # Encode images
        prefix_tokens, image_features = None, None
        if images is not None:
            prefix_tokens, image_features = self.encode_image(images)
        
        # Encode text
        text_embeddings, text_features = self.encode_text(input_ids, attention_mask)
        
        # Compute alignment loss if both modalities present
        if use_alignment and image_features is not None:
            alignment_loss = self.alignment_loss(image_features, text_features)
            outputs['alignment_loss'] = alignment_loss
        
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
        
        outputs['lm_loss'] = lm_outputs.loss
        outputs['logits'] = lm_outputs.logits
        outputs['hidden_states'] = lm_outputs.hidden_states
        
        # Compute total loss - ensure it's a tensor
        if lm_outputs.loss is not None:
            total_loss = lm_outputs.loss
        else:
            # Create zero tensor on same device as model
            device = next(self.parameters()).device
            total_loss = torch.tensor(0.0, device=device, requires_grad=True)
        
        if 'alignment_loss' in outputs:
            total_loss = total_loss + 0.1 * outputs['alignment_loss']
        
        if use_memory:
            # Add memory KL losses
            total_loss = total_loss + 0.01 * outputs['memory_kl']
            total_loss = total_loss + 0.001 * outputs['addressing_kl']
        
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
            generated_text = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        
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
        print(f"Language model frozen (last {unfreeze_last_n} layers unfrozen)")
    
    def get_trainable_params(self):
        """Get count of trainable parameters"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
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
                    quantize_4bit=False):
    """
    Factory function to create MicroVLM model
    
    Args:
        config: model configuration dictionary
        vision_checkpoint: path to DeiT checkpoint
        language_checkpoint: path to Qwen checkpoint or HF model name
        quantize_4bit: whether to apply 4-bit quantization
    
    Returns:
        model: MicroVLM instance
    """
    model = MicroVLM(
        config,
        vision_checkpoint=vision_checkpoint,
        language_checkpoint=language_checkpoint,
        quantize_4bit=quantize_4bit
    )
    return model
