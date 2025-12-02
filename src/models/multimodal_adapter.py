"""
Multimodal Adapter Module
Projects vision embeddings to language model space following EVO-1 methodology
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class MultimodalAdapter(nn.Module):
    """
    Adapts vision encoder outputs to language model input space
    Implements projection, pooling, and positional encoding
    
    Key insight: We preserve the original patch embeddings for visualization
    and fine-grained text-to-image attention, while also producing pooled
    prefix tokens for efficient LM processing.
    """
    
    def __init__(self, config):
        super().__init__()
        
        self.deit_dim = config.get('deit_embed_dim', 192)
        self.qwen_dim = config.get('qwen_hidden_dim', 896)
        self.num_patches = config.get('num_patches', 196)
        self.k_prefix = config.get('k_prefix', 25)
        
        # Projection layer: deit_dim -> qwen_dim
        self.projection = nn.Linear(self.deit_dim, self.qwen_dim)
        
        # Optional small MLP for additional capacity
        self.mlp = nn.Sequential(
            nn.Linear(self.qwen_dim, self.qwen_dim * 2),
            nn.GELU(),
            nn.Linear(self.qwen_dim * 2, self.qwen_dim)
        )
        
        # Pooling layer to reduce patches to K_prefix tokens
        # Using learned pooling with attention mechanism
        self.pooling_queries = nn.Parameter(torch.randn(self.k_prefix, self.qwen_dim))
        self.pooling_attn = nn.MultiheadAttention(
            embed_dim=self.qwen_dim,
            num_heads=4,
            batch_first=True
        )
        
        # Learned positional embeddings for prefix tokens
        self.prefix_pos_embeddings = nn.Parameter(torch.randn(self.k_prefix, self.qwen_dim))
        
        # Layer normalization
        self.layer_norm = nn.LayerNorm(self.qwen_dim)
        
        # Store last pooling attention weights for visualization
        # This shows which patches each prefix token attends to
        self._last_pooling_attn_weights = None
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights"""
        nn.init.xavier_uniform_(self.projection.weight)
        nn.init.zeros_(self.projection.bias)
        
        # Initialize pooling queries with small random values
        nn.init.normal_(self.pooling_queries, mean=0.0, std=0.02)
        nn.init.normal_(self.prefix_pos_embeddings, mean=0.0, std=0.02)
        
        # Initialize MLP
        for module in self.mlp.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(self, vision_embeddings, return_patch_embeddings=False):
        """
        Forward pass
        
        Args:
            vision_embeddings: (batch_size, num_patches, deit_dim)
            return_patch_embeddings: if True, also return projected patch embeddings
        
        Returns:
            prefix_tokens: (batch_size, k_prefix, qwen_dim)
            patch_embeddings_proj: (batch_size, num_patches, qwen_dim) if return_patch_embeddings
        """
        batch_size = vision_embeddings.size(0)
        
        # Project to qwen dimension: (B, num_patches, qwen_dim)
        projected = self.projection(vision_embeddings)
        
        # Apply MLP
        projected = self.mlp(projected)
        
        # Store projected patch embeddings before pooling (for attention visualization)
        patch_embeddings_proj = projected
        
        # Expand pooling queries for batch
        queries = self.pooling_queries.unsqueeze(0).expand(batch_size, -1, -1)
        
        # Apply cross-attention pooling with attention weights
        pooled, attn_weights = self.pooling_attn(
            query=queries,
            key=projected,
            value=projected,
            need_weights=True,
            average_attn_weights=True  # Average across heads
        )
        
        # Store attention weights for visualization: (B, k_prefix, num_patches)
        self._last_pooling_attn_weights = attn_weights.detach()
        
        # Add positional embeddings
        prefix_tokens = pooled + self.prefix_pos_embeddings.unsqueeze(0)
        
        # Layer normalization
        prefix_tokens = self.layer_norm(prefix_tokens)
        
        if return_patch_embeddings:
            return prefix_tokens, patch_embeddings_proj
        return prefix_tokens
    
    def get_pooling_attention_weights(self):
        """
        Get the last pooling attention weights showing which patches each prefix token attends to.
        
        Returns:
            attn_weights: (batch_size, k_prefix, num_patches) or None
        """
        return self._last_pooling_attn_weights


class ContrastiveAlignmentLoss(nn.Module):
    """
    CLIP-style contrastive alignment loss with learnable temperature
    Implements image-text alignment learning with improved stability
    """
    
    def __init__(self, temperature=0.07, learnable_temperature=True, label_smoothing=0.1):
        super().__init__()
        self.learnable_temperature = learnable_temperature
        self.label_smoothing = label_smoothing
        
        if learnable_temperature:
            # Learnable log temperature (CLIP style) - initialized to log(1/0.07) ≈ 2.66
            # Clamped to prevent instability
            self.logit_scale = nn.Parameter(torch.ones([]) * math.log(1 / temperature))
        else:
            self.register_buffer('logit_scale', torch.ones([]) * math.log(1 / temperature))
        
        # Use label smoothing for better generalization
        self.cross_entropy = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    
    def forward(self, image_features, text_features):
        """
        Compute contrastive alignment loss
        
        Args:
            image_features: (batch_size, hidden_dim)
            text_features: (batch_size, hidden_dim)
        
        Returns:
            loss: scalar tensor
        """
        # Normalize features (L2 normalization)
        image_features = F.normalize(image_features, p=2, dim=-1)
        text_features = F.normalize(text_features, p=2, dim=-1)
        
        # Clamp logit_scale to prevent NaN/explosion (CLIP uses 0-100 range)
        logit_scale = torch.clamp(self.logit_scale, min=0, max=4.6052)  # max = log(100)
        
        # Compute similarity matrix with learnable temperature
        # shape: (batch_size, batch_size)
        logits_per_image = logit_scale.exp() * torch.matmul(image_features, text_features.t())
        logits_per_text = logits_per_image.t()
        
        # Labels: diagonal elements are positive pairs
        batch_size = image_features.size(0)
        labels = torch.arange(batch_size, device=image_features.device)
        
        # Bidirectional loss (image->text and text->image)
        loss_i2t = self.cross_entropy(logits_per_image, labels)
        loss_t2i = self.cross_entropy(logits_per_text, labels)
        
        loss = (loss_i2t + loss_t2i) / 2
        
        return loss


class FineGrainedAlignmentLoss(nn.Module):
    """
    Fine-grained text-to-patch alignment loss
    
    Encourages each text token to attend to semantically relevant image patches.
    This provides explicit supervision for text-conditioned attention learning.
    
    Two components:
    1. Token-to-patch attention entropy regularization (encourages focused attention)
    2. Attention diversity loss (encourages different tokens to attend to different regions)
    """
    
    def __init__(self, temperature=0.1, entropy_weight=0.5, diversity_weight=0.5):
        super().__init__()
        self.temperature = temperature
        self.entropy_weight = entropy_weight
        self.diversity_weight = diversity_weight
    
    def forward(self, patch_embeddings, text_embeddings, attention_mask=None):
        """
        Compute fine-grained alignment loss
        
        Args:
            patch_embeddings: (B, num_patches, D) - projected patch embeddings
            text_embeddings: (B, seq_len, D) - text token embeddings
            attention_mask: (B, seq_len) - text attention mask
        
        Returns:
            loss: scalar tensor
            attention_weights: (B, seq_len, num_patches) for visualization
        """
        # Normalize embeddings
        patch_norm = F.normalize(patch_embeddings, p=2, dim=-1)
        text_norm = F.normalize(text_embeddings, p=2, dim=-1)
        
        # Compute text-to-patch attention: (B, seq_len, num_patches)
        attention_logits = torch.bmm(text_norm, patch_norm.transpose(1, 2)) / self.temperature
        attention_weights = F.softmax(attention_logits, dim=-1)
        
        # Apply attention mask if provided
        if attention_mask is not None:
            # Expand mask to (B, seq_len, 1) for broadcasting
            mask = attention_mask.unsqueeze(-1).float()
            attention_weights = attention_weights * mask
        
        # Component 1: Attention entropy loss
        # Low entropy = focused attention on specific patches (desirable)
        # We minimize negative entropy (maximize focus)
        eps = 1e-8
        entropy = -(attention_weights * torch.log(attention_weights + eps)).sum(dim=-1)  # (B, seq_len)
        max_entropy = math.log(patch_embeddings.size(1))  # Maximum possible entropy
        normalized_entropy = entropy / max_entropy  # Normalize to [0, 1]
        
        if attention_mask is not None:
            # Average only over valid tokens
            valid_tokens = attention_mask.sum(dim=-1).clamp(min=1)
            entropy_loss = (normalized_entropy * attention_mask).sum(dim=-1) / valid_tokens
        else:
            entropy_loss = normalized_entropy.mean(dim=-1)
        entropy_loss = entropy_loss.mean()  # Average over batch
        
        # Component 2: Attention diversity loss
        # Different tokens should attend to different patches
        # We want the attention patterns to be diverse across tokens
        if attention_mask is not None:
            # Compute mean attention pattern per sample
            valid_tokens = attention_mask.sum(dim=-1, keepdim=True).clamp(min=1)
            mean_attention = (attention_weights * attention_mask.unsqueeze(-1)).sum(dim=1) / valid_tokens
        else:
            mean_attention = attention_weights.mean(dim=1)  # (B, num_patches)
        
        # Diversity: maximize variance across patches (uniform is bad, peaked is good)
        # Low diversity means all tokens attend to the same patches
        diversity = mean_attention.var(dim=-1).mean()  # Higher is better
        diversity_loss = 1.0 - diversity.clamp(max=1.0)  # Convert to loss (minimize)
        
        # Combined loss
        loss = self.entropy_weight * entropy_loss + self.diversity_weight * diversity_loss
        
        return loss, attention_weights


class MultimodalFusion(nn.Module):
    """
    EVO-1 style multimodal fusion
    Combines image prefix tokens with text tokens
    """
    
    def __init__(self):
        super().__init__()
    
    def forward(self, prefix_tokens, text_embeddings, attention_mask=None):
        """
        Fuse image prefix tokens with text embeddings
        
        Args:
            prefix_tokens: (batch_size, k_prefix, hidden_dim)
            text_embeddings: (batch_size, seq_len, hidden_dim)
            attention_mask: (batch_size, seq_len) optional
        
        Returns:
            fused_embeddings: (batch_size, k_prefix + seq_len, hidden_dim)
            fused_attention_mask: (batch_size, k_prefix + seq_len)
        """
        # Concatenate: [prefix_tokens, text_embeddings]
        fused_embeddings = torch.cat([prefix_tokens, text_embeddings], dim=1)
        
        # Update attention mask if provided
        if attention_mask is not None:
            batch_size = prefix_tokens.size(0)
            k_prefix = prefix_tokens.size(1)
            
            # Prefix tokens are always attended to
            prefix_mask = torch.ones(batch_size, k_prefix, 
                                    dtype=attention_mask.dtype,
                                    device=attention_mask.device)
            
            fused_attention_mask = torch.cat([prefix_mask, attention_mask], dim=1)
        else:
            fused_attention_mask = None
        
        return fused_embeddings, fused_attention_mask
