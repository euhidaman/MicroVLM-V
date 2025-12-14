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
    
    NOTE: Reduced for compact model (<1GB target):
    - k_prefix: 25 -> 8
    - MLP expansion: 2x -> 1x (removed expansion)
    - Pooling heads: 4 -> 2
    """
    
    def __init__(self, config):
        super().__init__()
        
        self.deit_dim = getattr(config, 'deit_embed_dim', 192)
        self.qwen_dim = getattr(config, 'qwen_hidden_dim', 896)
        self.num_patches = getattr(config, 'num_patches', 196)
        self.k_prefix = getattr(config, 'k_prefix', 8)  # Reduced from 25

        # Projection layer: deit_dim -> qwen_dim
        self.projection = nn.Linear(self.deit_dim, self.qwen_dim)
        
        # Simplified MLP without expansion (reduced from 2x expansion)
        self.mlp = nn.Sequential(
            nn.Linear(self.qwen_dim, self.qwen_dim),
            nn.GELU(),
            nn.Linear(self.qwen_dim, self.qwen_dim)
        )
        
        # Pooling layer to reduce patches to K_prefix tokens
        # Using learned pooling with attention mechanism
        self.pooling_queries = nn.Parameter(torch.randn(self.k_prefix, self.qwen_dim))
        self.pooling_attn = nn.MultiheadAttention(
            embed_dim=self.qwen_dim,
            num_heads=2,  # Reduced from 4
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
        nn.init.xavier_uniform_(self.projection.weight, gain=0.01)
        nn.init.zeros_(self.projection.bias)
        
        # Initialize pooling queries with small random values
        nn.init.normal_(self.pooling_queries, mean=0.0, std=0.001)
        nn.init.normal_(self.prefix_pos_embeddings, mean=0.0, std=0.001)

        # Initialize MLP with very small weights
        for module in self.mlp.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight, gain=0.001)
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
        
        # Sanitize input
        vision_embeddings = torch.nan_to_num(vision_embeddings, nan=0.0, posinf=1e2, neginf=-1e2)
        vision_embeddings = torch.clamp(vision_embeddings, min=-1e2, max=1e2)

        # Project to qwen dimension: (B, num_patches, qwen_dim)
        projected = self.projection(vision_embeddings)
        projected = torch.nan_to_num(projected, nan=0.0, posinf=1e2, neginf=-1e2)
        projected = torch.clamp(projected, min=-1e2, max=1e2)

        # Apply MLP (removed residual to prevent instability)
        projected = self.mlp(projected)
        projected = torch.nan_to_num(projected, nan=0.0, posinf=1e2, neginf=-1e2)
        projected = torch.clamp(projected, min=-1e2, max=1e2)

        # Store projected patch embeddings before pooling (for attention visualization)
        patch_embeddings_proj = projected
        
        # Expand pooling queries for batch
        queries = self.pooling_queries.unsqueeze(0).expand(batch_size, -1, -1)
        queries = torch.nan_to_num(queries, nan=0.0, posinf=1e2, neginf=-1e2)

        # Apply cross-attention pooling with attention weights
        pooled, attn_weights = self.pooling_attn(
            query=queries,
            key=projected,
            value=projected,
            need_weights=True,
            average_attn_weights=True  # Average across heads
        )
        
        pooled = torch.nan_to_num(pooled, nan=0.0, posinf=1e2, neginf=-1e2)
        pooled = torch.clamp(pooled, min=-1e2, max=1e2)

        # Store attention weights for visualization: (B, k_prefix, num_patches)
        self._last_pooling_attn_weights = attn_weights.detach()
        
        # Add positional embeddings (no scaling due to small init)
        pos_emb = torch.nan_to_num(self.prefix_pos_embeddings, nan=0.0, posinf=1e2, neginf=-1e2)
        prefix_tokens = pooled + pos_emb.unsqueeze(0)
        prefix_tokens = torch.clamp(prefix_tokens, min=-1e2, max=1e2)

        # Layer normalization
        prefix_tokens = self.layer_norm(prefix_tokens)
        prefix_tokens = torch.nan_to_num(prefix_tokens, nan=0.0, posinf=1e2, neginf=-1e2)

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
    
    STABILITY FIXES:
    - Temperature bounded via soft clamping for smooth gradients
    - Logits clamped to prevent overflow
    """
    
    def __init__(self, temperature=0.07, learnable_temperature=True, label_smoothing=0.1,
                 min_temperature=0.01, max_temperature=0.5):
        super().__init__()
        self.learnable_temperature = learnable_temperature
        self.label_smoothing = label_smoothing
        
        # STABILITY FIX: Store bounds for soft clamping
        self.log_scale_min = math.log(1 / max_temperature)  # ~0.7
        self.log_scale_max = math.log(1 / min_temperature)  # ~4.6
        
        if learnable_temperature:
            # Initialize in middle of valid range
            init_log_scale = math.log(1 / temperature)
            self._logit_scale_raw = nn.Parameter(torch.tensor(init_log_scale))
        else:
            self.register_buffer('_logit_scale_raw', torch.tensor(math.log(1 / temperature)))
        
        # Use label smoothing for better generalization
        self.cross_entropy = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    
    @property
    def logit_scale(self):
        """Get bounded logit scale using soft clamping for smooth gradients"""
        range_size = self.log_scale_max - self.log_scale_min
        normalized = torch.sigmoid(self._logit_scale_raw - (self.log_scale_min + range_size / 2))
        return self.log_scale_min + normalized * range_size
    
    def forward(self, image_features, text_features):
        """
        Compute contrastive alignment loss
        
        Args:
            image_features: (batch_size, hidden_dim)
            text_features: (batch_size, hidden_dim)
        
        Returns:
            loss: scalar tensor
        """
        # Sanitize inputs
        image_features = torch.nan_to_num(image_features, nan=0.0, posinf=1e2, neginf=-1e2)
        text_features = torch.nan_to_num(text_features, nan=0.0, posinf=1e2, neginf=-1e2)

        # Normalize features (L2 normalization)
        image_features = F.normalize(image_features.float(), p=2, dim=-1, eps=1e-6)
        text_features = F.normalize(text_features.float(), p=2, dim=-1, eps=1e-6)

        # STABILITY FIX: Use property for bounded logit scale
        logit_scale = torch.clamp(self.logit_scale.exp(), min=1.0, max=100.0)

        # Compute similarity matrix with learnable temperature
        # shape: (batch_size, batch_size)
        logits_per_image = logit_scale * torch.matmul(image_features, text_features.t())
        logits_per_text = logits_per_image.t()
        
        # STABILITY FIX: Clamp logits to prevent extreme values
        logits_per_image = torch.clamp(logits_per_image, min=-50.0, max=50.0)
        logits_per_text = torch.clamp(logits_per_text, min=-50.0, max=50.0)

        # Labels: diagonal elements are positive pairs
        batch_size = image_features.size(0)
        labels = torch.arange(batch_size, device=image_features.device)
        
        # Bidirectional loss (image->text and text->image)
        try:
            loss_i2t = self.cross_entropy(logits_per_image, labels)
            loss_t2i = self.cross_entropy(logits_per_text, labels)
            loss = (loss_i2t + loss_t2i) / 2
        except RuntimeError:
            return torch.tensor(0.0, device=image_features.device, requires_grad=True)

        if torch.isnan(loss) or torch.isinf(loss):
            return torch.tensor(0.0, device=image_features.device, requires_grad=True)

        return loss


class FineGrainedAlignmentLoss(nn.Module):
    """
    Improved fine-grained text-to-patch alignment loss with spatial coherence.
    
    Encourages each text token to attend to semantically relevant image patches
    with spatially coherent attention patterns.
    
    STABILITY FIXES:
    - Removed conflicting sharpness vs smoothness objectives
    - Warmer temperature for smoother gradients
    - Soft focus target instead of pushing to extremes
    - Simplified contrastive grounding
    """
    
    def __init__(self, temperature=0.5, entropy_weight=0.30, diversity_weight=0.20,
                 spatial_weight=0.15, contrastive_weight=0.35,
                 grid_size=14):
        super().__init__()
        self.temperature = temperature
        self.entropy_weight = entropy_weight
        self.diversity_weight = diversity_weight
        self.spatial_weight = spatial_weight
        self.contrastive_weight = contrastive_weight
        self.grid_size = grid_size
    
    def forward(self, patch_embeddings, text_embeddings, attention_mask=None):
        """
        Compute improved fine-grained alignment loss
        
        Args:
            patch_embeddings: (B, num_patches, D) - projected patch embeddings
            text_embeddings: (B, seq_len, D) - text token embeddings
            attention_mask: (B, seq_len) - text attention mask
        
        Returns:
            loss: scalar tensor
            attention_weights: (B, seq_len, num_patches) for visualization
        """
        B, num_patches, D = patch_embeddings.shape
        seq_len = text_embeddings.size(1)
        device = patch_embeddings.device
        eps = 1e-8
        
        # Normalize embeddings - ensure same dtype for mixed precision training
        patch_norm = F.normalize(patch_embeddings.float(), p=2, dim=-1)
        text_norm = F.normalize(text_embeddings.float(), p=2, dim=-1)
        
        # STABILITY FIX: Clamp logits before softmax
        attention_logits = torch.bmm(text_norm, patch_norm.transpose(1, 2)) / self.temperature
        attention_logits = torch.clamp(attention_logits, min=-20.0, max=20.0)
        attention_weights = F.softmax(attention_logits, dim=-1)
        
        # Apply attention mask if provided
        if attention_mask is not None:
            mask = attention_mask.unsqueeze(-1).float()
            valid_tokens = attention_mask.sum(dim=-1, keepdim=True).clamp(min=1).float()
        else:
            mask = torch.ones(B, seq_len, 1, device=device)
            valid_tokens = torch.tensor(seq_len, device=device, dtype=torch.float32).view(1, 1)
        
        # ====== Component 1: Soft Focus Loss (Target moderate entropy) ======
        entropy = -(attention_weights * torch.log(attention_weights + eps)).sum(dim=-1)
        max_entropy = math.log(num_patches)
        normalized_entropy = entropy / max_entropy
        
        # STABILITY FIX: Target 40% entropy instead of minimum
        focus_loss = F.smooth_l1_loss(
            normalized_entropy,
            torch.full_like(normalized_entropy, 0.4),
            reduction='none'
        )
        focus_loss = (focus_loss * attention_mask.float()).sum(dim=-1) / valid_tokens.squeeze(-1) if attention_mask is not None else focus_loss.mean(dim=-1)
        focus_loss = focus_loss.mean()
        
        # ====== Component 2: Local Spatial Coherence ======
        H = W = self.grid_size
        if num_patches == H * W:
            attention_grid = attention_weights.view(B, seq_len, H, W)
            
            # Only penalize variation in significant attention regions
            mean_attn = attention_grid.mean(dim=(2, 3), keepdim=True)
            significant_mask = (attention_grid > mean_attn).float()
            
            tv_h = torch.abs(attention_grid[:, :, 1:, :] - attention_grid[:, :, :-1, :])
            tv_w = torch.abs(attention_grid[:, :, :, 1:] - attention_grid[:, :, :, :-1])
            
            masked_tv_h = (tv_h * significant_mask[:, :, 1:, :]).sum() / (significant_mask[:, :, 1:, :].sum() + eps)
            masked_tv_w = (tv_w * significant_mask[:, :, :, 1:]).sum() / (significant_mask[:, :, :, 1:].sum() + eps)
            
            spatial_loss = (masked_tv_h + masked_tv_w) * 0.25
        else:
            spatial_loss = torch.tensor(0.0, device=device)
        
        # ====== Component 3: Soft Diversity Loss ======
        mean_attention = (attention_weights * mask).sum(dim=1) / valid_tokens
        mean_attn_entropy = -(mean_attention * torch.log(mean_attention + eps)).sum(dim=-1)
        max_coverage_entropy = math.log(num_patches)
        target_coverage = 0.3 * max_coverage_entropy
        diversity_loss = F.relu(target_coverage - mean_attn_entropy).mean()
        
        # ====== Component 4: Simplified Contrastive Grounding ======
        attended_patches = torch.bmm(attention_weights, patch_norm)
        positive_sim = (text_norm * attended_patches).sum(dim=-1)
        
        # Use batch mean as simple negative
        batch_mean_patches = patch_norm.mean(dim=1, keepdim=True).expand(-1, seq_len, -1)
        baseline_sim = (text_norm * batch_mean_patches).sum(dim=-1)
        
        margin = 0.1
        contrastive_loss = F.relu(baseline_sim - positive_sim + margin)
        contrastive_loss = (contrastive_loss * attention_mask.float()).sum(dim=-1) / valid_tokens.squeeze(-1) if attention_mask is not None else contrastive_loss.mean(dim=-1)
        contrastive_loss = contrastive_loss.mean()
        
        # ====== Combine losses ======
        loss = (
            self.entropy_weight * focus_loss +
            self.spatial_weight * spatial_loss +
            self.diversity_weight * diversity_loss +
            self.contrastive_weight * contrastive_loss
        )
        
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
