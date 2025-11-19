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
    
    def forward(self, vision_embeddings):
        """
        Forward pass
        
        Args:
            vision_embeddings: (batch_size, num_patches, deit_dim)
        
        Returns:
            prefix_tokens: (batch_size, k_prefix, qwen_dim)
        """
        batch_size = vision_embeddings.size(0)
        
        # Project to qwen dimension: (B, num_patches, qwen_dim)
        projected = self.projection(vision_embeddings)
        
        # Apply MLP
        projected = self.mlp(projected)
        
        # Expand pooling queries for batch
        queries = self.pooling_queries.unsqueeze(0).expand(batch_size, -1, -1)
        
        # Apply cross-attention pooling
        # queries attend to all patch tokens
        pooled, _ = self.pooling_attn(
            query=queries,
            key=projected,
            value=projected
        )
        
        # Add positional embeddings
        prefix_tokens = pooled + self.prefix_pos_embeddings.unsqueeze(0)
        
        # Layer normalization
        prefix_tokens = self.layer_norm(prefix_tokens)
        
        return prefix_tokens


class ContrastiveAlignmentLoss(nn.Module):
    """
    EVO-1 style contrastive alignment loss
    Implements image-text alignment learning
    """
    
    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature
        self.cross_entropy = nn.CrossEntropyLoss()
    
    def forward(self, image_features, text_features):
        """
        Compute contrastive alignment loss
        
        Args:
            image_features: (batch_size, hidden_dim)
            text_features: (batch_size, hidden_dim)
        
        Returns:
            loss: scalar tensor
        """
        # Normalize features
        image_features = F.normalize(image_features, p=2, dim=-1)
        text_features = F.normalize(text_features, p=2, dim=-1)
        
        # Compute similarity matrix
        logits = torch.matmul(image_features, text_features.t()) / self.temperature
        
        # Labels: diagonal elements are positive pairs
        batch_size = image_features.size(0)
        labels = torch.arange(batch_size, device=image_features.device)
        
        # Bidirectional loss
        loss_i2t = self.cross_entropy(logits, labels)
        loss_t2i = self.cross_entropy(logits.t(), labels)
        
        loss = (loss_i2t + loss_t2i) / 2
        
        return loss


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
