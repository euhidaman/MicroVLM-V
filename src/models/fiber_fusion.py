"""
FIBER-style Fusion-in-Backbone Module for Vision-Language Alignment

Implements cross-modal attention blocks that inject text information into the vision backbone
and vision information into the text backbone, following Microsoft's FIBER approach.

Key differences from baseline:
1. Cross-modal attention at intermediate layers (not just at the top)
2. Bidirectional: Image-to-Text AND Text-to-Image attention
3. Learnable gating (alpha) to control fusion strength
4. Compatible with both DeiT-Tiny vision encoder and Qwen LM

Reference: FIBER: Coarse-to-Fine Vision-Language Pre-training
https://arxiv.org/abs/2206.07643
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple


class CrossModalAttention(nn.Module):
    """
    Cross-modal attention block for FIBER-style fusion.
    
    Computes attention from source modality to target modality:
    - Query from source (e.g., vision tokens)
    - Key/Value from target (e.g., text tokens)
    
    The output is added to the source with a learnable gating parameter (alpha).
    """
    
    def __init__(
        self,
        source_dim: int,
        target_dim: int,
        num_heads: int = 4,
        dropout: float = 0.0,
        use_layer_norm: bool = True
    ):
        super().__init__()
        
        self.source_dim = source_dim
        self.target_dim = target_dim
        self.num_heads = num_heads
        self.head_dim = source_dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        # Query projection from source
        self.q_proj = nn.Linear(source_dim, source_dim)
        
        # Key/Value projection from target to source dim
        self.k_proj = nn.Linear(target_dim, source_dim)
        self.v_proj = nn.Linear(target_dim, source_dim)
        
        # Output projection
        self.out_proj = nn.Linear(source_dim, source_dim)
        
        # Pre-attention layer norm for source
        if use_layer_norm:
            self.norm_source = nn.LayerNorm(source_dim)
        else:
            self.norm_source = nn.Identity()
        
        # Learnable gating parameter (FIBER uses alpha starting at 0)
        # This allows gradual introduction of cross-modal information
        self.alpha = nn.Parameter(torch.zeros(1))
        
        # Dropout
        self.attn_dropout = nn.Dropout(dropout)
        self.proj_dropout = nn.Dropout(dropout)
        
        # Store attention weights for visualization
        self._last_attention_weights = None
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights with small values for stable training"""
        nn.init.xavier_uniform_(self.q_proj.weight, gain=0.1)
        nn.init.xavier_uniform_(self.k_proj.weight, gain=0.1)
        nn.init.xavier_uniform_(self.v_proj.weight, gain=0.1)
        nn.init.xavier_uniform_(self.out_proj.weight, gain=0.1)
        nn.init.zeros_(self.q_proj.bias)
        nn.init.zeros_(self.k_proj.bias)
        nn.init.zeros_(self.v_proj.bias)
        nn.init.zeros_(self.out_proj.bias)
    
    def forward(
        self,
        source: torch.Tensor,
        target: torch.Tensor,
        target_mask: Optional[torch.Tensor] = None,
        return_attention: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass for cross-modal attention.
        
        Args:
            source: (B, seq_source, source_dim) - tokens to update
            target: (B, seq_target, target_dim) - tokens to attend to
            target_mask: (B, seq_target) - attention mask for target
            return_attention: whether to return attention weights
        
        Returns:
            updated_source: (B, seq_source, source_dim)
            attention_weights: (B, num_heads, seq_source, seq_target) if return_attention
        """
        B, seq_source, _ = source.shape
        seq_target = target.size(1)
        
        # Apply layer norm to source
        source_normed = self.norm_source(source)
        
        # Compute Q, K, V
        Q = self.q_proj(source_normed)  # (B, seq_source, source_dim)
        K = self.k_proj(target)          # (B, seq_target, source_dim)
        V = self.v_proj(target)          # (B, seq_target, source_dim)
        
        # Reshape for multi-head attention
        Q = Q.view(B, seq_source, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(B, seq_target, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(B, seq_target, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Compute attention scores
        attn = (Q @ K.transpose(-2, -1)) * self.scale  # (B, num_heads, seq_source, seq_target)
        
        # Apply attention mask if provided
        if target_mask is not None:
            # Expand mask: (B, seq_target) -> (B, 1, 1, seq_target)
            mask = target_mask.unsqueeze(1).unsqueeze(1)
            attn = attn.masked_fill(mask == 0, float('-inf'))
        
        # Softmax and dropout
        attn = F.softmax(attn, dim=-1)
        attn = self.attn_dropout(attn)
        
        # Store for visualization
        self._last_attention_weights = attn.detach()
        
        # Apply attention to values
        out = attn @ V  # (B, num_heads, seq_source, head_dim)
        
        # Reshape back
        out = out.transpose(1, 2).contiguous().view(B, seq_source, self.source_dim)
        
        # Output projection
        out = self.out_proj(out)
        out = self.proj_dropout(out)
        
        # Apply gating and residual
        updated_source = source + self.alpha * out
        
        if return_attention:
            return updated_source, attn
        return updated_source, None
    
    def get_attention_weights(self) -> Optional[torch.Tensor]:
        """Get the last computed attention weights"""
        return self._last_attention_weights


class FIBERFusionBlock(nn.Module):
    """
    FIBER-style bidirectional fusion block.
    
    Implements:
    1. Image-to-Text attention: Vision tokens attend to text tokens
    2. Text-to-Image attention: Text tokens attend to vision tokens
    
    Both directions are computed in parallel and applied with learnable gating.
    """
    
    def __init__(
        self,
        vision_dim: int = 192,
        text_dim: int = 896,
        num_heads: int = 4,
        dropout: float = 0.0,
        bidirectional: bool = True
    ):
        super().__init__()
        
        self.vision_dim = vision_dim
        self.text_dim = text_dim
        self.bidirectional = bidirectional
        
        # Image-to-Text: Vision queries, Text keys/values
        # Vision tokens get enriched with text information
        self.i2t_attention = CrossModalAttention(
            source_dim=vision_dim,
            target_dim=text_dim,
            num_heads=num_heads,
            dropout=dropout
        )
        
        # Text-to-Image: Text queries, Vision keys/values
        # Text tokens get enriched with vision information
        if bidirectional:
            self.t2i_attention = CrossModalAttention(
                source_dim=text_dim,
                target_dim=vision_dim,
                num_heads=num_heads,
                dropout=dropout
            )
        else:
            self.t2i_attention = None
    
    def forward(
        self,
        vision_tokens: torch.Tensor,
        text_tokens: torch.Tensor,
        text_mask: Optional[torch.Tensor] = None,
        return_attention: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[dict]]:
        """
        Forward pass for bidirectional fusion.
        
        Args:
            vision_tokens: (B, num_patches, vision_dim)
            text_tokens: (B, seq_len, text_dim)
            text_mask: (B, seq_len) attention mask for text
            return_attention: whether to return attention weights
        
        Returns:
            updated_vision: (B, num_patches, vision_dim)
            updated_text: (B, seq_len, text_dim)
            attention_dict: dict with 'i2t' and 't2i' attention weights
        """
        attention_dict = {}
        
        # Image-to-Text attention
        updated_vision, i2t_attn = self.i2t_attention(
            vision_tokens, text_tokens, text_mask, return_attention
        )
        if return_attention:
            attention_dict['i2t'] = i2t_attn
        
        # Text-to-Image attention (if bidirectional)
        if self.bidirectional and self.t2i_attention is not None:
            updated_text, t2i_attn = self.t2i_attention(
                text_tokens, vision_tokens, target_mask=None, return_attention=return_attention
            )
            if return_attention:
                attention_dict['t2i'] = t2i_attn
        else:
            updated_text = text_tokens
        
        return updated_vision, updated_text, attention_dict if return_attention else None


class FIBERVisionEncoder(nn.Module):
    """
    DeiT-Tiny vision encoder with FIBER-style fusion layers.
    
    Injects cross-modal attention blocks at specified layers of the transformer.
    This allows text information to guide visual feature extraction at intermediate levels.
    """
    
    def __init__(
        self,
        config: dict,
        pretrained_path: Optional[str] = None,
        fusion_layers: list = None,
        text_dim: int = 896,
        num_fusion_heads: int = 4
    ):
        super().__init__()
        
        self.image_size = config.get('image_size', 224)
        self.patch_size = config.get('patch_size', 16)
        self.hidden_size = config.get('deit_embed_dim', 192)
        self.num_patches = config.get('num_patches', 196)
        self.num_layers = 12  # DeiT-Tiny has 12 layers
        
        # Fusion layer indices (0-indexed, default: last 4 layers)
        self.fusion_layers = fusion_layers or [8, 9, 10, 11]
        
        # Image preprocessing
        from torchvision import transforms
        self.preprocess = transforms.Compose([
            transforms.Resize((self.image_size, self.image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        
        # Patch embedding layer
        self.patch_embed = nn.Conv2d(
            in_channels=3,
            out_channels=self.hidden_size,
            kernel_size=self.patch_size,
            stride=self.patch_size
        )
        
        # CLS token
        self.cls_token = nn.Parameter(torch.randn(1, 1, self.hidden_size))
        
        # Position embeddings
        self.pos_embed = nn.Parameter(
            torch.randn(1, 1 + self.num_patches, self.hidden_size)
        )
        
        # Transformer encoder layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.hidden_size,
            nhead=3,  # DeiT-Tiny uses 3 attention heads
            dim_feedforward=768,
            dropout=0.0,
            activation='gelu',
            batch_first=True,
            norm_first=False
        )
        
        # Create individual layers instead of TransformerEncoder for fusion injection
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=self.hidden_size,
                nhead=3,
                dim_feedforward=768,
                dropout=0.0,
                activation='gelu',
                batch_first=True,
                norm_first=False
            )
            for _ in range(self.num_layers)
        ])
        
        # FIBER fusion blocks at specified layers
        self.fusion_blocks = nn.ModuleDict()
        for layer_idx in self.fusion_layers:
            self.fusion_blocks[str(layer_idx)] = FIBERFusionBlock(
                vision_dim=self.hidden_size,
                text_dim=text_dim,
                num_heads=num_fusion_heads,
                dropout=0.0,
                bidirectional=False  # Only I2T for vision encoder
            )
        
        # Layer norm
        self.norm = nn.LayerNorm(self.hidden_size)
        
        self._init_weights()
        
        if pretrained_path:
            self.load_pretrained(pretrained_path)
    
    def _init_weights(self):
        """Initialize weights"""
        nn.init.xavier_uniform_(self.patch_embed.weight)
        nn.init.zeros_(self.patch_embed.bias)
        nn.init.normal_(self.cls_token, std=0.02)
        nn.init.normal_(self.pos_embed, std=0.02)
    
    def load_pretrained(self, checkpoint_path: str):
        """Load pretrained DeiT-Tiny weights"""
        print(f"Loading pretrained DeiT weights from {checkpoint_path}")
        state_dict = torch.load(checkpoint_path, map_location='cpu')
        
        if 'model' in state_dict:
            state_dict = state_dict['model']
        
        # Load with partial matching (fusion blocks won't match)
        self.load_state_dict(state_dict, strict=False)
        print("Pretrained weights loaded successfully")
    
    def forward(
        self,
        images: torch.Tensor,
        text_embeddings: Optional[torch.Tensor] = None,
        text_mask: Optional[torch.Tensor] = None,
        return_all_layers: bool = False
    ) -> Tuple[torch.Tensor, Optional[list], Optional[dict]]:
        """
        Forward pass with optional FIBER fusion.
        
        Args:
            images: (B, 3, H, W)
            text_embeddings: (B, seq_len, text_dim) - for fusion
            text_mask: (B, seq_len) - attention mask for text
            return_all_layers: whether to return intermediate layer outputs
        
        Returns:
            patch_tokens: (B, num_patches, hidden_size)
            layer_outputs: list of intermediate outputs if return_all_layers
            fusion_attention: dict of attention weights from fusion blocks
        """
        if isinstance(images, list):
            images = torch.stack([self.preprocess(img) for img in images])
        
        batch_size = images.size(0)
        
        # Patch embedding
        patch_embeds = self.patch_embed(images)
        patch_embeds = patch_embeds.flatten(2).transpose(1, 2)
        
        # Add CLS token
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        embeddings = torch.cat([cls_tokens, patch_embeds], dim=1)
        
        # Add positional embeddings
        embeddings = embeddings + self.pos_embed
        
        # Store layer outputs and fusion attention
        layer_outputs = [] if return_all_layers else None
        fusion_attention = {}
        
        # Process through transformer layers with FIBER fusion
        hidden_states = embeddings
        for layer_idx, layer in enumerate(self.layers):
            # Standard transformer layer
            hidden_states = layer(hidden_states)
            
            # FIBER fusion at specified layers
            if str(layer_idx) in self.fusion_blocks and text_embeddings is not None:
                # Extract patch tokens (exclude CLS)
                patch_tokens = hidden_states[:, 1:, :]
                
                # Apply cross-modal fusion
                fusion_block = self.fusion_blocks[str(layer_idx)]
                fused_patches, _, attn_dict = fusion_block(
                    patch_tokens, text_embeddings, text_mask, return_attention=True
                )
                
                # Reconstruct with CLS token
                hidden_states = torch.cat([hidden_states[:, :1, :], fused_patches], dim=1)
                
                if attn_dict:
                    fusion_attention[f'layer_{layer_idx}'] = attn_dict
            
            if return_all_layers:
                layer_outputs.append(hidden_states)
        
        # Layer norm
        hidden_states = self.norm(hidden_states)
        
        # Return only patch tokens (exclude CLS)
        patch_tokens = hidden_states[:, 1:, :]
        
        return patch_tokens, layer_outputs, fusion_attention
    
    def get_cls_token(self, images: torch.Tensor, text_embeddings: Optional[torch.Tensor] = None,
                      text_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Get CLS token for image-level representation"""
        if isinstance(images, list):
            images = torch.stack([self.preprocess(img) for img in images])
        
        batch_size = images.size(0)
        
        # Patch embedding
        patch_embeds = self.patch_embed(images)
        patch_embeds = patch_embeds.flatten(2).transpose(1, 2)
        
        # Add CLS token
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        embeddings = torch.cat([cls_tokens, patch_embeds], dim=1)
        embeddings = embeddings + self.pos_embed
        
        # Process through layers with fusion
        hidden_states = embeddings
        for layer_idx, layer in enumerate(self.layers):
            hidden_states = layer(hidden_states)
            
            if str(layer_idx) in self.fusion_blocks and text_embeddings is not None:
                patch_tokens = hidden_states[:, 1:, :]
                fusion_block = self.fusion_blocks[str(layer_idx)]
                fused_patches, _, _ = fusion_block(patch_tokens, text_embeddings, text_mask)
                hidden_states = torch.cat([hidden_states[:, :1, :], fused_patches], dim=1)
        
        hidden_states = self.norm(hidden_states)
        
        return hidden_states[:, 0, :]
    
    def freeze_base_vision(self, freeze: bool = True):
        """
        Freeze base vision encoder layers while keeping fusion blocks trainable.
        
        Args:
            freeze: whether to freeze (True) or unfreeze (False) base layers
        """
        # Freeze patch embedding
        for param in self.patch_embed.parameters():
            param.requires_grad = not freeze
        
        # Freeze positional embedding and CLS token
        self.cls_token.requires_grad = not freeze
        self.pos_embed.requires_grad = not freeze
        
        # Freeze transformer layers
        for layer in self.layers:
            for param in layer.parameters():
                param.requires_grad = not freeze
        
        # Freeze layer norm
        for param in self.norm.parameters():
            param.requires_grad = not freeze
        
        # Keep fusion blocks trainable
        for fusion_block in self.fusion_blocks.values():
            for param in fusion_block.parameters():
                param.requires_grad = True
        
        frozen_status = "frozen" if freeze else "unfrozen"
        print(f"🔒 Base vision encoder {frozen_status}, FIBER fusion blocks remain trainable")


class ImageTextMatchingHead(nn.Module):
    """
    Image-Text Matching (ITM) head for binary classification.
    
    Predicts whether an image-text pair is matched or not.
    Uses concatenated CLS features from both modalities.
    """
    
    def __init__(self, vision_dim: int, text_dim: int, hidden_dim: int = 512):
        super().__init__()
        
        # MLP classifier
        self.classifier = nn.Sequential(
            nn.Linear(vision_dim + text_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, 2)  # Binary: match / no-match
        )
        
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, vision_cls: torch.Tensor, text_cls: torch.Tensor) -> torch.Tensor:
        """
        Compute ITM logits.
        
        Args:
            vision_cls: (B, vision_dim) - vision CLS token
            text_cls: (B, text_dim) - text CLS token / pooled representation
        
        Returns:
            logits: (B, 2) - match/no-match logits
        """
        concat_features = torch.cat([vision_cls, text_cls], dim=-1)
        logits = self.classifier(concat_features)
        return logits


class FIBERAlignmentLoss(nn.Module):
    """
    Combined alignment loss for FIBER-style training.
    
    Includes:
    1. Image-Text Contrastive (ITC) loss - CLIP-style global alignment
    2. Image-Text Matching (ITM) loss - binary classification with hard negatives
    3. Token-level alignment loss (optional) - fine-grained supervision
    """
    
    def __init__(
        self,
        temperature: float = 0.07,
        learnable_temperature: bool = True,
        label_smoothing: float = 0.1,
        itc_weight: float = 1.0,
        itm_weight: float = 1.0,
        token_weight: float = 0.5
    ):
        super().__init__()
        
        self.itc_weight = itc_weight
        self.itm_weight = itm_weight
        self.token_weight = token_weight
        self.label_smoothing = label_smoothing
        
        # Learnable temperature for ITC
        if learnable_temperature:
            self.logit_scale = nn.Parameter(torch.ones([]) * math.log(1 / temperature))
        else:
            self.register_buffer('logit_scale', torch.ones([]) * math.log(1 / temperature))
        
        # ITM head
        self.itm_head = None  # Initialized externally based on model dims
        
        # Loss functions
        self.ce_loss = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
        self.bce_loss = nn.BCEWithLogitsLoss()
    
    def set_itm_head(self, vision_dim: int, text_dim: int):
        """Initialize ITM head with correct dimensions"""
        self.itm_head = ImageTextMatchingHead(vision_dim, text_dim)
    
    def compute_itc_loss(
        self,
        image_features: torch.Tensor,
        text_features: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute Image-Text Contrastive loss.
        
        Args:
            image_features: (B, D) normalized image features
            text_features: (B, D) normalized text features
        
        Returns:
            loss: scalar tensor
        """
        # Normalize - ensure float32 for mixed precision compatibility
        image_features = F.normalize(image_features.float(), p=2, dim=-1)
        text_features = F.normalize(text_features.float(), p=2, dim=-1)
        
        # Clamp temperature
        logit_scale = torch.clamp(self.logit_scale, min=0, max=4.6052).exp()
        
        # Compute similarity
        logits_per_image = logit_scale * (image_features @ text_features.t())
        logits_per_text = logits_per_image.t()
        
        # Labels (diagonal)
        batch_size = image_features.size(0)
        labels = torch.arange(batch_size, device=image_features.device)
        
        # Bidirectional loss
        loss_i2t = self.ce_loss(logits_per_image, labels)
        loss_t2i = self.ce_loss(logits_per_text, labels)
        
        return (loss_i2t + loss_t2i) / 2
    
    def compute_itm_loss(
        self,
        vision_cls: torch.Tensor,
        text_cls: torch.Tensor,
        image_features: torch.Tensor,
        text_features: torch.Tensor,
        hard_negative_ratio: float = 0.5
    ) -> torch.Tensor:
        """
        Compute Image-Text Matching loss with hard negatives.
        
        Hard negatives are selected based on high similarity but wrong pairing.
        
        Args:
            vision_cls: (B, vision_dim) vision CLS tokens
            text_cls: (B, text_dim) text CLS/pooled tokens  
            image_features: (B, D) for similarity computation
            text_features: (B, D) for similarity computation
            hard_negative_ratio: ratio of hard vs random negatives
        
        Returns:
            loss: scalar tensor
        """
        if self.itm_head is None:
            raise RuntimeError("ITM head not initialized. Call set_itm_head() first.")
        
        batch_size = vision_cls.size(0)
        device = vision_cls.device
        
        # Compute similarity for hard negative mining
        with torch.no_grad():
            sim_i2t = image_features @ text_features.t()  # (B, B)
            sim_t2i = sim_i2t.t()
            
            # Mask diagonal (positive pairs)
            mask = torch.eye(batch_size, device=device).bool()
            sim_i2t = sim_i2t.masked_fill(mask, -float('inf'))
            sim_t2i = sim_t2i.masked_fill(mask, -float('inf'))
            
            # Select hard negatives (highest similarity non-matching pairs)
            num_hard = int(batch_size * hard_negative_ratio)
            
            # Hard negative texts for each image
            hard_text_idx = sim_i2t.topk(k=min(num_hard, batch_size-1), dim=1)[1]
            
            # Hard negative images for each text  
            hard_image_idx = sim_t2i.topk(k=min(num_hard, batch_size-1), dim=1)[1]
        
        # Construct positive pairs
        pos_logits = self.itm_head(vision_cls, text_cls)  # (B, 2)
        pos_labels = torch.ones(batch_size, dtype=torch.long, device=device)
        
        # Construct hard negative pairs
        # For each image, pair with hard negative text
        neg_text_cls = text_cls[hard_text_idx[:, 0]]  # Take top hard negative
        neg_logits_i2t = self.itm_head(vision_cls, neg_text_cls)
        
        # For each text, pair with hard negative image
        neg_vision_cls = vision_cls[hard_image_idx[:, 0]]
        neg_logits_t2i = self.itm_head(neg_vision_cls, text_cls)
        
        neg_labels = torch.zeros(batch_size, dtype=torch.long, device=device)
        
        # Combined loss
        all_logits = torch.cat([pos_logits, neg_logits_i2t, neg_logits_t2i], dim=0)
        all_labels = torch.cat([pos_labels, neg_labels, neg_labels], dim=0)
        
        loss = self.ce_loss(all_logits, all_labels)
        
        return loss
    
    def compute_token_alignment_loss(
        self,
        patch_embeddings: torch.Tensor,
        text_embeddings: torch.Tensor,
        text_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute token-level alignment loss.
        
        Encourages text tokens to attend to relevant image patches.
        Similar to FineGrainedAlignmentLoss but integrated with FIBER.
        
        Args:
            patch_embeddings: (B, num_patches, D)
            text_embeddings: (B, seq_len, D)
            text_mask: (B, seq_len)
        
        Returns:
            loss: scalar tensor
            attention: (B, seq_len, num_patches) attention weights
        """
        # Normalize - ensure float32 for mixed precision compatibility
        patch_norm = F.normalize(patch_embeddings.float(), p=2, dim=-1)
        text_norm = F.normalize(text_embeddings.float(), p=2, dim=-1)
        
        # Compute attention scores
        attention_logits = torch.bmm(text_norm, patch_norm.transpose(1, 2))  # (B, seq_len, num_patches)
        attention = F.softmax(attention_logits / 0.1, dim=-1)
        
        # Entropy regularization (encourage focused attention)
        eps = 1e-8
        entropy = -(attention * torch.log(attention + eps)).sum(dim=-1)  # (B, seq_len)
        max_entropy = math.log(patch_embeddings.size(1))
        normalized_entropy = entropy / max_entropy
        
        # Diversity loss (different tokens should attend to different patches)
        if text_mask is not None:
            valid_tokens = text_mask.sum(dim=-1, keepdim=True).clamp(min=1)
            mean_attention = (attention * text_mask.unsqueeze(-1)).sum(dim=1) / valid_tokens
            entropy_loss = (normalized_entropy * text_mask).sum(dim=-1) / valid_tokens.squeeze(-1)
        else:
            mean_attention = attention.mean(dim=1)
            entropy_loss = normalized_entropy.mean(dim=-1)
        
        entropy_loss = entropy_loss.mean()
        
        # Diversity: variance across patches
        diversity = mean_attention.var(dim=-1).mean()
        diversity_loss = 1.0 - diversity.clamp(max=1.0)
        
        loss = 0.5 * entropy_loss + 0.5 * diversity_loss
        
        return loss, attention
    
    def forward(
        self,
        image_features: torch.Tensor,
        text_features: torch.Tensor,
        vision_cls: Optional[torch.Tensor] = None,
        text_cls: Optional[torch.Tensor] = None,
        patch_embeddings: Optional[torch.Tensor] = None,
        text_embeddings: Optional[torch.Tensor] = None,
        text_mask: Optional[torch.Tensor] = None,
        compute_itm: bool = True,
        compute_token: bool = True
    ) -> dict:
        """
        Compute all alignment losses.
        
        Args:
            image_features: (B, D) global image features
            text_features: (B, D) global text features
            vision_cls: (B, vision_dim) vision CLS for ITM
            text_cls: (B, text_dim) text CLS for ITM
            patch_embeddings: (B, num_patches, D) for token-level loss
            text_embeddings: (B, seq_len, D) for token-level loss
            text_mask: (B, seq_len) attention mask
            compute_itm: whether to compute ITM loss
            compute_token: whether to compute token-level loss
        
        Returns:
            dict with 'itc_loss', 'itm_loss', 'token_loss', 'total_loss'
        """
        losses = {}
        total_loss = 0.0
        
        # ITC loss
        itc_loss = self.compute_itc_loss(image_features, text_features)
        losses['itc_loss'] = itc_loss
        total_loss = total_loss + self.itc_weight * itc_loss
        
        # ITM loss
        if compute_itm and vision_cls is not None and text_cls is not None:
            if self.itm_head is not None:
                itm_loss = self.compute_itm_loss(
                    vision_cls, text_cls, image_features, text_features
                )
                losses['itm_loss'] = itm_loss
                total_loss = total_loss + self.itm_weight * itm_loss
        
        # Token-level loss
        if compute_token and patch_embeddings is not None and text_embeddings is not None:
            token_loss, token_attention = self.compute_token_alignment_loss(
                patch_embeddings, text_embeddings, text_mask
            )
            losses['token_loss'] = token_loss
            losses['token_attention'] = token_attention
            total_loss = total_loss + self.token_weight * token_loss
        
        losses['total_loss'] = total_loss
        
        return losses
