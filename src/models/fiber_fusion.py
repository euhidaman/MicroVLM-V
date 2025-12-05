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
    
    STABILITY FIXES:
    - Alpha is bounded via tanh to prevent unbounded growth
    - Target tokens are normalized before projection
    - Output is normalized before residual addition
    - Attention logits are clamped to prevent overflow
    """
    
    def __init__(
        self,
        source_dim: int,
        target_dim: int,
        num_heads: int = 4,
        dropout: float = 0.0,
        use_layer_norm: bool = True,
        alpha_max: float = 0.5  # Maximum alpha value for stability
    ):
        super().__init__()
        
        self.source_dim = source_dim
        self.target_dim = target_dim
        self.num_heads = num_heads
        self.head_dim = source_dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.alpha_max = alpha_max
        
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
        
        # STABILITY FIX: Add layer norm for target tokens
        self.norm_target = nn.LayerNorm(target_dim)
        
        # STABILITY FIX: Add output normalization
        self.norm_output = nn.LayerNorm(source_dim)
        
        # Learnable gating parameter - using raw value that will be passed through tanh
        # Initialize to small value so tanh(0.1) ≈ 0.1
        self._alpha_raw = nn.Parameter(torch.tensor(0.1))
        
        # Dropout
        self.attn_dropout = nn.Dropout(dropout)
        self.proj_dropout = nn.Dropout(dropout)
        
        # Store attention weights for visualization
        self._last_attention_weights = None
        
        self._init_weights()
    
    @property
    def alpha(self):
        """Get bounded alpha value using tanh for stability"""
        # tanh bounds output to [-1, 1], scale to [0, alpha_max]
        return self.alpha_max * (torch.tanh(self._alpha_raw) + 1) / 2
    
    def _init_weights(self):
        """Initialize weights with proper scaling for stable training"""
        # Use smaller gain for all projections for stability
        nn.init.xavier_uniform_(self.q_proj.weight, gain=0.5)
        nn.init.xavier_uniform_(self.k_proj.weight, gain=0.5)  # Reduced from 1.0
        nn.init.xavier_uniform_(self.v_proj.weight, gain=0.5)  # Reduced from 1.0
        nn.init.xavier_uniform_(self.out_proj.weight, gain=0.25)  # Even smaller for output
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
        
        # Ensure consistent dtype (needed for 4-bit quantized models that use float16)
        proj_dtype = self.q_proj.weight.dtype
        source = source.to(proj_dtype)
        target = target.to(proj_dtype)
        
        # Apply layer norm to source
        source_normed = self.norm_source(source)
        
        # STABILITY FIX: Normalize target before projection
        target_normed = self.norm_target(target)
        
        # Compute Q, K, V
        Q = self.q_proj(source_normed)  # (B, seq_source, source_dim)
        K = self.k_proj(target_normed)  # (B, seq_target, source_dim)  
        V = self.v_proj(target_normed)  # (B, seq_target, source_dim)
        
        # Reshape for multi-head attention
        Q = Q.view(B, seq_source, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(B, seq_target, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(B, seq_target, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Compute attention scores
        attn = (Q @ K.transpose(-2, -1)) * self.scale  # (B, num_heads, seq_source, seq_target)
        
        # STABILITY FIX: Clamp attention logits to prevent overflow
        attn = torch.clamp(attn, min=-50.0, max=50.0)
        
        # Apply attention mask if provided
        if target_mask is not None:
            # Expand mask: (B, seq_target) -> (B, 1, 1, seq_target)
            mask = target_mask.unsqueeze(1).unsqueeze(1)
            attn = attn.masked_fill(mask == 0, float('-inf'))
        
        # Softmax and dropout
        attn = F.softmax(attn, dim=-1)
        
        # STABILITY FIX: Handle NaN from softmax (all -inf case)
        attn = torch.nan_to_num(attn, nan=0.0)
        
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
        
        # STABILITY FIX: Normalize output before residual addition
        out = self.norm_output(out)
        
        # Apply bounded gating and residual
        # alpha is now bounded via tanh in the property
        updated_source = source + self.alpha * out
        
        if return_attention:
            return updated_source, attn
        return updated_source, None
    
    def get_attention_weights(self) -> Optional[torch.Tensor]:
        """Get the last computed attention weights"""
        return self._last_attention_weights


class FIBERFusionBlock(nn.Module):
    """
    FIBER-style bidirectional fusion block with dimension alignment.
    
    Implements:
    1. Image-to-Text attention: Vision tokens attend to text tokens
    2. Text-to-Image attention: Text tokens attend to vision tokens
    
    Both directions are computed in parallel and applied with learnable gating.
    
    DIMENSION ALIGNMENT FIX:
    - Vision (192-dim) and Text (896-dim) are projected to a shared fusion_dim (384)
    - This allows better cross-modal alignment without the 4.6x compression bottleneck
    - Adds only ~0.84 MB to model size
    """
    
    def __init__(
        self,
        vision_dim: int = 192,
        text_dim: int = 896,
        num_heads: int = 4,
        dropout: float = 0.0,
        bidirectional: bool = True,
        fusion_dim: int = 384  # Shared dimension for cross-modal fusion
    ):
        super().__init__()
        
        self.vision_dim = vision_dim
        self.text_dim = text_dim
        self.fusion_dim = fusion_dim
        self.bidirectional = bidirectional
        
        # ===== DIMENSION ALIGNMENT PROJECTIONS =====
        # Upscale vision: 192 -> 384 (expand to richer space)
        self.vision_up_proj = nn.Sequential(
            nn.Linear(vision_dim, fusion_dim),
            nn.LayerNorm(fusion_dim),
            nn.GELU()
        )
        
        # Compress text: 896 -> 384 (compress to shared space)
        self.text_down_proj = nn.Sequential(
            nn.Linear(text_dim, fusion_dim),
            nn.LayerNorm(fusion_dim),
            nn.GELU()
        )
        
        # Project back to vision dim after fusion: 384 -> 192
        self.vision_down_proj = nn.Linear(fusion_dim, vision_dim)
        
        # ===== CROSS-MODAL ATTENTION IN SHARED SPACE =====
        # Image-to-Text: Vision queries, Text keys/values (both in 384-dim)
        self.i2t_attention = CrossModalAttention(
            source_dim=fusion_dim,  # Now both in shared 384-dim space
            target_dim=fusion_dim,
            num_heads=num_heads,
            dropout=dropout
        )
        
        # Text-to-Image: Text queries, Vision keys/values
        if bidirectional:
            # Project back to text dim after fusion: 384 -> 896
            self.text_up_proj = nn.Linear(fusion_dim, text_dim)
            
            self.t2i_attention = CrossModalAttention(
                source_dim=fusion_dim,
                target_dim=fusion_dim,
                num_heads=num_heads,
                dropout=dropout
            )
        else:
            self.t2i_attention = None
            self.text_up_proj = None
        
        self._init_projection_weights()
    
    def _init_projection_weights(self):
        """Initialize projection layer weights"""
        for module in [self.vision_up_proj, self.text_down_proj]:
            for m in module.modules():
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight, gain=0.5)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)
        
        nn.init.xavier_uniform_(self.vision_down_proj.weight, gain=0.5)
        nn.init.zeros_(self.vision_down_proj.bias)
        
        if self.text_up_proj is not None:
            nn.init.xavier_uniform_(self.text_up_proj.weight, gain=0.5)
            nn.init.zeros_(self.text_up_proj.bias)
    
    def forward(
        self,
        vision_tokens: torch.Tensor,
        text_tokens: torch.Tensor,
        text_mask: Optional[torch.Tensor] = None,
        return_attention: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[dict]]:
        """
        Forward pass for bidirectional fusion with dimension alignment.
        
        Args:
            vision_tokens: (B, num_patches, vision_dim=192)
            text_tokens: (B, seq_len, text_dim=896)
            text_mask: (B, seq_len) attention mask for text
            return_attention: whether to return attention weights
        
        Returns:
            updated_vision: (B, num_patches, vision_dim=192)
            updated_text: (B, seq_len, text_dim=896)
            attention_dict: dict with 'i2t' and 't2i' attention weights
        """
        attention_dict = {}
        
        # ===== ENSURE CONSISTENT DTYPE =====
        # Get projection dtype and convert inputs to match
        proj_dtype = self.vision_up_proj[0].weight.dtype
        vision_tokens = vision_tokens.to(proj_dtype)
        text_tokens = text_tokens.to(proj_dtype)
        
        # ===== PROJECT TO SHARED FUSION SPACE =====
        # Vision: 192 -> 384
        vision_in_fusion = self.vision_up_proj(vision_tokens)
        # Text: 896 -> 384
        text_in_fusion = self.text_down_proj(text_tokens)
        
        # ===== CROSS-MODAL ATTENTION IN SHARED 384-DIM SPACE =====
        # Image-to-Text attention (vision attends to text)
        vision_fused, i2t_attn = self.i2t_attention(
            vision_in_fusion, text_in_fusion, text_mask, return_attention
        )
        if return_attention:
            attention_dict['i2t'] = i2t_attn
        
        # ===== PROJECT BACK TO ORIGINAL DIMENSIONS =====
        # Vision: 384 -> 192
        updated_vision = vision_tokens + self.vision_down_proj(vision_fused - vision_in_fusion)
        
        # Text-to-Image attention (if bidirectional)
        if self.bidirectional and self.t2i_attention is not None:
            text_fused, t2i_attn = self.t2i_attention(
                text_in_fusion, vision_in_fusion, target_mask=None, return_attention=return_attention
            )
            if return_attention:
                attention_dict['t2i'] = t2i_attn
            # Text: 384 -> 896
            updated_text = text_tokens + self.text_up_proj(text_fused - text_in_fusion)
        else:
            updated_text = text_tokens
        
        return updated_vision, updated_text, attention_dict if return_attention else None


class FIBERVisionEncoder(nn.Module):
    """
    DeiT-Tiny vision encoder with FIBER-style fusion layers.
    
    Injects cross-modal attention blocks at specified layers of the transformer.
    This allows text information to guide visual feature extraction at intermediate levels.
    
    NOTE: Reduced for compact model (<1GB target):
    - Default fusion_layers reduced from [8,9,10,11] to [9,11] (2 layers)
    - num_fusion_heads reduced from 4 to 2
    
    DIMENSION ALIGNMENT:
    - Uses fusion_dim (384) as shared space for cross-modal attention
    - Vision (192) and Text (896) are projected to this shared dimension
    """
    
    def __init__(
        self,
        config: dict,
        pretrained_path: Optional[str] = None,
        fusion_layers: list = None,
        text_dim: int = 896,
        num_fusion_heads: int = 2,  # Reduced from 4
        fusion_dim: int = 384  # Shared dimension for cross-modal fusion
    ):
        super().__init__()
        
        self.image_size = config.get('image_size', 224)
        self.patch_size = config.get('patch_size', 16)
        self.hidden_size = config.get('deit_embed_dim', 192)
        self.num_patches = config.get('num_patches', 196)
        self.num_layers = 12  # DeiT-Tiny has 12 layers
        self.fusion_dim = fusion_dim
        
        # Fusion layer indices (0-indexed, reduced from [8,9,10,11] to [9,11])
        self.fusion_layers = fusion_layers or [9, 11]
        
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
                bidirectional=False,  # Only I2T for vision encoder
                fusion_dim=fusion_dim  # Shared 384-dim space
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
    
    NOTE: Reduced hidden_dim from 512 to 256 for compact model.
    """
    
    def __init__(self, vision_dim: int, text_dim: int, hidden_dim: int = 256):
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
    1. Image-Text Contrastive (ITC) loss - CLIP-style global alignment with queue
    2. Image-Text Matching (ITM) loss - binary classification with hard negatives
    3. Token-level alignment loss (optional) - fine-grained supervision
    
    FEATURES (from FIBER):
    - ITC queue for better negative sampling (lightweight version)
    - Dedicated ITC projection heads for image and text
    - Learnable temperature with bounded optimization
    
    STABILITY FIXES:
    - Temperature has tighter bounds to prevent extreme values
    - Gradient scaling for temperature updates
    - Soft clamping for smoother optimization
    
    ANTI-COLLAPSE REGULARIZATION:
    - Feature variance regularization prevents embedding collapse
    - Spectral regularization maintains feature diversity
    - Attention entropy regularization prevents edge-detection mode
    """
    
    def __init__(
        self,
        temperature: float = 0.07,
        learnable_temperature: bool = True,
        label_smoothing: float = 0.2,  # Increased from 0.1 for better regularization
        itc_weight: float = 0.8,  # Reduced from 1.0 to prevent over-focus on pixel patterns
        itm_weight: float = 1.2,  # Increased for better semantic matching
        token_weight: float = 0.5,
        min_temperature: float = 0.05,  # Raised from 0.01 to prevent edge-detection mode
        max_temperature: float = 0.3,   # Tightened from 0.5 for more stable training
        # ITC Queue settings (lightweight for edge devices)
        use_itc_queue: bool = True,
        queue_size: int = 256,  # Reduced from FIBER's 4096 for compact model
        # ITC projection settings
        itc_embed_dim: int = 128,  # Compact ITC embedding dimension
        # Anti-collapse regularization
        use_anti_collapse: bool = True,
        variance_reg_weight: float = 0.04,  # Feature variance regularization
        covariance_reg_weight: float = 0.01,  # Covariance regularization (VICReg-style)
    ):
        super().__init__()
        
        self.itc_weight = itc_weight
        self.itm_weight = itm_weight
        self.token_weight = token_weight
        self.label_smoothing = label_smoothing
        
        # STABILITY FIX: Store bounds for soft clamping
        self.log_scale_min = math.log(1 / max_temperature)  # ~0.7
        self.log_scale_max = math.log(1 / min_temperature)  # ~4.6
        
        # Learnable temperature for ITC
        if learnable_temperature:
            # Initialize in middle of valid range for stability
            init_log_scale = math.log(1 / temperature)
            self._logit_scale_raw = nn.Parameter(torch.tensor(init_log_scale))
        else:
            self.register_buffer('_logit_scale_raw', torch.tensor(math.log(1 / temperature)))
        
        self.learnable_temperature = learnable_temperature
        
        # ===== Anti-Collapse Regularization =====
        self.use_anti_collapse = use_anti_collapse
        self.variance_reg_weight = variance_reg_weight
        self.covariance_reg_weight = covariance_reg_weight
        
        # ===== FIBER-style ITC Queue (lightweight version) =====
        self.use_itc_queue = use_itc_queue
        self.queue_size = queue_size
        self.itc_embed_dim = itc_embed_dim
        
        if use_itc_queue:
            # Register queue buffers (not parameters - no gradient)
            self.register_buffer("image_queue", torch.randn(itc_embed_dim, queue_size))
            self.register_buffer("text_queue", torch.randn(itc_embed_dim, queue_size))
            self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))
            self.register_buffer("queue_total", torch.zeros(1, dtype=torch.long))
            # Normalize queue embeddings
            self.image_queue = F.normalize(self.image_queue, dim=0)
            self.text_queue = F.normalize(self.text_queue, dim=0)
        
        # ===== FIBER-style ITC Projection Heads =====
        # These will be initialized when set_itc_heads is called
        self.image_proj_itc = None
        self.text_proj_itc = None
        
        # ITM head
        self.itm_head = None  # Initialized externally based on model dims
        
        # Loss functions
        self.ce_loss = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
        self.bce_loss = nn.BCEWithLogitsLoss()
    
    def compute_anti_collapse_loss(
        self,
        image_features: torch.Tensor,
        text_features: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute anti-collapse regularization to prevent feature collapse.
        
        Based on VICReg (Variance-Invariance-Covariance Regularization):
        1. Variance: Encourages feature variance to stay above a threshold
        2. Covariance: Decorrelates different feature dimensions
        
        This prevents the model from collapsing to trivial solutions where
        all embeddings become similar (causing attention to focus on low-level
        patterns like edges instead of semantic content).
        
        Args:
            image_features: (B, D) normalized image features
            text_features: (B, D) normalized text features
            
        Returns:
            loss: scalar regularization loss
        """
        if not self.use_anti_collapse:
            return torch.tensor(0.0, device=image_features.device)
        
        eps = 1e-4
        
        # ===== Variance Regularization =====
        # Encourage feature std to stay above 1 (prevent collapse)
        # hinge loss: relu(1 - std)
        def variance_loss(x):
            # x: (B, D)
            std = x.std(dim=0)  # (D,)
            return F.relu(1.0 - std).mean()
        
        var_loss_img = variance_loss(image_features)
        var_loss_txt = variance_loss(text_features)
        var_loss = (var_loss_img + var_loss_txt) / 2
        
        # ===== Covariance Regularization =====
        # Decorrelate feature dimensions to prevent redundant representations
        def covariance_loss(x):
            # x: (B, D)
            batch_size, dim = x.shape
            x_centered = x - x.mean(dim=0, keepdim=True)
            # Covariance matrix: (D, D)
            cov = (x_centered.T @ x_centered) / (batch_size - 1 + eps)
            # Zero out diagonal (we want to penalize off-diagonal elements)
            off_diag = cov - torch.diag(cov.diag())
            # Frobenius norm of off-diagonal elements
            return (off_diag ** 2).sum() / dim
        
        cov_loss_img = covariance_loss(image_features)
        cov_loss_txt = covariance_loss(text_features)
        cov_loss = (cov_loss_img + cov_loss_txt) / 2
        
        # Combined regularization loss
        total_reg_loss = (
            self.variance_reg_weight * var_loss +
            self.covariance_reg_weight * cov_loss
        )
        
        return total_reg_loss
    
    def compute_attention_entropy_loss(
        self,
        attention: torch.Tensor,
        min_entropy_ratio: float = 0.3
    ) -> torch.Tensor:
        """
        Compute attention entropy regularization to prevent edge-detection collapse.
        
        When attention entropy is too low, the model is focusing on very specific
        patterns (often edges/boundaries). This regularization encourages maintaining
        some entropy in attention distributions.
        
        Args:
            attention: (B, num_tokens, num_patches) attention weights
            min_entropy_ratio: minimum entropy as fraction of maximum possible
            
        Returns:
            loss: scalar entropy regularization loss
        """
        eps = 1e-8
        
        # Compute entropy per token
        # attention is already normalized (softmax output)
        entropy = -(attention * torch.log(attention + eps)).sum(dim=-1)  # (B, num_tokens)
        
        # Maximum possible entropy (uniform distribution)
        num_patches = attention.size(-1)
        max_entropy = math.log(num_patches)
        
        # Target minimum entropy
        min_target_entropy = min_entropy_ratio * max_entropy
        
        # Hinge loss: penalize if entropy drops below threshold
        entropy_loss = F.relu(min_target_entropy - entropy).mean()
        
        return entropy_loss
    
    @property
    def logit_scale(self):
        """Get bounded logit scale using soft clamping for smooth gradients"""
        # Use sigmoid-based soft clamping for smoother optimization
        range_size = self.log_scale_max - self.log_scale_min
        normalized = torch.sigmoid(self._logit_scale_raw - (self.log_scale_min + range_size / 2))
        return self.log_scale_min + normalized * range_size
    
    def set_itm_head(self, vision_dim: int, text_dim: int):
        """Initialize ITM head with correct dimensions"""
        self.itm_head = ImageTextMatchingHead(vision_dim, text_dim)
    
    def set_itc_heads(self, image_dim: int, text_dim: int):
        """
        Initialize FIBER-style dedicated ITC projection heads.
        
        These project image and text features to a shared ITC embedding space,
        separate from the main feature projections used elsewhere.
        
        Args:
            image_dim: dimension of input image features
            text_dim: dimension of input text features
        """
        self.image_proj_itc = nn.Sequential(
            nn.Linear(image_dim, self.itc_embed_dim),
            nn.LayerNorm(self.itc_embed_dim)
        )
        self.text_proj_itc = nn.Sequential(
            nn.Linear(text_dim, self.itc_embed_dim),
            nn.LayerNorm(self.itc_embed_dim)
        )
        # Initialize with small weights for stability
        for module in [self.image_proj_itc, self.text_proj_itc]:
            for m in module.modules():
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight, gain=0.5)
                    nn.init.zeros_(m.bias)
    
    @torch.no_grad()
    def _dequeue_and_enqueue(self, image_feat: torch.Tensor, text_feat: torch.Tensor):
        """
        Update the ITC queue with new features (FIBER-style momentum queue).
        
        Args:
            image_feat: (B, itc_embed_dim) normalized image features
            text_feat: (B, itc_embed_dim) normalized text features
        """
        if not self.use_itc_queue:
            return
        
        batch_size = image_feat.size(0)
        ptr = int(self.queue_ptr)
        
        # If batch is larger than queue, only use the last queue_size samples
        if batch_size >= self.queue_size:
            # Take only the most recent samples that fit in queue
            image_feat = image_feat[-self.queue_size:]
            text_feat = text_feat[-self.queue_size:]
            batch_size = self.queue_size
            # Fill the entire queue
            self.image_queue[:, :] = image_feat.T
            self.text_queue[:, :] = text_feat.T
            self.queue_ptr[0] = 0
            self.queue_total[0] = self.queue_size
            return
        
        # Handle wrap-around for smaller batches
        if ptr + batch_size <= self.queue_size:
            self.image_queue[:, ptr:ptr + batch_size] = image_feat.T
            self.text_queue[:, ptr:ptr + batch_size] = text_feat.T
        else:
            # Wrap around
            first_len = self.queue_size - ptr
            self.image_queue[:, ptr:] = image_feat[:first_len].T
            self.text_queue[:, ptr:] = text_feat[:first_len].T
            remaining = batch_size - first_len
            self.image_queue[:, :remaining] = image_feat[first_len:].T
            self.text_queue[:, :remaining] = text_feat[first_len:].T
        
        # Update pointers
        self.queue_ptr[0] = (ptr + batch_size) % self.queue_size
        self.queue_total[0] = min(self.queue_total[0] + batch_size, self.queue_size)
    
    def compute_itc_loss(
        self,
        image_features: torch.Tensor,
        text_features: torch.Tensor,
        use_queue: bool = True
    ) -> torch.Tensor:
        """
        Compute Image-Text Contrastive loss with optional queue for better negatives.
        
        When queue is enabled (FIBER-style), uses accumulated features from previous
        batches as additional negatives, improving contrastive learning quality.
        
        Args:
            image_features: (B, D) image features (will be projected if ITC heads exist)
            text_features: (B, D) text features (will be projected if ITC heads exist)
            use_queue: whether to use the ITC queue for additional negatives
        
        Returns:
            loss: scalar tensor
        """
        # Project through ITC heads if available (FIBER-style)
        if self.image_proj_itc is not None and self.text_proj_itc is not None:
            image_feat_itc = self.image_proj_itc(image_features.float())
            text_feat_itc = self.text_proj_itc(text_features.float())
        else:
            image_feat_itc = image_features.float()
            text_feat_itc = text_features.float()
        
        # Normalize - ensure float32 for mixed precision compatibility
        image_feat_itc = F.normalize(image_feat_itc, p=2, dim=-1)
        text_feat_itc = F.normalize(text_feat_itc, p=2, dim=-1)
        
        # STABILITY FIX: Use property for bounded logit scale
        logit_scale = self.logit_scale.exp()
        
        batch_size = image_feat_itc.size(0)
        
        # ===== FIBER-style Queue-based ITC =====
        if use_queue and self.use_itc_queue and self.queue_total[0] > 0:
            # Get queue features (detached - no gradient through queue)
            queue_size_actual = int(self.queue_total[0])
            image_queue = self.image_queue[:, :queue_size_actual].clone().detach()
            text_queue = self.text_queue[:, :queue_size_actual].clone().detach()
            
            # Concatenate current batch with queue
            # image_feat_all: (D, B + queue_size)
            image_feat_all = torch.cat([image_feat_itc.T, image_queue], dim=1)
            text_feat_all = torch.cat([text_feat_itc.T, text_queue], dim=1)
            
            # Compute similarity with all features (current + queue)
            sim_i2t = logit_scale * (image_feat_itc @ text_feat_all)  # (B, B + queue_size)
            sim_t2i = logit_scale * (text_feat_itc @ image_feat_all)  # (B, B + queue_size)
        else:
            # Standard in-batch contrastive (no queue)
            sim_i2t = logit_scale * (image_feat_itc @ text_feat_itc.T)  # (B, B)
            sim_t2i = sim_i2t.T
        
        # STABILITY FIX: Clamp logits to prevent extreme values
        # Tightened bounds to prevent over-confident predictions
        sim_i2t = torch.clamp(sim_i2t, min=-50.0, max=50.0)
        sim_t2i = torch.clamp(sim_t2i, min=-50.0, max=50.0)
        
        # Labels: diagonal elements are positive pairs (first B columns)
        labels = torch.arange(batch_size, device=image_feat_itc.device)
        
        # Bidirectional loss
        loss_i2t = self.ce_loss(sim_i2t, labels)
        loss_t2i = self.ce_loss(sim_t2i, labels)
        
        # Update queue with current batch features (during training)
        if self.training and self.use_itc_queue:
            self._dequeue_and_enqueue(image_feat_itc.detach(), text_feat_itc.detach())
        
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
        text_mask: Optional[torch.Tensor] = None,
        grid_size: int = 14
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute improved token-level alignment loss with spatial coherence.
        
        This loss encourages:
        1. Text tokens to attend to semantically relevant patches (grounding)
        2. Attention to be spatially coherent (attend to nearby patches)
        3. Different text tokens to attend to different regions (diversity)
        4. Sharp, focused attention (low entropy)
        
        STABILITY FIXES:
        - Reduced conflicting objectives (removed sharpness vs smoothness conflict)
        - Added gradient scaling for stable training
        - Warmer temperature for smoother attention gradients
        - Progressive loss scheduling via adaptive weights
        
        Args:
            patch_embeddings: (B, num_patches, D)
            text_embeddings: (B, seq_len, D)
            text_mask: (B, seq_len)
            grid_size: size of patch grid (default 14 for 14x14=196 patches)
        
        Returns:
            loss: scalar tensor
            attention: (B, seq_len, num_patches) attention weights
        """
        B, num_patches, D = patch_embeddings.shape
        seq_len = text_embeddings.size(1)
        device = patch_embeddings.device
        
        # Normalize - ensure float32 for mixed precision compatibility
        patch_norm = F.normalize(patch_embeddings.float(), p=2, dim=-1)
        text_norm = F.normalize(text_embeddings.float(), p=2, dim=-1)
        
        # STABILITY FIX: Use warmer temperature (0.5) for much smoother gradients
        # This prevents attention from becoming too sharp too quickly
        temperature = 0.5
        attention_logits = torch.bmm(text_norm, patch_norm.transpose(1, 2))  # (B, seq_len, num_patches)
        
        # STABILITY FIX: Clamp logits before softmax
        attention_logits = torch.clamp(attention_logits / temperature, min=-20.0, max=20.0)
        attention = F.softmax(attention_logits, dim=-1)
        
        eps = 1e-8
        
        # ====== Component 1: Soft Focus Loss (Entropy Regularization) ======
        # Encourage moderate focus - not too sharp, not too uniform
        # Target entropy: around 50% of max entropy for balanced attention
        entropy = -(attention * torch.log(attention + eps)).sum(dim=-1)  # (B, seq_len)
        max_entropy = math.log(num_patches)
        target_entropy = 0.4 * max_entropy  # Target: moderately focused
        
        # STABILITY FIX: Use smooth L1 loss instead of pushing to zero
        # This prevents attention from collapsing to single patches
        normalized_entropy = entropy / max_entropy
        focus_loss = F.smooth_l1_loss(
            normalized_entropy, 
            torch.full_like(normalized_entropy, 0.4),  # Target 40% of max entropy
            reduction='none'
        )
        
        if text_mask is not None:
            valid_tokens = text_mask.sum(dim=-1, keepdim=True).clamp(min=1)
            focus_loss = (focus_loss * text_mask).sum(dim=-1) / valid_tokens.squeeze(-1)
        else:
            focus_loss = focus_loss.mean(dim=-1)
        focus_loss = focus_loss.mean()
        
        # ====== Component 2: Local Coherence Loss (NOT global smoothness) ======
        # STABILITY FIX: Only encourage local coherence, not global smoothness
        # This means nearby patches that are BOTH attended should have similar attention
        H = W = grid_size
        if num_patches == H * W:
            attention_grid = attention.view(B, seq_len, H, W)
            
            # Only penalize variation where attention is significant (> mean)
            mean_attn = attention_grid.mean(dim=(2, 3), keepdim=True)
            significant_mask = (attention_grid > mean_attn).float()
            
            # Compute variation only in significant regions
            tv_h = torch.abs(attention_grid[:, :, 1:, :] - attention_grid[:, :, :-1, :])
            tv_w = torch.abs(attention_grid[:, :, :, 1:] - attention_grid[:, :, :, :-1])
            
            # Weight by significance
            masked_tv_h = (tv_h * significant_mask[:, :, 1:, :]).sum() / (significant_mask[:, :, 1:, :].sum() + eps)
            masked_tv_w = (tv_w * significant_mask[:, :, :, 1:]).sum() / (significant_mask[:, :, :, 1:].sum() + eps)
            
            spatial_coherence_loss = (masked_tv_h + masked_tv_w) * 0.25  # Reduced weight
        else:
            spatial_coherence_loss = torch.tensor(0.0, device=device)
        
        # ====== Component 3: Diversity Loss (Soft version) ======
        # STABILITY FIX: Use soft diversity that doesn't conflict with focus
        if text_mask is not None:
            mask_expanded = text_mask.unsqueeze(-1).float()
            sum_attention = (attention * mask_expanded).sum(dim=1)
            count = mask_expanded.sum(dim=1).clamp(min=1)
            mean_attention = sum_attention / count
        else:
            mean_attention = attention.mean(dim=1)
        
        # Encourage coverage: mean attention should not be too concentrated
        # Use entropy of mean attention as diversity measure
        mean_attn_entropy = -(mean_attention * torch.log(mean_attention + eps)).sum(dim=-1)
        max_coverage_entropy = math.log(num_patches)
        
        # Target: at least 30% coverage entropy (not too concentrated)
        target_coverage = 0.3 * max_coverage_entropy
        diversity_loss = F.relu(target_coverage - mean_attn_entropy).mean()
        
        # ====== Component 4: Contrastive Grounding Loss (Simplified) ======
        # STABILITY FIX: Simpler contrastive that doesn't require hard mining
        attended_patches = torch.bmm(attention, patch_norm)  # (B, seq_len, D)
        
        # Positive: text should be similar to attended patches
        positive_sim = (text_norm * attended_patches).sum(dim=-1)  # (B, seq_len)
        
        # STABILITY FIX: Use global negative instead of per-token hard negatives
        # This is more stable and still provides good grounding signal
        batch_mean_patches = patch_norm.mean(dim=1, keepdim=True).expand(-1, seq_len, -1)  # (B, seq_len, D)
        baseline_sim = (text_norm * batch_mean_patches).sum(dim=-1)  # (B, seq_len)
        
        # Loss: attended should be better than baseline by margin
        margin = 0.1  # Reduced margin for stability
        contrastive_loss = F.relu(baseline_sim - positive_sim + margin)
        if text_mask is not None:
            contrastive_loss = (contrastive_loss * text_mask).sum(dim=-1) / valid_tokens.squeeze(-1)
        else:
            contrastive_loss = contrastive_loss.mean(dim=-1)
        contrastive_loss = contrastive_loss.mean()
        
        # ====== Combine all losses with BALANCED weights ======
        # STABILITY FIX: Reduced total magnitude and balanced objectives
        loss = (
            0.30 * focus_loss +             # Moderate focus (not extreme)
            0.15 * spatial_coherence_loss + # Local smoothness only
            0.20 * diversity_loss +         # Soft coverage encouragement
            0.35 * contrastive_loss         # Primary grounding signal
        )
        
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
            
            # ===== Anti-Collapse Regularization =====
            # Add attention entropy regularization to prevent edge-detection mode
            if self.use_anti_collapse:
                attn_entropy_loss = self.compute_attention_entropy_loss(token_attention)
                losses['attention_entropy_loss'] = attn_entropy_loss
                total_loss = total_loss + 0.05 * attn_entropy_loss  # Small weight
        
        # ===== Feature Collapse Prevention =====
        # Add variance/covariance regularization to prevent embedding collapse
        if self.use_anti_collapse:
            anti_collapse_loss = self.compute_anti_collapse_loss(
                image_features, text_features
            )
            losses['anti_collapse_loss'] = anti_collapse_loss
            total_loss = total_loss + anti_collapse_loss
        
        losses['total_loss'] = total_loss
        
        return losses
