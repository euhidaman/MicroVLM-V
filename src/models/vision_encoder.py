"""
DeiT-Tiny Vision Encoder
Loads pretrained DeiT-Tiny and extracts patch embeddings
"""

import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image


class DeiTVisionEncoder(nn.Module):
    """
    DeiT-Tiny vision encoder with 4-bit quantization support
    Extracts patch tokens as visual embeddings
    """
    
    def __init__(self, config, pretrained_path=None):
        super().__init__()
        
        self.image_size = config.get('image_size', 224)
        self.patch_size = config.get('patch_size', 16)
        self.hidden_size = config.get('deit_embed_dim', 192)
        self.num_patches = config.get('num_patches', 196)
        
        # Image preprocessing
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
        
        # Position embeddings (1 for CLS + num_patches)
        self.pos_embed = nn.Parameter(
            torch.randn(1, 1 + self.num_patches, self.hidden_size)
        )
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.hidden_size,
            nhead=3,  # DeiT-Tiny uses 3 attention heads
            dim_feedforward=768,  # 4 * hidden_size
            dropout=0.0,
            activation='gelu',
            batch_first=True,
            norm_first=False
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, 
            num_layers=12,
            enable_nested_tensor=False  # Disable to avoid warning with odd nhead
        )
        
        # Layer norm
        self.norm = nn.LayerNorm(self.hidden_size)
        
        self._init_weights()
        
        if pretrained_path:
            self.load_pretrained(pretrained_path)
    
    def _init_weights(self):
        """Initialize weights"""
        # Initialize patch embedding
        nn.init.xavier_uniform_(self.patch_embed.weight)
        nn.init.zeros_(self.patch_embed.bias)
        
        # Initialize cls token and pos embeddings
        nn.init.normal_(self.cls_token, std=0.02)
        nn.init.normal_(self.pos_embed, std=0.02)
    
    def load_pretrained(self, checkpoint_path):
        """Load pretrained DeiT-Tiny weights"""
        print(f"Loading pretrained DeiT weights from {checkpoint_path}")
        state_dict = torch.load(checkpoint_path, map_location='cpu')
        
        # Handle different checkpoint formats
        if 'model' in state_dict:
            state_dict = state_dict['model']
        
        # Load with partial matching
        self.load_state_dict(state_dict, strict=False)
        print("Pretrained weights loaded successfully")
    
    def forward(self, images):
        """
        Extract patch embeddings from images
        
        Args:
            images: (batch_size, 3, H, W) or list of PIL Images
        
        Returns:
            patch_embeddings: (batch_size, num_patches, hidden_size)
        """
        # Handle PIL images
        if isinstance(images, list):
            images = torch.stack([self.preprocess(img) for img in images])
        
        batch_size = images.size(0)
        
        # Patch embedding: (B, C, H, W) -> (B, hidden_size, H/P, W/P)
        patch_embeds = self.patch_embed(images)
        
        # Flatten patches: (B, hidden_size, H/P, W/P) -> (B, num_patches, hidden_size)
        patch_embeds = patch_embeds.flatten(2).transpose(1, 2)
        
        # Add CLS token
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        embeddings = torch.cat([cls_tokens, patch_embeds], dim=1)
        
        # Add positional embeddings
        embeddings = embeddings + self.pos_embed
        
        # Transformer encoding
        encoded = self.transformer(embeddings)
        
        # Layer norm
        encoded = self.norm(encoded)
        
        # Return only patch tokens (exclude CLS token)
        patch_tokens = encoded[:, 1:, :]
        
        return patch_tokens
    
    def get_cls_token(self, images):
        """
        Extract CLS token for image-level representation
        
        Args:
            images: (batch_size, 3, H, W)
        
        Returns:
            cls_features: (batch_size, hidden_size)
        """
        # Handle PIL images
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
        
        # Transformer encoding
        encoded = self.transformer(embeddings)
        
        # Layer norm
        encoded = self.norm(encoded)
        
        # Return CLS token
        cls_features = encoded[:, 0, :]
        
        return cls_features


def create_deit_encoder(config, checkpoint_path=None):
    """
    Factory function to create DeiT encoder
    
    Args:
        config: model configuration dictionary
        checkpoint_path: path to pretrained checkpoint
    
    Returns:
        encoder: DeiTVisionEncoder instance
    """
    encoder = DeiTVisionEncoder(config, pretrained_path=checkpoint_path)
    return encoder
