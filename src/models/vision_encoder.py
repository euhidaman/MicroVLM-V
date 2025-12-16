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

    QUANTIZATION MODES:
        1. use_hf_model=True + quantize_4bit=True: HuggingFace DeiT in 4-bit (BitsAndBytes)
        2. use_hf_model=True + quantize_4bit=False: HuggingFace DeiT in FP32
        3. use_hf_model=False: Custom PyTorch implementation in FP32

    RECOMMENDED: use_hf_model=True + quantize_4bit=True for frozen backbone
    """
    
    def __init__(self, config, pretrained_path=None, quantize_4bit=False, use_hf_model=True):
        super().__init__()
        
        self.image_size = config.get('image_size', 224)
        self.patch_size = config.get('patch_size', 16)
        self.hidden_size = config.get('deit_embed_dim', 192)
        self.num_patches = config.get('num_patches', 196)
        self.quantize_4bit = quantize_4bit
        self.use_hf_model = use_hf_model

        # Image preprocessing
        self.preprocess = transforms.Compose([
            transforms.Resize((self.image_size, self.image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        
        # Choose between HuggingFace model (supports 4-bit) or custom implementation
        if use_hf_model:
            self._init_hf_model(pretrained_path, quantize_4bit)
        else:
            self._init_custom_model(pretrained_path)

    def _init_hf_model(self, pretrained_path=None, quantize_4bit=False):
        """Initialize HuggingFace DeiT model with optional 4-bit quantization"""
        from transformers import AutoModel, AutoImageProcessor

        model_name = pretrained_path or "facebook/deit-tiny-patch16-224"

        print(f"Loading DeiT from HuggingFace: {model_name}")

        if quantize_4bit:
            print("  ⚙️  Loading DeiT in 4-bit quantized format (BitsAndBytes NF4)")
            from transformers import BitsAndBytesConfig

            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4"
            )

            self.hf_model = AutoModel.from_pretrained(
                model_name,
                quantization_config=bnb_config,
                trust_remote_code=True,
                device_map="auto"
            )
            print("  ✅ DeiT loaded in 4-bit (frozen backbone)")
        else:
            print("  ⚙️  Loading DeiT in full precision (FP32)")
            self.hf_model = AutoModel.from_pretrained(
                model_name,
                torch_dtype=torch.float32,
                trust_remote_code=True
            )
            print("  ✅ DeiT loaded in FP32")

        # Get image processor
        self.image_processor = AutoImageProcessor.from_pretrained(model_name)

        # Mark as HF model
        self.patch_embed = None  # Not used in HF mode
        self.cls_token = None
        self.pos_embed = None
        self.transformer = None
        self.norm = None

    def _init_custom_model(self, pretrained_path=None):
        """Initialize custom PyTorch DeiT implementation (FP32 only)"""
        print("Using custom DeiT implementation (FP32)")

        # Mark as custom model
        self.hf_model = None
        self.image_processor = None

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
        # Use HuggingFace model if available
        if self.hf_model is not None:
            return self._forward_hf(images)
        else:
            return self._forward_custom(images)

    def _forward_hf(self, images):
        """Forward pass with HuggingFace model"""
        # Handle PIL images
        if isinstance(images, list):
            # Use HF image processor
            inputs = self.image_processor(images=images, return_tensors="pt")
            pixel_values = inputs['pixel_values'].to(self.hf_model.device)
        else:
            # Already tensors - no processing needed
            pixel_values = images

        # Get model outputs
        outputs = self.hf_model(pixel_values)

        # Extract hidden states (excluding CLS token)
        # DeiT output: [batch_size, num_patches + 1, hidden_size]
        hidden_states = outputs.last_hidden_state

        # Return only patch tokens (exclude CLS token at position 0)
        patch_tokens = hidden_states[:, 1:, :]

        return patch_tokens

    def _forward_custom(self, images):
        """Forward pass with custom PyTorch model"""
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
            images: (batch_size, 3, H, W) or list of PIL Images

        Returns:
            cls_features: (batch_size, hidden_size)
        """
        # Use HuggingFace model if available
        if self.hf_model is not None:
            return self._get_cls_token_hf(images)
        else:
            return self._get_cls_token_custom(images)

    def _get_cls_token_hf(self, images):
        """Get CLS token with HuggingFace model"""
        # Handle PIL images
        if isinstance(images, list):
            inputs = self.image_processor(images=images, return_tensors="pt")
            pixel_values = inputs['pixel_values'].to(self.hf_model.device)
        else:
            pixel_values = images

        # Get model outputs
        outputs = self.hf_model(pixel_values)

        # Extract CLS token (first token)
        cls_token = outputs.last_hidden_state[:, 0, :]

        return cls_token

    def _get_cls_token_custom(self, images):
        """Get CLS token with custom model"""
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
