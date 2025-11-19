"""
1.58-bit Quantization Implementation
Following BitNet 1.58-bit quantization methodology
"""

import torch
import torch.nn as nn
import numpy as np


def quantize_weights_158bit(weight):
    """
    Quantize weights to 1.58-bit: {-1, 0, +1}
    
    Args:
        weight: torch.Tensor of any shape
    
    Returns:
        quantized_weight: torch.Tensor with values in {-1, 0, 1}
        scale: scaling factor for dequantization
    """
    # Compute scale as mean absolute value
    scale = weight.abs().mean()
    
    if scale == 0:
        return torch.zeros_like(weight), scale
    
    # Normalize weights
    normalized = weight / scale
    
    # Quantize to {-1, 0, 1}
    # Use thresholds at -0.5 and 0.5
    quantized = torch.where(
        normalized > 0.5,
        torch.ones_like(normalized),
        torch.where(
            normalized < -0.5,
            -torch.ones_like(normalized),
            torch.zeros_like(normalized)
        )
    )
    
    return quantized, scale


def dequantize_weights_158bit(quantized_weight, scale):
    """
    Dequantize 1.58-bit weights
    
    Args:
        quantized_weight: torch.Tensor with values in {-1, 0, 1}
        scale: scaling factor
    
    Returns:
        weight: dequantized tensor
    """
    return quantized_weight * scale


class QuantizedLinear158(nn.Module):
    """
    Linear layer with 1.58-bit weight quantization
    """
    
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        
        self.in_features = in_features
        self.out_features = out_features
        
        # Store quantized weights and scale
        self.register_buffer('quantized_weight', 
                           torch.zeros(out_features, in_features, dtype=torch.int8))
        self.register_buffer('weight_scale', torch.ones(1))
        
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter('bias', None)
    
    def quantize_and_pack(self, weight):
        """
        Quantize weight matrix and pack
        
        Args:
            weight: (out_features, in_features) float tensor
        """
        quantized, scale = quantize_weights_158bit(weight)
        
        # Convert to int8 for storage
        self.quantized_weight.data = quantized.to(torch.int8)
        self.weight_scale.data = scale.view(1)
    
    def forward(self, x):
        """
        Forward pass with dequantization
        
        Args:
            x: (*, in_features)
        
        Returns:
            output: (*, out_features)
        """
        # Dequantize weights
        weight = self.quantized_weight.float() * self.weight_scale
        
        # Apply linear transformation
        output = F.linear(x, weight, self.bias)
        
        return output


def quantize_model_158bit(model):
    """
    Quantize entire model to 1.58-bit
    
    Args:
        model: nn.Module to quantize
    
    Returns:
        quantized_model: model with quantized linear layers
    """
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            # Create quantized version
            quantized_layer = QuantizedLinear158(
                module.in_features,
                module.out_features,
                bias=(module.bias is not None)
            )
            
            # Quantize weights
            quantized_layer.quantize_and_pack(module.weight.data)
            
            # Copy bias if exists
            if module.bias is not None:
                quantized_layer.bias.data = module.bias.data.clone()
            
            # Replace module
            parent_name = '.'.join(name.split('.')[:-1])
            child_name = name.split('.')[-1]
            
            if parent_name:
                parent = dict(model.named_modules())[parent_name]
                setattr(parent, child_name, quantized_layer)
            else:
                setattr(model, child_name, quantized_layer)
    
    return model


def estimate_model_size_158bit(model):
    """
    Estimate model size with 1.58-bit quantization
    
    Args:
        model: nn.Module
    
    Returns:
        size_mb: estimated size in megabytes
    """
    total_bits = 0
    
    for param in model.parameters():
        param_elements = param.numel()
        
        if param.dim() >= 2:  # Weight matrices
            # 1.58 bits per parameter + scale factor
            total_bits += param_elements * 1.58
            total_bits += 32  # Scale factor in float32
        else:  # Biases and other parameters
            # Keep in float32
            total_bits += param_elements * 32
    
    size_bytes = total_bits / 8
    size_mb = size_bytes / (1024 * 1024)
    
    return size_mb


class QuantizationAwareTraining:
    """
    Quantization-aware training utilities
    Simulates quantization during training
    """
    
    @staticmethod
    def apply_weight_quantization_noise(model, noise_level=0.1):
        """
        Apply quantization noise during training
        
        Args:
            model: model to add noise to
            noise_level: noise strength
        """
        for module in model.modules():
            if isinstance(module, nn.Linear):
                with torch.no_grad():
                    # Simulate quantization
                    quantized, scale = quantize_weights_158bit(module.weight.data)
                    dequantized = dequantize_weights_158bit(quantized, scale)
                    
                    # Add noise to simulate quantization error
                    noise = (module.weight.data - dequantized) * noise_level
                    module.weight.data = dequantized + noise
    
    @staticmethod
    def get_quantization_aware_loss(model, lambda_quant=0.01):
        """
        Compute quantization-aware regularization loss
        Encourages weights to be close to {-1, 0, 1}
        
        Args:
            model: model to compute loss for
            lambda_quant: regularization strength
        
        Returns:
            loss: quantization regularization loss
        """
        quant_loss = 0
        
        for module in model.modules():
            if isinstance(module, nn.Linear):
                weight = module.weight
                scale = weight.abs().mean()
                
                if scale > 0:
                    normalized = weight / scale
                    
                    # Penalty for not being in {-1, 0, 1}
                    deviation = torch.min(
                        torch.min(
                            (normalized - 1).abs(),
                            normalized.abs()
                        ),
                        (normalized + 1).abs()
                    )
                    
                    quant_loss += deviation.mean()
        
        return lambda_quant * quant_loss


def save_quantized_model(model, path):
    """
    Save quantized model with metadata
    
    Args:
        model: quantized model
        path: save path
    """
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'quantization': '1.58-bit',
        'model_size_mb': estimate_model_size_158bit(model)
    }
    
    torch.save(checkpoint, path)
    print(f"Quantized model saved to {path}")
    print(f"Estimated size: {checkpoint['model_size_mb']:.2f} MB")


def load_quantized_model(model, path):
    """
    Load quantized model
    
    Args:
        model: model instance to load weights into
        path: checkpoint path
    
    Returns:
        model: loaded model
    """
    checkpoint = torch.load(path, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    
    print(f"Quantized model loaded from {path}")
    print(f"Quantization: {checkpoint.get('quantization', 'unknown')}")
    print(f"Model size: {checkpoint.get('model_size_mb', 'unknown')} MB")
    
    return model
