"""
4-bit Quantization for Training
Implements symmetric 4-bit quantization for weights and activations
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def quantize_4bit_symmetric(tensor, bits=4):
    """
    Symmetric 4-bit quantization
    
    Args:
        tensor: input tensor
        bits: number of bits (default 4)
    
    Returns:
        quantized: quantized tensor
        scale: scaling factor
    """
    # Compute scale: max absolute value
    qmax = 2 ** (bits - 1) - 1  # 7 for 4-bit
    max_val = tensor.abs().max()

    # CRITICAL: Prevent division by zero with proper epsilon clamping
    scale = max_val / qmax
    scale = scale.clamp(min=1e-8)  # Ensure scale is never zero

    # Quantize (scale is guaranteed non-zero)
    quantized = torch.round(tensor / scale).clamp(-qmax - 1, qmax)
    
    return quantized, scale


def dequantize_4bit_symmetric(quantized, scale):
    """
    Dequantize 4-bit tensor
    
    Args:
        quantized: quantized tensor
        scale: scaling factor
    
    Returns:
        tensor: dequantized tensor
    """
    return quantized * scale


class FakeQuantize4bit(nn.Module):
    """
    Fake quantization module for QAT (Quantization-Aware Training)
    Simulates quantization during forward pass but maintains gradients
    """
    
    def __init__(self, bits=4):
        super().__init__()
        self.bits = bits
        self.qmax = 2 ** (bits - 1) - 1
    
    def forward(self, x):
        """
        Apply fake quantization
        
        Args:
            x: input tensor
        
        Returns:
            quantized: fake-quantized tensor (maintains gradients)
        """
        # Compute scale with safe minimum
        max_val = x.abs().max()
        scale = (max_val / self.qmax).clamp(min=1e-8)

        # Quantize and dequantize (scale is guaranteed non-zero)
        quantized = torch.round(x / scale).clamp(-self.qmax - 1, self.qmax)
        dequantized = quantized * scale
        
        # Straight-through estimator for gradients
        return x + (dequantized - x).detach()


class QuantizedLinear4bit(nn.Module):
    """
    4-bit quantized linear layer
    """
    
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        
        self.in_features = in_features
        self.out_features = out_features
        
        # Store quantized weights
        self.register_buffer('quantized_weight',
                           torch.zeros(out_features, in_features, dtype=torch.int8))
        self.register_buffer('weight_scale', torch.ones(1))
        
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter('bias', None)
        
        # Fake quantization for activations during training
        self.activation_quant = FakeQuantize4bit(bits=4)
    
    def quantize_and_pack(self, weight):
        """Quantize and pack weights"""
        quantized, scale = quantize_4bit_symmetric(weight, bits=4)
        
        self.quantized_weight.data = quantized.to(torch.int8)
        self.weight_scale.data = scale.view(1)
    
    def forward(self, x):
        """
        Forward with quantized weights and activation quantization
        
        Args:
            x: input tensor
        
        Returns:
            output: quantized output
        """
        # Dequantize weights
        weight = self.quantized_weight.float() * self.weight_scale
        
        # Quantize activations (fake quantization during training)
        if self.training:
            x = self.activation_quant(x)
        
        # Linear transformation
        output = F.linear(x, weight, self.bias)
        
        return output


def apply_4bit_quantization(model):
    """
    Apply 4-bit quantization to model
    
    Args:
        model: model to quantize
    
    Returns:
        quantized_model: model with 4-bit quantized layers
    """
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            # Create quantized layer
            quantized_layer = QuantizedLinear4bit(
                module.in_features,
                module.out_features,
                bias=(module.bias is not None)
            )
            
            # Quantize weights
            quantized_layer.quantize_and_pack(module.weight.data)
            
            # Copy bias
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


def estimate_model_size_4bit(model):
    """
    Estimate model size with 4-bit quantization
    
    Args:
        model: model to estimate
    
    Returns:
        size_mb: estimated size in MB
    """
    total_bits = 0
    
    for param in model.parameters():
        param_elements = param.numel()
        
        if param.dim() >= 2:  # Weight matrices
            # 4 bits per parameter + scale
            total_bits += param_elements * 4
            total_bits += 32  # Scale factor
        else:  # Biases
            total_bits += param_elements * 32
    
    size_bytes = total_bits / 8
    size_mb = size_bytes / (1024 * 1024)
    
    return size_mb


class QuantizationConfig:
    """Configuration for quantization"""
    
    def __init__(self):
        self.weight_bits = 4
        self.activation_bits = 4
        self.method = "symmetric"
        self.per_channel = False
        self.fake_quant = True  # Use fake quantization during training
    
    def to_dict(self):
        return {
            'weight_bits': self.weight_bits,
            'activation_bits': self.activation_bits,
            'method': self.method,
            'per_channel': self.per_channel,
            'fake_quant': self.fake_quant
        }


def save_4bit_model(model, path, config=None):
    """
    Save 4-bit quantized model
    
    Args:
        model: quantized model
        path: save path
        config: quantization configuration
    """
    if config is None:
        config = QuantizationConfig()
    
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'quantization': '4-bit',
        'quantization_config': config.to_dict() if hasattr(config, 'to_dict') else config,
        'model_size_mb': estimate_model_size_4bit(model)
    }
    
    torch.save(checkpoint, path)
    print(f"4-bit quantized model saved to {path}")
    print(f"Estimated size: {checkpoint['model_size_mb']:.2f} MB")


def load_4bit_model(model, path):
    """
    Load 4-bit quantized model
    
    Args:
        model: model instance
        path: checkpoint path
    
    Returns:
        model: loaded model
    """
    checkpoint = torch.load(path, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    
    print(f"4-bit model loaded from {path}")
    print(f"Quantization: {checkpoint.get('quantization', 'unknown')}")
    print(f"Model size: {checkpoint.get('model_size_mb', 'unknown')} MB")
    
    return model
