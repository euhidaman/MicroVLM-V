"""
1.58-bit Quantization Implementation
Following BitNet 1.58-bit quantization methodology
Reference: BitNet b1.58 - Ternary {-1, 0, +1} weight quantization with INT8 activations
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


def quantize_weights_158bit(weight):
    """
    Quantize weights to 1.58-bit: {-1, 0, +1} following BitNet methodology

    Reference: BitNet b1.58 paper - ternary weight quantization

    Args:
        weight: torch.Tensor of any shape
    
    Returns:
        quantized_weight: torch.Tensor with values in {-1, 0, 1} (same dtype as input)
        scale: scaling factor (gamma) for dequantization
    """
    # Store original dtype to preserve it
    original_dtype = weight.dtype

    # Compute scale (gamma) as mean absolute value following BitNet
    scale = weight.abs().mean().clamp(min=1e-5)

    # Normalize weights by gamma
    normalized = weight / scale
    
    # Quantize to {-1, 0, +1} using sign and threshold
    # BitNet uses: round(clip(w/gamma, -1, 1))
    # Use to(original_dtype) instead of .float() to preserve dtype
    quantized = torch.sign(normalized) * (normalized.abs() > 0.5).to(original_dtype)

    return quantized, scale


def quantize_activations_int8(activation):
    """
    Quantize activations to INT8 following BitNet activation quantization

    Args:
        activation: torch.Tensor of any shape

    Returns:
        quantized: INT8 tensor
        scale: per-token scaling factors
    """
    # Compute per-token scale: 127 / max(|x|)
    scale = 127.0 / activation.abs().amax(dim=-1, keepdim=True).clamp(min=1e-5)

    # Quantize to [-128, 127]
    quantized = (activation * scale).round().clamp(-128, 127).to(torch.int8)

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
    Linear layer with 1.58-bit weight quantization and INT8 activation quantization
    Follows BitNet b1.58 architecture
    """
    
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        
        self.in_features = in_features
        self.out_features = out_features
        
        # Store quantized weights {-1, 0, 1} as int8
        self.register_buffer('quantized_weight',
                           torch.zeros(out_features, in_features, dtype=torch.int8))
        self.register_buffer('weight_scale', torch.ones(1))
        
        # Bias kept in full precision (standard practice)
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter('bias', None)
    
    def quantize_and_pack(self, weight):
        """
        Quantize weight matrix to 1.58-bit and pack

        Args:
            weight: (out_features, in_features) float tensor
        """
        quantized, scale = quantize_weights_158bit(weight)
        
        # Store as int8 for efficient computation
        self.quantized_weight.data = quantized.to(torch.int8)
        self.weight_scale.data = scale.view(1)
    
    def forward(self, x):
        """
        Forward pass: INT8 activations × 1.58-bit weights

        Args:
            x: (*, in_features) float tensor

        Returns:
            output: (*, out_features) float tensor
        """
        # Step 1: Quantize input activations to INT8
        x_quant, act_scale = quantize_activations_int8(x)

        # Step 2: INT8 x INT2 (1.58-bit) matrix multiplication
        # Dequantize for computation (in practice, use optimized kernel)
        weight_dequant = self.quantized_weight.float() * self.weight_scale
        x_dequant = x_quant.float() / act_scale

        # Step 3: Compute output
        output = torch.nn.functional.linear(x_dequant, weight_dequant, self.bias)

        return output

    def get_quantization_info(self):
        """Get quantization statistics for monitoring"""
        weight_sparsity = (self.quantized_weight == 0).float().mean()
        weight_pos = (self.quantized_weight == 1).float().mean()
        weight_neg = (self.quantized_weight == -1).float().mean()

        return {
            'weight_scale': self.weight_scale.item(),
            'weight_sparsity': weight_sparsity.item(),
            'weight_positive_ratio': weight_pos.item(),
            'weight_negative_ratio': weight_neg.item(),
            'bits_per_param': 1.58
        }


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
