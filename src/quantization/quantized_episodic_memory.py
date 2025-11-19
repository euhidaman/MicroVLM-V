"""
Quantized Episodic Memory with 1.58-bit Weight Quantization
Applies BitNet-style quantization during training for memory efficiency
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from .quantize_158bit import quantize_weights_158bit, dequantize_weights_158bit

EPSILON = 1e-6


class QuantizedLinear158BitGrad(nn.Module):
    """
    1.58-bit linear layer with gradient-aware quantization
    Maintains full precision during backward pass for proper learning
    """
    
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        
        self.in_features = in_features
        self.out_features = out_features
        
        # Store full precision weights for training
        self.weight = nn.Parameter(torch.randn(out_features, in_features))
        
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter('bias', None)
        
        # Buffer to track quantization error for monitoring
        self.register_buffer('quant_error', torch.zeros(1))
        
    def forward(self, x):
        """
        Forward with straight-through estimator for gradients
        """
        # Quantize weights during forward pass
        quantized_weight, scale = quantize_weights_158bit(self.weight)
        dequantized_weight = dequantize_weights_158bit(quantized_weight, scale)
        
        # Track quantization error
        with torch.no_grad():
            self.quant_error.fill_((self.weight - dequantized_weight).abs().mean())
        
        # Straight-through estimator: forward uses quantized, backward uses full precision
        # This is done by adding zero in forward but preserving gradients
        quantized_weight_ste = dequantized_weight + (self.weight - self.weight.detach())
        
        return F.linear(x, quantized_weight_ste, self.bias)
    
    def get_quantization_stats(self):
        """Return quantization error for monitoring"""
        return {
            'quant_error': self.quant_error.item(),
            'weight_sparsity': (self.weight.abs() < 0.1).float().mean().item()
        }


def apply_158bit_quantization_to_memory(episodic_memory_module):
    """
    Replace Linear layers in episodic memory with quantized versions
    
    Args:
        episodic_memory_module: EpisodicMemory instance
    
    Returns:
        Modified module with quantized linear layers
    """
    # Quantize W_M projection (memory to KV)
    if hasattr(episodic_memory_module, 'W_M'):
        old_layer = episodic_memory_module.W_M
        new_layer = QuantizedLinear158BitGrad(
            old_layer.in_features,
            old_layer.out_features,
            bias=(old_layer.bias is not None)
        )
        new_layer.weight.data = old_layer.weight.data.clone()
        if old_layer.bias is not None:
            new_layer.bias.data = old_layer.bias.data.clone()
        episodic_memory_module.W_M = new_layer
    
    # Quantize ScopeDetector MLP
    if hasattr(episodic_memory_module, 'scope_detector'):
        scope_detector = episodic_memory_module.scope_detector
        if hasattr(scope_detector, 'mlp'):
            for i, module in enumerate(scope_detector.mlp):
                if isinstance(module, nn.Linear):
                    old_layer = module
                    new_layer = QuantizedLinear158BitGrad(
                        old_layer.in_features,
                        old_layer.out_features,
                        bias=(old_layer.bias is not None)
                    )
                    new_layer.weight.data = old_layer.weight.data.clone()
                    if old_layer.bias is not None:
                        new_layer.bias.data = old_layer.bias.data.clone()
                    scope_detector.mlp[i] = new_layer
    
    return episodic_memory_module


def get_memory_quantization_stats(episodic_memory_module):
    """
    Collect quantization statistics from all quantized layers
    
    Args:
        episodic_memory_module: EpisodicMemory with quantized layers
    
    Returns:
        dict: Statistics for monitoring
    """
    stats = {}
    
    # W_M stats
    if hasattr(episodic_memory_module, 'W_M') and isinstance(
        episodic_memory_module.W_M, QuantizedLinear158BitGrad
    ):
        w_m_stats = episodic_memory_module.W_M.get_quantization_stats()
        stats['W_M_quant_error'] = w_m_stats['quant_error']
        stats['W_M_sparsity'] = w_m_stats['weight_sparsity']
    
    # ScopeDetector stats
    if hasattr(episodic_memory_module, 'scope_detector'):
        scope_detector = episodic_memory_module.scope_detector
        if hasattr(scope_detector, 'mlp'):
            quant_errors = []
            sparsities = []
            for module in scope_detector.mlp:
                if isinstance(module, QuantizedLinear158BitGrad):
                    layer_stats = module.get_quantization_stats()
                    quant_errors.append(layer_stats['quant_error'])
                    sparsities.append(layer_stats['weight_sparsity'])
            
            if quant_errors:
                stats['scope_detector_quant_error'] = sum(quant_errors) / len(quant_errors)
                stats['scope_detector_sparsity'] = sum(sparsities) / len(sparsities)
    
    return stats
