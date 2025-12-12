"""
Quantized Episodic Memory with Mixed-Precision Quantization
- 1.58-bit for MLP projections (W_M, scope detector)
- 4-bit for memory slots (memory_mean, memory_logvar)
Applies BitNet-style quantization with 4-bit memory storage
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from .quantize_158bit import quantize_weights_158bit, dequantize_weights_158bit, quantize_activations_int8
from .quantize_4bit import quantize_4bit_symmetric, dequantize_4bit_symmetric

EPSILON = 1e-6


class QuantizedMemorySlots4bit(nn.Module):
    """
    4-bit quantized memory slots for episodic memory
    Stores memory_mean and memory_logvar in 4-bit precision with proper bit packing
    """

    def __init__(self, memory_size, code_size):
        super().__init__()

        self.memory_size = memory_size
        self.code_size = code_size

        # Store quantized memory slots (4-bit packed as uint8, 2 values per byte)
        packed_size = (code_size + 1) // 2  # Ceil division for 2 values per byte
        self.register_buffer('memory_mean_quantized',
                           torch.zeros(memory_size, packed_size, dtype=torch.uint8))
        self.register_buffer('memory_mean_scale',
                           torch.ones(memory_size, 1))

        self.register_buffer('memory_logvar_quantized',
                           torch.zeros(memory_size, packed_size, dtype=torch.uint8))
        self.register_buffer('memory_logvar_scale',
                           torch.ones(memory_size, 1))

    def quantize_memory(self, memory_mean, memory_logvar):
        """
        Quantize memory slots to 4-bit with proper bit packing

        Args:
            memory_mean: (memory_size, code_size) float tensor
            memory_logvar: (memory_size, code_size) float tensor OR (1,) scalar buffer
        """
        # Quantize memory_mean per-slot with bit packing
        for i in range(self.memory_size):
            quant, scale = quantize_4bit_symmetric(memory_mean[i], bits=4)
            # Pack two 4-bit values into each uint8
            packed = self._pack_4bit(quant.to(torch.int8))
            self.memory_mean_quantized[i] = packed
            self.memory_mean_scale[i] = scale.view(1)

        # Handle memory_logvar: check if it's a scalar buffer (shape (1,)) or full matrix
        if memory_logvar.numel() == 1:
            # Scalar buffer case: replicate the single value for all slots
            scalar_val = memory_logvar.item()
            for i in range(self.memory_size):
                dummy_tensor = torch.full((self.code_size,), scalar_val, device=memory_logvar.device)
                quant, scale = quantize_4bit_symmetric(dummy_tensor, bits=4)
                packed = self._pack_4bit(quant.to(torch.int8))
                self.memory_logvar_quantized[i] = packed
                self.memory_logvar_scale[i] = scale.view(1)
        else:
            # Full matrix case: quantize per-slot
            for i in range(self.memory_size):
                quant, scale = quantize_4bit_symmetric(memory_logvar[i], bits=4)
                packed = self._pack_4bit(quant.to(torch.int8))
                self.memory_logvar_quantized[i] = packed
                self.memory_logvar_scale[i] = scale.view(1)

    def _pack_4bit(self, values_int8):
        """Pack two 4-bit values into each uint8 byte"""
        # Ensure values are in [-8, 7] range for 4-bit signed
        values_int8 = values_int8.clamp(-8, 7)
        # Convert to unsigned [0, 15] for packing
        values_uint = (values_int8 + 8).to(torch.uint8)

        # Pad to even length
        if len(values_uint) % 2 == 1:
            values_uint = torch.cat([values_uint, torch.zeros(1, dtype=torch.uint8, device=values_uint.device)])

        # Pack: lower 4 bits from even indices, upper 4 bits from odd indices
        packed = values_uint[::2] | (values_uint[1::2] << 4)
        return packed

    def _unpack_4bit(self, packed, original_size):
        """Unpack 4-bit values from uint8 bytes"""
        # Unpack lower and upper nibbles
        lower = packed & 0x0F
        upper = (packed >> 4) & 0x0F

        # Interleave
        unpacked = torch.zeros(len(packed) * 2, dtype=torch.uint8, device=packed.device)
        unpacked[::2] = lower
        unpacked[1::2] = upper

        # Trim to original size and convert back to signed [-8, 7]
        unpacked = unpacked[:original_size]
        return (unpacked.to(torch.int8) - 8)

    def dequantize_memory(self):
        """
        Dequantize memory slots back to float with unpacking

        Returns:
            memory_mean: (memory_size, code_size) float tensor
            memory_logvar: (memory_size, code_size) float tensor
        """
        # Unpack and dequantize memory_mean
        memory_mean = torch.zeros(self.memory_size, self.code_size, device=self.memory_mean_quantized.device)
        for i in range(self.memory_size):
            unpacked = self._unpack_4bit(self.memory_mean_quantized[i], self.code_size)
            memory_mean[i] = unpacked.float() * self.memory_mean_scale[i]

        # Unpack and dequantize memory_logvar
        memory_logvar = torch.zeros(self.memory_size, self.code_size, device=self.memory_logvar_quantized.device)
        for i in range(self.memory_size):
            unpacked = self._unpack_4bit(self.memory_logvar_quantized[i], self.code_size)
            memory_logvar[i] = unpacked.float() * self.memory_logvar_scale[i]

        return memory_mean, memory_logvar

    def get_memory_quantization_stats(self):
        """Get quantization statistics for monitoring"""
        return {
            'memory_mean_scale_mean': self.memory_mean_scale.mean().item(),
            'memory_mean_scale_std': self.memory_mean_scale.std().item(),
            'memory_logvar_scale_mean': self.memory_logvar_scale.mean().item(),
            'memory_logvar_scale_std': self.memory_logvar_scale.std().item(),
            'memory_bits_per_param': 4.0,
            'memory_mean_sparsity': (self.memory_mean_quantized == 0).float().mean().item(),
        }


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
        Uses BitNet-style 1.58-bit weight quantization
        """
        # Save original dtype for output
        input_dtype = x.dtype

        # Convert input to float32 for stable computation
        x = x.to(torch.float32)

        # Quantize weights (operates on self.weight which should be float32)
        weight_float32 = self.weight.to(torch.float32)
        quantized_weight, scale = quantize_weights_158bit(weight_float32)
        dequantized_weight = dequantize_weights_158bit(quantized_weight, scale)
        
        # Track quantization error
        with torch.no_grad():
            self.quant_error.fill_((weight_float32 - dequantized_weight).abs().mean())

        # Straight-through estimator: forward uses quantized, backward uses full precision
        quantized_weight_ste = dequantized_weight + (weight_float32 - weight_float32.detach())

        # Optional: quantize activations for full BitNet emulation
        if self.training:
            x_quant, act_scale = quantize_activations_int8(x)
            x = x_quant.float() / act_scale

        # Perform linear operation in float32
        bias_float32 = self.bias.to(torch.float32) if self.bias is not None else None
        output = F.linear(x, quantized_weight_ste, bias_float32)

        # Convert output back to original dtype
        return output.to(input_dtype)

    def get_quantization_stats(self):
        """Return quantization error for monitoring"""
        quantized_weight, _ = quantize_weights_158bit(self.weight)
        sparsity = (quantized_weight == 0).float().mean()

        return {
            'quant_error': self.quant_error.item(),
            'weight_sparsity': sparsity.item(),
            'bits_per_param': 1.58
        }


def apply_mixed_precision_quantization_to_memory(episodic_memory_module, memory_size, code_size):
    """
    Apply mixed-precision quantization to episodic memory:
    - 1.58-bit for W_M and scope detector MLPs
    - 4-bit for memory slots (memory_mean, memory_logvar)

    Args:
        episodic_memory_module: EpisodicMemory instance
        memory_size: number of memory slots
        code_size: memory code dimension

    Returns:
        Modified module with quantized components
    """
    # Add 4-bit quantized memory slots
    quantized_memory = QuantizedMemorySlots4bit(memory_size, code_size)

    # If memory already exists, quantize it
    if hasattr(episodic_memory_module, 'memory_mean'):
        with torch.no_grad():
            memory_mean = episodic_memory_module.memory_mean.data
            memory_logvar = episodic_memory_module.memory_logvar.data
            quantized_memory.quantize_memory(memory_mean, memory_logvar)

    # Attach to module
    episodic_memory_module.quantized_memory_slots = quantized_memory

    # Quantize W_M projection with 1.58-bit (memory to KV)
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
    
    # Quantize ScopeDetector MLP with 1.58-bit
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
    Collect quantization statistics from all quantized components

    Args:
        episodic_memory_module: EpisodicMemory with quantized layers
    
    Returns:
        dict: Statistics for monitoring and WandB logging
    """
    stats = {}
    
    # 4-bit memory slot stats
    if hasattr(episodic_memory_module, 'quantized_memory_slots'):
        memory_stats = episodic_memory_module.quantized_memory_slots.get_memory_quantization_stats()
        stats.update({f'memory_slots/{k}': v for k, v in memory_stats.items()})

    # 1.58-bit W_M stats
    if hasattr(episodic_memory_module, 'W_M') and isinstance(
        episodic_memory_module.W_M, QuantizedLinear158BitGrad
    ):
        w_m_stats = episodic_memory_module.W_M.get_quantization_stats()
        stats['W_M_158bit/quant_error'] = w_m_stats['quant_error']
        stats['W_M_158bit/sparsity'] = w_m_stats['weight_sparsity']
        stats['W_M_158bit/bits_per_param'] = w_m_stats['bits_per_param']

    # 1.58-bit ScopeDetector stats
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
                stats['scope_detector_158bit/quant_error'] = sum(quant_errors) / len(quant_errors)
                stats['scope_detector_158bit/sparsity'] = sum(sparsities) / len(sparsities)
                stats['scope_detector_158bit/bits_per_param'] = 1.58

    return stats


def apply_158bit_quantization_to_memory(module, memory_size=None, code_size=None):
    """
    Backward-compatible wrapper for legacy imports.
    Delegates to apply_mixed_precision_quantization_to_memory for episodic memory modules.

    Args:
        module: Module to quantize (EpisodicMemory or ScopeDetector)
        memory_size: Number of memory slots (optional, will be inferred)
        code_size: Memory code dimension (optional, will be inferred)

    Returns:
        module: Quantized module
    """
    # EpisodicMemory-style module with memory buffers
    if hasattr(module, 'memory_mean') and hasattr(module, 'memory_logvar'):
        # Infer dimensions from module if not provided
        inferred_memory_size = memory_size or getattr(module, 'memory_size', None)
        inferred_code_size = code_size or getattr(module, 'code_size', None) or getattr(module, 'memory_dim', None)

        if inferred_memory_size is None or inferred_code_size is None:
            raise ValueError(
                f"Cannot quantize episodic memory: memory_size={inferred_memory_size}, "
                f"code_size={inferred_code_size}. Both must be specified or inferrable."
            )

        return apply_mixed_precision_quantization_to_memory(module, inferred_memory_size, inferred_code_size)

    # ScopeDetector or other modules with MLP
    if hasattr(module, 'mlp'):
        for i, submodule in enumerate(module.mlp):
            if isinstance(submodule, nn.Linear):
                quant_layer = QuantizedLinear158BitGrad(
                    submodule.in_features,
                    submodule.out_features,
                    bias=(submodule.bias is not None)
                )
                quant_layer.weight.data = submodule.weight.data.clone()
                if submodule.bias is not None:
                    quant_layer.bias.data = submodule.bias.data.clone()
                module.mlp[i] = quant_layer
        return module

    # Fallback: quantize any direct Linear children
    for name, submodule in module.named_children():
        if isinstance(submodule, nn.Linear):
            quant_layer = QuantizedLinear158BitGrad(
                submodule.in_features,
                submodule.out_features,
                bias=(submodule.bias is not None)
            )
            quant_layer.weight.data = submodule.weight.data.clone()
            if submodule.bias is not None:
                quant_layer.bias.data = submodule.bias.data.clone()
            setattr(module, name, quant_layer)

    return module


