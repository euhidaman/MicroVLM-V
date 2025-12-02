"""
MicroVLM-V Model Components
"""

from .multimodal_adapter import MultimodalAdapter
from .episodic_memory import EpisodicMemory, ScopeDetector
from .vision_encoder import DeiTVisionEncoder
from .language_model import Qwen2LanguageModel
from .microvlm import MicroVLM

# FIBER-style fusion components
from .fiber_fusion import (
    CrossModalAttention,
    FIBERFusionBlock,
    FIBERVisionEncoder,
    FIBERAlignmentLoss
)
from .microvlm_fiber import MicroVLM_FIBER, create_microvlm_fiber

# Alias for consistency
MicroVLMFIBER = MicroVLM_FIBER


def create_microvlm(config, language_checkpoint=None, vision_checkpoint=None, 
                    quantize_4bit=False, quantize_memory_158bit=False,
                    training_config=None, alignment_mode="baseline"):
    """
    Create MicroVLM model instance
    
    Args:
        config: model configuration dict
        language_checkpoint: path to Qwen2.5-0.5B checkpoint
        vision_checkpoint: path to DeiT-Tiny checkpoint
        quantize_4bit: whether to use 4-bit quantization
        quantize_memory_158bit: whether to use 1.58-bit quantization for memory
        training_config: optional training configuration
        alignment_mode: "baseline" or "fiber" for fusion-in-backbone
    
    Returns:
        model: MicroVLM or MicroVLMFIBER instance
    """
    if alignment_mode == "fiber":
        return create_microvlm_fiber(
            config=config,
            vision_checkpoint=vision_checkpoint,
            language_checkpoint=language_checkpoint,
            training_config=training_config
        )
    else:
        model = MicroVLM(
            config=config,
            vision_checkpoint=vision_checkpoint,
            language_checkpoint=language_checkpoint,
            quantize_4bit=quantize_4bit,
            quantize_memory_158bit=quantize_memory_158bit,
            training_config=training_config
        )
        return model


__all__ = [
    'MultimodalAdapter',
    'EpisodicMemory',
    'ScopeDetector',
    'DeiTVisionEncoder',
    'Qwen2LanguageModel',
    'MicroVLM',
    'create_microvlm',
    # FIBER components
    'CrossModalAttention',
    'FIBERFusionBlock',
    'FIBERVisionEncoder',
    'FIBERAlignmentLoss',
    'MicroVLM_FIBER',
    'MicroVLMFIBER',  # alias
    'create_microvlm_fiber',
]
