"""
MicroVLM-V Model Components
"""

from .multimodal_adapter import MultimodalAdapter
from .episodic_memory import EpisodicMemory, ScopeDetector
from .vision_encoder import DeiTVisionEncoder
from .language_model import Qwen2LanguageModel
from .microvlm import MicroVLM


def create_microvlm(config, language_checkpoint=None, vision_checkpoint=None, 
                    quantize_4bit=False, quantize_memory_158bit=False,
                    training_config=None):
    """
    Create MicroVLM model instance
    
    Args:
        config: model configuration dict
        language_checkpoint: path to Qwen2.5-0.5B checkpoint
        vision_checkpoint: path to DeiT-Tiny checkpoint
        quantize_4bit: whether to use 4-bit quantization
        quantize_memory_158bit: whether to use 1.58-bit quantization for memory
    
    Returns:
        model: MicroVLM instance
    """
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
    'create_microvlm'
]
