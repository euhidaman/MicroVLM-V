"""
MicroVLM-V Model Components
"""

from .multimodal_adapter import MultimodalAdapter
from .episodic_memory import EpisodicMemory, ScopeDetector
from .vision_encoder import DeiTVisionEncoder
from .language_model import Qwen2LanguageModel
from .microvlm import MicroVLM


def create_microvlm(config, language_checkpoint=None, vision_checkpoint=None, 
                    quantize_4bit=False):
    """
    Create MicroVLM model instance
    
    Args:
        config: model configuration dict
        language_checkpoint: path to Qwen2.5-0.5B checkpoint
        vision_checkpoint: path to DeiT-Tiny checkpoint
        quantize_4bit: whether to use 4-bit quantization
    
    Returns:
        model: MicroVLM instance
    """
    model = MicroVLM(
        config=config,
        vision_checkpoint=vision_checkpoint,
        language_checkpoint=language_checkpoint,
        quantize_4bit=quantize_4bit
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
