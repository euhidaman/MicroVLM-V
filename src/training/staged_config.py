"""
Training Configuration with Staged Learning and Quantization
"""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class TrainingConfig:
    """Base training configuration"""
    # Model paths
    vision_checkpoint: Optional[str] = None
    language_checkpoint: Optional[str] = "Qwen/Qwen2.5-0.5B"
    
    # Data
    data_dir: str = "./data/cc12m"
    train_metadata: str = "train_metadata.json"
    val_metadata: str = "val_metadata.json"
    batch_size: int = 8
    num_workers: int = 4
    
    # Training
    num_epochs: int = 5
    learning_rate: float = 1e-4
    warmup_steps: int = 1000
    weight_decay: float = 0.01
    gradient_clip: float = 1.0
    
    # Memory
    use_memory: bool = True
    episode_size: int = 1
    memory_kl_weight: float = 0.02
    addressing_kl_weight: float = 0.005
    lm_loss_weight: float = 1.0
    alignment_loss_weight: float = 0.2
    
    # Alignment
    use_alignment: bool = True
    alignment_temperature: float = 0.07
    freeze_vision: bool = False
    freeze_language: bool = True
    unfreeze_last_n_layers: int = 4
    
    # Quantization
    enable_quantization: bool = False
    quantize_memory_158bit: bool = False
    quantize_vision_4bit: bool = False
    quantize_language_4bit: bool = False
    
    # Optimization
    optimizer: str = "adamw"
    lr_scheduler: str = "cosine"
    
    # Logging
    log_interval: int = 50
    save_interval: int = 100
    visualize_interval: int = 50
    
    # Visualization
    num_viz_images: int = 3  # Number of random images for attention visualization
    viz_save_interval: int = 5000  # Save full attention visualizations every N steps
    
    # WandB
    use_wandb: bool = True
    wandb_project: str = "MicroVLM-V"
    wandb_run_name: Optional[str] = None
    
    # Paths
    output_dir: str = "./checkpoints"
    device: str = "cuda"


@dataclass
class Stage1Config(TrainingConfig):
    """
    Stage 1: Alignment Training Without Memory
    Focus on learning image-text alignment before introducing memory
    Based on EVO-1 methodology
    """
    # Override defaults for Stage 1
    use_memory: bool = False  # Disable memory in Stage 1
    num_epochs: int = 3
    learning_rate: float = 3e-4  # Higher LR for adapter training
    warmup_steps: int = 500
    
    # Adapter-focused training
    freeze_vision: bool = True
    freeze_language: bool = True
    train_adapter_only: bool = True
    unfreeze_last_n_layers: int = 0
    
    # Quantization - Enable for efficiency
    enable_quantization: bool = True
    quantize_vision_4bit: bool = True
    quantize_language_4bit: bool = True
    
    # More frequent monitoring during alignment learning
    visualize_interval: int = 50
    viz_save_interval: int = 2000
    
    wandb_run_name: str = "stage1_alignment"


@dataclass
class Stage2Config(TrainingConfig):
    """
    Stage 2: Memory Integration Training
    Introduce episodic memory after alignment is learned
    Based on Larimar methodology
    """
    # Override defaults for Stage 2
    use_memory: bool = True  # Enable memory in Stage 2
    episode_size: int = 4  # Longer episodes for memory learning
    num_epochs: int = 5
    learning_rate: float = 1e-4  # Lower LR with memory
    warmup_steps: int = 1000
    
    # Keep vision/language frozen, train adapter + memory
    freeze_vision: bool = False
    freeze_language: bool = True
    train_adapter_only: bool = False
    train_memory: bool = True
    unfreeze_last_n_layers: int = 4
    
    # Full quantization
    enable_quantization: bool = True
    quantize_memory_158bit: bool = True
    quantize_vision_4bit: bool = True
    quantize_language_4bit: bool = True
    
    # Memory-specific settings (from Larimar)
    memory_size: int = 512
    observation_noise_std: float = 0.000001
    pseudoinverse_steps: int = 15
    memory_kl_weight: float = 0.02
    addressing_kl_weight: float = 0.005
    
    # Visualization
    visualize_interval: int = 100
    viz_save_interval: int = 5000
    
    wandb_run_name: str = "stage2_memory"


@dataclass
class TestConfig(TrainingConfig):
    """Quick test configuration for debugging"""
    batch_size: int = 8
    num_epochs: int = 2
    learning_rate: float = 5e-5
    use_memory: bool = True
    enable_quantization: bool = False  # Disable for faster testing
    log_interval: int = 10
    visualize_interval: int = 50
    viz_save_interval: int = 200
    num_viz_images: int = 3
    wandb_run_name: str = "test_run"
    
    # Limit to downloaded samples
    data_dir: str = "./data/cc12m"
    train_metadata: str = "train_metadata.json"
    val_metadata: str = "val_metadata.json"


@dataclass
class FullQuantizedConfig(TrainingConfig):
    """
    Fully quantized training for maximum efficiency
    All quantization features enabled
    """
    # Full quantization
    enable_quantization: bool = True
    quantize_memory_158bit: bool = True
    quantize_vision_4bit: bool = True
    quantize_language_4bit: bool = True
    
    # Memory enabled
    use_memory: bool = True
    episode_size: int = 4
    
    # Training settings
    num_epochs: int = 10
    learning_rate: float = 5e-5
    warmup_steps: int = 2000
    batch_size: int = 16  # Larger batch with quantization
    
    # Monitoring
    visualize_interval: int = 100
    viz_save_interval: int = 5000
    num_viz_images: int = 3
    
    wandb_run_name: str = "full_quantized"


def load_config(config_name: str = "default") -> TrainingConfig:
    """
    Load configuration by name
    
    Args:
        config_name: One of ['stage1', 'stage2', 'test', 'full_quantized', 'default']
    
    Returns:
        TrainingConfig instance
    """
    configs = {
        'stage1': Stage1Config,
        'stage2': Stage2Config,
        'test': TestConfig,
        'full_quantized': FullQuantizedConfig,
        'default': TrainingConfig,
    }
    
    if config_name not in configs:
        raise ValueError(f"Unknown config: {config_name}. Choose from {list(configs.keys())}")
    
    return configs[config_name]()
