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
    qwen_model: str = "Qwen/Qwen2.5-0.5B"  # Alias for compatibility
    deit_checkpoint: Optional[str] = None
    
    # Data
    data_dir: str = "./data/cc12m"
    train_metadata: str = "train_metadata.json"
    val_metadata: str = "val_metadata.json"
    batch_size: int = 8  # Default reduced for 12GB GPU
    num_workers: int = 4  # Reduced to save system RAM
    gradient_accumulation_steps: int = 1  # No accumulation by default
    max_samples: Optional[int] = None
    max_val_samples: Optional[int] = None
    
    # Computed paths (will be set from data_dir)
    @property
    def train_metadata_file(self):
        return f"{self.data_dir}/{self.train_metadata}"
    
    @property
    def val_metadata_file(self):
        return f"{self.data_dir}/{self.val_metadata}"
    
    # Training
    num_epochs: int = 5
    learning_rate: float = 1e-4
    warmup_steps: int = 1000
    weight_decay: float = 0.01
    gradient_clip: float = 0.5
    
    # Memory
    use_memory: bool = True
    episode_size: int = 1
    memory_kl_weight: float = 0.02
    addressing_kl_weight: float = 0.005
    lm_loss_weight: float = 1.0
    alignment_loss_weight: float = 1.0
    
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
    wandb_username: str = "aman-derax20"
    wandb_run_name: Optional[str] = None
    
    # HuggingFace
    hf_username: str = "euhidaman"
    hf_repo_name: str = "MicroVLM-V"
    
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
    num_epochs: int = 15  # Increased for stable alignment convergence
    learning_rate: float = 5e-5  # Lower LR for contrastive learning stability
    warmup_steps: int = 3000  # ~5-10% of total steps for gradual warmup
    batch_size: int = 8  # Reduced for 12GB GPU (was 64)
    num_workers: int = 4  # Reduced to save RAM
    gradient_clip: float = 0.3  # Tighter clipping for alignment-only training
    alignment_loss_weight: float = 1.0  # Full weight since it's the only loss
    gradient_accumulation_steps: int = 8  # Effective batch = 8*8=64
    
    # Adapter-focused training
    freeze_vision: bool = True
    freeze_language: bool = True
    train_adapter_only: bool = True
    unfreeze_last_n_layers: int = 0
    
    # Quantization - Disabled for higher GPU compute load
    enable_quantization: bool = False
    quantize_vision_4bit: bool = False
    quantize_language_4bit: bool = False
    
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
    batch_size: int = 4  # Reduced for memory component (was 64)
    num_workers: int = 4
    gradient_accumulation_steps: int = 16  # Effective batch = 4*16=64
    
    # Keep vision/language frozen, train adapter + memory
    freeze_vision: bool = False
    freeze_language: bool = True
    train_adapter_only: bool = False
    train_memory: bool = True
    unfreeze_last_n_layers: int = 4
    
    # Disable quantization for higher compute (memory still uses 1.58-bit)
    enable_quantization: bool = True
    quantize_memory_158bit: bool = True
    quantize_vision_4bit: bool = False
    quantize_language_4bit: bool = False
    
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
    batch_size: int = 2  # Minimal for 12GB GPU
    num_epochs: int = 2
    learning_rate: float = 5e-5
    use_memory: bool = False  # Disable memory for testing to save VRAM
    enable_quantization: bool = False  # Disable for faster testing
    gradient_accumulation_steps: int = 4  # Effective batch = 2*4=8
    log_interval: int = 10
    visualize_interval: int = 50
    viz_save_interval: int = 200
    num_viz_images: int = 3
    wandb_run_name: str = "test_run"
    
    # Limit to 5000 samples for quick testing
    max_samples: int = 5000
    max_val_samples: int = 5000
    
    # Override output dir for test
    output_dir: str = "./checkpoints"
    device: str = "cuda"
    train_metadata: str = "train_metadata.json"
    val_metadata: str = "val_metadata.json"
    max_samples: int = 5000


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
    batch_size: int = 4  # Reduced for 12GB GPU even with quantization
    gradient_accumulation_steps: int = 16  # Effective batch = 4*16=64
    
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
