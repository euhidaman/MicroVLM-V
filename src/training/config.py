"""
Training Configuration
Defines training hyperparameters and settings
"""

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class TrainingConfig:
    """Main training configuration"""
    
    # Model paths
    qwen_model: str = "Qwen/Qwen2.5-0.5B"
    deit_checkpoint: Optional[str] = None
    
    # Data
    metadata_file: str = "./data/cc12m/cc12m_metadata_0.tsv"
    image_dir: str = "./data/cc12m/images"
    max_samples: Optional[int] = None  # None for full dataset
    
    # Training
    batch_size: int = 32
    num_epochs: int = 10
    learning_rate: float = 1e-4
    warmup_steps: int = 1000
    weight_decay: float = 0.01
    gradient_clip: float = 1.0
    
    # Episode settings for memory
    episode_size: int = 4
    
    # Loss weights
    lm_loss_weight: float = 1.0
    alignment_loss_weight: float = 0.1
    memory_kl_weight: float = 0.01
    addressing_kl_weight: float = 0.001
    
    # Quantization
    use_4bit: bool = False
    
    # Optimization
    optimizer: str = "adamw"
    scheduler: str = "cosine"
    
    # Logging
    log_interval: int = 100
    eval_interval: int = 1000
    save_interval: int = 5000
    visualize_interval: int = 5000
    
    # WandB
    use_wandb: bool = True
    wandb_project: str = "MicroVLM-V"
    wandb_username: str = "aman-derax20"
    wandb_run_name: Optional[str] = None
    
    # Output
    output_dir: str = "./checkpoints"
    
    # Hardware
    device: str = "cuda"
    num_workers: int = 4
    mixed_precision: bool = True
    
    # Freezing strategy
    freeze_vision: bool = True
    freeze_language: bool = True
    unfreeze_last_n_layers: int = 0  # 0 means fully frozen
    
    # Memory
    use_memory: bool = True
    use_alignment: bool = True


@dataclass
class SmallScaleTestConfig(TrainingConfig):
    """Configuration for small-scale testing with 1000 images"""
    
    metadata_file: str = "./data/cc12m/cc12m_test_1000.tsv"
    max_samples: int = 1000
    batch_size: int = 16
    num_epochs: int = 5
    log_interval: int = 10
    eval_interval: int = 50
    save_interval: int = 100
    visualize_interval: int = 50
    episode_size: int = 2


@dataclass
class Stage1Config(TrainingConfig):
    """Stage 1: Train adapters and memory only"""
    
    freeze_vision: bool = True
    freeze_language: bool = True
    unfreeze_last_n_layers: int = 0
    
    num_epochs: int = 10
    learning_rate: float = 1e-4


@dataclass
class Stage2Config(TrainingConfig):
    """Stage 2: Unfreeze last N language layers"""
    
    freeze_vision: bool = True
    freeze_language: bool = False  # Will unfreeze last N
    unfreeze_last_n_layers: int = 4
    
    num_epochs: int = 5
    learning_rate: float = 1e-5  # Lower LR for fine-tuning


def load_config(config_name: str = "default"):
    """
    Load configuration by name
    
    Args:
        config_name: one of ['test', 'stage1', 'stage2', 'default']
    
    Returns:
        config: TrainingConfig instance
    """
    if config_name == "test":
        return SmallScaleTestConfig()
    elif config_name == "stage1":
        return Stage1Config()
    elif config_name == "stage2":
        return Stage2Config()
    else:
        return TrainingConfig()


def get_run_counter(counter_file="./checkpoints/run_counter.txt"):
    """
    Get and increment run counter
    
    Args:
        counter_file: path to counter file
    
    Returns:
        counter: current run number
    """
    import os
    from pathlib import Path
    
    Path(counter_file).parent.mkdir(parents=True, exist_ok=True)
    
    if os.path.exists(counter_file):
        with open(counter_file, 'r') as f:
            counter = int(f.read().strip())
    else:
        counter = 0
    
    # Increment
    new_counter = counter + 1
    with open(counter_file, 'w') as f:
        f.write(str(new_counter))
    
    return new_counter


def create_run_name(config_name: str = "default"):
    """
    Create run name with counter
    
    Args:
        config_name: configuration name
    
    Returns:
        run_name: formatted run name
    """
    from datetime import datetime
    
    counter = get_run_counter()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    run_name = f"run_{counter}_{config_name}_{timestamp}"
    
    return run_name, counter
