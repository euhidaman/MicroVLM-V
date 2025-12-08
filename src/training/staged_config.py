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
    batch_size: int = 64
    num_workers: int = 12
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
    # Skip LM forward pass (for alignment-only training)
    skip_lm_loss: bool = False

    # Alignment
    use_alignment: bool = True
    alignment_mode: str = 'baseline'  # 'baseline' or 'fiber'
    alignment_temperature: float = 0.07
    freeze_vision: bool = False
    freeze_language: bool = True
    unfreeze_last_n_layers: int = 4

    # Quantization - 4-bit for language model to reduce size < 1GB
    enable_quantization: bool = True
    quantize_memory_158bit: bool = False
    quantize_vision_4bit: bool = False
    quantize_language_4bit: bool = True  # Reduces Qwen from ~2GB to ~250MB

    # Optimization
    optimizer: str = "adamw"
    lr_scheduler: str = "cosine"

    # Logging
    log_interval: int = 50
    save_interval: int = 100
    visualize_interval: int = 300  # Save attention visualizations every N steps
    viz_save_interval: int = 5000  # Save full attention visualizations every N steps

    # Early Stopping (loss-based convergence detection)
    use_early_stopping: bool = True  # Enable early stopping on loss plateau
    early_stop_patience: int = 2  # Stop after N epochs without significant improvement
    early_stop_min_delta: float = 0.01  # Minimum loss change to count as improvement

    # Alignment-specific safeguards
    alignment_early_stop: bool = False
    alignment_patience: int = 0
    alignment_improve_min_delta: float = 0.01
    alignment_stop_threshold: float = 0.0
    alignment_negative_patience: int = 100
    save_best_alignment_checkpoint: bool = True
    alignment_save_cooldown: int = 100  # min steps between best-checkpoint saves
    alignment_min_stop_steps: int = 0  # do not stop on alignment before this step
    alignment_min_stop_steps: int = 0  # do not stop on alignment before this step

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
    Based on CLIP/EVO-1 methodology

    KEY INSIGHT: Since both vision and language models are FROZEN,
    we only train the multimodal adapter using contrastive alignment loss.
    LM loss is disabled because a frozen LM cannot improve.

    HYPERPARAMETERS tuned for stable contrastive learning:
    - Lower LR (5e-5) with proper warmup prevents oscillation
    - Large batch size for better negative sampling and GPU utilization
    - Label smoothing in loss helps generalization
    - Learnable temperature adapts to data

    ATTENTION QUALITY MONITORING:
    - Monitors entropy, edge ratio, and spatial coherence
    - Auto-stops training if attention degrades to edge-detection mode
    - Anti-collapse regularization prevents feature collapse
    
    EARLY STOPPING:
    - Loss typically converges to ~0.3-0.5 within 1-2 epochs
    - Training auto-stops when loss plateaus or reaches threshold
    - Max 5 epochs is safety cap (rarely reached with early stopping)
    """
    # Override defaults for Stage 1
    alignment_mode: str = 'fiber'  # Use FIBER fusion for alignment
    use_memory: bool = False  # Disable memory in Stage 1
    num_epochs: int = 5  # Max epochs (early stopping usually triggers earlier)
    # Lower LR for stable contrastive learning (CLIP uses 5e-4 for full model)
    learning_rate: float = 5e-5
    warmup_steps: int = 2000  # ~10% of epoch, critical for contrastive stability
    batch_size: int = 512  # Optimized for 2x A100 80GB (320 per GPU)
    num_workers: int = 32  # Match GPU throughput
    gradient_clip: float = 1.0  # Standard clipping
    weight_decay: float = 0.1  # Higher weight decay for regularization

    # Loss weights for Stage 1 - ALIGNMENT ONLY
    lm_loss_weight: float = 0.0  # DISABLED - frozen LM can't improve
    # Full weight - this is the ONLY trainable objective
    alignment_loss_weight: float = 1.0
    fine_grained_loss_weight: float = 0.5  # Text-to-patch attention supervision
    skip_lm_loss: bool = True  # Skip LM forward pass (saves compute)

    # Contrastive learning specific
    # Initial temperature (will be learned)
    alignment_temperature: float = 0.07

    # Adapter-focused training
    freeze_vision: bool = True
    freeze_language: bool = True
    train_adapter_only: bool = True
    unfreeze_last_n_layers: int = 0

    # Quantization - 4-bit for language model to reduce size < 1GB
    enable_quantization: bool = True
    quantize_vision_4bit: bool = False
    quantize_language_4bit: bool = True  # Reduces Qwen from ~2GB to ~250MB

    # Attention Quality Monitoring (prevents attention degradation to edge-detection)
    use_attention_monitor: bool = True
    # Stop if quality drops below this (0-1 scale)
    attention_quality_threshold: float = 0.25
    # Stop if degradation rate exceeds this
    attention_degradation_threshold: float = 0.15
    attention_min_steps: int = 2000  # Minimum steps before early stopping is allowed

    # Early Stopping - stop when loss plateaus
    use_early_stopping: bool = True
    early_stop_patience: int = 2  # Stop after 2 epochs without improvement
    early_stop_min_delta: float = 0.01  # Loss must improve by at least this

    # Alignment safeguards
    alignment_early_stop: bool = True
    alignment_patience: int = 300  # ~300 steps ≈ 1/3 epoch at batch 512
    alignment_improve_min_delta: float = 0.01  # require noticeable jump to count
    alignment_stop_threshold: float = 0.0
    alignment_negative_patience: int = 75
    alignment_save_cooldown: int = 100  # min steps between best-checkpoint saves
    alignment_min_stop_steps: int = 1000  # allow training until step 1000 before stopping

    # Visualization settings
    log_interval: int = 25  # More frequent logging to track loss
    visualize_interval: int = 300  # Save attention visualizations every 300 steps
    viz_save_interval: int = 5000
    num_viz_images: int = 3  # Number of images to visualize

    wandb_run_name: str = "stage1_alignment"


@dataclass
class Stage2Config(TrainingConfig):
    """
    Stage 2: Memory Integration Training
    Introduce episodic memory after alignment is learned
    Based on Larimar methodology
    """
    # Override defaults for Stage 2
    alignment_mode: str = 'fiber'  # Must match Stage 1 checkpoint
    use_memory: bool = True  # Enable memory in Stage 2
    episode_size: int = 4  # Longer episodes for memory learning
    num_epochs: int = 10  # More epochs for memory learning
    learning_rate: float = 1e-4  # Lower LR with memory
    warmup_steps: int = 1000
    batch_size: int = 64
    num_workers: int = 12

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

    # Stage 2 output directory and HF repo
    output_dir: str = "./checkpoints/stage2"
    hf_repo_name: str = "MicroVLM-V-stage2"

    # Early stopping for Stage 2 (loss-based)
    use_early_stopping: bool = True
    early_stop_patience: int = 3  # More patience for memory learning
    early_stop_min_delta: float = 0.005  # Smaller delta for finer convergence

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
        raise ValueError(
            f"Unknown config: {config_name}. Choose from {list(configs.keys())}")

    return configs[config_name]()
