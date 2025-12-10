"""
Staged Training Configuration for MicroVLM-V

Supports both baseline and FIBER-style training with configurable stages.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
from enum import Enum


class AlignmentMode(Enum):
    """Alignment training mode"""
    BASELINE = "baseline"
    FIBER = "fiber"


class TrainingStage(Enum):
    """Training stage definitions"""
    STAGE1_ALIGNMENT = "stage1_alignment"
    STAGE2_FINETUNING = "stage2_finetuning"
    STAGE3_INSTRUCT = "stage3_instruct"


@dataclass
class BaseModelConfig:
    """Base model architecture configuration"""
    # Vision encoder
    vision_model: str = "facebook/deit-tiny-patch16-224"
    vision_hidden_size: int = 192
    vision_num_patches: int = 196  # 14x14 patches
    vision_num_layers: int = 12
    vision_num_heads: int = 3
    image_size: int = 224
    patch_size: int = 16
    
    # Language model
    language_model: str = "Qwen/Qwen2.5-0.5B"
    language_hidden_size: int = 896
    language_num_layers: int = 24
    language_num_heads: int = 14
    language_vocab_size: int = 151936
    
    # Adapter
    num_prefix_tokens: int = 25
    adapter_hidden_size: int = 512


@dataclass
class FIBERConfig:
    """FIBER-style fusion-in-backbone configuration"""
    # Enable FIBER mode
    enabled: bool = True
    
    # Fusion layers in vision encoder (DeiT has 12 layers, 0-indexed)
    # Default: fuse in later layers [6, 8, 10] for coarse-to-fine
    fiber_fusion_layers: List[int] = field(default_factory=lambda: [6, 8, 10])
    
    # Cross-modal attention configuration
    cross_attention_heads: int = 4
    cross_attention_dim: int = 192  # Match vision hidden size
    cross_attention_dropout: float = 0.1
    
    # Bidirectional fusion (both I2T and T2I attention)
    use_bidirectional: bool = True
    
    # Alpha gating initialization (0.0 = start from pure vision, 1.0 = full fusion)
    alpha_init: float = 0.0
    
    # ITC (Image-Text Contrastive) loss
    itc_enabled: bool = True
    itc_weight: float = 1.0
    itc_temperature_init: float = 0.07
    itc_queue_size: int = 0  # 0 = no queue (in-batch negatives only)
    itc_label_smoothing: float = 0.1
    
    # ITM (Image-Text Matching) loss
    itm_enabled: bool = True
    itm_weight: float = 0.5
    itm_hard_negative_ratio: float = 0.5  # Ratio of hard negatives
    
    # Projection dimensions
    embed_dim: int = 256  # Shared embedding space dimension
    
    # Gradient settings
    freeze_vision_until_layer: int = 0  # Freeze first N vision layers
    fusion_only_grad: bool = False  # Only train fusion layers, freeze backbones


@dataclass
class Stage1Config:
    """Stage 1: Alignment pre-training configuration"""
    # Training settings
    learning_rate: float = 1e-4
    warmup_ratio: float = 0.1
    warmup_steps: int = 0  # Use ratio if 0
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    
    # Batch settings - optimized for 2x A100 80GB GPUs
    # With ~34GB used at batch_size=32, we can safely increase to 96
    # This gives ~60-70GB usage per GPU (leaving headroom for peaks)
    batch_size: int = 96
    gradient_accumulation_steps: int = 1  # Reduced since batch is larger
    
    # Epochs
    num_epochs: int = 15
    
    # Loss weights
    lm_loss_weight: float = 0.0  # Disabled in stage 1
    contrastive_loss_weight: float = 1.0
    alignment_loss_weight: float = 0.5
    fine_grained_loss_weight: float = 0.3
    
    # FIBER-specific (only used if fiber_config.enabled)
    itc_loss_weight: float = 1.0
    itm_loss_weight: float = 0.5
    
    # Frozen components
    freeze_vision_encoder: bool = True  # Freeze DeiT backbone
    freeze_language_model: bool = True  # Freeze Qwen
    freeze_vision_in_fiber: bool = False  # Unfreeze for FIBER fusion
    
    # Scheduler
    lr_scheduler: str = "cosine"  # cosine, linear, constant
    
    # Logging
    log_interval: int = 50
    save_interval: int = 1  # Save every N epochs
    eval_interval: int = 500  # Eval every N steps


@dataclass
class Stage2Config:
    """Stage 2: Language model fine-tuning configuration"""
    learning_rate: float = 2e-5
    warmup_ratio: float = 0.05
    warmup_steps: int = 0
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    
    batch_size: int = 16
    gradient_accumulation_steps: int = 4
    
    num_epochs: int = 5
    
    # Loss weights (enable LM loss)
    lm_loss_weight: float = 1.0
    contrastive_loss_weight: float = 0.1
    alignment_loss_weight: float = 0.1
    
    # Sliding Window Early Stopping
    use_sliding_window_early_stop: bool = True
    sliding_window_size: int = 1000  # Number of steps in window (larger = more stable, ignores short-term noise)
    sliding_window_min_delta: float = 0.01  # Minimum improvement threshold (0.01 = requires real progress beyond noise)
    sliding_window_patience_steps: int = 2000  # Steps without improvement before stopping (reduced for faster stopping)

    # HuggingFace Auto-Push on Early Stop
    push_to_hf_on_stop: bool = True
    hf_stage2_repo_name: str = "MicroVLM-V-stage2-final"

    # Frozen components
    freeze_vision_encoder: bool = True
    freeze_language_model: bool = False  # Unfreeze for fine-tuning
    freeze_adapter: bool = False
    
    lr_scheduler: str = "cosine"
    
    log_interval: int = 50
    save_interval: int = 1
    eval_interval: int = 500


@dataclass
class Stage3Config:
    """Stage 3: Instruction tuning configuration"""
    learning_rate: float = 1e-5
    warmup_ratio: float = 0.03
    warmup_steps: int = 0
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    
    batch_size: int = 8
    gradient_accumulation_steps: int = 8
    
    num_epochs: int = 3
    
    lm_loss_weight: float = 1.0
    contrastive_loss_weight: float = 0.0
    alignment_loss_weight: float = 0.0
    
    freeze_vision_encoder: bool = True
    freeze_language_model: bool = False
    freeze_adapter: bool = False
    
    # LoRA settings (optional)
    use_lora: bool = False
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_target_modules: List[str] = field(
        default_factory=lambda: ["q_proj", "v_proj", "k_proj", "o_proj"]
    )
    
    lr_scheduler: str = "cosine"
    
    log_interval: int = 50
    save_interval: int = 1
    eval_interval: int = 500


@dataclass
class DataConfig:
    """Dataset configuration"""
    # Dataset paths
    train_data_path: str = ""
    val_data_path: str = ""
    image_root: str = ""
    
    # Data loading
    num_workers: int = 4
    pin_memory: bool = True
    prefetch_factor: int = 2
    
    # Preprocessing
    max_text_length: int = 128
    image_size: int = 224
    
    # Augmentation
    use_augmentation: bool = True
    random_crop: bool = True
    random_flip: bool = True
    color_jitter: float = 0.4
    
    # Caption settings
    caption_field: str = "caption"
    image_field: str = "image"


@dataclass
class VisualizationConfig:
    """Attention visualization configuration"""
    enabled: bool = True
    save_interval: int = 500  # Save attention maps every N steps
    num_samples: int = 4
    output_dir: str = "attention_vis"
    
    # Which attention to visualize
    show_cross_modal: bool = True
    show_patch_text: bool = True
    show_cls_attention: bool = True
    
    # FIBER-specific
    show_fiber_attention: bool = True
    show_alpha_values: bool = True


@dataclass
class CheckpointConfig:
    """Checkpoint saving/loading configuration"""
    save_dir: str = "checkpoints"
    save_total_limit: int = 3
    save_on_each_node: bool = False
    
    # Resume
    resume_from: Optional[str] = None
    resume_optimizer: bool = True
    resume_scheduler: bool = True
    
    # Best model tracking
    save_best: bool = True
    best_metric: str = "val_loss"
    best_mode: str = "min"


@dataclass
class StagedTrainingConfig:
    """Complete staged training configuration"""
    # Model
    model: BaseModelConfig = field(default_factory=BaseModelConfig)
    
    # FIBER configuration
    fiber: FIBERConfig = field(default_factory=FIBERConfig)
    
    # Stage configurations
    stage1: Stage1Config = field(default_factory=Stage1Config)
    stage2: Stage2Config = field(default_factory=Stage2Config)
    stage3: Stage3Config = field(default_factory=Stage3Config)
    
    # Data
    data: DataConfig = field(default_factory=DataConfig)
    
    # Visualization
    visualization: VisualizationConfig = field(default_factory=VisualizationConfig)
    
    # Checkpoints
    checkpoint: CheckpointConfig = field(default_factory=CheckpointConfig)
    
    # Training mode
    alignment_mode: AlignmentMode = AlignmentMode.BASELINE
    current_stage: TrainingStage = TrainingStage.STAGE1_ALIGNMENT
    
    # Quantization settings - ENABLE 4-bit for compact model (<1GB target)
    quantize_language_4bit: bool = True   # 4-bit Qwen reduces ~2GB to ~250MB
    quantize_vision_4bit: bool = False    # DeiT stays FP16 (only 23MB)
    quantize_memory_158bit: bool = False  # Optional 1.58-bit memory
    
    # Carbon/Compute Tracking
    track_carbon: bool = True    # Track CO2 emissions via CodeCarbon
    track_flops: bool = True     # Track FLOPs computation
    track_gpu: bool = True       # Track GPU utilization, memory, power
    country_iso_code: str = "USA"  # ISO country code for carbon intensity
    
    # Distributed training
    use_ddp: bool = True
    ddp_backend: str = "nccl"
    find_unused_parameters: bool = True
    
    # Mixed precision
    use_amp: bool = True
    amp_dtype: str = "bfloat16"  # bfloat16 or float16
    
    # Reproducibility
    seed: int = 42
    deterministic: bool = False
    
    # Logging
    use_wandb: bool = True
    wandb_project: str = "microvlm-v"
    wandb_run_name: Optional[str] = None
    
    def get_stage_config(self) -> Any:
        """Get current stage configuration"""
        if self.current_stage == TrainingStage.STAGE1_ALIGNMENT:
            return self.stage1
        elif self.current_stage == TrainingStage.STAGE2_FINETUNING:
            return self.stage2
        else:
            return self.stage3
    
    def use_fiber(self) -> bool:
        """Check if FIBER mode is enabled"""
        return self.alignment_mode == AlignmentMode.FIBER and self.fiber.enabled


def create_baseline_config() -> StagedTrainingConfig:
    """Create configuration for baseline training"""
    config = StagedTrainingConfig()
    config.alignment_mode = AlignmentMode.BASELINE
    config.fiber.enabled = False
    return config


def create_fiber_config(
    fusion_layers: List[int] = [6, 8, 10],
    itc_weight: float = 1.0,
    itm_weight: float = 0.5,
    bidirectional: bool = True
) -> StagedTrainingConfig:
    """Create configuration for FIBER training"""
    config = StagedTrainingConfig()
    config.alignment_mode = AlignmentMode.FIBER
    config.fiber.enabled = True
    config.fiber.fiber_fusion_layers = fusion_layers
    config.fiber.itc_weight = itc_weight
    config.fiber.itm_weight = itm_weight
    config.fiber.use_bidirectional = bidirectional
    
    # Adjust stage 1 for FIBER
    config.stage1.freeze_vision_encoder = False  # Need gradients for fusion
    config.stage1.freeze_vision_in_fiber = False
    config.stage1.itc_loss_weight = itc_weight
    config.stage1.itm_loss_weight = itm_weight
    
    return config


def config_to_dict(config: StagedTrainingConfig) -> Dict[str, Any]:
    """Convert config to dictionary for serialization"""
    from dataclasses import asdict
    d = asdict(config)
    # Convert enums to strings
    d['alignment_mode'] = config.alignment_mode.value
    d['current_stage'] = config.current_stage.value
    return d


def load_config_from_dict(d: Dict[str, Any]) -> StagedTrainingConfig:
    """Load config from dictionary"""
    config = StagedTrainingConfig()
    
    # Handle enums
    if 'alignment_mode' in d:
        config.alignment_mode = AlignmentMode(d['alignment_mode'])
    if 'current_stage' in d:
        config.current_stage = TrainingStage(d['current_stage'])
    
    # Load nested configs
    for key, value in d.items():
        if hasattr(config, key) and isinstance(value, dict):
            nested_config = getattr(config, key)
            for k, v in value.items():
                if hasattr(nested_config, k):
                    setattr(nested_config, k, v)
        elif hasattr(config, key) and key not in ['alignment_mode', 'current_stage']:
            setattr(config, key, value)
    
    return config
