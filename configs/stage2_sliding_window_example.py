"""
Example Stage 2 Configuration with Sliding Window Early Stopping

This configuration demonstrates how to use the sliding window early stopping mechanism
to automatically stop training when the model plateaus and push the final model to HuggingFace.
"""

from staged_config import Stage2Config
from dataclasses import dataclass

@dataclass
class Stage2SlidingWindowConfig(Stage2Config):
    """
    Stage 2 configuration with sliding window early stopping enabled.

    This configuration will:
    1. Monitor training loss in a sliding window of recent steps
    2. Stop training automatically when no meaningful improvement is detected
    3. Push the final model to a separate HuggingFace repository
    """

    # ===== Sliding Window Early Stopping Parameters =====

    # Enable sliding window early stopping
    use_sliding_window_early_stop: bool = True

    # Window size: number of recent training steps to consider
    # Larger window = more stable but slower to detect plateau
    # Recommended: 200-400 for Stage 2
    sliding_window_size: int = 300

    # Minimum delta: minimum loss improvement to count as meaningful
    # If (best_window_loss - current_window_loss) < min_delta, no improvement
    # Recommended: 0.001-0.003 for Stage 2
    sliding_window_min_delta: float = 0.002

    # Patience: number of steps without improvement before stopping
    # Training stops if no improvement for this many steps
    # Recommended: 2000-4000 for Stage 2 (prevents premature stopping)
    sliding_window_patience_steps: int = 3000

    # ===== HuggingFace Auto-Push Parameters =====

    # Automatically push final model to HuggingFace when early stopping triggers
    push_to_hf_on_stop: bool = True

    # Repository name for Stage 2 final model (separate from main repo)
    # Will be created as: <hf_username>/<hf_stage2_repo_name>
    hf_stage2_repo_name: str = "MicroVLM-V-stage2-final"

    # Your HuggingFace username (must set in main config or here)
    hf_username: str = "euhidaman"

    # ===== Other Stage 2 Parameters =====

    # Learning rate for Stage 2 fine-tuning
    learning_rate: float = 2e-5

    # Batch size (adjust based on GPU memory)
    batch_size: int = 16
    gradient_accumulation_steps: int = 4

    # Maximum epochs (early stopping may trigger before reaching this)
    num_epochs: int = 10  # Set high, early stopping will handle it

    # Loss weights
    lm_loss_weight: float = 1.0
    contrastive_loss_weight: float = 0.1
    alignment_loss_weight: float = 0.1

    # Frozen components (Stage 2 unfreezes language model)
    freeze_vision_encoder: bool = True
    freeze_language_model: bool = False  # Unfreeze for fine-tuning
    freeze_adapter: bool = False

    # Memory module (enable for Stage 2)
    use_memory: bool = True

    # Quantization
    quantize_language_4bit: bool = True
    quantize_memory_158bit: bool = False

    # Logging
    log_interval: int = 50
    save_interval: int = 1

    # WandB logging
    use_wandb: bool = True
    wandb_project: str = "microvlm-v-stage2"


# ===== Usage Example =====
"""
To use this configuration:

1. Save this file as configs/stage2_sliding_window_example.py

2. Run training with:
   torchrun --nproc_per_node=2 scripts/train.py --config stage2 --use-staged-config

3. The training will:
   - Monitor loss in a 300-step sliding window
   - Stop when no improvement (≥0.002) for 3000 steps
   - Automatically save final checkpoint
   - Push final model to HuggingFace under <username>/MicroVLM-V-stage2-final

4. Monitor progress in WandB:
   - sliding_window/best_window_loss
   - sliding_window/current_window_loss
   - sliding_window/steps_without_improvement

5. The final HuggingFace repo will include:
   - model.pt (final checkpoint)
   - metrics.json (all early stopping metrics)
   - README.md (model card with stop reason and metrics)
"""


# ===== Recommended Configurations for Different Scenarios =====

@dataclass
class AggressiveEarlyStopConfig(Stage2Config):
    """
    Aggressive early stopping - stops quickly to save compute.
    Use when you want to stop as soon as plateau is detected.
    """
    use_sliding_window_early_stop: bool = True
    sliding_window_size: int = 200  # Smaller window = faster detection
    sliding_window_min_delta: float = 0.003  # Higher threshold
    sliding_window_patience_steps: int = 2000  # Less patience
    push_to_hf_on_stop: bool = True


@dataclass
class ConservativeEarlyStopConfig(Stage2Config):
    """
    Conservative early stopping - waits longer before stopping.
    Use when training is noisy or you want to be sure of plateau.
    """
    use_sliding_window_early_stop: bool = True
    sliding_window_size: int = 400  # Larger window = more stable
    sliding_window_min_delta: float = 0.001  # Lower threshold
    sliding_window_patience_steps: int = 4000  # More patience
    push_to_hf_on_stop: bool = True


@dataclass
class NoEarlyStopConfig(Stage2Config):
    """
    Disable early stopping - train for full num_epochs.
    Use for debugging or when you want full control.
    """
    use_sliding_window_early_stop: bool = False
    use_early_stopping: bool = False  # Disable epoch-based too
    num_epochs: int = 5

