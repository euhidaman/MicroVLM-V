"""
Main Training Script for MicroVLM-V
Supports small-scale testing, full training stages, and FIBER-style training
"""

from src.quantization.quantized_episodic_memory import get_memory_quantization_stats
from src.quantization.quantize_4bit import QuantizationConfig
from src.visualization.wandb_logger import WandBLogger
from src.visualization.attention_vis import create_attention_visualizer
from src.training.staged_config import load_config as load_staged_config
from src.training.config import load_config, create_run_name
from src.data.cc12m_loader import create_dataloaders
from src.models import create_microvlm, create_microvlm_fiber, MicroVLMFIBER
import os
import sys
import json
from datetime import datetime
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from pathlib import Path
import argparse
from tqdm import tqdm
import wandb
from huggingface_hub import HfApi, create_repo, upload_file

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def count_parameters(model):
    """Count total and trainable parameters"""
    # Handle DDP/DataParallel wrapped models
    base_model = model.module if hasattr(model, 'module') else model
    total_params = sum(p.numel() for p in base_model.parameters())
    trainable_params = sum(p.numel()
                           for p in base_model.parameters() if p.requires_grad)
    frozen_params = total_params - trainable_params
    return {
        'total': total_params,
        'trainable': trainable_params,
        'frozen': frozen_params
    }


def get_model_statistics(model, config):
    """Get comprehensive model statistics"""
    # Unwrap DDP model if necessary
    base_model = model.module if hasattr(model, 'module') else model
    
    stats = {}

    # Overall parameters (use base_model for accurate counting)
    overall = count_parameters(base_model)
    stats['total_parameters'] = overall['total']
    stats['trainable_parameters'] = overall['trainable']
    stats['frozen_parameters'] = overall['frozen']
    stats['trainable_percentage'] = 100.0 * overall['trainable'] / \
        overall['total'] if overall['total'] > 0 else 0.0

    # Vision encoder parameters
    if hasattr(base_model, 'vision_encoder'):
        vision_stats = count_parameters(base_model.vision_encoder)
        stats['vision_total_params'] = vision_stats['total']
        stats['vision_trainable_params'] = vision_stats['trainable']
        stats['vision_frozen_params'] = vision_stats['frozen']

    # Language model parameters
    if hasattr(base_model, 'language_model'):
        lang_stats = count_parameters(base_model.language_model)
        stats['language_total_params'] = lang_stats['total']
        stats['language_trainable_params'] = lang_stats['trainable']
        stats['language_frozen_params'] = lang_stats['frozen']

    # Adapter parameters
    if hasattr(base_model, 'multimodal_adapter'):
        adapter_stats = count_parameters(base_model.multimodal_adapter)
        stats['adapter_total_params'] = adapter_stats['total']
        stats['adapter_trainable_params'] = adapter_stats['trainable']

    # Memory parameters
    if hasattr(base_model, 'episodic_memory'):
        memory_stats = count_parameters(base_model.episodic_memory)
        stats['memory_total_params'] = memory_stats['total']
        stats['memory_trainable_params'] = memory_stats['trainable']

    # Quantization info
    stats['vision_4bit_quantized'] = getattr(
        config, 'quantize_vision_4bit', False)
    stats['language_4bit_quantized'] = getattr(
        config, 'quantize_language_4bit', False)
    stats['memory_158bit_quantized'] = getattr(
        config, 'quantize_memory_158bit', False)

    # Estimate memory size (approximate)
    # 4-bit: 0.5 bytes per param, 1.58-bit: ~0.2 bytes per param, fp16: 2 bytes, fp32: 4 bytes
    total_size_mb = 0
    if stats.get('vision_4bit_quantized'):
        total_size_mb += stats.get('vision_total_params', 0) * 0.5 / 1e6
    else:
        total_size_mb += stats.get('vision_total_params', 0) * 2 / 1e6  # fp16

    if stats.get('language_4bit_quantized'):
        total_size_mb += stats.get('language_total_params', 0) * 0.5 / 1e6
    else:
        total_size_mb += stats.get('language_total_params',
                                   0) * 2 / 1e6  # fp16

    if stats.get('memory_158bit_quantized'):
        total_size_mb += stats.get('memory_total_params', 0) * 0.2 / 1e6
    else:
        total_size_mb += stats.get('memory_total_params', 0) * 4 / 1e6  # fp32

    # Add other components (adapter, etc.)
    total_size_mb += stats.get('adapter_total_params', 0) * 4 / 1e6

    stats['estimated_model_size_mb'] = round(total_size_mb, 2)

    return stats


def print_model_statistics(stats, epoch):
    """Print model statistics in a formatted way"""
    print("\n" + "="*80)
    print(f"MODEL STATISTICS - Epoch {epoch}")
    print("="*80)

    print("\n📊 Overall Parameters:")
    print(f"  Total:      {stats['total_parameters']:>15,}")
    print(
        f"  Trainable:  {stats['trainable_parameters']:>15,} ({stats['trainable_percentage']:.2f}%)")
    print(f"  Frozen:     {stats['frozen_parameters']:>15,}")

    print("\n🖼️  Vision Encoder:")
    print(f"  Total:      {stats.get('vision_total_params', 0):>15,}")
    print(f"  Trainable:  {stats.get('vision_trainable_params', 0):>15,}")
    print(f"  Frozen:     {stats.get('vision_frozen_params', 0):>15,}")

    print("\n💬 Language Model:")
    print(f"  Total:      {stats.get('language_total_params', 0):>15,}")
    print(f"  Trainable:  {stats.get('language_trainable_params', 0):>15,}")
    print(f"  Frozen:     {stats.get('language_frozen_params', 0):>15,}")

    print("\n🔗 Multimodal Adapter:")
    print(f"  Total:      {stats.get('adapter_total_params', 0):>15,}")
    print(f"  Trainable:  {stats.get('adapter_trainable_params', 0):>15,}")

    print("\n🧠 Episodic Memory:")
    print(f"  Total:      {stats.get('memory_total_params', 0):>15,}")
    print(f"  Trainable:  {stats.get('memory_trainable_params', 0):>15,}")

    print("\n⚙️  Quantization:")
    print(
        f"  Vision 4-bit:     {'✓' if stats.get('vision_4bit_quantized') else '✗'}")
    print(
        f"  Language 4-bit:   {'✓' if stats.get('language_4bit_quantized') else '✗'}")
    print(
        f"  Memory 1.58-bit:  {'✓' if stats.get('memory_158bit_quantized') else '✗'}")

    print("\n💾 Estimated Model Size:")
    print(f"  ~{stats['estimated_model_size_mb']:.2f} MB")

    print("="*80 + "\n")


def save_epoch_checkpoint(model, optimizer, epoch, global_step, config, stage_name="default"):
    """Save model checkpoint after each epoch"""
    checkpoint_dir = Path(config.output_dir) / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # Get and print model statistics
    stats = get_model_statistics(model, config)
    print_model_statistics(stats, epoch)

    # Save checkpoint with epoch number
    checkpoint_path = checkpoint_dir / f"epoch_{epoch}_checkpoint.pt"

    torch.save({
        'epoch': epoch,
        'global_step': global_step,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'config': vars(config),
        'model_statistics': stats
    }, checkpoint_path)

    print(f"✅ Checkpoint saved: {checkpoint_path}")

    # Save statistics as JSON
    stats_path = checkpoint_dir / f"epoch_{epoch}_statistics.json"
    with open(stats_path, 'w') as f:
        json.dump(stats, f, indent=2)
    print(f"📊 Statistics saved: {stats_path}")

    return checkpoint_path, stats


def generate_model_card(stats, epoch, total_epochs, stage_name, config, training_history=None):
    """Generate comprehensive model card with technical details and training progress"""

    # Helper to format large numbers
    def format_params(n):
        if n >= 1e9:
            return f"{n/1e9:.2f}B"
        elif n >= 1e6:
            return f"{n/1e6:.2f}M"
        elif n >= 1e3:
            return f"{n/1e3:.2f}K"
        return str(n)

    # Calculate percentages
    total = stats.get('total_parameters', 0)
    vision_pct = 100.0 * \
        stats.get('vision_total_params', 0) / total if total > 0 else 0
    lang_pct = 100.0 * \
        stats.get('language_total_params', 0) / total if total > 0 else 0
    adapter_pct = 100.0 * \
        stats.get('adapter_total_params', 0) / total if total > 0 else 0
    memory_pct = 100.0 * \
        stats.get('memory_total_params', 0) / total if total > 0 else 0

    card = f"""---
license: apache-2.0
tags:
- vision-language
- multimodal
- episodic-memory
- 1.58-bit
- pytorch
library_name: transformers
---

# MicroVLM-V: Vision-Language Model with Episodic Memory

## 🔄 Training Progress: {stage_name}

**Current Status:** Epoch {epoch}/{total_epochs}

> ⚠️ **Note:** This repository contains ONLY the latest checkpoint. Each epoch overwrites previous weights.

---

## 📊 Model Architecture

### Parameter Distribution

| Component | Total Parameters | Trainable | Percentage |
|-----------|-----------------|-----------|------------|
| **Total Model** | **{format_params(total)}** | **{format_params(stats.get('trainable_parameters', 0))}** | **{stats.get('trainable_percentage', 0):.1f}%** |
| Vision Encoder | {format_params(stats.get('vision_total_params', 0))} | {format_params(stats.get('vision_trainable_params', 0))} | {vision_pct:.1f}% |
| Language Model | {format_params(stats.get('language_total_params', 0))} | {format_params(stats.get('language_trainable_params', 0))} | {lang_pct:.1f}% |
| Multimodal Adapter | {format_params(stats.get('adapter_total_params', 0))} | {format_params(stats.get('adapter_trainable_params', 0))} | {adapter_pct:.1f}% |
| Episodic Memory | {format_params(stats.get('memory_total_params', 0))} | {format_params(stats.get('memory_trainable_params', 0))} | {memory_pct:.1f}% |

### Technical Specifications

- **Vision Encoder:** DeiT-Tiny (192-dim embeddings)
  - Quantization: {'4-bit' if stats.get('vision_4bit_quantized') else 'FP16'}
  - Status: {'Frozen' if stats.get('vision_frozen_params', 0) > 0 else 'Trainable'}
  
- **Language Model:** Qwen2.5-0.5B (896-dim embeddings)
  - Quantization: {'4-bit' if stats.get('language_4bit_quantized') else 'FP16'}
  - Trainable Layers: {stats.get('language_trainable_params', 0) // 1000}K params
  
- **Multimodal Adapter:**
  - Architecture: Linear projection + Layer Norm
  - Mapping: 192-dim (vision) → 896-dim (language)
  - Parameters: {format_params(stats.get('adapter_total_params', 0))}
  
- **Episodic Memory:**
  - Type: BitLinear 1.58-bit quantized
  - Quantization: {'Enabled' if stats.get('memory_158bit_quantized') else 'Disabled'}
  - Parameters: {format_params(stats.get('memory_total_params', 0))}

### Model Size

- **Estimated Size:** {stats.get('estimated_model_size_mb', 0):.2f} MB
- **Memory Footprint:** ~{stats.get('estimated_model_size_mb', 0) * 1.5:.0f} MB (with activation)

---

## 🎯 Training Methodology

### {stage_name} Configuration

"""

    # Add stage-specific details
    if 'Stage 1' in stage_name or 'stage1' in stage_name.lower():
        card += f"""**Focus:** Contrastive alignment learning (vision ↔ language)

**Training Strategy:**
- Vision encoder: **Frozen** (pretrained DeiT weights preserved)
- Language model: **Frozen** (pretrained Qwen weights preserved)  
- Multimodal adapter: **Trainable** (learning alignment mapping)
- Episodic memory: **Disabled** (not used in Stage 1)

**Loss Function:** Contrastive alignment loss only
- Aligns vision and language embeddings in shared space
- InfoNCE-style loss for image-text matching

"""
    elif 'Stage 2' in stage_name or 'stage2' in stage_name.lower():
        card += f"""**Focus:** Episodic memory integration

**Training Strategy:**
- Vision encoder: **Frozen**
- Language model: **Partially unfrozen** (last 2 layers)
- Multimodal adapter: **Trainable** (initialized from Stage 1)
- Episodic memory: **Enabled** (1.58-bit quantization)

**Loss Function:** Alignment + Memory losses
- Continues alignment refinement
- Adds memory read/write/retrieval objectives

"""
    elif 'Stage 3' in stage_name or 'stage3' in stage_name.lower():
        card += f"""**Focus:** Full fine-tuning with all components

**Training Strategy:**
- Vision encoder: **Frozen** (preserves pretrained features)
- Language model: **Fully unfrozen** (all layers trainable)
- Multimodal adapter: **Trainable**
- Episodic memory: **Enabled** (1.58-bit quantization)

**Loss Function:** Language modeling + Alignment + Memory
- Full next-token prediction
- Maintains vision-language alignment
- Memory-enhanced generation

"""

    card += f"""**Hyperparameters:**
- Learning Rate: {getattr(config, 'learning_rate', 'N/A')}
- Batch Size: {getattr(config, 'batch_size', 'N/A')}
- Warmup Steps: {getattr(config, 'warmup_steps', 'N/A')}
- Gradient Clipping: {getattr(config, 'gradient_clip', 'N/A')}
- Optimizer: {getattr(config, 'optimizer', 'adamw').upper()}
- Scheduler: {getattr(config, 'scheduler', getattr(config, 'lr_scheduler', 'cosine'))}

**Hardware:**
- GPU: NVIDIA RTX 6000 Ada (48GB)
- Precision: Mixed FP16/FP32
- Distributed: Single GPU

---

## 📈 Training Statistics (Epoch {epoch})

"""

    # Add training history if available
    if training_history and len(training_history) > 0:
        latest = training_history[-1]
        card += f"""**Latest Metrics:**
- Training Loss: {latest.get('train_loss', 'N/A'):.4f}
- Alignment Loss: {latest.get('alignment_loss', 'N/A'):.4f}
- Learning Rate: {latest.get('learning_rate', 'N/A'):.2e}
- Gradient Norm: {latest.get('gradient_norm', 'N/A'):.4f}

"""

    card += f"""**Timestamp:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')}

---

## 💻 Usage

### Loading the Model

```python
import torch
from pathlib import Path

# Download model weights
checkpoint = torch.load('model.pt', map_location='cpu')

# Load model state
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Move to GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
```

### Inference Example

```python
from PIL import Image
import torchvision.transforms as transforms

# Prepare image
image = Image.open('example.jpg').convert('RGB')
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                       std=[0.229, 0.224, 0.225])
])
image_tensor = transform(image).unsqueeze(0).to(device)

# Prepare text
text = "A photo of a cat"
tokens = tokenizer(text, return_tensors='pt', padding=True).to(device)

# Forward pass
with torch.no_grad():
    outputs = model(
        images=image_tensor,
        input_ids=tokens['input_ids'],
        attention_mask=tokens['attention_mask']
    )
```

### Model Input/Output Format

**Inputs:**
- `images`: Tensor [B, 3, 224, 224] - RGB images normalized
- `input_ids`: Tensor [B, seq_len] - Tokenized text
- `attention_mask`: Tensor [B, seq_len] - Attention mask

**Outputs:**
- `lm_loss`: Language modeling loss (if labels provided)
- `alignment_loss`: Vision-language alignment loss
- `memory_loss`: Episodic memory loss (Stage 2/3 only)
- `logits`: Next token predictions [B, seq_len, vocab_size]

---

## ⚙️ Requirements

```bash
pip install torch>=2.0.0
pip install transformers>=4.30.0
pip install timm  # For DeiT vision encoder
pip install Pillow  # For image processing
```

---

## 📜 License

Apache 2.0 License

---

## 🔗 Links

- **GitHub Repository:** [euhidaman/MicroVLM-V](https://github.com/euhidaman/MicroVLM-V)
- **Paper:** Coming soon
- **Demo:** Coming soon

---

## ⚠️ Limitations

- **Training in Progress:** This model is still under active training
- **Checkpoint Volatility:** Only latest epoch is preserved - download if needed
- **Stage-Specific:** Capabilities depend on training stage
  - Stage 1: Alignment only, no generation
  - Stage 2: Basic generation with memory
  - Stage 3: Full capabilities

---

## 📧 Contact

For questions or issues, please open an issue on GitHub.

---

*Last updated: Epoch {epoch}/{total_epochs} - {datetime.now().strftime('%Y-%m-%d')}*
"""

    return card


def push_to_huggingface(checkpoint_path, stats, epoch, total_epochs, stage_name, config, training_history=None):
    """Push ONLY latest model weights and statistics to HuggingFace Hub (replaces previous)"""
    try:
        # Get HF token from environment or huggingface_hub cache
        hf_token = os.environ.get(
            'HF_TOKEN') or os.environ.get('HUGGINGFACE_TOKEN')

        # Try to get token from huggingface_hub if not in env
        if not hf_token:
            try:
                from huggingface_hub import HfFolder
                hf_token = HfFolder.get_token()
            except:
                pass

        if not hf_token:
            print("⚠️  Warning: No HuggingFace token found.")
            print("   Run: huggingface-cli login")
            print("   Or set HF_TOKEN environment variable")
            print("   Get token from: https://huggingface.co/settings/tokens")
            return False

        # Get repo config
        repo_name = getattr(config, 'hf_repo_name', 'MicroVLM-V')
        username = getattr(config, 'hf_username', 'euhidaman')
        repo_id = f"{username}/{repo_name}"

        print(f"\n🤗 Uploading latest checkpoint to HuggingFace: {repo_id}")
        print(f"   📝 This will REPLACE previous epoch files")

        # Create repo if it doesn't exist
        api = HfApi()
        try:
            api.create_repo(repo_id=repo_id, token=hf_token,
                            exist_ok=True, repo_type="model")
            print(f"   ✓ Repository ready: https://huggingface.co/{repo_id}")
        except Exception as e:
            print(f"   ℹ Repository check: {e}")

        # Generate comprehensive model card
        model_card = generate_model_card(
            stats, epoch, total_epochs, stage_name, config, training_history)

        # Save model card locally
        readme_path = Path(checkpoint_path).parent / "README.md"
        with open(readme_path, 'w', encoding='utf-8') as f:
            f.write(model_card)
        print(f"   ✓ Generated comprehensive model card")

        # Save statistics to JSON with consistent filename
        stats_json_path = Path(checkpoint_path).parent / "statistics.json"
        with open(stats_json_path, 'w') as f:
            json.dump({
                **stats,
                'epoch': epoch,
                'total_epochs': total_epochs,
                'stage': stage_name,
                'timestamp': datetime.now().isoformat(),
                # Last 10 epochs
                'training_history': training_history[-10:] if training_history else []
            }, f, indent=2)
        print(f"   ✓ Prepared statistics.json")

        # Create temporary model.pt (copy of checkpoint with consistent name)
        model_pt_path = Path(checkpoint_path).parent / "model.pt"
        import shutil
        shutil.copy(checkpoint_path, model_pt_path)

        # Upload files sequentially (will replace previous versions)
        commit_message = f"{stage_name}: Epoch {epoch}/{total_epochs} - Latest checkpoint"

        print(f"   ⏳ Uploading model.pt...")
        api.upload_file(
            path_or_fileobj=str(model_pt_path),
            path_in_repo="model.pt",
            repo_id=repo_id,
            repo_type="model",
            commit_message=commit_message,
            token=hf_token
        )

        print(f"   ⏳ Uploading statistics.json...")
        api.upload_file(
            path_or_fileobj=str(stats_json_path),
            path_in_repo="statistics.json",
            repo_id=repo_id,
            repo_type="model",
            commit_message=commit_message,
            token=hf_token
        )

        print(f"   ⏳ Uploading README.md...")
        api.upload_file(
            path_or_fileobj=str(readme_path),
            path_in_repo="README.md",
            repo_id=repo_id,
            repo_type="model",
            commit_message=commit_message,
            token=hf_token
        )

        # Cleanup temporary files
        model_pt_path.unlink(missing_ok=True)

        print(f"   ✅ Successfully uploaded latest checkpoint!")
        print(f"   📝 Commit: \"{commit_message}\"")
        print(f"   🔗 View at: https://huggingface.co/{repo_id}")
        print(
            f"   ℹ️  Repository contains ONLY epoch {epoch} (previous epochs replaced)")
        return True

    except Exception as e:
        print(f"❌ Error pushing to HuggingFace: {e}")
        import traceback
        traceback.print_exc()
        print("   Continuing training...")
        return False


def setup_wandb(config, run_name):
    """Initialize WandB logging"""
    if config.use_wandb:
        try:
            # Initialize with project (will auto-create if doesn't exist)
            wandb.init(
                project=config.wandb_project,
                name=run_name,
                config=vars(config),
                resume='allow'  # Allow resuming runs
            )
            print(f"WandB initialized: {run_name}")
            print(f"WandB project: {config.wandb_project}")
            return wandb
        except Exception as e:
            print(f"WARNING: WandB initialization failed: {e}")
            print("Continuing without WandB logging...")
            print("To enable WandB:")
            print("  1. Run: wandb login")
            print("  2. Or disable in config: use_wandb=False")
            return None
    return None


def create_optimizer(model, config):
    """Create optimizer"""
    # Get trainable parameters
    trainable_params = [p for p in model.parameters() if p.requires_grad]

    if config.optimizer == "adamw":
        optimizer = torch.optim.AdamW(
            trainable_params,
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
            betas=(0.9, 0.999)
        )
    elif config.optimizer == "adam":
        optimizer = torch.optim.Adam(
            trainable_params,
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )
    else:
        raise ValueError(f"Unknown optimizer: {config.optimizer}")

    return optimizer


def create_scheduler(optimizer, config, total_steps):
    """Create learning rate scheduler with warmup for stable contrastive learning"""
    from torch.optim.lr_scheduler import LambdaLR
    
    warmup_steps = getattr(config, 'warmup_steps', 1000)
    scheduler_name = getattr(config, 'scheduler', getattr(
        config, 'lr_scheduler', 'cosine'))
    
    if scheduler_name == "cosine":
        # Cosine annealing with linear warmup (CLIP-style)
        def lr_lambda(current_step):
            if current_step < warmup_steps:
                # Linear warmup
                return float(current_step) / float(max(1, warmup_steps))
            # Cosine decay after warmup
            import math
            progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
            return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))
        
        scheduler = LambdaLR(optimizer, lr_lambda)
    elif scheduler_name == "linear":
        # Linear decay with warmup
        def lr_lambda(current_step):
            if current_step < warmup_steps:
                return float(current_step) / float(max(1, warmup_steps))
            progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
            return max(0.0, 1.0 - progress)
        
        scheduler = LambdaLR(optimizer, lr_lambda)
    else:
        scheduler = None

    return scheduler


def compute_gradient_flow_metrics(model):
    """
    Compute gradient flow metrics for monitoring training
    Based on EVO-1 parameter inspection methodology

    Returns:
        Dictionary with gradient statistics by module
    """
    metrics = {}

    # Key module groups to monitor
    module_groups = {
        'adapter': [],
        'memory': [],
        'projection': [],
        'scope_detector': [],
        'vision': [],
        'language': []
    }

    # Collect gradients by module group
    for name, param in model.named_parameters():
        if param.grad is None:
            continue

        grad_norm = param.grad.norm().item()
        grad_mean = param.grad.mean().item()
        grad_std = param.grad.std().item()
        grad_max = param.grad.abs().max().item()

        # Categorize parameter
        if 'multimodal_adapter' in name or 'adapter' in name:
            module_groups['adapter'].append({
                'name': name,
                'norm': grad_norm,
                'mean': grad_mean,
                'std': grad_std,
                'max': grad_max
            })
        elif 'episodic_memory' in name or 'memory' in name:
            module_groups['memory'].append({
                'name': name,
                'norm': grad_norm,
                'mean': grad_mean,
                'std': grad_std,
                'max': grad_max
            })
        elif 'scope_detector' in name:
            module_groups['scope_detector'].append({
                'name': name,
                'norm': grad_norm,
                'mean': grad_mean,
                'std': grad_std,
                'max': grad_max
            })
        elif 'proj' in name.lower() or 'projection' in name:
            module_groups['projection'].append({
                'name': name,
                'norm': grad_norm,
                'mean': grad_mean,
                'std': grad_std,
                'max': grad_max
            })
        elif 'vision' in name:
            module_groups['vision'].append({
                'name': name,
                'norm': grad_norm,
                'mean': grad_mean,
                'std': grad_std,
                'max': grad_max
            })
        elif 'language' in name or 'qwen' in name.lower():
            module_groups['language'].append({
                'name': name,
                'norm': grad_norm,
                'mean': grad_mean,
                'std': grad_std,
                'max': grad_max
            })

    # Aggregate statistics per group
    for group_name, grads in module_groups.items():
        if not grads:
            continue

        total_norm = sum(g['norm'] for g in grads)
        avg_mean = sum(g['mean'] for g in grads) / len(grads)
        avg_std = sum(g['std'] for g in grads) / len(grads)
        max_grad = max(g['max'] for g in grads)

        metrics[f'grad_flow/{group_name}/total_norm'] = total_norm
        metrics[f'grad_flow/{group_name}/avg_mean'] = avg_mean
        metrics[f'grad_flow/{group_name}/avg_std'] = avg_std
        metrics[f'grad_flow/{group_name}/max_grad'] = max_grad
        metrics[f'grad_flow/{group_name}/num_params'] = len(grads)

    return metrics


def train_epoch(model, train_loader, optimizer, scheduler, config, visualizer,
                epoch, global_step, wandb_run=None, wandb_logger=None,
                is_distributed=False, is_main_process=True):
    """Train for one epoch
    
    Args:
        model: The model (may be wrapped in DDP/DP)
        train_loader: Training dataloader
        optimizer: Optimizer
        scheduler: Learning rate scheduler
        config: Training configuration
        visualizer: Attention visualizer
        epoch: Current epoch number
        global_step: Current global step
        wandb_run: WandB run object
        wandb_logger: WandB logger object
        is_distributed: Whether using distributed training
        is_main_process: Whether this is the main process (rank 0)
    """
    model.train()

    total_loss = 0
    lm_loss_total = 0
    alignment_loss_total = 0
    memory_kl_total = 0
    itc_loss_total = 0
    itm_loss_total = 0

    # Check if using FIBER mode
    use_fiber = getattr(config, 'alignment_mode', 'baseline') == 'fiber'

    # Only show progress bar on main process
    if is_main_process:
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}")
    else:
        pbar = train_loader

    for batch_idx, batch in enumerate(pbar):
        # Move to device
        images = batch['image'].to(config.device)
        input_ids = batch['input_ids'].to(config.device)
        attention_mask = batch['attention_mask'].to(config.device)

        # Labels for language modeling (shifted in model)
        labels = input_ids.clone()

        # Forward pass - different path for FIBER vs baseline
        # Get base model (unwrap DDP/DP if needed)
        base_model = model.module if hasattr(model, 'module') else model
        
        if use_fiber and isinstance(base_model, MicroVLMFIBER):
            # FIBER forward pass with ITC + ITM losses
            outputs = model(
                images=images,
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
                use_memory=config.use_memory,
                use_alignment=True  # Always use alignment in FIBER mode
            )
        else:
            # Baseline forward pass
            outputs = model(
                images=images,
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
                use_memory=config.use_memory,
                use_alignment=config.use_alignment,
                episode_size=config.episode_size
            )

        loss = outputs['loss']

        # Debug: Check for NaN/Inf in loss components
        if torch.isnan(loss) or torch.isinf(loss):
            if is_main_process:
                print(f"\n⚠️ Invalid loss detected at step {global_step}:")
                print(f"  Total loss: {loss.item()}")
                if 'lm_loss' in outputs and outputs['lm_loss'] is not None:
                    print(f"  LM loss: {outputs['lm_loss'].item()}")
                if 'alignment_loss' in outputs:
                    print(f"  Alignment loss: {outputs['alignment_loss'].item()}")
                if 'itc_loss' in outputs:
                    print(f"  ITC loss: {outputs['itc_loss'].item()}")
                if 'itm_loss' in outputs:
                    print(f"  ITM loss: {outputs['itm_loss'].item()}")
                if 'memory_kl' in outputs:
                    print(f"  Memory KL: {outputs['memory_kl'].item()}")
                if 'addressing_kl' in outputs:
                    print(f"  Addressing KL: {outputs['addressing_kl'].item()}")

                # Check for NaN in model parameters
                for name, param in model.named_parameters():
                    if param.grad is not None and (torch.isnan(param.grad).any() or torch.isinf(param.grad).any()):
                        print(f"  NaN/Inf gradient in: {name}")

                print("  Skipping this batch...")
            continue

        # Backward pass
        optimizer.zero_grad()
        loss.backward()

        # Compute gradient norm before clipping
        total_norm = 0.0
        for p in model.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        total_norm = total_norm ** 0.5

        # Gradient clipping
        if config.gradient_clip > 0:
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), config.gradient_clip)

        # Log gradient norm periodically (main process only)
        if is_main_process and global_step % 100 == 0:
            print(f"\n  [Step {global_step}] Gradient norm: {total_norm:.4f}")

        optimizer.step()

        if scheduler is not None:
            scheduler.step()

        # Accumulate losses
        total_loss += loss.item()

        # Safely extract loss values
        lm_loss_val = outputs.get('lm_loss')
        lm_loss_total += lm_loss_val.item() if lm_loss_val is not None else 0

        alignment_loss_val = outputs.get('alignment_loss')
        alignment_loss_total += alignment_loss_val.item() if alignment_loss_val is not None else 0

        memory_kl_val = outputs.get('memory_kl')
        memory_kl_total += memory_kl_val.item() if memory_kl_val is not None else 0
        
        # FIBER-specific losses
        itc_loss_val = outputs.get('itc_loss')
        itc_loss_total += itc_loss_val.item() if itc_loss_val is not None else 0
        
        itm_loss_val = outputs.get('itm_loss')
        itm_loss_total += itm_loss_val.item() if itm_loss_val is not None else 0

        global_step += 1

        # Update progress bar (main process only)
        if is_main_process:
            lm_loss_display = outputs.get('lm_loss')
            if lm_loss_display is None:
                lm_display_val = 'none'
            elif torch.isnan(lm_loss_display) or torch.isinf(lm_loss_display):
                lm_display_val = 'nan'
            else:
                lm_display_val = f"{lm_loss_display.item():.3f}"

            # Show alignment loss if present (critical for Stage1)
            align_loss_display = outputs.get('alignment_loss')
            if align_loss_display is not None:
                align_display_val = f"{align_loss_display.item():.3f}"
            else:
                align_display_val = 'none'
            
            # Build postfix dict based on training mode
            if use_fiber:
                # FIBER mode: show ITC and ITM losses
                itc_val = outputs.get('itc_loss')
                itm_val = outputs.get('itm_loss')
                pbar.set_postfix({
                    'loss': f"{loss.item():.3f}",
                    'itc': f"{itc_val.item():.3f}" if itc_val is not None else 'none',
                    'itm': f"{itm_val.item():.3f}" if itm_val is not None else 'none',
                    'lr': f"{optimizer.param_groups[0]['lr']:.2e}"
                })
            else:
                # Baseline mode: show LM and alignment losses
                pbar.set_postfix({
                    'loss': f"{loss.item():.3f}",
                    'lm': lm_display_val,
                    'align': align_display_val,
                    'lr': f"{optimizer.param_groups[0]['lr']:.2e}"
                })

        # Logging (main process only)
        if is_main_process and global_step % config.log_interval == 0:
            # Use comprehensive WandB logger
            if wandb_logger:
                # Core training metrics
                wandb_logger.log_training_metrics(
                    outputs, optimizer, epoch, global_step)

                # Gradient metrics
                wandb_logger.log_gradient_metrics(model, global_step)

                # Gradient flow monitoring (every 50 steps)
                if global_step % 50 == 0:
                    grad_flow_metrics = compute_gradient_flow_metrics(model)
                    wandb_logger.log_metrics(
                        grad_flow_metrics, step=global_step)

                # Language model metrics
                wandb_logger.log_language_model_metrics(
                    outputs, input_ids, global_step)

            # Legacy simple logging (fallback)
            elif wandb_run:
                # Build metrics dict with safe access
                lm_loss_metric = outputs.get('lm_loss')
                metrics = {
                    'train/loss': loss.item(),
                    'train/lm_loss': lm_loss_metric.item() if lm_loss_metric is not None else 0.0,
                    'train/learning_rate': optimizer.param_groups[0]['lr'],
                    'train/epoch': epoch,
                    'train/global_step': global_step
                }

                if 'alignment_loss' in outputs:
                    metrics['train/alignment_loss'] = outputs['alignment_loss'].item()

                if 'memory_kl' in outputs:
                    metrics['train/memory_kl'] = outputs['memory_kl'].item()
                    metrics['train/addressing_kl'] = outputs['addressing_kl'].item()

                wandb_run.log(metrics, step=global_step)

        # Visualization (main process only)
        if is_main_process and global_step % config.visualize_interval == 0 and (visualizer is not None or wandb_logger):
            model.eval()
            # Get underlying model for DDP/DP wrapped models
            viz_model = model.module if hasattr(model, 'module') else model
            with torch.no_grad():
                # Extract features for visualization
                # Handle both baseline (2 returns) and FIBER (4 returns) models
                encode_result = viz_model.encode_image(images[:4])
                if len(encode_result) == 4:
                    # FIBER model returns: prefix_tokens, image_features, patch_embeddings_proj, fiber_attention
                    prefix_tokens, image_features, _, _ = encode_result
                else:
                    # Baseline model returns: prefix_tokens, image_features
                    prefix_tokens, image_features = encode_result
                text_embeddings, text_features = viz_model.encode_text(
                    input_ids[:4], attention_mask[:4]
                )

                # Vision encoder visualizations
                if wandb_logger:
                    wandb_logger.log_vision_encoder_metrics(
                        viz_model, images, global_step)

                # Alignment visualizations
                if wandb_logger and 'alignment_loss' in outputs:
                    # Get the alignment_loss module for temperature logging
                    alignment_loss_module = getattr(viz_model, 'alignment_loss', None)
                    wandb_logger.log_alignment_metrics(
                        image_features, text_features, global_step,
                        alignment_loss_module=alignment_loss_module
                    )

                # Cross-modal attention analysis + side-by-side visualization
                if visualizer:
                    num_viz_images = getattr(config, 'num_viz_images', 3)
                    # Cache first batch for consistent visualization
                    if not hasattr(visualizer, '_fixed_samples'):
                        cache_samples = min(num_viz_images, images.size(0))
                        if cache_samples > 0:
                            visualizer._fixed_samples = {
                                'images': images[:cache_samples].detach().cpu().clone(),
                                'input_ids': input_ids[:cache_samples].detach().cpu().clone(),
                                'attention_mask': attention_mask[:cache_samples].detach().cpu().clone() if attention_mask is not None else None
                            }
                    fixed_samples = getattr(visualizer, '_fixed_samples', None)
                    if fixed_samples:
                        num_samples = min(
                            num_viz_images, fixed_samples['images'].size(0))
                        device = images.device
                        text_device = input_ids.device
                        viz_images = fixed_samples['images'][:num_samples].to(
                            device)
                        viz_input_ids = fixed_samples['input_ids'][:num_samples].to(
                            text_device)
                        viz_attention_mask = None
                        if fixed_samples['attention_mask'] is not None:
                            viz_attention_mask = fixed_samples['attention_mask'][:num_samples].to(
                                text_device)

                        # Handle both baseline (2 returns) and FIBER (4 returns) models
                        viz_encode_result = viz_model.encode_image(viz_images)
                        if len(viz_encode_result) == 4:
                            viz_prefix_tokens, _, _, _ = viz_encode_result
                        else:
                            viz_prefix_tokens, _ = viz_encode_result
                        viz_text_embeddings, _ = viz_model.encode_text(
                            viz_input_ids, viz_attention_mask)
                        attention_from_lm = None
                        try:
                            # Temporarily set attention implementation to eager for visualization
                            original_attn_impl = None
                            if hasattr(viz_model.language_model, 'model') and hasattr(viz_model.language_model.model, 'config'):
                                original_attn_impl = getattr(
                                    viz_model.language_model.model.config, '_attn_implementation', None)
                                viz_model.language_model.model.config._attn_implementation = 'eager'

                            fused_embeddings, fused_mask = viz_model.fusion(
                                viz_prefix_tokens, viz_text_embeddings, viz_attention_mask
                            )
                            model_dtype = None
                            if hasattr(viz_model.language_model, 'model') and viz_model.language_model.model is not None:
                                model_dtype = next(
                                    viz_model.language_model.model.parameters()).dtype
                            elif hasattr(viz_model.language_model, 'embed_tokens') and viz_model.language_model.embed_tokens is not None:
                                model_dtype = viz_model.language_model.embed_tokens.weight.dtype
                            if model_dtype is not None and fused_embeddings.dtype != model_dtype:
                                fused_embeddings = fused_embeddings.to(
                                    model_dtype)
                            lm_viz_outputs = viz_model.language_model(
                                inputs_embeds=fused_embeddings,
                                attention_mask=fused_mask,
                                output_attentions=True,
                                output_hidden_states=False,
                                use_cache=False,
                                return_dict=True
                            )
                            attentions = getattr(
                                lm_viz_outputs, 'attentions', None)
                            if attentions:
                                attn = attentions[-1].detach()
                                prefix_len = viz_prefix_tokens.size(1)
                                attention_from_lm = attn[:,
                                                         :, prefix_len:, :prefix_len]

                            # Restore original attention implementation
                            if original_attn_impl is not None and hasattr(viz_model.language_model, 'model'):
                                viz_model.language_model.model.config._attn_implementation = original_attn_impl
                        except Exception as exc:
                            print(
                                f"Warning: unable to compute attention maps from LM: {exc}")
                            attention_from_lm = None
                        stats, attention = visualizer.analyze_cross_modal_attention(
                            viz_prefix_tokens, viz_text_embeddings, attention_weights=attention_from_lm
                        )

                        if not hasattr(visualizer, '_caption_tokenizer'):
                            from transformers import AutoTokenizer
                            tokenizer_name = getattr(config, 'text_model', None) or getattr(
                                config, 'qwen_model', None)
                            if tokenizer_name is None:
                                raise ValueError(
                                    "No text model or tokenizer specified in config")
                            visualizer._caption_tokenizer = AutoTokenizer.from_pretrained(
                                tokenizer_name)
                        tokenizer = visualizer._caption_tokenizer
                        captions = []
                        for i in range(num_samples):
                            if viz_attention_mask is not None:
                                valid_len = int(
                                    viz_attention_mask[i].sum().item())
                                token_ids = viz_input_ids[i][:valid_len].tolist(
                                )
                            else:
                                token_ids = viz_input_ids[i].tolist()
                            caption = tokenizer.decode(
                                token_ids, skip_special_tokens=True)
                            captions.append(
                                caption if caption.strip() else "(empty caption)")

                        # Get text-to-patch attention if available (much better for visualization)
                        text_to_patch_attn = None
                        if hasattr(viz_model, 'get_text_to_patch_attention'):
                            text_to_patch_attn = viz_model.get_text_to_patch_attention()
                            if text_to_patch_attn is not None:
                                text_to_patch_attn = text_to_patch_attn[:num_samples]

                        sbs_path = Path(config.output_dir) / "visualizations" / \
                            f"attention_sidebyside_step_{global_step}.png"
                        visualizer.visualize_attention_side_by_side(
                            images=viz_images.cpu(),
                            image_tokens=viz_prefix_tokens,
                            text_tokens=viz_text_embeddings,
                            attention_weights=attention[:num_samples],
                            captions=captions,
                            save_path=str(sbs_path),
                            title=f"Text-Conditioned Attention (Step {global_step})",
                            num_images=num_samples,
                            text_to_patch_attention=text_to_patch_attn,
                            num_patches=196  # DeiT-Tiny: 14x14 patches
                        )

                        print(
                            f"Saved side-by-side attention visualization to {sbs_path}")
                        print(
                            f"  Mean attention entropy: {stats['attention_entropy']:.4f}")
                        print(
                            f"  Mean attention sparsity: {stats['attention_sparsity']:.4f}")

                        if wandb_logger and wandb_logger.wandb_run:
                            import wandb
                            wandb_logger.wandb_run.log({
                                'enhanced_viz/attention_sidebyside': wandb.Image(str(sbs_path)),
                                'enhanced_viz/attention_entropy': stats['attention_entropy'],
                                'enhanced_viz/attention_sparsity': stats['attention_sparsity'],
                                'enhanced_viz/divergence': stats['divergence_statistic']
                            }, step=global_step)
                            
                            # Log text-to-patch attention metrics if available
                            if text_to_patch_attn is not None:
                                wandb_logger.log_text_to_patch_attention_metrics(
                                    text_to_patch_attn, viz_attention_mask, global_step
                                )
                        elif wandb_run:
                            wandb_run.log({
                                'attention/mean': stats['mean_attention'],
                                'attention/max': stats['max_attention'],
                                'attention/entropy': stats['attention_entropy'],
                                'attention/sparsity': stats['attention_sparsity'],
                                'attention/divergence': stats['divergence_statistic']
                            }, step=global_step)

                # Memory visualizations
                if config.use_memory and wandb_logger and viz_model.memory_state is not None:
                    # Ensure memory_state is valid tuple with data
                    if isinstance(viz_model.memory_state, tuple) and len(viz_model.memory_state) == 2 and viz_model.memory_state[0] is not None:
                        # Get batch size from memory state
                        M_batch_size = viz_model.memory_state[0].shape[0]
                        # Use matching batch size for validation
                        batch_size_for_vis = min(M_batch_size, len(input_ids))
                        z_for_memory = viz_model.encode_text(input_ids[:batch_size_for_vis], attention_mask[:batch_size_for_vis])[
                            0].mean(dim=1).unsqueeze(0)
                        w_mean = viz_model.episodic_memory._solve_w_mean(
                            z_for_memory, viz_model.memory_state[0][:batch_size_for_vis])

                        wandb_logger.log_memory_heatmap(
                            (viz_model.memory_state[0][:batch_size_for_vis],
                             viz_model.memory_state[1][:batch_size_for_vis]), w_mean, global_step
                        )

            model.train()

        # Remove step-based checkpointing (only save at end of epoch)

    avg_loss = total_loss / len(train_loader)
    return avg_loss, global_step


def count_parameters(model):
    """Count total and trainable parameters"""
    # Handle DDP/DataParallel wrapped models
    base_model = model.module if hasattr(model, 'module') else model
    total_params = sum(p.numel() for p in base_model.parameters())
    trainable_params = sum(p.numel()
                           for p in base_model.parameters() if p.requires_grad)
    frozen_params = total_params - trainable_params
    return {
        'total': total_params,
        'trainable': trainable_params,
        'frozen': frozen_params
    }


def get_model_statistics(model, config):
    """Get comprehensive model statistics"""
    # Unwrap DDP model if necessary
    base_model = model.module if hasattr(model, 'module') else model
    
    stats = {}

    # Overall parameters (use base_model for accurate counting)
    overall = count_parameters(base_model)
    stats['total_parameters'] = overall['total']
    stats['trainable_parameters'] = overall['trainable']
    stats['frozen_parameters'] = overall['frozen']
    stats['trainable_percentage'] = 100.0 * overall['trainable'] / \
        overall['total'] if overall['total'] > 0 else 0.0

    # Vision encoder parameters
    if hasattr(base_model, 'vision_encoder'):
        vision_stats = count_parameters(base_model.vision_encoder)
        stats['vision_total_params'] = vision_stats['total']
        stats['vision_trainable_params'] = vision_stats['trainable']
        stats['vision_frozen_params'] = vision_stats['frozen']

    # Language model parameters
    if hasattr(base_model, 'language_model'):
        lang_stats = count_parameters(base_model.language_model)
        stats['language_total_params'] = lang_stats['total']
        stats['language_trainable_params'] = lang_stats['trainable']
        stats['language_frozen_params'] = lang_stats['frozen']

    # Adapter parameters
    if hasattr(base_model, 'multimodal_adapter'):
        adapter_stats = count_parameters(base_model.multimodal_adapter)
        stats['adapter_total_params'] = adapter_stats['total']
        stats['adapter_trainable_params'] = adapter_stats['trainable']

    # Memory parameters
    if hasattr(base_model, 'episodic_memory'):
        memory_stats = count_parameters(base_model.episodic_memory)
        stats['memory_total_params'] = memory_stats['total']
        stats['memory_trainable_params'] = memory_stats['trainable']

    # Quantization info
    stats['vision_4bit_quantized'] = getattr(
        config, 'quantize_vision_4bit', False)
    stats['language_4bit_quantized'] = getattr(
        config, 'quantize_language_4bit', False)
    stats['memory_158bit_quantized'] = getattr(
        config, 'quantize_memory_158bit', False)

    # Estimate memory size (approximate)
    # 4-bit: 0.5 bytes per param, 1.58-bit: ~0.2 bytes per param, fp16: 2 bytes, fp32: 4 bytes
    total_size_mb = 0
    if stats.get('vision_4bit_quantized'):
        total_size_mb += stats.get('vision_total_params', 0) * 0.5 / 1e6
    else:
        total_size_mb += stats.get('vision_total_params', 0) * 2 / 1e6  # fp16

    if stats.get('language_4bit_quantized'):
        total_size_mb += stats.get('language_total_params', 0) * 0.5 / 1e6
    else:
        total_size_mb += stats.get('language_total_params',
                                   0) * 2 / 1e6  # fp16

    if stats.get('memory_158bit_quantized'):
        total_size_mb += stats.get('memory_total_params', 0) * 0.2 / 1e6
    else:
        total_size_mb += stats.get('memory_total_params', 0) * 4 / 1e6  # fp32

    # Add other components (adapter, etc.)
    total_size_mb += stats.get('adapter_total_params', 0) * 4 / 1e6

    stats['estimated_model_size_mb'] = round(total_size_mb, 2)

    return stats


def print_model_statistics(stats, epoch):
    """Print model statistics in a formatted way"""
    print("\n" + "="*80)
    print(f"MODEL STATISTICS - Epoch {epoch}")
    print("="*80)

    print("\n📊 Overall Parameters:")
    print(f"  Total:      {stats['total_parameters']:>15,}")
    print(
        f"  Trainable:  {stats['trainable_parameters']:>15,} ({stats['trainable_percentage']:.2f}%)")
    print(f"  Frozen:     {stats['frozen_parameters']:>15,}")

    print("\n🖼️  Vision Encoder:")
    print(f"  Total:      {stats.get('vision_total_params', 0):>15,}")
    print(f"  Trainable:  {stats.get('vision_trainable_params', 0):>15,}")
    print(f"  Frozen:     {stats.get('vision_frozen_params', 0):>15,}")

    print("\n💬 Language Model:")
    print(f"  Total:      {stats.get('language_total_params', 0):>15,}")
    print(f"  Trainable:  {stats.get('language_trainable_params', 0):>15,}")
    print(f"  Frozen:     {stats.get('language_frozen_params', 0):>15,}")

    print("\n🔗 Multimodal Adapter:")
    print(f"  Total:      {stats.get('adapter_total_params', 0):>15,}")
    print(f"  Trainable:  {stats.get('adapter_trainable_params', 0):>15,}")

    print("\n🧠 Episodic Memory:")
    print(f"  Total:      {stats.get('memory_total_params', 0):>15,}")
    print(f"  Trainable:  {stats.get('memory_trainable_params', 0):>15,}")

    print("\n⚙️  Quantization:")
    print(
        f"  Vision 4-bit:     {'✓' if stats.get('vision_4bit_quantized') else '✗'}")
    print(
        f"  Language 4-bit:   {'✓' if stats.get('language_4bit_quantized') else '✗'}")
    print(
        f"  Memory 1.58-bit:  {'✓' if stats.get('memory_158bit_quantized') else '✗'}")

    print("\n💾 Estimated Model Size:")
    print(f"  ~{stats['estimated_model_size_mb']:.2f} MB")

    print("="*80 + "\n")


def save_epoch_checkpoint(model, optimizer, epoch, global_step, config, stage_name="default"):
    """Save model checkpoint after each epoch"""
    checkpoint_dir = Path(config.output_dir) / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # Get and print model statistics
    stats = get_model_statistics(model, config)
    print_model_statistics(stats, epoch)

    # Save checkpoint with epoch number
    checkpoint_path = checkpoint_dir / f"epoch_{epoch}_checkpoint.pt"

    torch.save({
        'epoch': epoch,
        'global_step': global_step,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'config': vars(config),
        'model_statistics': stats
    }, checkpoint_path)

    print(f"✅ Checkpoint saved: {checkpoint_path}")

    # Save statistics as JSON
    stats_path = checkpoint_dir / f"epoch_{epoch}_statistics.json"
    with open(stats_path, 'w') as f:
        json.dump(stats, f, indent=2)
    print(f"📊 Statistics saved: {stats_path}")

    return checkpoint_path, stats


def main():
    parser = argparse.ArgumentParser(description="Train MicroVLM-V")
    parser.add_argument('--config', type=str, default='default',
                        choices=['test', 'stage1', 'stage2',
                                 'default', 'full_quantized'],
                        help='Training configuration')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume from')
    parser.add_argument('--use-staged-config', action='store_true',
                        help='Use staged configuration system')
    parser.add_argument('--num_images', type=int, default=None,
                        help='Limit total dataset size to N images. '
                             'Auto-splits into train/val (95%%/5%%). '
                             'E.g., --num_images=2000000 uses 1.9M train, 100K val.')
    parser.add_argument('--val_split', type=float, default=0.05,
                        help='Validation split ratio when using --num_images (default: 0.05)')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of DataLoader workers (default: 4, max: 32)')
    parser.add_argument('--multi_gpu', action='store_true',
                        help='Enable multi-GPU training with DistributedDataParallel')
    parser.add_argument('--local_rank', type=int, default=-1,
                        help='Local rank for distributed training (set automatically by torchrun)')
    parser.add_argument('--alignment_mode', type=str, default='baseline',
                        choices=['baseline', 'fiber'],
                        help='Alignment mode: baseline (prefix-only) or fiber (fusion-in-backbone)')
    parser.add_argument('--fiber_layers', type=str, default='6,8,10',
                        help='Comma-separated list of vision encoder layers for FIBER fusion (default: 6,8,10)')
    parser.add_argument('--itc_weight', type=float, default=1.0,
                        help='Weight for ITC (image-text contrastive) loss in FIBER mode')
    parser.add_argument('--itm_weight', type=float, default=0.5,
                        help='Weight for ITM (image-text matching) loss in FIBER mode')

    args = parser.parse_args()

    # Parse FIBER layers
    fiber_fusion_layers = [int(x.strip()) for x in args.fiber_layers.split(',')]

    # Cap num_workers at 32
    args.num_workers = min(max(args.num_workers, 0), 32)

    # ========== Multi-GPU / DDP Setup ==========
    is_distributed = False
    local_rank = 0
    world_size = 1
    
    # Check for torchrun environment variables (preferred method)
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        is_distributed = True
        local_rank = int(os.environ.get('LOCAL_RANK', 0))
        world_size = int(os.environ['WORLD_SIZE'])
        rank = int(os.environ['RANK'])
        
        # Initialize distributed process group
        dist.init_process_group(backend='nccl', init_method='env://')
        torch.cuda.set_device(local_rank)
        
        if rank == 0:
            print(f"\n{'='*60}")
            print(f"  DISTRIBUTED TRAINING ENABLED")
            print(f"  World size: {world_size} GPUs")
            print(f"  Backend: NCCL")
            print(f"{'='*60}\n")
    elif args.multi_gpu and torch.cuda.device_count() > 1:
        # Fallback: DataParallel mode (simpler but less efficient)
        print(f"\n{'='*60}")
        print(f"  MULTI-GPU MODE (DataParallel)")
        print(f"  Available GPUs: {torch.cuda.device_count()}")
        print(f"  NOTE: For best performance, use torchrun:")
        print(f"    torchrun --nproc_per_node={torch.cuda.device_count()} scripts/train.py ...")
        print(f"{'='*60}\n")
    
    # Helper to check if current process is main
    is_main_process = (not is_distributed) or (dist.get_rank() == 0)

    # Load configuration - prioritize staged config if requested
    if args.use_staged_config or args.config in ['stage1', 'stage2', 'full_quantized']:
        config = load_staged_config(args.config)
        if is_main_process:
            print(f"Loaded STAGED configuration: {args.config}")
    else:
        config = load_config(args.config)
        if is_main_process:
            print(f"Loaded configuration: {args.config}")

    # Create output directories
    Path(config.output_dir).mkdir(parents=True, exist_ok=True)
    Path(config.output_dir).joinpath("checkpoints").mkdir(exist_ok=True)
    Path(config.output_dir).joinpath("visualizations").mkdir(exist_ok=True)

    # Load model configuration
    model_config_path = Path(__file__).parent.parent / \
        "configs" / "model_config.json"
    if not model_config_path.exists():
        print(f"ERROR: Model config not found: {model_config_path}")
        print("Run: python scripts/extract_model_config.py")
        sys.exit(1)

    with open(model_config_path, 'r') as f:
        model_config = json.load(f)

    # Create run name
    run_name, run_counter = create_run_name(args.config)
    if hasattr(config, 'wandb_run_name') and config.wandb_run_name:
        run_name = f"{config.wandb_run_name}_{run_counter}"
    config.wandb_run_name = run_name

    if is_main_process:
        print(f"Run name: {run_name} (counter: {run_counter})")

    # Setup WandB (main process only to avoid duplicate logging)
    wandb_run = None
    wandb_logger = None
    if is_main_process:
        wandb_run = setup_wandb(config, run_name)
        wandb_logger = WandBLogger(config, wandb_run) if wandb_run else None

    # Create model with quantization settings
    if is_main_process:
        print("Creating model...")
        print(f"  - Alignment mode: {args.alignment_mode}")
        if args.alignment_mode == 'fiber':
            print(f"  - FIBER fusion layers: {fiber_fusion_layers}")
            print(f"  - ITC weight: {args.itc_weight}, ITM weight: {args.itm_weight}")
        print(f"  - 4-bit quantization: Vision={getattr(config, 'quantize_vision_4bit', False)}, "
              f"Language={getattr(config, 'quantize_language_4bit', False)}")
        print(
            f"  - 1.58-bit memory quantization: {getattr(config, 'quantize_memory_158bit', False)}")

    if args.alignment_mode == 'fiber':
        # Create FIBER model with fusion-in-backbone alignment
        # Build FIBER config
        fiber_config = {
            'fiber_fusion_layers': fiber_fusion_layers,
            'cross_attention_heads': 4,
            'cross_attention_dim': model_config['model_dimensions'].get('vision_hidden_size', 192),
            'use_bidirectional': True,
            'alpha_init': 0.0,
            'embed_dim': 256,
            'itc_weight': args.itc_weight,
            'itm_weight': args.itm_weight,
        }
        
        model = create_microvlm_fiber(
            config=model_config['model_dimensions'],
            vision_checkpoint=getattr(
                config, 'vision_checkpoint', config.deit_checkpoint),
            language_checkpoint=getattr(
                config, 'language_checkpoint', config.qwen_model),
            training_config=config,
            fiber_config=fiber_config
        )
        
        if is_main_process:
            print(f"  ✓ Created MicroVLMFIBER model with fusion at layers {fiber_fusion_layers}")
    else:
        # Create baseline MicroVLM model
        model = create_microvlm(
            config=model_config['model_dimensions'],
            language_checkpoint=getattr(
                config, 'language_checkpoint', config.qwen_model),
            vision_checkpoint=getattr(
                config, 'vision_checkpoint', config.deit_checkpoint),
            quantize_4bit=(getattr(config, 'quantize_vision_4bit', False) or
                           getattr(config, 'quantize_language_4bit', False)),
            quantize_memory_158bit=getattr(
                config, 'quantize_memory_158bit', False),
            training_config=config
        )

    # Log quantization statistics if enabled
    if getattr(config, 'quantize_memory_158bit', False) and hasattr(model, 'episodic_memory'):
        quant_stats = get_memory_quantization_stats(model.episodic_memory)
        print("\n=== Memory Quantization Statistics ===")
        for key, value in quant_stats.items():
            print(f"  {key}: {value}")
        print("=" * 40 + "\n")

        if wandb_logger:
            wandb_logger.log_metrics(
                quant_stats, step=0, prefix="quantization")

    # Apply freezing strategy (different for FIBER vs baseline)
    if args.alignment_mode == 'fiber':
        # For FIBER: freeze language model but keep vision encoder trainable for fusion
        if config.freeze_language:
            model.freeze_language_model(
                unfreeze_last_n=config.unfreeze_last_n_layers)
        # Note: FIBER vision encoder has fusion layers that need to be trainable
        # The base DeiT layers can optionally be frozen
        if config.freeze_vision:
            # Freeze base vision encoder but keep fusion layers trainable
            if hasattr(model, 'fiber_vision_encoder'):
                model.fiber_vision_encoder.freeze_base_vision(freeze=True)
    else:
        # Baseline freezing strategy
        if config.freeze_vision:
            model.freeze_vision_encoder()

        if config.freeze_language:
            model.freeze_language_model(
                unfreeze_last_n=config.unfreeze_last_n_layers)

    # Print trainable parameters
    trainable_params = model.get_trainable_params()
    total_params = sum(p.numel() for p in model.parameters())
    if is_main_process:
        print(f"Trainable parameters: {trainable_params:,} / {total_params:,} "
              f"({100*trainable_params/total_params:.2f}%)")
        if args.alignment_mode == 'fiber':
            print("  ✓ FIBER fusion layers are trainable")

    # Move to device (use local_rank for DDP, otherwise config.device)
    if is_distributed:
        device = torch.device(f'cuda:{local_rank}')
        model = model.to(device)
        # Wrap model with DDP
        model = DDP(model, device_ids=[local_rank], output_device=local_rank,
                    find_unused_parameters=True)
        if is_main_process:
            print(f"Model wrapped with DistributedDataParallel on GPU {local_rank}")
    elif args.multi_gpu and torch.cuda.device_count() > 1:
        # DataParallel fallback
        device = torch.device('cuda:0')
        model = model.to(device)
        model = nn.DataParallel(model)
        print(f"Model wrapped with DataParallel across {torch.cuda.device_count()} GPUs")
    else:
        device = torch.device(config.device)
        model = model.to(device)
    
    # Update config device for consistency
    config.device = device
    
    # Store alignment mode in config for use in training loop
    config.alignment_mode = args.alignment_mode

    # Create visualizer
    visualizer = create_attention_visualizer(model_config['model_dimensions'])

    # Determine max_samples: CLI --num_images takes precedence over config
    if args.num_images is not None:
        # User specified --num_images, auto-split into train/val
        num_val = int(args.num_images * args.val_split)
        num_train = args.num_images - num_val
        effective_max_samples = num_train
        effective_max_val_samples = num_val
        if is_main_process:
            print(f"\n=== Dataset Limiting (--num_images={args.num_images:,}) ===")
            print(f"  Train samples: {num_train:,}")
            print(f"  Val samples:   {num_val:,}")
            print(f"  Val split:     {args.val_split:.1%}")
            print("=" * 50 + "\n")
    else:
        # Use config values
        effective_max_samples = config.max_samples
        effective_max_val_samples = getattr(config, 'max_val_samples', None)

    # Determine effective num_workers (CLI takes precedence)
    effective_num_workers = args.num_workers
    if is_main_process:
        print(f"Using {effective_num_workers} DataLoader workers (max: 32)")

    # Get the underlying model for tokenizer access (handles DDP/DP wrapping)
    base_model = model.module if hasattr(model, 'module') else model

    # Create dataloaders with distributed support
    if is_main_process:
        print("Creating dataloaders...")
    
    train_loader, val_loader, train_sampler = create_dataloaders(
        train_metadata_file=config.train_metadata_file,
        val_metadata_file=config.val_metadata_file,
        tokenizer=base_model.language_model.tokenizer,
        batch_size=config.batch_size,
        num_workers=effective_num_workers,
        max_samples=effective_max_samples,
        max_val_samples=effective_max_val_samples,
        distributed=is_distributed,
        world_size=world_size,
        rank=local_rank if is_distributed else 0
    )

    # Validate first batch to check data quality (main process only)
    if is_main_process:
        print("\nValidating first training batch...")
        first_batch = next(iter(train_loader))
        print(f"  Image shape: {first_batch['image'].shape}")
        print(f"  Input IDs shape: {first_batch['input_ids'].shape}")
        print(f"  Caption samples: {first_batch['caption'][:2]}")
        print(
            f"  Caption lengths: min={first_batch['attention_mask'].sum(dim=1).min().item()}, max={first_batch['attention_mask'].sum(dim=1).max().item()}")
        del first_batch

    # Create optimizer and scheduler
    optimizer = create_optimizer(model, config)
    total_steps = len(train_loader) * config.num_epochs
    scheduler = create_scheduler(optimizer, config, total_steps)

    # Resume from checkpoint if specified
    start_epoch = 0
    global_step = 0

    if args.resume:
        if is_main_process:
            print(f"Resuming from checkpoint: {args.resume}")
        checkpoint = torch.load(args.resume, map_location=config.device)
        # Handle DDP/DP wrapped models
        if hasattr(model, 'module'):
            model.module.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch']
        global_step = checkpoint['global_step']

    # Training loop
    if is_main_process:
        print(f"Starting training for {config.num_epochs} epochs...")

    # Track training history for model card
    training_history = []

    for epoch in range(start_epoch, config.num_epochs):
        # Set epoch for distributed sampler (ensures proper shuffling)
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        
        avg_loss, global_step = train_epoch(
            model=model,
            train_loader=train_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            config=config,
            visualizer=visualizer,
            epoch=epoch,
            global_step=global_step,
            wandb_run=wandb_run,
            wandb_logger=wandb_logger,
            is_distributed=is_distributed,
            is_main_process=is_main_process
        )

        if is_main_process:
            print(f"Epoch {epoch} completed. Average loss: {avg_loss:.4f}")

            # Log weight histograms at end of epoch (if wandb_logger available)
            if wandb_logger and (epoch % max(1, config.num_epochs // 5) == 0):
                wandb_logger.log_model_weights_histogram(model, global_step)

            # Determine stage name for HF commit message
            stage_name = args.config if hasattr(args, 'config') else 'default'

            # Save epoch checkpoint with statistics
            checkpoint_path, stats = save_epoch_checkpoint(
                model, optimizer, epoch, global_step, config, stage_name
            )

            # Add to training history (for model card)
            training_history.append({
                'epoch': epoch,
                'train_loss': avg_loss,
                'alignment_loss': stats.get('alignment_loss', 0),
                'learning_rate': optimizer.param_groups[0]['lr'],
                'gradient_norm': stats.get('gradient_norm', 0)
            })

            # Log statistics to WandB
            if wandb_logger:
                wandb_logger.log_metrics(
                    stats, step=global_step, prefix="model_stats")

            # Push ONLY latest checkpoint to HuggingFace (replaces previous)
            push_to_huggingface(
                checkpoint_path=checkpoint_path,
                stats=stats,
                epoch=epoch,
                total_epochs=config.num_epochs,
                stage_name=stage_name,
                config=config,
                training_history=training_history
            )
        
        # Synchronize all processes before next epoch
        if is_distributed:
            dist.barrier()

    # Final save (main process only)
    if is_main_process:
        final_path = Path(config.output_dir) / "final_model.pt"
        base_model = model.module if hasattr(model, 'module') else model
        base_model.save_checkpoint(str(final_path))
        print(f"Training complete! Final model saved to {final_path}")

        if wandb_run:
            wandb_run.finish()
    
    # Clean up distributed training
    if is_distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
