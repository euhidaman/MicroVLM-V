"""
Main Training Script for MicroVLM-V
Supports small-scale testing and full training stages
"""

import os
import sys
import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pathlib import Path
import argparse
from tqdm import tqdm
import wandb

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models import create_microvlm
from src.data.cc12m_loader import create_dataloaders
from src.training.config import load_config, create_run_name
from src.visualization.attention_vis import create_attention_visualizer
from src.quantization.quantize_4bit import QuantizationConfig


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
    """Create learning rate scheduler"""
    if config.scheduler == "cosine":
        from torch.optim.lr_scheduler import CosineAnnealingLR
        scheduler = CosineAnnealingLR(optimizer, T_max=total_steps)
    elif config.scheduler == "linear":
        from torch.optim.lr_scheduler import LinearLR
        scheduler = LinearLR(optimizer, start_factor=1.0, end_factor=0.1, 
                           total_iters=total_steps)
    else:
        scheduler = None
    
    return scheduler


def train_epoch(model, train_loader, optimizer, scheduler, config, visualizer,
                epoch, global_step, wandb_run=None):
    """Train for one epoch"""
    model.train()
    
    total_loss = 0
    lm_loss_total = 0
    alignment_loss_total = 0
    memory_kl_total = 0
    
    pbar = tqdm(train_loader, desc=f"Epoch {epoch}")
    
    for batch_idx, batch in enumerate(pbar):
        # Move to device
        images = batch['image'].to(config.device)
        input_ids = batch['input_ids'].to(config.device)
        attention_mask = batch['attention_mask'].to(config.device)
        
        # Labels for language modeling (shifted in model)
        labels = input_ids.clone()
        
        # Forward pass
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
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping
        if config.gradient_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip)
        
        optimizer.step()
        
        if scheduler is not None:
            scheduler.step()
        
        # Accumulate losses
        total_loss += loss.item()
        lm_loss_total += outputs.get('lm_loss', 0).item() if outputs.get('lm_loss') is not None else 0
        alignment_loss_total += outputs.get('alignment_loss', 0).item() if 'alignment_loss' in outputs else 0
        memory_kl_total += outputs.get('memory_kl', 0).item() if 'memory_kl' in outputs else 0
        
        global_step += 1
        
        # Update progress bar
        pbar.set_postfix({
            'loss': loss.item(),
            'lm': outputs.get('lm_loss', 0).item() if outputs.get('lm_loss') is not None else 0,
            'lr': optimizer.param_groups[0]['lr']
        })
        
        # Logging
        if global_step % config.log_interval == 0:
            metrics = {
                'train/loss': loss.item(),
                'train/lm_loss': outputs.get('lm_loss', 0).item() if outputs.get('lm_loss') is not None else 0,
                'train/learning_rate': optimizer.param_groups[0]['lr'],
                'train/epoch': epoch,
                'train/global_step': global_step
            }
            
            if 'alignment_loss' in outputs:
                metrics['train/alignment_loss'] = outputs['alignment_loss'].item()
            
            if 'memory_kl' in outputs:
                metrics['train/memory_kl'] = outputs['memory_kl'].item()
                metrics['train/addressing_kl'] = outputs['addressing_kl'].item()
            
            if wandb_run:
                wandb_run.log(metrics, step=global_step)
        
        # Visualization
        if global_step % config.visualize_interval == 0 and visualizer is not None:
            model.eval()
            with torch.no_grad():
                # Analyze attention
                prefix_tokens, image_features = model.encode_image(images[:1])
                text_embeddings, text_features = model.encode_text(
                    input_ids[:1], attention_mask[:1]
                )
                
                stats, attention = visualizer.analyze_cross_modal_attention(
                    prefix_tokens, text_embeddings
                )
                
                # Save visualization
                save_path = Path(config.output_dir) / "visualizations" / f"attention_step_{global_step}.png"
                visualizer.visualize_attention(
                    attention[0],
                    save_path=str(save_path),
                    title=f"Cross-Modal Attention (Step {global_step})"
                )
                
                # Log to wandb
                if wandb_run:
                    wandb_run.log({
                        'attention/mean': stats['mean_attention'],
                        'attention/max': stats['max_attention'],
                        'attention/entropy': stats['attention_entropy'],
                        'attention/sparsity': stats['attention_sparsity'],
                        'attention/divergence': stats['divergence_statistic']
                    }, step=global_step)
            
            model.train()
        
        # Save checkpoint
        if global_step % config.save_interval == 0:
            save_checkpoint(model, optimizer, epoch, global_step, config)
    
    avg_loss = total_loss / len(train_loader)
    return avg_loss, global_step


def save_checkpoint(model, optimizer, epoch, global_step, config):
    """Save model checkpoint"""
    checkpoint_dir = Path(config.output_dir) / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    checkpoint_path = checkpoint_dir / f"checkpoint_step_{global_step}.pt"
    
    torch.save({
        'epoch': epoch,
        'global_step': global_step,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'config': vars(config)
    }, checkpoint_path)
    
    print(f"Checkpoint saved: {checkpoint_path}")


def main():
    parser = argparse.ArgumentParser(description="Train MicroVLM-V")
    parser.add_argument('--config', type=str, default='default',
                       choices=['test', 'stage1', 'stage2', 'default'],
                       help='Training configuration')
    parser.add_argument('--resume', type=str, default=None,
                       help='Path to checkpoint to resume from')
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    print(f"Loaded configuration: {args.config}")
    
    # Create output directories
    Path(config.output_dir).mkdir(parents=True, exist_ok=True)
    Path(config.output_dir).joinpath("checkpoints").mkdir(exist_ok=True)
    Path(config.output_dir).joinpath("visualizations").mkdir(exist_ok=True)
    
    # Load model configuration
    model_config_path = Path(__file__).parent.parent / "configs" / "model_config.json"
    if not model_config_path.exists():
        print(f"ERROR: Model config not found: {model_config_path}")
        print("Run: python scripts/extract_model_config.py")
        sys.exit(1)
    
    with open(model_config_path, 'r') as f:
        model_config = json.load(f)
    
    # Create run name
    run_name, run_counter = create_run_name(args.config)
    config.wandb_run_name = run_name
    
    print(f"Run name: {run_name} (counter: {run_counter})")
    
    # Setup WandB
    wandb_run = setup_wandb(config, run_name)
    
    # Create model
    print("Creating model...")
    model = create_microvlm(
        config=model_config['model_dimensions'],
        language_checkpoint=config.qwen_model,
        vision_checkpoint=config.deit_checkpoint,
        quantize_4bit=config.use_4bit
    )
    
    # Apply freezing strategy
    if config.freeze_vision:
        model.freeze_vision_encoder()
    
    if config.freeze_language:
        model.freeze_language_model(unfreeze_last_n=config.unfreeze_last_n_layers)
    
    # Print trainable parameters
    trainable_params = model.get_trainable_params()
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Trainable parameters: {trainable_params:,} / {total_params:,} "
          f"({100*trainable_params/total_params:.2f}%)")
    
    # Move to device
    model = model.to(config.device)
    
    # Create visualizer
    visualizer = create_attention_visualizer(model_config['model_dimensions'])
    
    # Create dataloaders
    print("Creating dataloaders...")
    train_loader, val_loader = create_dataloaders(
        train_metadata_file=config.train_metadata_file,
        val_metadata_file=config.val_metadata_file,
        tokenizer=model.language_model.tokenizer,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        max_samples=config.max_samples
    )
    
    # Create optimizer and scheduler
    optimizer = create_optimizer(model, config)
    total_steps = len(train_loader) * config.num_epochs
    scheduler = create_scheduler(optimizer, config, total_steps)
    
    # Resume from checkpoint if specified
    start_epoch = 0
    global_step = 0
    
    if args.resume:
        print(f"Resuming from checkpoint: {args.resume}")
        checkpoint = torch.load(args.resume)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch']
        global_step = checkpoint['global_step']
    
    # Training loop
    print(f"Starting training for {config.num_epochs} epochs...")
    
    for epoch in range(start_epoch, config.num_epochs):
        avg_loss, global_step = train_epoch(
            model=model,
            train_loader=train_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            config=config,
            visualizer=visualizer,
            epoch=epoch,
            global_step=global_step,
            wandb_run=wandb_run
        )
        
        print(f"Epoch {epoch} completed. Average loss: {avg_loss:.4f}")
        
        # Save epoch checkpoint
        save_checkpoint(model, optimizer, epoch, global_step, config)
    
    # Final save
    final_path = Path(config.output_dir) / "final_model.pt"
    model.save_checkpoint(str(final_path))
    
    print(f"Training complete! Final model saved to {final_path}")
    
    if wandb_run:
        wandb_run.finish()


if __name__ == "__main__":
    main()
