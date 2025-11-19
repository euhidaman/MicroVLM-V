"""
Attention Visualization Module
Integrates LeJEPA statistical testing for attention analysis
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import distributed as dist
from torch.distributed._functional_collectives import all_reduce as functional_all_reduce
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path


def all_reduce(x, op="AVG"):
    """Distributed all-reduce helper"""
    if dist.is_available() and dist.is_initialized():
        return functional_all_reduce(x, op.lower(), dist.group.WORLD)
    else:
        return x


class FastEppsPulley(nn.Module):
    """
    Fast Epps-Pulley univariate test statistic
    For testing distributional differences
    """
    
    def __init__(self, t_max=5.0, n_points=21):
        super().__init__()
        self.t_max = t_max
        self.n_points = n_points
        
        # Precompute test points
        self.register_buffer('t_points', 
                           torch.linspace(-t_max, t_max, n_points))
    
    def forward(self, x):
        """
        Compute Epps-Pulley statistic
        
        Args:
            x: (*, N, K) samples
        
        Returns:
            statistic: (*, K) test statistics
        """
        # x: (*, N, K)
        # Expand for test points: (*, N, K, n_points)
        x_expanded = x.unsqueeze(-1)  # (*, N, K, 1)
        t_expanded = self.t_points.view(1, 1, 1, -1)  # (1, 1, 1, n_points)
        
        # Compute characteristic function
        char_fn = torch.exp(1j * x_expanded * t_expanded)  # (*, N, K, n_points)
        
        # Mean over samples
        mean_char = char_fn.mean(dim=-3)  # (*, K, n_points)
        
        # Compute statistic
        statistic = (mean_char.abs() ** 2).mean(dim=-1)  # (*, K)
        
        return statistic


class SlicingUnivariateTest(nn.Module):
    """
    LeJEPA-style slicing univariate test
    Projects multivariate data to random 1D directions
    """
    
    def __init__(self, univariate_test, num_slices=256, reduction='mean',
                 sampler='gaussian', clip_value=0.01):
        super().__init__()
        
        self.univariate_test = univariate_test
        self.num_slices = num_slices
        self.reduction = reduction
        self.sampler = sampler
        self.clip_value = clip_value
        
        self.register_buffer('global_step', torch.zeros((), dtype=torch.long))
        
        # Generator cache
        self._generator = None
        self._generator_device = None
    
    def _get_generator(self, device, seed):
        """Get or create generator"""
        if self._generator is None or self._generator_device != device:
            self._generator = torch.Generator(device=device)
            self._generator_device = device
        self._generator.manual_seed(seed)
        return self._generator
    
    def forward(self, x):
        """
        Apply slicing test
        
        Args:
            x: (*, N, D) multivariate samples
        
        Returns:
            statistic: aggregated test statistic
        """
        with torch.no_grad():
            # Sync global step
            global_step_sync = all_reduce(self.global_step.clone(), op='MAX')
            seed = global_step_sync.item()
            
            # Generate random projections
            g = self._get_generator(x.device, seed)
            proj_shape = (x.size(-1), self.num_slices)
            A = torch.randn(proj_shape, device=x.device, generator=g)
            A = A / A.norm(p=2, dim=0)  # Normalize
            
            self.global_step.add_(1)
        
        # Project: (*, N, D) @ (D, K) -> (*, N, K)
        x_projected = x @ A
        
        # Apply univariate test
        stats = self.univariate_test(x_projected)
        
        # Clip
        if self.clip_value is not None:
            stats = torch.where(stats < self.clip_value, 
                              torch.zeros_like(stats), stats)
        
        # Reduce
        if self.reduction == 'mean':
            return stats.mean()
        elif self.reduction == 'sum':
            return stats.sum()
        else:
            return stats


class AttentionVisualizer(nn.Module):
    """
    Attention visualization and analysis
    Generates heatmaps and statistics for cross-modal attention
    """
    
    def __init__(self, config):
        super().__init__()
        
        self.config = config
        
        # Statistical test for attention analysis
        univariate_test = FastEppsPulley(t_max=5.0, n_points=21)
        self.attention_test = SlicingUnivariateTest(
            univariate_test=univariate_test,
            num_slices=256,
            reduction='mean',
            sampler='gaussian',
            clip_value=0.01
        )
        
        self.visualizations = []
    
    def compute_attention_maps(self, query, key, mask=None):
        """
        Compute attention maps
        
        Args:
            query: (batch, num_heads, seq_q, head_dim)
            key: (batch, num_heads, seq_k, head_dim)
            mask: optional attention mask
        
        Returns:
            attention_weights: (batch, num_heads, seq_q, seq_k)
        """
        # Compute attention scores
        scores = torch.matmul(query, key.transpose(-2, -1))
        scores = scores / np.sqrt(query.size(-1))
        
        # Apply mask if provided
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        # Softmax
        attention_weights = F.softmax(scores, dim=-1)
        
        return attention_weights
    
    def analyze_cross_modal_attention(self, image_tokens, text_tokens):
        """
        Analyze attention between image and text
        
        Args:
            image_tokens: (batch, k_prefix, hidden_dim)
            text_tokens: (batch, seq_len, hidden_dim)
        
        Returns:
            statistics: dictionary of attention statistics
        """
        batch_size = image_tokens.size(0)
        
        # Compute attention from text to image
        query = text_tokens  # (B, seq_len, D)
        key = image_tokens  # (B, k_prefix, D)
        
        # Simple attention (without multi-head split)
        scores = torch.matmul(query, key.transpose(-2, -1))
        scores = scores / np.sqrt(query.size(-1))
        attention = F.softmax(scores, dim=-1)  # (B, seq_len, k_prefix)
        
        # Statistics
        stats = {
            'mean_attention': attention.mean().item(),
            'max_attention': attention.max().item(),
            'attention_entropy': self._compute_entropy(attention).mean().item(),
            'attention_sparsity': (attention < 0.01).float().mean().item()
        }
        
        # Apply slicing test
        # Flatten batch and reshape for test
        attention_flat = attention.reshape(-1, attention.size(-1))  # (B*seq_len, k_prefix)
        test_stat = self.attention_test(attention_flat.unsqueeze(1))  # Add sample dim
        stats['divergence_statistic'] = test_stat.item()
        
        return stats, attention
    
    def _compute_entropy(self, probs, eps=1e-10):
        """Compute entropy of probability distribution"""
        return -(probs * torch.log(probs + eps)).sum(dim=-1)
    
    def visualize_attention(self, attention_weights, text_tokens_str=None, 
                          save_path=None, title="Cross-Modal Attention"):
        """
        Visualize attention heatmap
        
        Args:
            attention_weights: (seq_len, k_prefix) or (batch, seq_len, k_prefix)
            text_tokens_str: list of token strings
            save_path: path to save figure
            title: plot title
        """
        # Handle batch dimension
        if attention_weights.dim() == 3:
            attention_weights = attention_weights[0]  # Take first in batch
        
        # Convert to numpy
        attn_np = attention_weights.cpu().detach().numpy()
        
        # Create heatmap
        fig, ax = plt.subplots(figsize=(10, 8))
        
        im = ax.imshow(attn_np, cmap='viridis', aspect='auto')
        
        ax.set_xlabel('Image Tokens (Prefix)')
        ax.set_ylabel('Text Tokens')
        ax.set_title(title)
        
        # Add colorbar
        plt.colorbar(im, ax=ax)
        
        # Add text labels if provided
        if text_tokens_str is not None:
            ax.set_yticks(range(len(text_tokens_str)))
            ax.set_yticklabels(text_tokens_str, fontsize=8)
        
        plt.tight_layout()
        
        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Attention heatmap saved to {save_path}")
        
        # Store for WandB logging
        self.visualizations.append({
            'figure': fig,
            'title': title,
            'path': save_path
        })
        
        plt.close()
        
        return fig
    
    def visualize_memory_addressing(self, addressing_weights, save_path=None,
                                   title="Episodic Memory Addressing"):
        """
        Visualize memory addressing weights
        
        Args:
            addressing_weights: (episode_size, batch, memory_size)
            save_path: path to save
            title: plot title
        """
        # Convert to numpy
        if isinstance(addressing_weights, torch.Tensor):
            w_np = addressing_weights.cpu().detach().numpy()
        else:
            w_np = np.array(addressing_weights)
        
        # Average over batch
        if w_np.ndim == 3:
            w_np = w_np.mean(axis=1)  # (episode_size, memory_size)
        
        # Create heatmap
        fig, ax = plt.subplots(figsize=(12, 6))
        
        im = ax.imshow(w_np, cmap='coolwarm', aspect='auto')
        
        ax.set_xlabel('Memory Slots')
        ax.set_ylabel('Episode Steps')
        ax.set_title(title)
        
        plt.colorbar(im, ax=ax, label='Addressing Weight')
        
        plt.tight_layout()
        
        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Memory addressing heatmap saved to {save_path}")
        
        self.visualizations.append({
            'figure': fig,
            'title': title,
            'path': save_path
        })
        
        plt.close()
        
        return fig
    
    def get_visualizations(self):
        """Get all stored visualizations"""
        return self.visualizations
    
    def clear_visualizations(self):
        """Clear stored visualizations"""
        self.visualizations = []


def create_attention_visualizer(config):
    """Factory function for attention visualizer"""
    return AttentionVisualizer(config)
