"""
Attention Visualization Module
Integrates LeJEPA statistical testing for attention analysis
"""

import os
import warnings
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import distributed as dist
from torch.distributed._functional_collectives import all_reduce as functional_all_reduce
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path


def all_reduce(x, op="AVG"):
    """Distributed all-reduce helper
    
    Note: For visualization which only runs on main process,
    we skip distributed sync to avoid CPU tensor issues.
    """
    if dist.is_available() and dist.is_initialized() and x.is_cuda:
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
        # Convert to float32 to avoid ComplexHalf warning
        x_float = x.float()

        # Expand for test points: (*, N, K, n_points)
        x_expanded = x_float.unsqueeze(-1)  # (*, N, K, 1)
        t_expanded = self.t_points.to(x.device).view(
            1, 1, 1, -1)  # (1, 1, 1, n_points)

        # Compute characteristic function
        # (*, N, K, n_points)
        char_fn = torch.exp(1j * x_expanded * t_expanded)

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
            A = torch.randn(proj_shape, device=x.device,
                            generator=g, dtype=x.dtype)
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

    def analyze_cross_modal_attention(self, image_tokens, text_tokens, attention_weights=None):
        """
        Analyze attention between image and text

        Args:
            image_tokens: (batch, k_prefix, hidden_dim)
            text_tokens: (batch, seq_len, hidden_dim)
            attention_weights: optional precomputed attention (batch, seq_len, k_prefix)

        Returns:
            statistics: dictionary of attention statistics
        """
        batch_size = image_tokens.size(0)

        if attention_weights is None:
            attention = self._compute_embedding_attention(
                image_tokens, text_tokens)
        else:
            attention = attention_weights
            if attention.dim() == 4:
                # If heads dimension present, average across heads
                attention = attention.mean(dim=1)
            # Renormalize over prefix tokens to interpret as probability
            attn_sum = attention.sum(dim=-1, keepdim=True).clamp_min(1e-8)
            attention = attention / attn_sum
            if torch.isnan(attention).any() or torch.isinf(attention).any():
                # This is expected in Stage1 with frozen LM - use embedding-based fallback
                # Only warn once to avoid spam
                if not hasattr(self, '_nan_warning_shown'):
                    warnings.warn(
                        "LM attention contains NaN (expected in Stage1 with frozen backbone). "
                        "Using embedding-based attention fallback for visualization."
                    )
                    self._nan_warning_shown = True
                attention = self._compute_embedding_attention(
                    image_tokens, text_tokens)

        # Statistics
        entropy = self._compute_entropy(attention)

        # Check for NaN in entropy
        valid_entropy = entropy[~torch.isnan(entropy)]
        if len(valid_entropy) == 0:
            # Fallback to zero if all NaN
            mean_entropy = 0.0
        else:
            mean_entropy = valid_entropy.mean().item()

        max_entropy = max(np.log(attention.size(-1)), 1e-8)
        normalized_entropy = (entropy / max_entropy).clamp(min=0.0, max=1.0)

        # Handle NaN in sparsity calculation
        valid_norm_entropy = normalized_entropy[~torch.isnan(
            normalized_entropy)]
        if len(valid_norm_entropy) == 0:
            attention_sparsity = 0.0
        else:
            attention_sparsity = (1.0 - valid_norm_entropy).mean().item()

        stats = {
            'mean_attention': attention.mean().item(),
            'max_attention': attention.max().item(),
            'attention_entropy': mean_entropy,
            'attention_sparsity': attention_sparsity
        }

        # Apply slicing test
        # Flatten batch and reshape for test
        # (B*seq_len, k_prefix)
        attention_flat = attention.reshape(-1, attention.size(-1))
        test_stat = self.attention_test(
            attention_flat.unsqueeze(1))  # Add sample dim
        stats['divergence_statistic'] = test_stat.item()

        return stats, attention

    def _compute_embedding_attention(self, image_tokens, text_tokens):
        """Compute proxy attention using image/text embeddings when LM attention is unavailable.

        This is the primary attention mechanism in Stage1 where the LM is frozen and may not
        produce stable attention weights for fused vision-text inputs.
        """
        if text_tokens is None or image_tokens is None:
            raise ValueError(
                "Both text_tokens and image_tokens are required to compute fallback attention")

        query = text_tokens  # (batch, seq_len, hidden_dim)
        key = image_tokens    # (batch, k_prefix, hidden_dim)

        # Ensure dtype compatibility
        if query.dtype != key.dtype:
            key = key.to(query.dtype)

        # Scaled dot-product attention
        dim_scale = np.sqrt(max(query.size(-1), 1))
        scores = torch.matmul(query, key.transpose(-2, -1)) / \
            dim_scale  # (batch, seq_len, k_prefix)

        # Softmax to get attention distribution
        attention = F.softmax(scores, dim=-1)

        # Sanitize any remaining NaN (shouldn't happen, but safety check)
        if torch.isnan(attention).any() or torch.isinf(attention).any():
            attention = torch.nan_to_num(
                attention, nan=1.0 / max(key.size(-1), 1))

        return attention

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

    def visualize_attention_side_by_side(
        self,
        images: torch.Tensor,
        image_tokens: torch.Tensor,
        text_tokens: torch.Tensor,
        attention_weights: torch.Tensor,
        captions: list,
        save_path: str,
        title: str = "Text-Conditioned Attention",
        num_images: int = 3
    ):
        """
        Create a side-by-side visualization of images with their text-conditioned attention heatmaps

        Layout for each image:
        - Left: Original image with caption below
        - Right: Attention heatmap overlay

        Args:
            images: (B, 3, H, W) tensor of images
            image_tokens: (B, k_prefix, D) image token embeddings
            text_tokens: (B, seq_len, D) text token embeddings  
            attention_weights: (B, seq_len, k_prefix) attention from text to image tokens
            captions: list of text captions for each image
            save_path: path to save the grid image
            title: title for the visualization
            num_images: number of images to visualize (default: 3)

        Returns:
            fig: matplotlib figure
        """
        # Select first num_images
        num_images = min(num_images, images.size(0), len(captions))
        images = images[:num_images]
        attention_weights = attention_weights[:num_images]
        captions = captions[:num_images]

        # Convert images to numpy (denormalize if needed)
        images_np = images.cpu().detach().numpy()

        # Denormalize from ImageNet mean/std
        mean = np.array([0.485, 0.456, 0.406]).reshape(1, 3, 1, 1)
        std = np.array([0.229, 0.224, 0.225]).reshape(1, 3, 1, 1)
        images_np = images_np * std + mean
        images_np = np.clip(images_np, 0, 1)

        # Transpose to (B, H, W, C)
        images_np = images_np.transpose(0, 2, 3, 1)

        # Get attention maps: average over text sequence dimension to get (B, k_prefix)
        attention_maps = attention_weights.mean(
            dim=1).cpu().detach().numpy()  # (B, k_prefix)

        # Reshape attention to spatial grid
        k_prefix = attention_maps.shape[1]
        grid_size = int(np.sqrt(k_prefix))

        if grid_size * grid_size != k_prefix:
            # Handle DeiT-Tiny with 197 tokens (196 patches + 1 CLS)
            if k_prefix == 197:
                grid_size = 14  # 14x14 = 196
                attention_maps = attention_maps[:, 1:]  # Remove CLS token
            elif k_prefix == 196:
                grid_size = 14
            else:
                grid_size = int(np.ceil(np.sqrt(k_prefix)))
                pad_size = grid_size * grid_size - k_prefix
                if pad_size > 0:
                    attention_maps = np.pad(
                        attention_maps, ((0, 0), (0, pad_size)), mode='constant')

        attention_maps = attention_maps.reshape(
            num_images, grid_size, grid_size)

        # Upsample attention maps to image resolution
        H, W = images_np.shape[1:3]
        attention_maps_upsampled = []
        for i in range(num_images):
            attn_tensor = torch.from_numpy(
                attention_maps[i]).unsqueeze(0).unsqueeze(0).float()
            attn_upsampled = F.interpolate(attn_tensor, size=(
                H, W), mode='bilinear', align_corners=False)
            attention_maps_upsampled.append(attn_upsampled.squeeze().numpy())

        attention_maps_upsampled = np.array(attention_maps_upsampled)

        # Create figure with num_images rows, 2 columns (original | attention)
        fig, axes = plt.subplots(num_images, 2, figsize=(12, num_images * 4))

        if num_images == 1:
            axes = axes.reshape(1, 2)

        for i in range(num_images):
            # Left column: original image with caption
            axes[i, 0].imshow(images_np[i])
            axes[i, 0].axis('off')

            # Add caption below image
            caption_text = captions[i] if i < len(captions) else ""
            # Wrap text if too long
            import textwrap
            wrapped_caption = "\n".join(textwrap.wrap(caption_text, width=40))
            axes[i, 0].text(0.5, -0.05, wrapped_caption,
                            transform=axes[i, 0].transAxes,
                            ha='center', va='top', fontsize=10,
                            wrap=True, style='italic')

            if i == 0:
                axes[i, 0].set_title(
                    'Original Image + Caption', fontsize=12, fontweight='bold', pad=10)

            # Right column: attention heatmap overlay
            axes[i, 1].imshow(images_np[i])

            # Create colorful attention overlay
            attn = attention_maps_upsampled[i]
            # Normalize attention to [0, 1]
            attn_norm = (attn - attn.min()) / (attn.max() - attn.min() + 1e-8)

            # Apply colormap (use 'jet' for vibrant colors)
            im = axes[i, 1].imshow(attn_norm, cmap='jet', alpha=0.6,
                                   interpolation='bilinear')
            axes[i, 1].axis('off')

            if i == 0:
                axes[i, 1].set_title(
                    'Text-Conditioned Attention', fontsize=12, fontweight='bold', pad=10)

        # Add overall title
        fig.suptitle(title, fontsize=14, fontweight='bold', y=0.98)

        # Add colorbar on the right
        cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
        cbar = fig.colorbar(im, cax=cbar_ax)
        cbar.set_label('Attention Weight', rotation=270, labelpad=20)

        # Use constrained_layout instead of tight_layout to avoid warning
        try:
            plt.tight_layout(rect=[0, 0, 0.9, 0.96])
        except:
            # Fallback if tight_layout fails
            pass

        # Save figure
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()

        self.visualizations.append(save_path)
        print(f"Saved side-by-side attention visualization to {save_path}")

        return fig

    def visualize_attention_grid(self, images, image_tokens, text_tokens,
                                 attention_weights, save_path=None,
                                 title="Text-Conditioned Attention Grid",
                                 num_images=3):
        """
        Create a grid visualization of images with attention heatmap overlays
        Similar to LeJEPA style visualization

        Args:
            images: (B, 3, H, W) tensor of images
            image_tokens: (B, k_prefix, D) image token embeddings
            text_tokens: (B, seq_len, D) text token embeddings  
            attention_weights: (B, seq_len, k_prefix) attention from text to image tokens
            save_path: path to save the grid image
            title: title for the visualization
            num_images: number of images to visualize (default: 3)

        Returns:
            fig: matplotlib figure
        """
        # Select first num_images
        num_images = min(num_images, images.size(0))
        images = images[:num_images]
        attention_weights = attention_weights[:num_images]

        # Convert images to numpy (denormalize if needed)
        images_np = images.cpu().detach().numpy()

        # Denormalize from ImageNet mean/std
        mean = np.array([0.485, 0.456, 0.406]).reshape(1, 3, 1, 1)
        std = np.array([0.229, 0.224, 0.225]).reshape(1, 3, 1, 1)
        images_np = images_np * std + mean
        images_np = np.clip(images_np, 0, 1)

        # Transpose to (B, H, W, C)
        images_np = images_np.transpose(0, 2, 3, 1)

        # Get attention maps: average over text sequence dimension to get (B, k_prefix)
        attention_maps = attention_weights.mean(
            dim=1).cpu().detach().numpy()  # (B, k_prefix)

        # Reshape attention to spatial grid
        # Assuming k_prefix tokens come from a square grid of patches
        k_prefix = attention_maps.shape[1]
        grid_size = int(np.sqrt(k_prefix))

        if grid_size * grid_size != k_prefix:
            # Not a perfect square, use closest approximation
            # For DeiT-Tiny with 197 tokens (196 patches + 1 CLS), use 196 patches
            if k_prefix == 197:
                grid_size = 14  # 14x14 = 196
                attention_maps = attention_maps[:, 1:]  # Remove CLS token
            elif k_prefix == 196:
                grid_size = 14
            else:
                grid_size = int(np.ceil(np.sqrt(k_prefix)))
                # Pad if needed
                pad_size = grid_size * grid_size - k_prefix
                if pad_size > 0:
                    attention_maps = np.pad(
                        attention_maps, ((0, 0), (0, pad_size)), mode='constant')

        attention_maps = attention_maps.reshape(
            num_images, grid_size, grid_size)

        # Upsample attention maps to image resolution
        H, W = images_np.shape[1:3]
        attention_maps_upsampled = []
        for i in range(num_images):
            # Use bilinear interpolation to upsample
            attn_tensor = torch.from_numpy(
                attention_maps[i]).unsqueeze(0).unsqueeze(0).float()
            attn_upsampled = F.interpolate(attn_tensor, size=(
                H, W), mode='bilinear', align_corners=False)
            attention_maps_upsampled.append(attn_upsampled.squeeze().numpy())

        attention_maps_upsampled = np.array(attention_maps_upsampled)

        # Create figure with 2 rows: original images and attention overlays
        fig, axes = plt.subplots(2, num_images, figsize=(num_images * 4, 8))

        if num_images == 1:
            axes = axes.reshape(2, 1)

        for i in range(num_images):
            # Top row: original images
            axes[0, i].imshow(images_np[i])
            axes[0, i].axis('off')
            if i == num_images // 2:
                axes[0, i].set_title(
                    'Original Images', fontsize=12, fontweight='bold')

            # Bottom row: attention overlay
            axes[1, i].imshow(images_np[i])

            # Create colorful attention overlay (like the example image)
            attn = attention_maps_upsampled[i]
            # Normalize attention to [0, 1]
            attn_norm = (attn - attn.min()) / (attn.max() - attn.min() + 1e-8)

            # Apply colormap (use 'jet' for vibrant colors like the example)
            im = axes[1, i].imshow(attn_norm, cmap='jet', alpha=0.6,
                                   interpolation='bilinear')
            axes[1, i].axis('off')
            if i == num_images // 2:
                axes[1, i].set_title(
                    'Text-Conditioned Attention', fontsize=12, fontweight='bold')

        # Add colorbar
        cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.3])
        cbar = fig.colorbar(im, cax=cbar_ax)
        cbar.set_label('Attention Weight', rotation=270, labelpad=20)

        # Overall title
        fig.suptitle(title, fontsize=14, fontweight='bold', y=0.98)

        plt.tight_layout(rect=[0, 0, 0.9, 0.96])

        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Attention grid visualization saved to {save_path}")

        self.visualizations.append({
            'figure': fig,
            'title': title,
            'path': save_path
        })

        plt.close()

        return fig


def create_attention_visualizer(config):
    """Factory function for attention visualizer"""
    return AttentionVisualizer(config)
