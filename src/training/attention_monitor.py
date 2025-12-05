"""
Attention Quality Monitor for MicroVLM-V Training

Monitors attention patterns during training and can pause/stop training
if attention quality degrades significantly.

Key metrics tracked:
1. Attention entropy - should stay in healthy range (not too sharp, not too uniform)
2. Spatial coherence - attention should be spatially coherent, not scattered
3. Background ratio - attention should focus on foreground, not background
4. Consistency - attention patterns should be stable across similar inputs
"""

import torch
import torch.nn.functional as F
import numpy as np
from collections import deque
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Tuple
import warnings


@dataclass
class AttentionHealthMetrics:
    """Container for attention health metrics"""
    step: int = 0
    
    # Entropy metrics (0 = collapsed, 1 = uniform)
    mean_entropy_ratio: float = 0.0  # entropy / max_entropy
    entropy_std: float = 0.0
    
    # Spatial metrics
    spatial_coherence: float = 0.0  # Higher = more spatially coherent
    edge_ratio: float = 0.0  # Ratio of attention on edges vs centers
    
    # Focus metrics
    peak_attention: float = 0.0  # Max attention value
    top_k_concentration: float = 0.0  # Fraction of attention in top-k patches
    
    # Health flags
    is_collapsed: bool = False  # Attention too sharp (single patch)
    is_uniform: bool = False  # Attention too spread out
    is_edge_biased: bool = False  # Attention focusing on edges
    
    # Overall health score (0-1, higher is better)
    health_score: float = 1.0


class AttentionQualityMonitor:
    """
    Monitors attention quality during training and can trigger alerts/stops.
    
    Usage:
        monitor = AttentionQualityMonitor(
            min_health_score=0.3,
            patience=5,
            auto_pause=True
        )
        
        # During training:
        metrics = monitor.check_attention(attention_weights, step)
        if monitor.should_stop():
            print("Attention quality degraded! Stopping training.")
            break
    """
    
    def __init__(
        self,
        # Health thresholds
        min_health_score: float = 0.25,
        target_entropy_ratio: float = 0.35,  # Target: 35% of max entropy
        entropy_tolerance: float = 0.25,  # ±25% tolerance around target
        max_edge_ratio: float = 0.6,  # Max 60% attention on edges
        min_spatial_coherence: float = 0.3,
        
        # Monitoring settings
        history_size: int = 100,  # Steps to track
        patience: int = 10,  # Consecutive bad steps before alert
        
        # Action settings
        auto_pause: bool = True,
        auto_adjust_lr: bool = True,
        lr_reduction_factor: float = 0.5,
        
        # Grid settings (for spatial analysis)
        grid_size: int = 14,
        
        # Logging
        verbose: bool = True
    ):
        self.min_health_score = min_health_score
        self.target_entropy_ratio = target_entropy_ratio
        self.entropy_tolerance = entropy_tolerance
        self.max_edge_ratio = max_edge_ratio
        self.min_spatial_coherence = min_spatial_coherence
        
        self.history_size = history_size
        self.patience = patience
        
        self.auto_pause = auto_pause
        self.auto_adjust_lr = auto_adjust_lr
        self.lr_reduction_factor = lr_reduction_factor
        
        self.grid_size = grid_size
        self.verbose = verbose
        
        # State
        self.metrics_history: deque = deque(maxlen=history_size)
        self.consecutive_bad_steps = 0
        self.should_pause = False
        self.lr_reductions = 0
        self.max_lr_reductions = 3
        
        # Baseline metrics (from early training)
        self.baseline_metrics: Optional[AttentionHealthMetrics] = None
        self.baseline_steps = 500  # Steps to establish baseline
        self.baseline_collected = False
    
    def compute_entropy_metrics(
        self, 
        attention: torch.Tensor
    ) -> Tuple[float, float]:
        """
        Compute attention entropy metrics.
        
        Args:
            attention: (B, seq_len, num_patches) attention weights
            
        Returns:
            mean_entropy_ratio: mean entropy / max_entropy
            entropy_std: std of entropy across tokens
        """
        eps = 1e-8
        num_patches = attention.size(-1)
        max_entropy = np.log(num_patches)
        
        # Compute entropy per token
        entropy = -(attention * torch.log(attention + eps)).sum(dim=-1)  # (B, seq_len)
        
        # Normalize by max entropy
        entropy_ratio = entropy / max_entropy
        
        mean_ratio = entropy_ratio.mean().item()
        std_ratio = entropy_ratio.std().item()
        
        return mean_ratio, std_ratio
    
    def compute_spatial_metrics(
        self, 
        attention: torch.Tensor
    ) -> Tuple[float, float]:
        """
        Compute spatial coherence and edge ratio.
        
        Args:
            attention: (B, seq_len, num_patches)
            
        Returns:
            spatial_coherence: measure of spatial smoothness (0-1)
            edge_ratio: fraction of attention on edge patches
        """
        B, seq_len, num_patches = attention.shape
        H = W = self.grid_size
        
        if num_patches != H * W:
            return 0.5, 0.5  # Can't compute without proper grid
        
        # Reshape to grid
        attn_grid = attention.view(B, seq_len, H, W)
        
        # === Spatial Coherence ===
        # Compute total variation (lower = more coherent)
        tv_h = torch.abs(attn_grid[:, :, 1:, :] - attn_grid[:, :, :-1, :]).mean()
        tv_w = torch.abs(attn_grid[:, :, :, 1:] - attn_grid[:, :, :, :-1]).mean()
        total_variation = (tv_h + tv_w).item()
        
        # Normalize to 0-1 (lower TV = higher coherence)
        # Max possible TV is ~2 (for checkerboard pattern)
        spatial_coherence = max(0, 1 - total_variation / 0.5)
        
        # === Edge Ratio ===
        # Create edge mask (outer 2 rows/cols)
        edge_mask = torch.zeros(H, W, device=attention.device)
        edge_mask[:2, :] = 1  # Top
        edge_mask[-2:, :] = 1  # Bottom
        edge_mask[:, :2] = 1  # Left
        edge_mask[:, -2:] = 1  # Right
        
        center_mask = 1 - edge_mask
        
        # Compute attention on edges vs center
        edge_mask_expanded = edge_mask.view(1, 1, H, W).expand(B, seq_len, H, W)
        center_mask_expanded = center_mask.view(1, 1, H, W).expand(B, seq_len, H, W)
        
        edge_attention = (attn_grid * edge_mask_expanded).sum() / edge_mask_expanded.sum()
        center_attention = (attn_grid * center_mask_expanded).sum() / center_mask_expanded.sum()
        
        # Edge ratio (normalized so that uniform attention gives 0.5)
        total = edge_attention + center_attention + 1e-8
        edge_ratio = (edge_attention / total).item()
        
        return spatial_coherence, edge_ratio
    
    def compute_focus_metrics(
        self, 
        attention: torch.Tensor,
        top_k: int = 10
    ) -> Tuple[float, float]:
        """
        Compute focus/concentration metrics.
        
        Args:
            attention: (B, seq_len, num_patches)
            top_k: number of top patches to consider
            
        Returns:
            peak_attention: max attention value
            top_k_concentration: fraction in top-k patches
        """
        # Peak attention
        peak = attention.max().item()
        
        # Top-k concentration
        top_k_values, _ = attention.topk(min(top_k, attention.size(-1)), dim=-1)
        top_k_sum = top_k_values.sum(dim=-1).mean().item()
        
        return peak, top_k_sum
    
    def compute_health_score(
        self, 
        metrics: AttentionHealthMetrics
    ) -> float:
        """
        Compute overall health score from individual metrics.
        
        Returns:
            score: 0-1, where 1 is perfect health
        """
        scores = []
        
        # Entropy score (penalize both collapse and uniformity)
        entropy_diff = abs(metrics.mean_entropy_ratio - self.target_entropy_ratio)
        entropy_score = max(0, 1 - entropy_diff / self.entropy_tolerance)
        scores.append(entropy_score * 0.35)  # 35% weight
        
        # Edge ratio score (penalize high edge attention)
        edge_score = max(0, 1 - metrics.edge_ratio / self.max_edge_ratio)
        scores.append(edge_score * 0.35)  # 35% weight
        
        # Spatial coherence score
        coherence_score = metrics.spatial_coherence
        scores.append(coherence_score * 0.20)  # 20% weight
        
        # Focus score (penalize extreme concentration)
        # Good range: top-10 patches have 30-70% of attention
        focus_diff = abs(metrics.top_k_concentration - 0.5)
        focus_score = max(0, 1 - focus_diff / 0.3)
        scores.append(focus_score * 0.10)  # 10% weight
        
        return sum(scores)
    
    def check_attention(
        self, 
        attention: torch.Tensor,
        step: int
    ) -> AttentionHealthMetrics:
        """
        Check attention quality and return health metrics.
        
        Args:
            attention: (B, seq_len, num_patches) attention weights
            step: current training step
            
        Returns:
            AttentionHealthMetrics with all computed metrics
        """
        with torch.no_grad():
            metrics = AttentionHealthMetrics(step=step)
            
            # Compute metrics
            metrics.mean_entropy_ratio, metrics.entropy_std = self.compute_entropy_metrics(attention)
            metrics.spatial_coherence, metrics.edge_ratio = self.compute_spatial_metrics(attention)
            metrics.peak_attention, metrics.top_k_concentration = self.compute_focus_metrics(attention)
            
            # Set flags
            metrics.is_collapsed = metrics.mean_entropy_ratio < 0.1
            metrics.is_uniform = metrics.mean_entropy_ratio > 0.8
            metrics.is_edge_biased = metrics.edge_ratio > self.max_edge_ratio
            
            # Compute health score
            metrics.health_score = self.compute_health_score(metrics)
            
            # Update history
            self.metrics_history.append(metrics)
            
            # Establish baseline from early training
            if not self.baseline_collected and step >= self.baseline_steps:
                self._establish_baseline()
            
            # Check for degradation
            self._check_degradation(metrics)
            
            return metrics
    
    def _establish_baseline(self):
        """Establish baseline metrics from early training"""
        if len(self.metrics_history) < 10:
            return
        
        # Use metrics from steps 100-500 as baseline
        early_metrics = [m for m in self.metrics_history if 100 <= m.step <= self.baseline_steps]
        
        if len(early_metrics) < 5:
            return
        
        # Average the early metrics
        self.baseline_metrics = AttentionHealthMetrics(
            step=0,
            mean_entropy_ratio=np.mean([m.mean_entropy_ratio for m in early_metrics]),
            entropy_std=np.mean([m.entropy_std for m in early_metrics]),
            spatial_coherence=np.mean([m.spatial_coherence for m in early_metrics]),
            edge_ratio=np.mean([m.edge_ratio for m in early_metrics]),
            peak_attention=np.mean([m.peak_attention for m in early_metrics]),
            top_k_concentration=np.mean([m.top_k_concentration for m in early_metrics]),
            health_score=np.mean([m.health_score for m in early_metrics])
        )
        
        self.baseline_collected = True
        
        if self.verbose:
            print(f"\n[AttentionMonitor] Baseline established at step {self.baseline_steps}:")
            print(f"  Entropy ratio: {self.baseline_metrics.mean_entropy_ratio:.3f}")
            print(f"  Edge ratio: {self.baseline_metrics.edge_ratio:.3f}")
            print(f"  Health score: {self.baseline_metrics.health_score:.3f}")
    
    def _check_degradation(self, metrics: AttentionHealthMetrics):
        """Check if attention has degraded and update state"""
        is_bad = metrics.health_score < self.min_health_score
        
        # Also check for significant degradation from baseline
        if self.baseline_collected and self.baseline_metrics is not None:
            baseline_score = self.baseline_metrics.health_score
            degradation = (baseline_score - metrics.health_score) / max(baseline_score, 0.1)
            if degradation > 0.4:  # 40% degradation from baseline
                is_bad = True
        
        if is_bad:
            self.consecutive_bad_steps += 1
            
            if self.verbose and self.consecutive_bad_steps == 1:
                print(f"\n⚠️  [AttentionMonitor] Warning at step {metrics.step}:")
                print(f"    Health score: {metrics.health_score:.3f} (min: {self.min_health_score})")
                print(f"    Entropy ratio: {metrics.mean_entropy_ratio:.3f}")
                print(f"    Edge ratio: {metrics.edge_ratio:.3f}")
        else:
            self.consecutive_bad_steps = 0
        
        # Check if we should pause
        if self.consecutive_bad_steps >= self.patience:
            if self.auto_pause:
                self.should_pause = True
                if self.verbose:
                    print(f"\n🛑 [AttentionMonitor] PAUSING TRAINING at step {metrics.step}")
                    print(f"    Attention quality degraded for {self.patience} consecutive checks")
                    print(f"    Health score: {metrics.health_score:.3f}")
    
    def should_stop(self) -> bool:
        """Check if training should be stopped/paused"""
        return self.should_pause
    
    def get_lr_adjustment(self) -> Optional[float]:
        """
        Get learning rate adjustment factor if needed.
        
        Returns:
            Adjustment factor (e.g., 0.5 to halve LR) or None
        """
        if not self.auto_adjust_lr:
            return None
        
        # If attention is degrading, suggest LR reduction
        if self.consecutive_bad_steps >= self.patience // 2:
            if self.lr_reductions < self.max_lr_reductions:
                self.lr_reductions += 1
                self.consecutive_bad_steps = 0  # Reset counter
                return self.lr_reduction_factor
        
        return None
    
    def reset_pause(self):
        """Reset pause flag (e.g., after LR adjustment)"""
        self.should_pause = False
        self.consecutive_bad_steps = 0
    
    def get_wandb_metrics(self, metrics: AttentionHealthMetrics) -> Dict[str, float]:
        """Convert metrics to WandB-ready dictionary"""
        return {
            'attention_health/entropy_ratio': metrics.mean_entropy_ratio,
            'attention_health/entropy_std': metrics.entropy_std,
            'attention_health/spatial_coherence': metrics.spatial_coherence,
            'attention_health/edge_ratio': metrics.edge_ratio,
            'attention_health/peak_attention': metrics.peak_attention,
            'attention_health/top_k_concentration': metrics.top_k_concentration,
            'attention_health/health_score': metrics.health_score,
            'attention_health/is_collapsed': float(metrics.is_collapsed),
            'attention_health/is_edge_biased': float(metrics.is_edge_biased),
        }
    
    def get_summary(self) -> Dict[str, float]:
        """Get summary statistics from history"""
        if len(self.metrics_history) == 0:
            return {}
        
        recent = list(self.metrics_history)[-20:]  # Last 20 checks
        
        return {
            'mean_health_score': np.mean([m.health_score for m in recent]),
            'min_health_score': np.min([m.health_score for m in recent]),
            'mean_entropy_ratio': np.mean([m.mean_entropy_ratio for m in recent]),
            'mean_edge_ratio': np.mean([m.edge_ratio for m in recent]),
            'degradation_events': self.lr_reductions,
        }
