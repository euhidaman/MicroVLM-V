"""
Loss Smoothing Utilities for Training Visualization

Provides sliding window smoothing, EMA smoothing, and running minimum tracking
for cleaner loss visualization without affecting training dynamics.
"""

from collections import deque
from typing import Optional, Dict, Tuple
import math


class LossSmoother:
    """
    Track and smooth loss values for visualization.
    
    Supports:
    - Sliding window average
    - Exponential moving average (EMA)
    - Running minimum (monotonic decreasing envelope)
    
    None of these affect training - they are purely for visualization/logging.
    """
    
    def __init__(
        self,
        window_size: int = 50,
        ema_alpha: Optional[float] = 0.1,
        track_components: bool = True
    ):
        """
        Initialize loss smoother.
        
        Args:
            window_size: Size of sliding window for averaging (default: 50)
            ema_alpha: EMA decay factor. Smaller = smoother. Set to None to disable.
                       EMA_t = alpha * loss_t + (1 - alpha) * EMA_{t-1}
            track_components: Whether to track individual loss components
        """
        self.window_size = max(1, window_size)
        self.ema_alpha = ema_alpha
        self.track_components = track_components
        
        # Main loss tracking
        self._window: deque = deque(maxlen=self.window_size)
        self._running_min: float = float('inf')
        self._ema: Optional[float] = None
        self._step_count: int = 0
        
        # Component loss tracking (for ITC, ITM, alignment, etc.)
        self._component_windows: Dict[str, deque] = {}
        self._component_mins: Dict[str, float] = {}
        self._component_emas: Dict[str, Optional[float]] = {}
        
    def update(self, loss: float, components: Optional[Dict[str, float]] = None) -> Dict[str, float]:
        """
        Update smoother with new loss value.
        
        Args:
            loss: Current raw loss value
            components: Optional dict of component losses (e.g., {'itc': 0.5, 'itm': 0.3})
            
        Returns:
            Dict with all smoothed values:
            - raw_loss: The input loss value
            - smoothed_loss: Sliding window average
            - running_min_loss: Monotonic decreasing minimum
            - ema_loss: Exponential moving average (if enabled)
            - <component>_smoothed: Smoothed component losses
            - <component>_min: Running min for components
        """
        self._step_count += 1
        
        # Skip NaN/Inf values
        if math.isnan(loss) or math.isinf(loss):
            return self._get_current_values()
        
        # Update main loss tracking
        self._window.append(loss)
        self._running_min = min(self._running_min, loss)
        
        # Update EMA
        if self.ema_alpha is not None:
            if self._ema is None:
                self._ema = loss
            else:
                self._ema = self.ema_alpha * loss + (1 - self.ema_alpha) * self._ema
        
        # Update component tracking
        if self.track_components and components:
            for name, value in components.items():
                if math.isnan(value) or math.isinf(value):
                    continue
                    
                # Initialize tracking for new components
                if name not in self._component_windows:
                    self._component_windows[name] = deque(maxlen=self.window_size)
                    self._component_mins[name] = float('inf')
                    self._component_emas[name] = None
                
                # Update window
                self._component_windows[name].append(value)
                
                # Update running min
                self._component_mins[name] = min(self._component_mins[name], value)
                
                # Update EMA
                if self.ema_alpha is not None:
                    if self._component_emas[name] is None:
                        self._component_emas[name] = value
                    else:
                        self._component_emas[name] = (
                            self.ema_alpha * value + 
                            (1 - self.ema_alpha) * self._component_emas[name]
                        )
        
        return self._get_current_values()
    
    def _get_current_values(self) -> Dict[str, float]:
        """Get current smoothed values for all tracked losses."""
        result = {
            'raw_loss': self._window[-1] if self._window else 0.0,
            'smoothed_loss': self.get_smoothed(),
            'running_min_loss': self.get_running_min(),
        }
        
        if self.ema_alpha is not None:
            result['ema_loss'] = self.get_ema()
        
        # Add component values
        for name in self._component_windows:
            result[f'{name}_smoothed'] = self._get_component_smoothed(name)
            result[f'{name}_min'] = self._component_mins.get(name, 0.0)
            if self.ema_alpha is not None:
                result[f'{name}_ema'] = self._component_emas.get(name, 0.0)
        
        return result
    
    def get_smoothed(self) -> float:
        """Get sliding window average of loss."""
        if not self._window:
            return 0.0
        return sum(self._window) / len(self._window)
    
    def get_running_min(self) -> float:
        """Get running minimum loss (monotonic decreasing envelope)."""
        if self._running_min == float('inf'):
            return 0.0
        return self._running_min
    
    def get_ema(self) -> float:
        """Get exponential moving average of loss."""
        if self._ema is None:
            return 0.0
        return self._ema
    
    def _get_component_smoothed(self, name: str) -> float:
        """Get sliding window average for a component loss."""
        window = self._component_windows.get(name)
        if not window:
            return 0.0
        return sum(window) / len(window)
    
    def get_wandb_metrics(self, prefix: str = 'train') -> Dict[str, float]:
        """
        Get metrics formatted for W&B logging.
        
        Args:
            prefix: Metric prefix (e.g., 'train' -> 'train/smoothed_loss')
            
        Returns:
            Dict ready for wandb.log()
        """
        metrics = {}
        
        # Main loss metrics
        if self._window:
            metrics[f'{prefix}/smoothed_loss'] = self.get_smoothed()
            metrics[f'{prefix}/running_min_loss'] = self.get_running_min()
            
            if self.ema_alpha is not None and self._ema is not None:
                metrics[f'{prefix}/ema_loss'] = self.get_ema()
        
        # Component metrics
        for name in self._component_windows:
            metrics[f'{prefix}/{name}_smoothed'] = self._get_component_smoothed(name)
            metrics[f'{prefix}/{name}_min'] = self._component_mins.get(name, 0.0)
            
            if self.ema_alpha is not None:
                ema_val = self._component_emas.get(name)
                if ema_val is not None:
                    metrics[f'{prefix}/{name}_ema'] = ema_val
        
        return metrics
    
    def get_stats(self) -> Dict[str, float]:
        """Get summary statistics for logging/printing."""
        return {
            'steps': self._step_count,
            'window_size': self.window_size,
            'ema_alpha': self.ema_alpha,
            'current_smoothed': self.get_smoothed(),
            'current_min': self.get_running_min(),
            'current_ema': self.get_ema() if self.ema_alpha else None,
            'window_fill': len(self._window) / self.window_size,
        }
    
    def reset(self):
        """Reset all tracking (e.g., between epochs if desired)."""
        self._window.clear()
        self._running_min = float('inf')
        self._ema = None
        self._step_count = 0
        self._component_windows.clear()
        self._component_mins.clear()
        self._component_emas.clear()


def create_loss_smoother(
    window_size: int = 50,
    ema_alpha: Optional[float] = 0.1,
    enabled: bool = True
) -> Optional[LossSmoother]:
    """
    Factory function to create a loss smoother.
    
    Args:
        window_size: Sliding window size
        ema_alpha: EMA decay factor (None to disable)
        enabled: If False, returns None
        
    Returns:
        LossSmoother instance or None
    """
    if not enabled:
        return None
    return LossSmoother(window_size=window_size, ema_alpha=ema_alpha)
