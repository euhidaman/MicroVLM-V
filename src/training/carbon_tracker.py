"""
Carbon Emissions and Compute Tracking for MicroVLM-V Training

This module provides comprehensive tracking of:
- Carbon emissions (via CodeCarbon)
- Energy consumption
- GPU utilization and memory
- FLOPs computation

Designed to be DDP-compatible (rank-0 only tracking)
"""

import time
import torch
import os
from typing import Dict, Optional, Any, Tuple
from dataclasses import dataclass, field
from contextlib import contextmanager
from pathlib import Path

# Lazy imports for optional dependencies
_codecarbon_available = None
_pynvml_available = None
_fvcore_available = None


def _check_codecarbon():
    global _codecarbon_available
    if _codecarbon_available is None:
        try:
            from codecarbon import EmissionsTracker
            _codecarbon_available = True
        except ImportError:
            _codecarbon_available = False
    return _codecarbon_available


def _check_pynvml():
    global _pynvml_available
    if _pynvml_available is None:
        try:
            import pynvml
            pynvml.nvmlInit()
            _pynvml_available = True
        except (ImportError, Exception):
            _pynvml_available = False
    return _pynvml_available


def _check_fvcore():
    global _fvcore_available
    if _fvcore_available is None:
        try:
            from fvcore.nn import FlopCountAnalysis
            _fvcore_available = True
        except ImportError:
            _fvcore_available = False
    return _fvcore_available


@dataclass
class ComputeMetrics:
    """Container for compute metrics at a given point in time"""
    timestamp: float = 0.0
    step: int = 0
    
    # Carbon metrics
    emissions_kg: float = 0.0
    energy_consumed_kwh: float = 0.0
    
    # GPU metrics (per-GPU)
    gpu_utilization: Dict[int, float] = field(default_factory=dict)
    gpu_memory_used_gb: Dict[int, float] = field(default_factory=dict)
    gpu_memory_total_gb: Dict[int, float] = field(default_factory=dict)
    gpu_power_watts: Dict[int, float] = field(default_factory=dict)
    gpu_temperature_c: Dict[int, float] = field(default_factory=dict)
    
    # FLOPs metrics
    batch_flops: float = 0.0
    cumulative_flops: float = 0.0
    
    # Timing metrics
    batch_time_seconds: float = 0.0
    epoch_time_seconds: float = 0.0
    total_time_seconds: float = 0.0
    
    # Derived metrics
    flops_per_second: float = 0.0
    samples_per_second: float = 0.0


class CarbonComputeTracker:
    """
    Comprehensive tracker for carbon emissions, energy, GPU, and FLOPs metrics.
    
    Features:
    - CodeCarbon integration for emissions tracking
    - NVIDIA pynvml for GPU metrics
    - fvcore for FLOPs computation
    - DDP-compatible (only rank-0 tracks)
    - WandB integration ready
    
    Usage:
        tracker = CarbonComputeTracker(
            is_main_process=True,
            output_dir="./logs",
            project_name="microvlm-v-training"
        )
        
        tracker.start_training()
        
        for epoch in range(num_epochs):
            tracker.start_epoch()
            for batch in dataloader:
                tracker.start_batch()
                # ... training code ...
                metrics = tracker.end_batch(batch_size=batch_size)
            tracker.end_epoch()
        
        final_metrics = tracker.end_training()
    """
    
    def __init__(
        self,
        is_main_process: bool = True,
        output_dir: Optional[str] = None,
        project_name: str = "microvlm-v",
        country_iso_code: str = "USA",  # For CodeCarbon
        track_carbon: bool = True,
        track_gpu: bool = True,
        track_flops: bool = True,
        model: Optional[torch.nn.Module] = None,
        sample_input: Optional[Dict[str, torch.Tensor]] = None
    ):
        self.is_main_process = is_main_process
        self.output_dir = Path(output_dir) if output_dir else Path("./carbon_logs")
        self.project_name = project_name
        self.country_iso_code = country_iso_code
        
        # Feature flags
        self.track_carbon = track_carbon and is_main_process
        self.track_gpu = track_gpu and is_main_process
        self.track_flops = track_flops and is_main_process
        
        # State
        self.is_tracking = False
        self.current_epoch = 0
        self.current_step = 0
        
        # Timing
        self._training_start_time = None
        self._epoch_start_time = None
        self._batch_start_time = None
        
        # Cumulative metrics
        self._cumulative_flops = 0.0
        self._cumulative_emissions = 0.0
        self._cumulative_energy = 0.0
        self._total_samples = 0
        
        # CodeCarbon tracker
        self._emissions_tracker = None
        
        # FLOPs per forward pass (computed once)
        self._flops_per_forward: Optional[float] = None
        self._model = model
        self._sample_input = sample_input
        
        # GPU handles
        self._gpu_handles = []
        self._num_gpus = 0
        
        # Initialize components
        if self.is_main_process:
            self._init_components()
    
    def _init_components(self):
        """Initialize tracking components"""
        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize CodeCarbon
        if self.track_carbon and _check_codecarbon():
            try:
                from codecarbon import EmissionsTracker
                self._emissions_tracker = EmissionsTracker(
                    project_name=self.project_name,
                    output_dir=str(self.output_dir),
                    output_file="emissions.csv",
                    country_iso_code=self.country_iso_code,
                    save_to_file=True,
                    tracking_mode="process"
                )
                print(f"[CarbonTracker] CodeCarbon initialized (country: {self.country_iso_code})")
            except Exception as e:
                print(f"[CarbonTracker] Warning: CodeCarbon init failed: {e}")
                self._emissions_tracker = None
        
        # Initialize pynvml for GPU metrics
        if self.track_gpu and _check_pynvml():
            try:
                import pynvml
                self._num_gpus = pynvml.nvmlDeviceGetCount()
                self._gpu_handles = [
                    pynvml.nvmlDeviceGetHandleByIndex(i) 
                    for i in range(self._num_gpus)
                ]
                print(f"[CarbonTracker] pynvml initialized ({self._num_gpus} GPUs)")
            except Exception as e:
                print(f"[CarbonTracker] Warning: pynvml init failed: {e}")
                self._gpu_handles = []
        
        # Compute FLOPs if model provided
        if self.track_flops and self._model is not None:
            self._compute_model_flops()
    
    def _compute_model_flops(self):
        """Compute FLOPs for a single forward pass"""
        if not _check_fvcore():
            print("[CarbonTracker] Warning: fvcore not available, FLOPs tracking disabled")
            self._flops_per_forward = None
            return
        
        if self._sample_input is None:
            print("[CarbonTracker] Warning: No sample input provided, FLOPs tracking disabled")
            return
        
        try:
            from fvcore.nn import FlopCountAnalysis
            
            model = self._model
            if hasattr(model, 'module'):
                model = model.module
            
            model.eval()
            with torch.no_grad():
                # Create tuple of inputs for FlopCountAnalysis
                # This depends on model signature
                flop_analysis = FlopCountAnalysis(model, self._sample_input)
                self._flops_per_forward = flop_analysis.total()
            
            model.train()
            
            # Format nicely
            flops_str = self._format_flops(self._flops_per_forward)
            print(f"[CarbonTracker] Model FLOPs per forward: {flops_str}")
            
        except Exception as e:
            print(f"[CarbonTracker] Warning: FLOPs computation failed: {e}")
            self._flops_per_forward = None
    
    def set_model_flops(self, flops_per_forward: float):
        """Manually set FLOPs per forward pass"""
        self._flops_per_forward = flops_per_forward
        if self.is_main_process:
            flops_str = self._format_flops(flops_per_forward)
            print(f"[CarbonTracker] Model FLOPs set manually: {flops_str}")
    
    def estimate_flops_from_params(self, num_params: int, seq_len: int = 512):
        """
        Estimate FLOPs using 6 * num_params * seq_len rule of thumb
        This is a rough approximation for transformer models
        """
        estimated_flops = 6 * num_params * seq_len
        self._flops_per_forward = estimated_flops
        if self.is_main_process:
            flops_str = self._format_flops(estimated_flops)
            print(f"[CarbonTracker] Estimated FLOPs from params: {flops_str}")
        return estimated_flops
    
    @staticmethod
    def _format_flops(flops: float) -> str:
        """Format FLOPs with appropriate unit"""
        if flops >= 1e15:
            return f"{flops / 1e15:.2f} PFLOPs"
        elif flops >= 1e12:
            return f"{flops / 1e12:.2f} TFLOPs"
        elif flops >= 1e9:
            return f"{flops / 1e9:.2f} GFLOPs"
        elif flops >= 1e6:
            return f"{flops / 1e6:.2f} MFLOPs"
        else:
            return f"{flops:.0f} FLOPs"
    
    def start_training(self):
        """Start training-level tracking"""
        if not self.is_main_process:
            return
        
        self._training_start_time = time.time()
        self.is_tracking = True
        
        # Start CodeCarbon tracker
        if self._emissions_tracker is not None:
            self._emissions_tracker.start()
        
        print(f"[CarbonTracker] Training tracking started")
    
    def start_epoch(self, epoch: int = None):
        """Start epoch-level tracking"""
        if not self.is_main_process:
            return
        
        self._epoch_start_time = time.time()
        if epoch is not None:
            self.current_epoch = epoch
    
    def start_batch(self):
        """Start batch-level timing"""
        if not self.is_main_process:
            return
        
        self._batch_start_time = time.time()
    
    def end_batch(self, batch_size: int = 1) -> ComputeMetrics:
        """
        End batch tracking and collect metrics
        
        Args:
            batch_size: Number of samples in this batch
            
        Returns:
            ComputeMetrics with current batch stats
        """
        if not self.is_main_process:
            return ComputeMetrics()
        
        self.current_step += 1
        self._total_samples += batch_size
        
        metrics = ComputeMetrics(
            timestamp=time.time(),
            step=self.current_step
        )
        
        # Batch timing
        if self._batch_start_time is not None:
            metrics.batch_time_seconds = time.time() - self._batch_start_time
            if metrics.batch_time_seconds > 0:
                metrics.samples_per_second = batch_size / metrics.batch_time_seconds
        
        # Total time
        if self._training_start_time is not None:
            metrics.total_time_seconds = time.time() - self._training_start_time
        
        # Epoch time
        if self._epoch_start_time is not None:
            metrics.epoch_time_seconds = time.time() - self._epoch_start_time
        
        # FLOPs (forward + backward = 3x forward)
        if self._flops_per_forward is not None:
            # Forward + backward (gradient) = 3x forward FLOPs
            batch_flops = self._flops_per_forward * batch_size * 3
            self._cumulative_flops += batch_flops
            metrics.batch_flops = batch_flops
            metrics.cumulative_flops = self._cumulative_flops
            
            if metrics.batch_time_seconds > 0:
                metrics.flops_per_second = batch_flops / metrics.batch_time_seconds
        
        # GPU metrics
        if self.track_gpu and self._gpu_handles:
            self._collect_gpu_metrics(metrics)
        
        # Carbon metrics (sample periodically to reduce overhead)
        if self.track_carbon and self._emissions_tracker is not None:
            if self.current_step % 10 == 0:  # Sample every 10 steps
                self._collect_carbon_metrics(metrics)
        
        return metrics
    
    def _collect_gpu_metrics(self, metrics: ComputeMetrics):
        """Collect GPU metrics via pynvml"""
        if not _check_pynvml():
            return
        
        try:
            import pynvml
            
            for i, handle in enumerate(self._gpu_handles):
                # Utilization
                util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                metrics.gpu_utilization[i] = util.gpu
                
                # Memory
                mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                metrics.gpu_memory_used_gb[i] = mem_info.used / (1024**3)
                metrics.gpu_memory_total_gb[i] = mem_info.total / (1024**3)
                
                # Power
                try:
                    power = pynvml.nvmlDeviceGetPowerUsage(handle)
                    metrics.gpu_power_watts[i] = power / 1000.0  # mW to W
                except pynvml.NVMLError:
                    pass
                
                # Temperature
                try:
                    temp = pynvml.nvmlDeviceGetTemperature(
                        handle, pynvml.NVML_TEMPERATURE_GPU
                    )
                    metrics.gpu_temperature_c[i] = temp
                except pynvml.NVMLError:
                    pass
                    
        except Exception as e:
            # Silently fail - GPU metrics are optional
            pass
    
    def _collect_carbon_metrics(self, metrics: ComputeMetrics):
        """Collect carbon emission metrics from CodeCarbon"""
        if self._emissions_tracker is None:
            return
        
        try:
            # Get current emissions data
            # Note: CodeCarbon updates internally, we read accumulated values
            if hasattr(self._emissions_tracker, '_emissions'):
                emissions_data = self._emissions_tracker._emissions
                if emissions_data is not None:
                    metrics.emissions_kg = getattr(emissions_data, 'emissions', 0.0)
                    metrics.energy_consumed_kwh = getattr(emissions_data, 'energy_consumed', 0.0)
        except Exception:
            pass
    
    def end_epoch(self) -> ComputeMetrics:
        """End epoch tracking and return epoch summary"""
        if not self.is_main_process:
            return ComputeMetrics()
        
        metrics = ComputeMetrics(
            timestamp=time.time(),
            step=self.current_step
        )
        
        if self._epoch_start_time is not None:
            metrics.epoch_time_seconds = time.time() - self._epoch_start_time
        
        if self._training_start_time is not None:
            metrics.total_time_seconds = time.time() - self._training_start_time
        
        metrics.cumulative_flops = self._cumulative_flops
        
        # Collect final GPU state
        if self.track_gpu and self._gpu_handles:
            self._collect_gpu_metrics(metrics)
        
        # Collect carbon metrics
        if self.track_carbon and self._emissions_tracker is not None:
            self._collect_carbon_metrics(metrics)
        
        self.current_epoch += 1
        
        return metrics
    
    def end_training(self) -> ComputeMetrics:
        """
        End training tracking and return final summary
        
        Returns:
            ComputeMetrics with final training statistics
        """
        if not self.is_main_process:
            return ComputeMetrics()
        
        metrics = ComputeMetrics(
            timestamp=time.time(),
            step=self.current_step
        )
        
        if self._training_start_time is not None:
            metrics.total_time_seconds = time.time() - self._training_start_time
        
        metrics.cumulative_flops = self._cumulative_flops
        
        # Stop CodeCarbon and get final emissions
        if self._emissions_tracker is not None:
            try:
                final_emissions = self._emissions_tracker.stop()
                metrics.emissions_kg = final_emissions if final_emissions else 0.0
                
                # Get energy from tracker data
                if hasattr(self._emissions_tracker, '_total_energy'):
                    metrics.energy_consumed_kwh = self._emissions_tracker._total_energy.kWh
                    
            except Exception as e:
                print(f"[CarbonTracker] Warning: Error stopping emissions tracker: {e}")
        
        # Final GPU metrics
        if self.track_gpu and self._gpu_handles:
            self._collect_gpu_metrics(metrics)
        
        # Compute throughput
        if metrics.total_time_seconds > 0:
            metrics.samples_per_second = self._total_samples / metrics.total_time_seconds
            if self._cumulative_flops > 0:
                metrics.flops_per_second = self._cumulative_flops / metrics.total_time_seconds
        
        self.is_tracking = False
        
        # Print summary
        self._print_summary(metrics)
        
        return metrics
    
    def _print_summary(self, metrics: ComputeMetrics):
        """Print training summary to console"""
        print("\n" + "="*60)
        print("CARBON & COMPUTE TRACKING SUMMARY")
        print("="*60)
        
        hours = metrics.total_time_seconds / 3600
        print(f"Total Training Time:     {hours:.2f} hours ({metrics.total_time_seconds:.0f} seconds)")
        print(f"Total Samples Processed: {self._total_samples:,}")
        print(f"Throughput:              {metrics.samples_per_second:.2f} samples/sec")
        
        print("\n--- FLOPs ---")
        print(f"Total FLOPs:             {self._format_flops(metrics.cumulative_flops)}")
        print(f"FLOPs/second:            {self._format_flops(metrics.flops_per_second)}/s")
        
        if metrics.emissions_kg > 0:
            print("\n--- Carbon Emissions ---")
            print(f"Total Emissions:         {metrics.emissions_kg * 1000:.4f} g CO2eq")
            print(f"Energy Consumed:         {metrics.energy_consumed_kwh:.4f} kWh")
            
            # Equivalents for context
            if metrics.emissions_kg > 0:
                km_car = metrics.emissions_kg / 0.21  # ~210g CO2/km for average car
                print(f"Equivalent Car Distance: {km_car:.2f} km")
        
        if metrics.gpu_utilization:
            print("\n--- GPU Metrics (Final) ---")
            for gpu_id in metrics.gpu_utilization:
                util = metrics.gpu_utilization.get(gpu_id, 0)
                mem_used = metrics.gpu_memory_used_gb.get(gpu_id, 0)
                mem_total = metrics.gpu_memory_total_gb.get(gpu_id, 0)
                power = metrics.gpu_power_watts.get(gpu_id, 0)
                temp = metrics.gpu_temperature_c.get(gpu_id, 0)
                print(f"  GPU {gpu_id}: {util:.1f}% util, {mem_used:.1f}/{mem_total:.1f} GB, {power:.0f}W, {temp}°C")
        
        print("="*60 + "\n")
    
    def get_wandb_metrics(self, metrics: ComputeMetrics, prefix: str = "compute") -> Dict[str, Any]:
        """
        Convert ComputeMetrics to WandB-ready dictionary
        
        Args:
            metrics: ComputeMetrics object
            prefix: Metric prefix for WandB grouping
            
        Returns:
            Dictionary ready for wandb.log()
        """
        wandb_dict = {
            f"{prefix}/step": metrics.step,
            f"{prefix}/batch_time_seconds": metrics.batch_time_seconds,
            f"{prefix}/samples_per_second": metrics.samples_per_second,
        }
        
        # FLOPs
        if metrics.batch_flops > 0:
            wandb_dict[f"{prefix}/batch_flops"] = metrics.batch_flops
            wandb_dict[f"{prefix}/cumulative_flops"] = metrics.cumulative_flops
            wandb_dict[f"{prefix}/flops_per_second"] = metrics.flops_per_second
            # TFLOPs for easier reading
            wandb_dict[f"{prefix}/batch_tflops"] = metrics.batch_flops / 1e12
            wandb_dict[f"{prefix}/cumulative_tflops"] = metrics.cumulative_flops / 1e12
        
        # Carbon
        if metrics.emissions_kg > 0:
            wandb_dict[f"carbon/emissions_g"] = metrics.emissions_kg * 1000
            wandb_dict[f"carbon/emissions_kg"] = metrics.emissions_kg
            wandb_dict[f"carbon/energy_kwh"] = metrics.energy_consumed_kwh
        
        # GPU metrics (aggregate across GPUs)
        if metrics.gpu_utilization:
            avg_util = sum(metrics.gpu_utilization.values()) / len(metrics.gpu_utilization)
            total_mem_used = sum(metrics.gpu_memory_used_gb.values())
            total_power = sum(metrics.gpu_power_watts.values()) if metrics.gpu_power_watts else 0
            avg_temp = sum(metrics.gpu_temperature_c.values()) / len(metrics.gpu_temperature_c) if metrics.gpu_temperature_c else 0
            
            wandb_dict[f"gpu/avg_utilization"] = avg_util
            wandb_dict[f"gpu/total_memory_gb"] = total_mem_used
            wandb_dict[f"gpu/total_power_watts"] = total_power
            wandb_dict[f"gpu/avg_temperature_c"] = avg_temp
            
            # Per-GPU metrics
            for gpu_id in metrics.gpu_utilization:
                wandb_dict[f"gpu/gpu{gpu_id}_utilization"] = metrics.gpu_utilization[gpu_id]
                wandb_dict[f"gpu/gpu{gpu_id}_memory_gb"] = metrics.gpu_memory_used_gb.get(gpu_id, 0)
                if gpu_id in metrics.gpu_power_watts:
                    wandb_dict[f"gpu/gpu{gpu_id}_power_watts"] = metrics.gpu_power_watts[gpu_id]
        
        # Timing
        wandb_dict[f"{prefix}/total_time_hours"] = metrics.total_time_seconds / 3600
        if metrics.epoch_time_seconds > 0:
            wandb_dict[f"{prefix}/epoch_time_minutes"] = metrics.epoch_time_seconds / 60
        
        return wandb_dict
    
    def cleanup(self):
        """Clean up resources"""
        if self._emissions_tracker is not None:
            try:
                if self.is_tracking:
                    self._emissions_tracker.stop()
            except Exception:
                pass
        
        # Shutdown pynvml
        if _check_pynvml():
            try:
                import pynvml
                pynvml.nvmlShutdown()
            except Exception:
                pass


# Convenience function for quick FLOPs estimation
def estimate_model_flops(model: torch.nn.Module, input_shape: Tuple[int, ...] = None) -> float:
    """
    Estimate FLOPs for a model using parameter count approximation
    
    For transformer-based models:
    FLOPs ≈ 6 * num_params * seq_len (forward + backward)
    
    Args:
        model: PyTorch model
        input_shape: Optional input shape for more accurate estimation
        
    Returns:
        Estimated FLOPs per forward pass
    """
    num_params = sum(p.numel() for p in model.parameters())
    
    # Default sequence length for VLM
    seq_len = input_shape[1] if input_shape and len(input_shape) > 1 else 512
    
    # 2x for forward (rough approximation)
    estimated_flops = 2 * num_params * seq_len
    
    return estimated_flops
