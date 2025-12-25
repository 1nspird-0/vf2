"""GPU utilities for optimized training.

Features:
- GPU memory monitoring and optimization
- Mixed precision setup
- Multi-GPU support
- Performance benchmarking
"""

import os
import sys
from typing import Optional, Tuple, Dict, Any

import torch
import torch.nn as nn


def setup_gpu(
    device_id: int = 0,
    seed: int = 42,
    cudnn_benchmark: bool = True,
    deterministic: bool = False,
    tf32: bool = True,
) -> torch.device:
    """Setup GPU with optimal settings.
    
    Args:
        device_id: GPU device ID to use.
        seed: Random seed for reproducibility.
        cudnn_benchmark: Enable cuDNN benchmark mode.
        deterministic: Enable deterministic algorithms.
        tf32: Enable TensorFloat-32 for Ampere+ GPUs.
        
    Returns:
        Configured torch device.
    """
    if not torch.cuda.is_available():
        print("CUDA not available, using CPU")
        return torch.device("cpu")
    
    device = torch.device(f"cuda:{device_id}")
    torch.cuda.set_device(device)
    
    # Set random seeds
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    # cuDNN settings
    torch.backends.cudnn.benchmark = cudnn_benchmark
    torch.backends.cudnn.deterministic = deterministic
    
    # TensorFloat-32 for Ampere GPUs (RTX 30xx, 40xx, 50xx)
    if tf32 and hasattr(torch.backends.cuda, 'matmul'):
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    
    return device


def get_gpu_info() -> Dict[str, Any]:
    """Get detailed GPU information.
    
    Returns:
        Dict with GPU info.
    """
    if not torch.cuda.is_available():
        return {"available": False}
    
    props = torch.cuda.get_device_properties(0)
    
    return {
        "available": True,
        "name": props.name,
        "compute_capability": f"{props.major}.{props.minor}",
        "total_memory_gb": props.total_memory / (1024**3),
        "multi_processor_count": props.multi_processor_count,
        "cuda_version": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version(),
        "device_count": torch.cuda.device_count(),
    }


def get_memory_stats() -> Dict[str, float]:
    """Get current GPU memory statistics.
    
    Returns:
        Dict with memory stats in GB.
    """
    if not torch.cuda.is_available():
        return {}
    
    allocated = torch.cuda.memory_allocated() / (1024**3)
    reserved = torch.cuda.memory_reserved() / (1024**3)
    max_allocated = torch.cuda.max_memory_allocated() / (1024**3)
    
    props = torch.cuda.get_device_properties(0)
    total = props.total_memory / (1024**3)
    
    return {
        "allocated_gb": allocated,
        "reserved_gb": reserved,
        "max_allocated_gb": max_allocated,
        "total_gb": total,
        "free_gb": total - reserved,
        "utilization_pct": (allocated / total) * 100,
    }


def estimate_batch_size(
    model: nn.Module,
    sample_input: Dict[str, torch.Tensor],
    target_memory_usage: float = 0.8,
    start_batch: int = 8,
    max_batch: int = 512,
) -> int:
    """Estimate optimal batch size for available GPU memory.
    
    Args:
        model: The model to estimate for.
        sample_input: Sample input dict (single sample).
        target_memory_usage: Target GPU memory usage (0-1).
        start_batch: Starting batch size for search.
        max_batch: Maximum batch size to try.
        
    Returns:
        Estimated optimal batch size.
    """
    if not torch.cuda.is_available():
        return start_batch
    
    device = next(model.parameters()).device
    model.eval()
    
    # Clear cache
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    
    total_memory = torch.cuda.get_device_properties(0).total_memory
    target_bytes = total_memory * target_memory_usage
    
    # Test with start_batch
    batch_input = {k: v.expand(start_batch, *v.shape[1:]) for k, v in sample_input.items()}
    
    try:
        with torch.no_grad():
            _ = model(batch_input)
        
        torch.cuda.synchronize()
        memory_per_batch = torch.cuda.max_memory_allocated() / start_batch
        
        estimated_batch = int(target_bytes / memory_per_batch)
        return min(max(estimated_batch, start_batch), max_batch)
        
    except RuntimeError:
        return start_batch // 2


def clear_memory():
    """Clear GPU memory cache."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def optimize_model_for_inference(model: nn.Module) -> nn.Module:
    """Optimize model for inference.
    
    Args:
        model: Model to optimize.
        
    Returns:
        Optimized model.
    """
    model.eval()
    
    # Disable gradient computation
    for param in model.parameters():
        param.requires_grad = False
    
    # Try to compile with torch.compile (PyTorch 2.0+)
    if hasattr(torch, 'compile'):
        try:
            model = torch.compile(model, mode="reduce-overhead")
        except Exception:
            pass
    
    return model


def benchmark_model(
    model: nn.Module,
    sample_input: Dict[str, torch.Tensor],
    batch_size: int = 32,
    num_warmup: int = 10,
    num_iterations: int = 100,
) -> Dict[str, float]:
    """Benchmark model inference speed.
    
    Args:
        model: Model to benchmark.
        sample_input: Sample input dict.
        batch_size: Batch size to test.
        num_warmup: Number of warmup iterations.
        num_iterations: Number of timed iterations.
        
    Returns:
        Dict with timing stats.
    """
    import time
    
    device = next(model.parameters()).device
    model.eval()
    
    # Prepare batch
    batch_input = {
        k: v.expand(batch_size, *v.shape[1:]).to(device) 
        for k, v in sample_input.items()
    }
    
    # Warmup
    with torch.no_grad():
        for _ in range(num_warmup):
            _ = model(batch_input)
    
    if device.type == "cuda":
        torch.cuda.synchronize()
    
    # Benchmark
    times = []
    with torch.no_grad():
        for _ in range(num_iterations):
            start = time.perf_counter()
            _ = model(batch_input)
            if device.type == "cuda":
                torch.cuda.synchronize()
            times.append(time.perf_counter() - start)
    
    times = times[len(times)//10:]  # Remove outliers
    
    return {
        "mean_ms": sum(times) / len(times) * 1000,
        "std_ms": (sum((t - sum(times)/len(times))**2 for t in times) / len(times))**0.5 * 1000,
        "throughput_samples_per_sec": batch_size / (sum(times) / len(times)),
        "batch_size": batch_size,
    }


def print_gpu_info():
    """Print GPU information to console."""
    info = get_gpu_info()
    
    if not info["available"]:
        print("No GPU available")
        return
    
    print("\n" + "="*50)
    print("GPU INFORMATION")
    print("="*50)
    print(f"Device: {info['name']}")
    print(f"Compute Capability: {info['compute_capability']}")
    print(f"Total Memory: {info['total_memory_gb']:.1f} GB")
    print(f"CUDA Version: {info['cuda_version']}")
    print(f"cuDNN Version: {info['cudnn_version']}")
    
    mem = get_memory_stats()
    print(f"\nMemory Usage:")
    print(f"  Allocated: {mem['allocated_gb']:.2f} GB")
    print(f"  Reserved:  {mem['reserved_gb']:.2f} GB")
    print(f"  Free:      {mem['free_gb']:.2f} GB")
    print("="*50 + "\n")


class MemoryTracker:
    """Context manager to track memory usage."""
    
    def __init__(self, label: str = ""):
        self.label = label
        self.start_allocated = 0
        self.start_reserved = 0
    
    def __enter__(self):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            self.start_allocated = torch.cuda.memory_allocated()
            self.start_reserved = torch.cuda.memory_reserved()
        return self
    
    def __exit__(self, *args):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            end_allocated = torch.cuda.memory_allocated()
            end_reserved = torch.cuda.memory_reserved()
            
            delta_alloc = (end_allocated - self.start_allocated) / (1024**2)
            delta_res = (end_reserved - self.start_reserved) / (1024**2)
            
            print(f"[{self.label}] Memory: +{delta_alloc:.1f} MB allocated, +{delta_res:.1f} MB reserved")

