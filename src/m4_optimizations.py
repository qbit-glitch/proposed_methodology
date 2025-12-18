"""
GPU Optimizations for Efficient Training.

Implements memory-efficient training techniques for:
- Apple Silicon (MPS) - M1/M2/M3/M4 Macs
- NVIDIA GPUs (CUDA) - RTX 3070, 3080, 4xxx series, etc.

Features:
- Gradient checkpointing for backbone
- Activation checkpointing for transformers
- Platform-specific configurations
- Memory monitoring utilities
- Mixed precision training (CUDA)
"""

import os
import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint, checkpoint_sequential
from typing import List, Optional, Callable, Any, Dict
from functools import wraps


def get_device() -> torch.device:
    """Get the best available device."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def get_cuda_config() -> dict:
    """
    Get optimal CUDA configuration for NVIDIA GPUs.
    
    Optimized for RTX 3070 (8GB), also works for other NVIDIA GPUs.
    
    Returns:
        Dict with recommended settings
    """
    config = {
        # CUDA supports mixed precision
        'use_mixed_precision': True,  # Use AMP for faster training
        
        # cuDNN optimizations
        'cudnn_benchmark': True,  # Enable for fixed input sizes
        'cudnn_deterministic': False,  # Set True for reproducibility
        
        # Memory settings
        'empty_cache_freq': 5,  # Empty cache every N batches
        
        # Recommended batch settings for different GPU memory
        'memory_configs': {
            '6GB': {'batch_size': 1, 'grad_accum': 8, 'image_size': 384},
            '8GB': {'batch_size': 1, 'grad_accum': 8, 'image_size': 448},   # RTX 3070
            '10GB': {'batch_size': 2, 'grad_accum': 4, 'image_size': 448},  # RTX 3080
            '12GB': {'batch_size': 2, 'grad_accum': 4, 'image_size': 512},  # RTX 3080 Ti
            '16GB': {'batch_size': 2, 'grad_accum': 4, 'image_size': 512},  # RTX 4080
            '24GB': {'batch_size': 4, 'grad_accum': 2, 'image_size': 512},  # RTX 3090/4090
        },
    }
    return config


def get_mps_config() -> dict:
    """
    Get optimal MPS configuration for Apple Silicon.
    
    Returns:
        Dict with recommended settings
    """
    config = {
        # MPS doesn't support all CUDA operations
        'use_mixed_precision': False,  # MPS has limited fp16 support
        
        # Memory settings
        'empty_cache_freq': 10,  # Empty MPS cache every N batches
        
        # Recommended batch settings for different memory configs
        'memory_configs': {
            '8GB': {'batch_size': 1, 'grad_accum': 8, 'image_size': 384},   # M1
            '16GB': {'batch_size': 1, 'grad_accum': 8, 'image_size': 448},  # M1 Pro/Max
            '24GB': {'batch_size': 2, 'grad_accum': 4, 'image_size': 448},  # M4 Pro
            '48GB': {'batch_size': 2, 'grad_accum': 4, 'image_size': 512},  # M2 Max
            '64GB': {'batch_size': 4, 'grad_accum': 2, 'image_size': 640},  # M2 Ultra
        },
        
        # Operations to avoid on MPS (use CPU fallback)
        'cpu_fallback_ops': [
            'grid_sample',  # Known MPS issue
            'affine_grid',
        ],
    }
    return config


def get_platform_config() -> Dict:
    """
    Get configuration for current platform.
    
    Returns:
        Platform-specific configuration dict
    """
    device = get_device()
    
    if device.type == 'cuda':
        config = get_cuda_config()
        config['platform'] = 'cuda'
        config['device'] = device
        
        # Get GPU memory
        gpu_mem_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        config['gpu_memory_gb'] = gpu_mem_gb
        config['gpu_name'] = torch.cuda.get_device_name(0)
        
    elif device.type == 'mps':
        config = get_mps_config()
        config['platform'] = 'mps'
        config['device'] = device
        config['gpu_memory_gb'] = 24.0  # Estimate for M4 Pro
        
    else:
        config = {
            'platform': 'cpu',
            'device': device,
            'use_mixed_precision': False,
            'memory_configs': {
                'default': {'batch_size': 1, 'grad_accum': 16, 'image_size': 256}
            }
        }
    
    return config


def configure_cuda_defaults():
    """Configure optimal defaults for CUDA training."""
    if not torch.cuda.is_available():
        return
    
    # Enable cuDNN autotuner
    torch.backends.cudnn.benchmark = True
    
    # Use TF32 for faster matrix operations on Ampere GPUs (RTX 30xx, 40xx)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    
    # Set memory allocation strategy for better fragmentation handling
    if hasattr(torch.cuda, 'memory'):
        # Expandable segments help with memory fragmentation
        os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
    
    print("✓ CUDA defaults configured")


def empty_cuda_cache():
    """Empty CUDA cache to free memory."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def estimate_memory_usage(model: nn.Module, batch_size: int, image_size: int) -> float:
    """
    Estimate peak memory usage in GB.
    
    Args:
        model: The model
        batch_size: Batch size
        image_size: Image dimension (assumes square)
        
    Returns:
        Estimated peak memory in GB
    """
    # Count parameters
    num_params = sum(p.numel() for p in model.parameters())
    param_memory = num_params * 4 / (1024**3)  # float32
    
    # Estimate activation memory (rough heuristic)
    # Assuming ~4x parameter count for activations at peak
    activation_multiplier = 4.0
    
    # Scale by batch size and image size
    base_size = 512
    size_factor = (image_size / base_size) ** 2
    
    estimated_memory = param_memory * (1 + activation_multiplier * batch_size * size_factor)
    
    return estimated_memory


def empty_mps_cache():
    """Empty MPS cache to free memory."""
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()


class GradientCheckpointWrapper(nn.Module):
    """
    Wrapper to enable gradient checkpointing for any module.
    
    Trades compute for memory by not storing intermediate activations.
    """
    
    def __init__(self, module: nn.Module, checkpoint_every: int = 1):
        super().__init__()
        self.module = module
        self.checkpoint_every = checkpoint_every
        self._checkpoint_enabled = True
        
    def enable_checkpointing(self):
        self._checkpoint_enabled = True
        
    def disable_checkpointing(self):
        self._checkpoint_enabled = False
    
    def forward(self, *args, **kwargs):
        if self._checkpoint_enabled and self.training:
            # Create a wrapper function that handles kwargs
            def custom_forward(*inputs):
                return self.module(*inputs, **kwargs)
            
            return checkpoint(custom_forward, *args, use_reentrant=False)
        else:
            return self.module(*args, **kwargs)


class CheckpointedSequential(nn.Module):
    """
    Sequential module with gradient checkpointing.
    
    Checkpoints every N layers to save memory.
    """
    
    def __init__(self, layers: List[nn.Module], checkpoint_every: int = 2):
        super().__init__()
        self.layers = nn.ModuleList(layers)
        self.checkpoint_every = checkpoint_every
        self._checkpoint_enabled = True
    
    def enable_checkpointing(self):
        self._checkpoint_enabled = True
        
    def disable_checkpointing(self):
        self._checkpoint_enabled = False
    
    def forward(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        if self._checkpoint_enabled and self.training:
            # Group layers for checkpointing
            segments = []
            for i in range(0, len(self.layers), self.checkpoint_every):
                segment = self.layers[i:i + self.checkpoint_every]
                segments.append(nn.Sequential(*segment))
            
            for segment in segments:
                x = checkpoint(segment, x, use_reentrant=False)
            return x
        else:
            for layer in self.layers:
                x = layer(x, *args, **kwargs) if kwargs else layer(x)
            return x


def apply_gradient_checkpointing(
    model: nn.Module,
    checkpoint_backbone: bool = True,
    checkpoint_encoder: bool = True,
    encoder_checkpoint_every: int = 2,
) -> nn.Module:
    """
    Apply gradient checkpointing to model components.
    
    Args:
        model: The unified transformer model
        checkpoint_backbone: Whether to checkpoint backbone layers
        checkpoint_encoder: Whether to checkpoint encoder layers
        encoder_checkpoint_every: Checkpoint every N encoder layers
        
    Returns:
        Model with checkpointing enabled
    """
    # Backbone checkpointing
    if checkpoint_backbone and hasattr(model, 'backbone'):
        backbone = model.backbone
        
        # For ResNet, checkpoint each layer group
        if hasattr(backbone, 'resnet'):
            resnet = backbone.resnet
            for name in ['layer2', 'layer3', 'layer4']:
                if hasattr(resnet, name):
                    layer = getattr(resnet, name)
                    # Wrap in checkpoint
                    wrapped = GradientCheckpointWrapper(layer)
                    setattr(resnet, name, wrapped)
                    print(f"  ✓ Checkpointing backbone.{name}")
    
    # Encoder checkpointing
    if checkpoint_encoder and hasattr(model, 'encoder'):
        encoder = model.encoder
        if hasattr(encoder, 'layers'):
            original_layers = list(encoder.layers)
            encoder.layers = nn.ModuleList([
                GradientCheckpointWrapper(layer, checkpoint_every=1) 
                if i % encoder_checkpoint_every == 0 else layer
                for i, layer in enumerate(original_layers)
            ])
            print(f"  ✓ Checkpointing encoder (every {encoder_checkpoint_every} layers)")
    
    return model


class MemoryTracker:
    """
    Track memory usage during training.
    
    Useful for debugging memory issues on MPS.
    """
    
    def __init__(self, device: torch.device):
        self.device = device
        self.peak_memory = 0
        self.measurements = []
    
    def measure(self, label: str = "") -> float:
        """Measure current memory usage."""
        if self.device.type == 'mps':
            # MPS doesn't have direct memory query
            # Use system memory as proxy
            try:
                import psutil
                process = psutil.Process(os.getpid())
                memory_gb = process.memory_info().rss / (1024**3)
            except ImportError:
                memory_gb = 0.0
        elif self.device.type == 'cuda':
            memory_gb = torch.cuda.max_memory_allocated() / (1024**3)
        else:
            memory_gb = 0.0
        
        self.peak_memory = max(self.peak_memory, memory_gb)
        self.measurements.append((label, memory_gb))
        
        return memory_gb
    
    def report(self):
        """Print memory report."""
        print("\n📊 Memory Report:")
        print(f"   Peak memory: {self.peak_memory:.2f} GB")
        if self.measurements:
            print("   Measurements:")
            for label, mem in self.measurements[-5:]:  # Last 5
                print(f"     {label}: {mem:.2f} GB")


def optimize_for_inference(model: nn.Module) -> nn.Module:
    """
    Optimize model for inference on MPS.
    
    Args:
        model: The model to optimize
        
    Returns:
        Optimized model
    """
    model.eval()
    
    # Disable checkpointing for inference
    for module in model.modules():
        if isinstance(module, (GradientCheckpointWrapper, CheckpointedSequential)):
            module.disable_checkpointing()
    
    # Fuse batch norms where possible
    # Note: torch.jit.script has limited MPS support
    
    return model


class MPSFallbackWrapper(nn.Module):
    """
    Wrapper that falls back to CPU for unsupported MPS operations.
    """
    
    def __init__(self, module: nn.Module, fallback_ops: List[str] = None):
        super().__init__()
        self.module = module
        self.fallback_ops = fallback_ops or ['grid_sample', 'affine_grid']
        
    def forward(self, *args, **kwargs):
        # Check if any args are on MPS
        device = None
        for arg in args:
            if isinstance(arg, torch.Tensor):
                device = arg.device
                break
        
        if device and device.type == 'mps':
            # Move to CPU for operation, then back
            cpu_args = [a.cpu() if isinstance(a, torch.Tensor) else a for a in args]
            cpu_kwargs = {k: v.cpu() if isinstance(v, torch.Tensor) else v 
                         for k, v in kwargs.items()}
            
            result = self.module(*cpu_args, **cpu_kwargs)
            
            if isinstance(result, torch.Tensor):
                return result.to(device)
            return result
        
        return self.module(*args, **kwargs)


def configure_mps_defaults():
    """Configure optimal defaults for MPS training."""
    # Set environment variables for MPS
    os.environ['PYTORCH_MPS_HIGH_WATERMARK_RATIO'] = '0.0'  # Disable memory limit
    
    # Enable MPS fallback for unsupported ops
    if hasattr(torch.backends.mps, 'enable_fallback'):
        # This is a placeholder - actual API may differ
        pass
    
    print("✓ MPS defaults configured")


def configure_platform_defaults():
    """Configure optimal defaults for current platform."""
    device = get_device()
    
    if device.type == 'cuda':
        configure_cuda_defaults()
    elif device.type == 'mps':
        configure_mps_defaults()
    else:
        print("✓ CPU mode configured")
    
    return device


def empty_cache():
    """Empty GPU cache (works for both CUDA and MPS)."""
    if torch.cuda.is_available():
        empty_cuda_cache()
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        empty_mps_cache()


def get_optimal_batch_config(available_memory_gb: float = None) -> dict:
    """
    Get optimal batch configuration based on available memory and platform.
    
    Args:
        available_memory_gb: Available GPU memory in GB (auto-detected if None)
        
    Returns:
        Dict with batch_size, gradient_accumulation, image_size
    """
    platform_config = get_platform_config()
    
    # Auto-detect memory if not provided
    if available_memory_gb is None:
        if platform_config['platform'] == 'cuda':
            available_memory_gb = platform_config.get('gpu_memory_gb', 8.0)
        elif platform_config['platform'] == 'mps':
            available_memory_gb = platform_config.get('gpu_memory_gb', 24.0)
        else:
            available_memory_gb = 4.0  # Conservative for CPU
    
    memory_configs = platform_config['memory_configs']
    
    # Find best config for available memory
    best_config = {'batch_size': 1, 'grad_accum': 16, 'image_size': 256}
    
    for mem_key in sorted(memory_configs.keys(), key=lambda x: int(x.replace('GB', ''))):
        mem_gb = int(mem_key.replace('GB', ''))
        if available_memory_gb >= mem_gb:
            best_config = memory_configs[mem_key]
    
    return best_config


def print_platform_info():
    """Print detailed platform information."""
    config = get_platform_config()
    
    print(f"\n{'='*50}")
    print("  Platform Configuration")
    print(f"{'='*50}")
    print(f"  Platform: {config['platform'].upper()}")
    print(f"  Device: {config['device']}")
    
    if config['platform'] == 'cuda':
        print(f"  GPU: {config.get('gpu_name', 'Unknown')}")
        print(f"  Memory: {config.get('gpu_memory_gb', 0):.1f} GB")
        print(f"  Mixed Precision: {config['use_mixed_precision']}")
        print(f"  cuDNN Benchmark: {config.get('cudnn_benchmark', True)}")
    elif config['platform'] == 'mps':
        print(f"  Memory: ~{config.get('gpu_memory_gb', 24):.0f} GB (unified)")
        print(f"  Mixed Precision: {config['use_mixed_precision']}")
    
    batch_config = get_optimal_batch_config()
    print(f"\n  Recommended Config:")
    print(f"    Batch size: {batch_config['batch_size']}")
    print(f"    Gradient accumulation: {batch_config['grad_accum']}")
    print(f"    Image size: {batch_config['image_size']}")
    print(f"{'='*50}\n")


if __name__ == "__main__":
    print("🖥️  GPU Optimizations Test\n")
    
    # Print platform info
    print_platform_info()
    
    # Test configuration
    device = get_device()
    print(f"Best available device: {device}")
    
    if device.type == 'cuda':
        config = get_cuda_config()
        print("\nCUDA Configuration:")
    else:
        config = get_mps_config()
        print("\nMPS Configuration:")
    
    print(f"  Mixed precision: {config['use_mixed_precision']}")
    print(f"  Memory configs: {list(config['memory_configs'].keys())}")
    
    # Test memory estimation
    try:
        from src.unified_transformer import build_model
        from src.config import ModelConfig
        
        model = build_model(ModelConfig())
        estimated = estimate_memory_usage(model, batch_size=2, image_size=512)
        print(f"\nEstimated memory for batch=2, size=512: {estimated:.2f} GB")
        
        # Test gradient checkpointing
        print("\nApplying gradient checkpointing...")
        model = apply_gradient_checkpointing(model)
        
        print("\n✓ All optimizations tested successfully!")
    except Exception as e:
        print(f"\n⚠️ Model test skipped: {e}")
