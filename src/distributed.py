"""
Distributed Training Utilities for Multi-GPU Training.

Implements automatic GPU detection and DistributedDataParallel (DDP) setup.
Supports:
- Single GPU training (automatic fallback)
- Multi-GPU training with DDP
- Mixed precision training with GradScaler

Usage:
    # Single script launch (auto-detects GPUs)
    python train_complete.py --epochs 15 --data-dir datasets
    
    # Manual multi-GPU launch
    torchrun --nproc_per_node=2 train_complete.py --epochs 15 --data-dir datasets
"""

import os
import sys
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from torch.cuda.amp import GradScaler, autocast
from typing import Optional, Tuple, Dict, Any
from contextlib import contextmanager


def get_world_size() -> int:
    """Get the number of processes in distributed training."""
    if dist.is_initialized():
        return dist.get_world_size()
    return 1


def get_rank() -> int:
    """Get the rank of current process."""
    if dist.is_initialized():
        return dist.get_rank()
    return 0


def get_local_rank() -> int:
    """Get the local rank (GPU index on this node)."""
    return int(os.environ.get('LOCAL_RANK', 0))


def is_main_process() -> bool:
    """Check if this is the main process (rank 0)."""
    return get_rank() == 0


def setup_distributed() -> Tuple[int, int, torch.device]:
    """
    Initialize distributed training environment.
    
    Automatically detects:
    - Number of available GPUs
    - Whether running with torchrun/distributed launch
    - Falls back to single GPU/MPS if no distributed setup
    
    Returns:
        Tuple of (rank, world_size, device)
    """
    # Check if already launched with torchrun/distributed
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        local_rank = int(os.environ.get('LOCAL_RANK', 0))
        
        # Initialize process group
        if not dist.is_initialized():
            dist.init_process_group(
                backend='nccl',  # Best for NVIDIA GPUs
                init_method='env://',
            )
        
        # Set device
        torch.cuda.set_device(local_rank)
        device = torch.device(f'cuda:{local_rank}')
        
        if rank == 0:
            print(f"✓ Distributed training initialized")
            print(f"  World size: {world_size} GPUs")
        
        return rank, world_size, device
    
    # Single GPU or MPS mode
    if torch.cuda.is_available():
        num_gpus = torch.cuda.device_count()
        if num_gpus > 1:
            print(f"⚠️ Found {num_gpus} GPUs but running in single-GPU mode")
            print(f"   To use all GPUs, run with: torchrun --nproc_per_node={num_gpus} train_complete.py ...")
        device = torch.device('cuda:0')
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
    
    return 0, 1, device


def cleanup_distributed():
    """Clean up distributed training."""
    if dist.is_initialized():
        dist.destroy_process_group()


def wrap_model_ddp(
    model: nn.Module,
    device: torch.device,
    find_unused_parameters: bool = True,
) -> nn.Module:
    """
    Wrap model with DDP if in distributed mode.
    
    Args:
        model: The model to wrap
        device: Target device
        find_unused_parameters: Whether to find unused params (needed for some models)
        
    Returns:
        DDP-wrapped model if distributed, otherwise original model
    """
    model = model.to(device)
    
    if dist.is_initialized():
        local_rank = get_local_rank()
        model = DDP(
            model,
            device_ids=[local_rank],
            output_device=local_rank,
            find_unused_parameters=find_unused_parameters,
        )
        if is_main_process():
            print(f"✓ Model wrapped with DistributedDataParallel")
    
    return model


def get_distributed_sampler(
    dataset,
    shuffle: bool = True,
    drop_last: bool = True,
) -> Optional[DistributedSampler]:
    """
    Create a DistributedSampler if in distributed mode.
    
    Args:
        dataset: The dataset to sample from
        shuffle: Whether to shuffle
        drop_last: Whether to drop last incomplete batch
        
    Returns:
        DistributedSampler if distributed, None otherwise
    """
    if dist.is_initialized():
        return DistributedSampler(
            dataset,
            num_replicas=get_world_size(),
            rank=get_rank(),
            shuffle=shuffle,
            drop_last=drop_last,
        )
    return None


def create_dataloader(
    dataset,
    batch_size: int,
    shuffle: bool = True,
    num_workers: int = 4,
    pin_memory: bool = True,
    drop_last: bool = True,
    collate_fn=None,
) -> Tuple[DataLoader, Optional[DistributedSampler]]:
    """
    Create a DataLoader with optional distributed sampler.
    
    Args:
        dataset: Dataset to load
        batch_size: Batch size per GPU
        shuffle: Whether to shuffle (only used if not distributed)
        num_workers: Number of data loading workers
        pin_memory: Whether to pin memory (recommended for CUDA)
        drop_last: Drop last incomplete batch
        collate_fn: Custom collate function
        
    Returns:
        Tuple of (DataLoader, sampler or None)
    """
    sampler = get_distributed_sampler(dataset, shuffle=shuffle, drop_last=drop_last)
    
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(shuffle and sampler is None),
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=pin_memory and torch.cuda.is_available(),
        drop_last=drop_last,
        collate_fn=collate_fn,
    )
    
    return loader, sampler


class MixedPrecisionTrainer:
    """
    Helper class for mixed precision training with automatic scaling.
    
    Handles:
    - Automatic mixed precision (FP16/BF16) on CUDA
    - Gradient scaling for numerical stability
    - Falls back to FP32 on MPS/CPU
    """
    
    def __init__(self, enabled: bool = True, device: torch.device = None):
        self.enabled = enabled and torch.cuda.is_available()
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        if self.enabled:
            self.scaler = GradScaler()
            if is_main_process():
                print("✓ Mixed precision training enabled (FP16)")
        else:
            self.scaler = None
    
    @contextmanager
    def autocast_context(self):
        """Context manager for automatic mixed precision."""
        if self.enabled:
            with autocast():
                yield
        else:
            yield
    
    def scale_loss(self, loss: torch.Tensor) -> torch.Tensor:
        """Scale loss for mixed precision training."""
        if self.enabled:
            return self.scaler.scale(loss)
        return loss
    
    def step(self, optimizer: torch.optim.Optimizer):
        """Optimizer step with gradient unscaling."""
        if self.enabled:
            self.scaler.step(optimizer)
            self.scaler.update()
        else:
            optimizer.step()
    
    def unscale_gradients(self, optimizer: torch.optim.Optimizer):
        """Unscale gradients for gradient clipping."""
        if self.enabled:
            self.scaler.unscale_(optimizer)


def all_reduce_mean(tensor: torch.Tensor) -> torch.Tensor:
    """
    All-reduce a tensor and compute mean across all processes.
    
    Args:
        tensor: Tensor to reduce
        
    Returns:
        Mean tensor across all processes
    """
    if not dist.is_initialized():
        return tensor
    
    tensor = tensor.clone()
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    tensor /= get_world_size()
    return tensor


def broadcast_object(obj: Any, src: int = 0) -> Any:
    """
    Broadcast a Python object from source rank to all processes.
    
    Args:
        obj: Object to broadcast
        src: Source rank
        
    Returns:
        Broadcasted object
    """
    if not dist.is_initialized():
        return obj
    
    object_list = [obj]
    dist.broadcast_object_list(object_list, src=src)
    return object_list[0]


def synchronize():
    """Synchronize all processes."""
    if dist.is_initialized():
        dist.barrier()


def print_rank0(*args, **kwargs):
    """Print only on rank 0."""
    if is_main_process():
        print(*args, **kwargs)


def get_gpu_info() -> Dict[str, Any]:
    """
    Get information about available GPUs.
    
    Returns:
        Dictionary with GPU information
    """
    info = {
        'cuda_available': torch.cuda.is_available(),
        'mps_available': hasattr(torch.backends, 'mps') and torch.backends.mps.is_available(),
        'num_gpus': 0,
        'gpus': [],
    }
    
    if torch.cuda.is_available():
        info['num_gpus'] = torch.cuda.device_count()
        for i in range(info['num_gpus']):
            props = torch.cuda.get_device_properties(i)
            info['gpus'].append({
                'index': i,
                'name': props.name,
                'memory_gb': props.total_memory / (1024**3),
                'compute_capability': f"{props.major}.{props.minor}",
            })
    
    return info


def print_gpu_info():
    """Print detailed GPU information."""
    info = get_gpu_info()
    
    print("\n" + "="*50)
    print("  GPU Information")
    print("="*50)
    
    if info['cuda_available']:
        print(f"  CUDA available: Yes")
        print(f"  Number of GPUs: {info['num_gpus']}")
        for gpu in info['gpus']:
            print(f"\n  GPU {gpu['index']}: {gpu['name']}")
            print(f"    Memory: {gpu['memory_gb']:.1f} GB")
            print(f"    Compute: {gpu['compute_capability']}")
    elif info['mps_available']:
        print(f"  MPS (Apple Silicon) available: Yes")
        print(f"  Note: Multi-GPU not supported on MPS")
    else:
        print(f"  No GPU available, using CPU")
    
    print("="*50 + "\n")


if __name__ == "__main__":
    # Test utilities
    print_gpu_info()
    
    rank, world_size, device = setup_distributed()
    print(f"Rank: {rank}, World size: {world_size}, Device: {device}")
    
    cleanup_distributed()
