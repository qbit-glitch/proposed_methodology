#!/usr/bin/env python3
"""
Smart Training Launcher for Multi-GPU Training.

Automatically detects available GPUs and launches training with:
- Single GPU: Regular training
- Multi-GPU: Distributed Data Parallel (DDP) with torchrun

Usage:
    python train.py --epochs 15 --data-dir datasets
    
    # Force single GPU
    python train.py --single-gpu --epochs 15
    
    # Specify GPU count
    python train.py --num-gpus 2 --epochs 15
"""

import os
import sys
import subprocess
import argparse


def get_num_gpus() -> int:
    """Detect number of available NVIDIA GPUs."""
    try:
        import torch
        if torch.cuda.is_available():
            return torch.cuda.device_count()
    except:
        pass
    
    # Fallback to nvidia-smi
    try:
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=count', '--format=csv,noheader'],
            capture_output=True, text=True
        )
        if result.returncode == 0:
            return int(result.stdout.strip().split('\n')[0])
    except:
        pass
    
    return 0


def main():
    parser = argparse.ArgumentParser(
        description='Smart Training Launcher (auto-detects GPUs)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Auto-detect GPUs and train
    python train.py --epochs 15 --data-dir datasets
    
    # Force single GPU
    python train.py --single-gpu --epochs 15
    
    # Use specific number of GPUs
    python train.py --num-gpus 2 --epochs 15
    
    # Demo mode
    python train.py --demo
        """
    )
    
    # Launcher-specific arguments
    parser.add_argument('--single-gpu', action='store_true',
                        help='Force single GPU mode even if multiple GPUs available')
    parser.add_argument('--num-gpus', type=int, default=None,
                        help='Number of GPUs to use (default: all available)')
    parser.add_argument('--master-port', type=int, default=29500,
                        help='Port for distributed training (default: 29500)')
    
    # Parse known args (rest go to train_complete.py)
    args, remaining = parser.parse_known_args()
    
    # Detect GPUs
    num_gpus = get_num_gpus()
    
    # Determine how many GPUs to use
    if args.single_gpu:
        use_gpus = 1
    elif args.num_gpus is not None:
        use_gpus = min(args.num_gpus, num_gpus)
    else:
        use_gpus = num_gpus
    
    # Print GPU info
    print("=" * 60)
    print("  🚀 Smart Training Launcher")
    print("=" * 60)
    
    if num_gpus == 0:
        print("  GPUs detected: 0 (will use CPU or MPS)")
        use_gpus = 0
    else:
        print(f"  GPUs detected: {num_gpus}")
        print(f"  GPUs to use: {use_gpus}")
    
    print("=" * 60 + "\n")
    
    # Build command
    train_script = os.path.join(os.path.dirname(__file__), 'train_complete.py')
    
    if use_gpus > 1:
        # Multi-GPU: use torchrun
        cmd = [
            sys.executable, '-m', 'torch.distributed.launch',
            '--nproc_per_node', str(use_gpus),
            '--master_port', str(args.master_port),
            train_script,
        ] + remaining
        
        print(f"🔥 Launching distributed training on {use_gpus} GPUs...\n")
    else:
        # Single GPU or CPU/MPS
        cmd = [sys.executable, train_script] + remaining
        
        if use_gpus == 1:
            print("🔥 Launching single-GPU training...\n")
        else:
            print("🔥 Launching training (CPU/MPS)...\n")
    
    # Print command
    print(f"Command: {' '.join(cmd)}\n")
    print("-" * 60 + "\n")
    
    # Execute
    try:
        result = subprocess.run(cmd)
        sys.exit(result.returncode)
    except KeyboardInterrupt:
        print("\n⚠️ Training interrupted by user")
        sys.exit(1)


if __name__ == '__main__':
    main()
