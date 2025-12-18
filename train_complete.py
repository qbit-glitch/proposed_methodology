#!/usr/bin/env python3
"""
Complete Training Script for Unified Multi-Task Transformer.

Main entry point for training the parking violation detection model.
Supports:
- Three-stage training pipeline
- Stage selection (init, pretrain, joint)
- Checkpoint resume
- Synthetic data demo mode
- Full evaluation

Optimized for:
- Apple Silicon (MPS) - M1/M2/M3/M4 Macs
- NVIDIA GPUs (CUDA) - RTX 3070, 3080, 3090, 4xxx series

Usage:
    # Demo with synthetic data
    python train_complete.py --demo
    
    # Full Stage 3 joint training
    python train_complete.py --stage 3 --epochs 100
    
    # Resume from checkpoint
    python train_complete.py --resume outputs/checkpoints/best_model.pt
"""

import os
import sys
import argparse
import time
from datetime import datetime
from typing import Optional
from pathlib import Path

import torch
from torch.utils.data import DataLoader

# Add src to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.config import ModelConfig, TrainingConfig, get_device
from src.unified_transformer import build_model
from src.staged_trainer import StagedTrainer, StagedTrainingConfig
from src.trainer import SyntheticDataset, collate_fn
from src.m4_optimizations import (
    get_platform_config, 
    get_optimal_batch_config,
    configure_platform_defaults,
    print_platform_info,
)
from src.evaluation import MultiTaskEvaluator


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Train Unified Multi-Task Transformer for Parking Violation Detection"
    )
    
    # Training mode
    parser.add_argument(
        '--demo', action='store_true',
        help='Run quick demo with synthetic data (2 epochs)'
    )
    parser.add_argument(
        '--stage', type=int, choices=[1, 2, 3], default=3,
        help='Training stage: 1=init, 2=pretrain, 3=joint (default: 3)'
    )
    
    # Training parameters
    parser.add_argument(
        '--epochs', type=int, default=None,
        help='Number of training epochs for Stage 3 (default: 100)'
    )
    parser.add_argument(
        '--pretrain-epochs', type=int, default=15,
        help='Number of epochs for Stage 2 decoder pretraining (default: 15)'
    )
    parser.add_argument(
        '--batch-size', type=int, default=None,
        help='Batch size (auto-configured for M4 Pro if not specified)'
    )
    parser.add_argument(
        '--grad-accum', type=int, default=None,
        help='Gradient accumulation steps'
    )
    parser.add_argument(
        '--lr', type=float, default=1e-4,
        help='Learning rate'
    )
    
    # Checkpoints
    parser.add_argument(
        '--resume', type=str, default=None,
        help='Resume from checkpoint path'
    )
    parser.add_argument(
        '--output-dir', type=str, default='outputs',
        help='Output directory for checkpoints and logs'
    )
    
    # Data
    parser.add_argument(
        '--data-dir', type=str, default=None,
        help='Path to training data directory'
    )
    parser.add_argument(
        '--synthetic-samples', type=int, default=100,
        help='Number of synthetic samples for demo mode'
    )
    parser.add_argument(
        '--subset-ratio', type=float, default=1.0,
        help='Fraction of dataset to use (0.0-1.0), e.g., 0.05 for 5%%'
    )
    
    # Model
    parser.add_argument(
        '--image-size', type=int, default=512,
        help='Input image size (default: 512)'
    )
    parser.add_argument(
        '--hidden-dim', type=int, default=256,
        help='Transformer hidden dimension'
    )
    
    # Optimization
    parser.add_argument(
        '--no-checkpoint', action='store_true',
        help='Disable gradient checkpointing'
    )
    parser.add_argument(
        '--no-uncertainty', action='store_true',
        help='Disable uncertainty weighting for losses'
    )
    
    # Advanced Optimizations (from optimizations.md)
    parser.add_argument(
        '--optimizer', type=str, default='lookahead_adam',
        choices=['adam', 'radam', 'lookahead_adam', 'lookahead_radam'],
        help='Optimizer type (default: lookahead_adam for better convergence)'
    )
    parser.add_argument(
        '--curriculum', action='store_true',
        help='Enable curriculum learning (easy-to-hard scheduling)'
    )
    parser.add_argument(
        '--use-fpn', action='store_true',
        help='Use Feature Pyramid Network for multi-scale features'
    )
    parser.add_argument(
        '--use-deformable', action='store_true',
        help='Use deformable attention (more efficient for spatial features)'
    )
    
    # Misc
    parser.add_argument(
        '--seed', type=int, default=42,
        help='Random seed'
    )
    parser.add_argument(
        '--workers', type=int, default=0,
        help='Number of data loading workers (0 for MPS recommended)'
    )
    
    return parser.parse_args()


def setup_environment(args):
    """Setup training environment."""
    # Set random seed
    torch.manual_seed(args.seed)
    
    # Create output directories
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, 'checkpoints'), exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, 'logs'), exist_ok=True)
    
    # Get device
    device = get_device()
    print(f"\n🖥️  Device: {device}")
    
    # Auto-configure batch size for M4 Pro
    if args.batch_size is None or args.grad_accum is None:
        # Estimate available memory (assume 24GB for M4 Pro)
        batch_config = get_optimal_batch_config(24.0)
        
        if args.batch_size is None:
            args.batch_size = batch_config['batch_size']
        if args.grad_accum is None:
            args.grad_accum = batch_config['grad_accum']
        
        print(f"📊 Auto-configured: batch_size={args.batch_size}, grad_accum={args.grad_accum}")
    
    return device


def create_configs(args, device):
    """Create model and training configurations."""
    # Model config
    model_config = ModelConfig(
        image_size=(args.image_size, args.image_size),
        hidden_dim=args.hidden_dim,
    )
    
    # Training config
    training_config = TrainingConfig(
        device=device,
        batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        output_dir=args.output_dir,
        checkpoint_dir=os.path.join(args.output_dir, 'checkpoints'),
    )
    
    # Set epochs
    if args.demo:
        training_config.max_epochs = 2
    elif args.epochs:
        training_config.max_epochs = args.epochs
    
    # Staged training config
    staged_config = StagedTrainingConfig(
        model_config=model_config,
        training_config=training_config,
        use_gradient_checkpointing=not args.no_checkpoint,
        use_uncertainty_weighting=not args.no_uncertainty,
    )
    
    if args.epochs:
        staged_config.joint_training_epochs = args.epochs
    elif args.demo:
        staged_config.joint_training_epochs = 2
    
    return model_config, training_config, staged_config


def create_data_loaders(args, training_config):
    """Create training and validation data loaders."""
    from src.datasets import (
        COCODetectionDataset, CityscapesDataset, CCPDDataset, MOT17Dataset,
        MultiTaskDataset, multi_task_collate_fn,
        create_coco_loaders, create_cityscapes_loaders, create_ccpd_loaders, create_mot17_loaders,
    )
    
    subset_ratio = getattr(args, 'subset_ratio', 1.0)
    
    if args.data_dir and os.path.exists(args.data_dir):
        data_dir = Path(args.data_dir)
        image_size = (args.image_size, args.image_size)
        
        # Check which datasets are available (check multiple possible names)
        datasets_found = {}
        
        # COCO for detection (check coco, coco2017, COCO)
        for name in ['coco', 'coco2017', 'COCO']:
            coco_path = data_dir / name
            if coco_path.exists():
                print(f"📂 Found COCO dataset at {coco_path}")
                datasets_found['coco'] = coco_path
                break
        
        # Cityscapes for segmentation
        for name in ['cityscapes', 'Cityscapes']:
            cityscapes_path = data_dir / name
            if cityscapes_path.exists():
                print(f"📂 Found Cityscapes dataset at {cityscapes_path}")
                datasets_found['cityscapes'] = cityscapes_path
                break
        
        # CCPD for plate detection and OCR
        for name in ['ccpd', 'CCPD2019', 'CCPD', 'ccpd2019']:
            ccpd_path = data_dir / name
            if ccpd_path.exists():
                print(f"📂 Found CCPD dataset at {ccpd_path}")
                datasets_found['ccpd'] = ccpd_path
                break
        
        # MOT17 for tracking
        for name in ['mot17', 'MOT17', 'MOT17Det']:
            mot17_path = data_dir / name
            if mot17_path.exists():
                print(f"📂 Found MOT17 dataset at {mot17_path}")
                datasets_found['mot17'] = mot17_path
                break
        
        if not datasets_found:
            print("⚠️ No datasets found, falling back to synthetic data")
            return create_synthetic_loaders(args, training_config)
        
        # Single dataset mode (for Stage 2 pretraining)
        if hasattr(args, 'task') and args.task:
            task = args.task
            # For now, skip single dataset mode and use multi-task
        
        # Multi-task mode: combine available datasets
        train_datasets = {}
        val_datasets = {}
        
        if 'coco' in datasets_found:
            coco_path = datasets_found['coco']
            train_datasets['detection'] = COCODetectionDataset(str(coco_path), 'train', image_size, subset_ratio=subset_ratio)
            val_datasets['detection'] = COCODetectionDataset(str(coco_path), 'val', image_size, subset_ratio=subset_ratio)
        
        if 'cityscapes' in datasets_found:
            cityscapes_path = datasets_found['cityscapes']
            train_datasets['segmentation'] = CityscapesDataset(str(cityscapes_path), 'train', image_size, subset_ratio=subset_ratio)
            val_datasets['segmentation'] = CityscapesDataset(str(cityscapes_path), 'val', image_size, subset_ratio=subset_ratio)
        
        if 'ccpd' in datasets_found:
            ccpd_path = datasets_found['ccpd']
            train_datasets['plate'] = CCPDDataset(str(ccpd_path), 'train', image_size, subset_ratio=subset_ratio)
            val_datasets['plate'] = CCPDDataset(str(ccpd_path), 'val', image_size, subset_ratio=subset_ratio)
        
        if 'mot17' in datasets_found:
            mot17_path = datasets_found['mot17']
            train_datasets['tracking'] = MOT17Dataset(str(mot17_path), 'train', image_size, subset_ratio=subset_ratio)
            val_datasets['tracking'] = MOT17Dataset(str(mot17_path), 'train', image_size, subset_ratio=subset_ratio)
        
        # Use detection as primary if available, else first found
        primary = 'detection' if 'detection' in train_datasets else list(train_datasets.keys())[0]
        
        train_dataset = MultiTaskDataset(train_datasets, primary_task=primary)
        val_dataset = MultiTaskDataset(val_datasets, primary_task=primary)
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=training_config.batch_size,
            shuffle=True,
            collate_fn=multi_task_collate_fn,
            num_workers=args.workers,
            pin_memory=False,
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=training_config.batch_size,
            shuffle=False,
            collate_fn=multi_task_collate_fn,
            num_workers=args.workers,
        )
        
        return train_loader, val_loader
    else:
        return create_synthetic_loaders(args, training_config)


def create_task_specific_loaders(args, training_config):
    """
    Create separate task-specific data loaders for Stage 2 pretraining.
    
    Each decoder gets its own loader with the appropriate dataset:
    - Detection: COCO
    - Segmentation: Cityscapes
    - Plate + OCR: CCPD
    - Tracking: MOT17
    
    Returns:
        dict: Task name -> (train_loader, val_loader) tuples
    """
    from src.datasets import (
        create_coco_loaders, create_cityscapes_loaders, 
        create_ccpd_loaders, create_mot17_loaders,
    )
    
    subset_ratio = getattr(args, 'subset_ratio', 1.0)
    loaders = {}
    
    if not args.data_dir or not os.path.exists(args.data_dir):
        print("⚠️ No data directory specified, using synthetic loaders for all tasks")
        return None
    
    data_dir = Path(args.data_dir)
    image_size = (args.image_size, args.image_size)
    batch_size = training_config.batch_size
    
    # COCO for detection
    coco_candidates = ['coco2017', 'coco', 'COCO']
    for name in coco_candidates:
        coco_path = data_dir / name
        if coco_path.exists():
            print(f"📦 Loading COCO for detection from {coco_path}")
            loaders['detection'] = create_coco_loaders(
                str(coco_path), batch_size, args.workers, image_size, subset_ratio
            )
            break
    
    # Cityscapes for segmentation
    cityscapes_candidates = ['cityscapes', 'Cityscapes', 'CITYSCAPES']
    for name in cityscapes_candidates:
        cityscapes_path = data_dir / name
        if cityscapes_path.exists():
            print(f"📦 Loading Cityscapes for segmentation from {cityscapes_path}")
            loaders['segmentation'] = create_cityscapes_loaders(
                str(cityscapes_path), batch_size, args.workers, image_size, subset_ratio
            )
            break
    
    # License Plates for plate detection (prefer over CCPD)
    license_plate_candidates = ['license_plates', 'License_Plates', 'plates']
    plate_loaded = False
    for name in license_plate_candidates:
        lp_path = data_dir / name
        if lp_path.exists():
            print(f"📦 Loading LicensePlates for plate detection from {lp_path}")
            from src.datasets import create_license_plate_loaders
            loaders['plate'] = create_license_plate_loaders(
                str(lp_path), batch_size, args.workers, image_size, subset_ratio
            )
            plate_loaded = True
            break
    
    # CCPD for OCR (always load if available) and plate (fallback)
    ccpd_candidates = ['CCPD2019', 'ccpd', 'CCPD']
    for name in ccpd_candidates:
        ccpd_path = data_dir / name
        if ccpd_path.exists():
            # Use smaller subset for CCPD since it's much larger
            ccpd_ratio = min(subset_ratio * 0.2, 1.0)  # 5% of 5% = 1% for CCPD
            
            # Always load OCR from CCPD (has plate text annotations)
            print(f"📦 Loading CCPD for OCR from {ccpd_path}")
            loaders['ocr'] = create_ccpd_loaders(
                str(ccpd_path), batch_size, args.workers, image_size, ccpd_ratio
            )
            
            # Also use CCPD for plate if license_plates not found
            if not plate_loaded:
                print(f"📦 Loading CCPD for plate from {ccpd_path}")
                loaders['plate'] = loaders['ocr']
            break
    
    # MOT17 for tracking
    mot17_candidates = ['mot17', 'MOT17', 'MOT17-train']
    for name in mot17_candidates:
        mot17_path = data_dir / name
        if mot17_path.exists():
            print(f"📦 Loading MOT17 for tracking from {mot17_path}")
            loaders['tracking'] = create_mot17_loaders(
                str(mot17_path), batch_size, args.workers, image_size, subset_ratio
            )
            break
    
    print(f"✓ Created task-specific loaders: {list(loaders.keys())}")
    return loaders


def create_synthetic_loaders(args, training_config):
    """Create synthetic data loaders for testing."""
    print(f"🔧 Using synthetic data ({args.synthetic_samples} samples)")
    
    train_dataset = SyntheticDataset(
        num_samples=args.synthetic_samples,
        image_size=(512, 512),
    )
    
    # Split for validation
    val_size = max(1, args.synthetic_samples // 10)
    val_dataset = SyntheticDataset(num_samples=val_size)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=training_config.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=args.workers,
        pin_memory=False,
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=training_config.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=args.workers,
    )
    
    return train_loader, val_loader


def run_training(args):
    """Main training function."""
    start_time = time.time()
    
    print("\n" + "="*70)
    print("  🚗 Unified Multi-Task Transformer Training")
    print("  📊 Parking Violation Detection System")
    print("="*70)
    
    # Setup
    device = setup_environment(args)
    model_config, training_config, staged_config = create_configs(args, device)
    
    # Print configuration
    print(f"\n📋 Configuration:")
    print(f"   Stage: {args.stage}")
    print(f"   Epochs: {staged_config.joint_training_epochs}")
    print(f"   Batch size: {training_config.batch_size}")
    print(f"   Gradient accumulation: {training_config.gradient_accumulation_steps}")
    print(f"   Effective batch size: {training_config.batch_size * training_config.gradient_accumulation_steps}")
    print(f"   Learning rate: {training_config.learning_rate}")
    print(f"   Image size: {model_config.image_size}")
    print(f"   Gradient checkpointing: {staged_config.use_gradient_checkpointing}")
    print(f"   Uncertainty weighting: {staged_config.use_uncertainty_weighting}")
    
    # Create data loaders
    train_loader, val_loader = create_data_loaders(args, training_config)
    print(f"\n📊 Data:")
    print(f"   Training batches: {len(train_loader)}")
    print(f"   Validation batches: {len(val_loader)}")
    
    # Create trainer
    trainer = StagedTrainer(staged_config)
    
    # Resume from checkpoint if specified
    if args.resume:
        print(f"\n📂 Resuming from {args.resume}")
        trainer.load_checkpoint(args.resume)
    
    # Run training based on stage
    results = {}
    
    if args.stage >= 1:
        # Stage 1: Initialization
        trainer.initialize_from_pretrained()
        results['stage1'] = {'status': 'complete'}
    
    if args.stage >= 2:
        # Stage 2: Task-specific pretraining for each decoder
        print("\n" + "="*60)
        print("  Stage 2: Task-specific Decoder Pretraining")
        print("="*60)
        
        # Set pretrain epochs for Stage 2
        if args.demo:
            pretrain_epochs = 1
        else:
            pretrain_epochs = args.pretrain_epochs  # Use command line arg (default: 15)
        
        # Create task-specific loaders (each decoder gets its OWN dataset)
        task_loaders = create_task_specific_loaders(args, training_config)
        
        # Fallback to multi-task loader if task-specific loaders not available
        if task_loaders is None:
            print("⚠️ Using multi-task loader (less effective for Stage 2)")
            task_loaders = {}
        
        # Pretrain each decoder with its appropriate dataset
        det_result, seg_result, plate_result, ocr_result, track_result = {}, {}, {}, {}, {}
        
        if 'detection' in task_loaders:
            print("\n📦 Detection decoder (COCO)...")
            det_train, det_val = task_loaders['detection']
            det_result = trainer.pretrain_detection(det_train, val_loader=det_val, epochs=pretrain_epochs)
        else:
            print("\n⚠️ Detection decoder skipped (no COCO data)")
        
        if 'segmentation' in task_loaders:
            print("\n📦 Segmentation decoder (Cityscapes)...")
            seg_train, seg_val = task_loaders['segmentation']
            seg_result = trainer.pretrain_segmentation(seg_train, val_loader=seg_val, epochs=pretrain_epochs)
        else:
            print("\n⚠️ Segmentation decoder skipped (no Cityscapes data)")
        
        if 'plate' in task_loaders:
            print("\n📦 Plate decoder (CCPD)...")
            plate_train, plate_val = task_loaders['plate']
            plate_result = trainer.pretrain_plate(plate_train, val_loader=plate_val, epochs=pretrain_epochs)
        else:
            print("\n⚠️ Plate decoder skipped (no CCPD data)")
        
        if 'ocr' in task_loaders:
            print("\n📦 OCR decoder (CCPD)...")
            ocr_train, ocr_val = task_loaders['ocr']
            ocr_result = trainer.pretrain_ocr(ocr_train, val_loader=ocr_val, epochs=pretrain_epochs)
        else:
            print("\n⚠️ OCR decoder skipped (no CCPD data)")
        
        if 'tracking' in task_loaders:
            print("\n📦 Tracking decoder (MOT17)...")
            track_train, track_val = task_loaders['tracking']
            track_result = trainer.pretrain_tracking(track_train, val_loader=track_val, epochs=pretrain_epochs)
        else:
            print("\n⚠️ Tracking decoder skipped (no MOT17 data)")
        
        results['stage2'] = {
            'status': 'complete',
            'detection': det_result,
            'segmentation': seg_result,
            'plate': plate_result,
            'ocr': ocr_result,
            'tracking': track_result,
        }
        print("\n  Stage 2 Complete ✓")
    
    if args.stage >= 3:
        # Stage 3: Joint training
        joint_results = trainer.train_joint(
            train_loader=train_loader,
            val_loader=val_loader,
            epochs=staged_config.joint_training_epochs,
        )
        results['stage3'] = joint_results
    
    # Training summary
    total_time = time.time() - start_time
    
    print("\n" + "="*70)
    print("  🎉 Training Complete!")
    print("="*70)
    print(f"\n⏱️  Total time: {total_time/60:.1f} minutes")
    
    if 'stage3' in results and 'best_loss' in results['stage3']:
        print(f"📉 Best loss: {results['stage3']['best_loss']:.4f}")
    
    print(f"💾 Checkpoints saved to: {training_config.checkpoint_dir}")
    
    # Run final evaluation
    if val_loader:
        print("\n📊 Running final evaluation...")
        trainer.model.eval()
        trainer.evaluator.reset()
        
        with torch.no_grad():
            for images, targets in val_loader:
                images = images.to(device)
                # Move targets to device
                targets_on_device = {}
                for key, val in targets.items():
                    if isinstance(val, torch.Tensor):
                        targets_on_device[key] = val.to(device)
                    elif isinstance(val, list):
                        targets_on_device[key] = val
                    else:
                        targets_on_device[key] = val
                
                outputs = trainer.model(images)
                
                # Add batch to evaluator
                try:
                    trainer.evaluator.add_batch(outputs, targets_on_device)
                except Exception as e:
                    print(f"  Warning: evaluation error for batch: {e}")
                
        # Print summary
        print("\n" + trainer.evaluator.summarize())
    
    return results


def main():
    """Entry point."""
    args = parse_args()
    
    # Quick demo mode
    if args.demo:
        print("🚀 Running quick demo mode (2 epochs, synthetic data)")
        args.epochs = 2
        args.synthetic_samples = 20
    
    try:
        results = run_training(args)
        
        print("\n✅ Training completed successfully!")
        return 0
        
    except KeyboardInterrupt:
        print("\n\n⚠️ Training interrupted by user")
        return 1
        
    except Exception as e:
        print(f"\n❌ Training failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
