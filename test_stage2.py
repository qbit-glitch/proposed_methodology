#!/usr/bin/env python3
"""
Test Stage 2: Individual Decoder Training.

Tests the task-specific pretraining for each decoder separately:
- Detection decoder
- Segmentation decoder
- Plate decoder
- OCR decoder
- Tracking decoder
"""

import os
import sys

# Add src to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
from torch.utils.data import DataLoader

from src.config import ModelConfig, TrainingConfig
from src.staged_trainer import StagedTrainer, StagedTrainingConfig
from src.trainer import SyntheticDataset, collate_fn


def test_stage2_decoder_training():
    """Test individual decoder pretraining."""
    print("\n" + "="*70)
    print("  Testing Stage 2: Individual Decoder Training")
    print("="*70)
    
    # Create config with short epochs for testing
    config = StagedTrainingConfig()
    config.detection_pretrain_epochs = 2
    config.segmentation_pretrain_epochs = 2
    config.plate_pretrain_epochs = 2
    config.ocr_pretrain_epochs = 2
    config.tracking_pretrain_epochs = 2
    
    # Create trainer
    print("\n📦 Creating trainer...")
    trainer = StagedTrainer(config)
    
    # Stage 1: Initialize
    trainer.initialize_from_pretrained()
    
    # Create synthetic data
    print("\n📊 Creating synthetic data...")
    train_dataset = SyntheticDataset(num_samples=10)
    train_loader = DataLoader(
        train_dataset,
        batch_size=1,
        shuffle=True,
        collate_fn=collate_fn,
    )
    
    results = {}
    
    # Test each decoder
    decoders_to_test = [
        ('detection', trainer.pretrain_detection),
        ('segmentation', trainer.pretrain_segmentation),
        ('plate', trainer.pretrain_plate),
        ('ocr', trainer.pretrain_ocr),
        ('tracking', trainer.pretrain_tracking),
    ]
    
    for name, pretrain_fn in decoders_to_test:
        print(f"\n{'='*50}")
        print(f"  Testing {name.upper()} decoder pretraining")
        print(f"{'='*50}")
        
        try:
            result = pretrain_fn(
                train_loader=train_loader,
                val_loader=None,
                epochs=2,
            )
            results[name] = {
                'status': 'SUCCESS',
                'best_loss': result.get('best_loss', 'N/A'),
            }
            print(f"\n  ✓ {name} decoder pretraining: SUCCESS")
            print(f"    Best loss: {result.get('best_loss', 'N/A')}")
        except Exception as e:
            results[name] = {
                'status': 'FAILED',
                'error': str(e),
            }
            print(f"\n  ✗ {name} decoder pretraining: FAILED")
            print(f"    Error: {e}")
    
    # Summary
    print("\n" + "="*70)
    print("  Stage 2 Test Summary")
    print("="*70)
    
    for name, result in results.items():
        status = result['status']
        if status == 'SUCCESS':
            print(f"  ✓ {name}: {status} (loss: {result.get('best_loss', 'N/A'):.4f})")
        else:
            print(f"  ✗ {name}: {status} ({result.get('error', '')})")
    
    # Check if all passed
    all_passed = all(r['status'] == 'SUCCESS' for r in results.values())
    
    print("\n" + "="*70)
    if all_passed:
        print("  🎉 All Stage 2 decoder training tests PASSED!")
    else:
        print("  ⚠️  Some Stage 2 decoder training tests FAILED!")
    print("="*70)
    
    return all_passed, results


if __name__ == "__main__":
    success, results = test_stage2_decoder_training()
    sys.exit(0 if success else 1)
