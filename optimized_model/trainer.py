"""
Training Pipeline for Unified Multi-Task Transformer.

Implements Algorithm 10: Training Procedure with:
- Multi-task loss optimization
- Gradient accumulation for memory efficiency
- Learning rate scheduler (warmup + cosine decay)
- Mixed precision training (on supported devices)
- Checkpointing and visualization callbacks
"""

import os
import time
import math
from typing import Dict, Optional, Tuple, List, Callable
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import TrainingConfig, ModelConfig, get_device
from src.unified_transformer import UnifiedMultiTaskTransformer, build_model
from src.losses.multi_task_loss import MultiTaskLoss


def get_cosine_schedule_with_warmup(
    optimizer,
    num_warmup_steps: int,
    num_training_steps: int,
    min_lr_ratio: float = 0.01,
):
    """
    Create a schedule with warmup + cosine decay.
    
    η(t) = η_min + 0.5*(η_base - η_min)*(1 + cos(π*(t-T_warmup)/(T_max-T_warmup)))
    """
    def lr_lambda(current_step: int) -> float:
        if current_step < num_warmup_steps:
            # Linear warmup
            return float(current_step) / float(max(1, num_warmup_steps))
        # Cosine decay
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        return max(min_lr_ratio, 0.5 * (1.0 + math.cos(math.pi * progress)))
    
    return LambdaLR(optimizer, lr_lambda)


class SyntheticDataset(Dataset):
    """
    Synthetic dataset for testing the training pipeline.
    
    Generates random images with random annotations.
    """
    
    def __init__(
        self,
        num_samples: int = 100,
        image_size: Tuple[int, int] = (512, 512),
        num_vehicle_classes: int = 6,
        max_objects: int = 10,
    ):
        self.num_samples = num_samples
        self.image_size = image_size
        self.num_vehicle_classes = num_vehicle_classes
        self.max_objects = max_objects
    
    def __len__(self) -> int:
        return self.num_samples
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Dict]:
        H, W = self.image_size
        
        # Generate random image
        image = torch.rand(3, H, W)
        
        # Generate random targets
        num_objects = torch.randint(1, self.max_objects + 1, (1,)).item()
        
        # Generate detection boxes (cx, cy, w, h normalized)
        det_boxes = torch.rand(num_objects, 4)
        det_boxes[:, 2:] = det_boxes[:, 2:] * 0.3 + 0.05  # Reasonable box sizes
        
        # Fixed number of tracking queries to match model output
        num_queries = 100
        
        targets = {
            'detection': {
                'labels': torch.randint(0, self.num_vehicle_classes, (num_objects,)),
                'boxes': det_boxes,
            },
            'segmentation': torch.randint(0, 3, (H, W)),  # Random seg mask
            'plate': {
                'labels': torch.zeros(num_objects, dtype=torch.long),  # All plates
                'boxes': torch.rand(num_objects, 4),
            },
            'ocr': [self._random_plate_text() for _ in range(num_objects)],
            # Tracking targets (fixed size 100 to match model queries)
            'tracking': {
                # Track IDs for each query (-1 for invalid/background)
                'track_ids': torch.cat([torch.arange(num_objects), torch.full((num_queries - num_objects,), -1)]),
                # Previous track IDs
                'prev_track_ids': torch.cat([torch.arange(num_objects), torch.full((num_queries - num_objects,), -1)]),
                # Association matrix: [num_queries, num_queries] - diagonal for valid, zeros for padding
                'associations': self._make_padded_associations(num_objects, num_queries),
                # Trajectory deltas (dx, dy, dw, dh) - zeros for padding
                'trajectory': torch.cat([torch.randn(num_objects, 4) * 0.02, torch.zeros(num_queries - num_objects, 4)]),
                # Previous boxes
                'prev_boxes': torch.cat([det_boxes + torch.randn(num_objects, 4) * 0.02, torch.zeros(num_queries - num_objects, 4)]),
            },
        }
        
        return image, targets
    
    def _random_plate_text(self) -> str:
        """Generate random plate text."""
        letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        digits = "0123456789"
        # Format: XX00XX0000
        text = (
            letters[torch.randint(0, 26, (1,)).item()] +
            letters[torch.randint(0, 26, (1,)).item()] +
            digits[torch.randint(0, 10, (1,)).item()] +
            digits[torch.randint(0, 10, (1,)).item()] +
            letters[torch.randint(0, 26, (1,)).item()] +
            letters[torch.randint(0, 26, (1,)).item()] +
            digits[torch.randint(0, 10, (1,)).item()] +
            digits[torch.randint(0, 10, (1,)).item()] +
            digits[torch.randint(0, 10, (1,)).item()] +
            digits[torch.randint(0, 10, (1,)).item()]
        )
        return text
    
    def _make_padded_associations(self, num_objects: int, num_queries: int) -> torch.Tensor:
        """
        Create padded association matrix.
        
        Returns [num_queries, num_queries] matrix where:
        - Diagonal entries for valid objects = 1 (same track matches same detection)
        - All other entries = 0
        """
        # Create identity matrix for valid objects, zeros elsewhere
        associations = torch.zeros(num_queries, num_queries, dtype=torch.long)
        associations[:num_objects, :num_objects] = torch.eye(num_objects, dtype=torch.long)
        return associations


def collate_fn(batch: List[Tuple]) -> Tuple[torch.Tensor, Dict]:
    """Custom collate function for variable-length targets."""
    images = torch.stack([item[0] for item in batch])
    
    targets = {
        'detection': [item[1]['detection'] for item in batch],
        'segmentation': torch.stack([item[1]['segmentation'] for item in batch]),
        'plate': [item[1]['plate'] for item in batch],
        'ocr': [text for item in batch for text in item[1]['ocr']],
        'tracking': [item[1].get('tracking', {}) for item in batch],
    }
    
    return images, targets


class Trainer:
    """
    Trainer for Unified Multi-Task Transformer.
    
    Features:
    - Gradient accumulation
    - Mixed precision training
    - Warmup + cosine LR schedule
    - Checkpointing
    - Visualization callbacks
    """
    
    def __init__(
        self,
        model: UnifiedMultiTaskTransformer,
        config: TrainingConfig,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
    ):
        self.model = model
        self.config = config
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = config.device
        
        # Move model to device
        self.model = self.model.to(self.device)
        
        # Loss function
        self.loss_fn = MultiTaskLoss(
            num_vehicle_classes=6,
            num_seg_classes=3,
            lambda_det=config.lambda_det,
            lambda_seg=config.lambda_seg,
            lambda_plate=config.lambda_plate,
            lambda_ocr=config.lambda_ocr,
            lambda_track=config.lambda_track,
        )
        
        # Optimizer with different LR for backbone
        backbone_params = list(self.model.backbone.parameters())
        other_params = [p for n, p in self.model.named_parameters() 
                       if 'backbone' not in n]
        
        self.optimizer = AdamW([
            {'params': backbone_params, 'lr': config.backbone_lr},
            {'params': other_params, 'lr': config.learning_rate},
        ], weight_decay=config.weight_decay)
        
        # Calculate training steps
        self.num_training_steps = len(train_loader) * config.max_epochs
        
        # LR scheduler
        self.scheduler = get_cosine_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=config.warmup_steps,
            num_training_steps=self.num_training_steps,
        )
        
        # Mixed precision (for CUDA, MPS has limited support)
        self.use_amp = config.use_mixed_precision and self.device.type == 'cuda'
        self.scaler = torch.cuda.amp.GradScaler() if self.use_amp else None
        
        # Tracking
        self.global_step = 0
        self.epoch = 0
        self.best_loss = float('inf')
        
        # Create checkpoint directory
        os.makedirs(config.checkpoint_dir, exist_ok=True)
        
        print(f"✓ Trainer initialized")
        print(f"  - Device: {self.device}")
        print(f"  - Training steps: {self.num_training_steps}")
        print(f"  - Gradient accumulation: {config.gradient_accumulation_steps}")
        print(f"  - Effective batch size: {config.batch_size * config.gradient_accumulation_steps}")
        print(f"  - Mixed precision: {self.use_amp}")
    
    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        epoch_losses = {}
        num_batches = 0
        
        self.optimizer.zero_grad()
        
        for batch_idx, (images, targets) in enumerate(self.train_loader):
            # Move data to device
            images = images.to(self.device)
            targets = self._move_targets_to_device(targets)
            
            # Forward pass
            if self.use_amp:
                with torch.cuda.amp.autocast():
                    outputs = self._forward_and_loss(images, targets)
            else:
                outputs = self._forward_and_loss(images, targets)
            
            loss = outputs['loss'] / self.config.gradient_accumulation_steps
            loss_dict = outputs['loss_dict']
            
            # Backward pass
            if self.use_amp:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()
            
            # Gradient accumulation
            if (batch_idx + 1) % self.config.gradient_accumulation_steps == 0:
                # Clip gradients
                if self.config.clip_grad_norm > 0:
                    if self.use_amp:
                        self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), 
                        self.config.clip_grad_norm
                    )
                
                # Optimizer step
                if self.use_amp:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    self.optimizer.step()
                
                self.optimizer.zero_grad()
                self.scheduler.step()
                self.global_step += 1
            
            # Accumulate losses
            for k, v in loss_dict.items():
                if isinstance(v, torch.Tensor):
                    if k not in epoch_losses:
                        epoch_losses[k] = 0.0
                    epoch_losses[k] += v.item()
            num_batches += 1
            
            # Progress
            if batch_idx % 10 == 0:
                lr = self.scheduler.get_last_lr()[0]
                print(f"  Batch {batch_idx}/{len(self.train_loader)} | "
                      f"Loss: {loss.item() * self.config.gradient_accumulation_steps:.4f} | "
                      f"LR: {lr:.2e}")
        
        # Average losses
        for k in epoch_losses:
            epoch_losses[k] /= num_batches
        
        return epoch_losses
    
    def _forward_and_loss(
        self, 
        images: torch.Tensor, 
        targets: Dict
    ) -> Dict:
        """Forward pass and loss computation."""
        # Model forward
        outputs = self.model(images)
        
        # Prepare outputs for loss
        loss_outputs = {
            'detection': outputs['detection'],
            'segmentation': outputs['segmentation'],
            'plate': outputs['plate'],
            'ocr': outputs['ocr'],
            'tracking': outputs['tracking'],
        }
        
        # Compute loss
        total_loss, loss_dict = self.loss_fn(loss_outputs, targets)
        
        return {'loss': total_loss, 'loss_dict': loss_dict}
    
    def _move_targets_to_device(self, targets: Dict) -> Dict:
        """Move targets to device."""
        moved = {}
        
        # Detection: list of dicts
        if 'detection' in targets:
            moved['detection'] = [
                {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                 for k, v in t.items()}
                for t in targets['detection']
            ]
        
        # Segmentation: tensor
        if 'segmentation' in targets:
            moved['segmentation'] = targets['segmentation'].to(self.device)
        
        # Plate: list of dicts
        if 'plate' in targets:
            moved['plate'] = [
                {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                 for k, v in t.items()}
                for t in targets['plate']
            ]
        
        # OCR: list of strings (no device needed)
        if 'ocr' in targets:
            moved['ocr'] = targets['ocr']
        
        return moved
    
    def validate(self) -> Dict[str, float]:
        """Validate on validation set."""
        if self.val_loader is None:
            return {}
        
        self.model.eval()
        val_losses = {}
        num_batches = 0
        
        with torch.no_grad():
            for images, targets in self.val_loader:
                images = images.to(self.device)
                targets = self._move_targets_to_device(targets)
                
                outputs = self._forward_and_loss(images, targets)
                
                for k, v in outputs['loss_dict'].items():
                    if isinstance(v, torch.Tensor):
                        if k not in val_losses:
                            val_losses[k] = 0.0
                        val_losses[k] += v.item()
                num_batches += 1
        
        for k in val_losses:
            val_losses[k] /= num_batches
        
        return val_losses
    
    def save_checkpoint(self, path: str, is_best: bool = False):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': self.epoch,
            'global_step': self.global_step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_loss': self.best_loss,
        }
        
        torch.save(checkpoint, path)
        print(f"✓ Saved checkpoint: {path}")
        
        if is_best:
            best_path = path.replace('.pt', '_best.pt')
            torch.save(checkpoint, best_path)
            print(f"✓ Saved best model: {best_path}")
    
    def load_checkpoint(self, path: str):
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.epoch = checkpoint['epoch']
        self.global_step = checkpoint['global_step']
        self.best_loss = checkpoint['best_loss']
        
        print(f"✓ Loaded checkpoint from epoch {self.epoch}")
    
    def train(
        self, 
        num_epochs: Optional[int] = None,
        callbacks: Optional[List[Callable]] = None,
    ):
        """
        Main training loop.
        
        Args:
            num_epochs: Number of epochs (default: config.max_epochs)
            callbacks: Optional list of callback functions
        """
        num_epochs = num_epochs or self.config.max_epochs
        
        print(f"\n{'='*60}")
        print(f"  Starting Training for {num_epochs} epochs")
        print(f"{'='*60}\n")
        
        for epoch in range(self.epoch, num_epochs):
            self.epoch = epoch
            epoch_start = time.time()
            
            print(f"Epoch {epoch + 1}/{num_epochs}")
            print("-" * 40)
            
            # Train
            train_losses = self.train_epoch()
            
            # Validate
            val_losses = self.validate()
            
            # Print summary
            epoch_time = time.time() - epoch_start
            print(f"\nEpoch {epoch + 1} Summary (time: {epoch_time:.1f}s):")
            print(f"  Train Loss: {train_losses.get('total_loss', 0):.4f}")
            if val_losses:
                print(f"  Val Loss:   {val_losses.get('total_loss', 0):.4f}")
            
            # Check for best model
            current_loss = val_losses.get('total_loss', train_losses.get('total_loss', float('inf')))
            is_best = current_loss < self.best_loss
            if is_best:
                self.best_loss = current_loss
            
            # Save checkpoint
            if (epoch + 1) % self.config.save_every == 0:
                path = os.path.join(
                    self.config.checkpoint_dir,
                    f"checkpoint_epoch_{epoch + 1}.pt"
                )
                self.save_checkpoint(path, is_best)
            
            # Callbacks
            if callbacks:
                for callback in callbacks:
                    callback(self, epoch, train_losses, val_losses)
            
            print()
        
        print(f"\n{'='*60}")
        print(f"  Training Complete!")
        print(f"  Best Loss: {self.best_loss:.4f}")
        print(f"{'='*60}")


def train_demo():
    """Run a quick training demo."""
    print("🚀 Training Demo\n")
    
    # Config
    model_config = ModelConfig()
    train_config = TrainingConfig()
    train_config.max_epochs = 2
    train_config.batch_size = 1
    train_config.gradient_accumulation_steps = 2
    
    # Build model
    model = build_model(model_config)
    
    # Create synthetic dataset
    train_dataset = SyntheticDataset(num_samples=20)
    train_loader = DataLoader(
        train_dataset,
        batch_size=train_config.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=0,
    )
    
    # Create trainer
    trainer = Trainer(
        model=model,
        config=train_config,
        train_loader=train_loader,
    )
    
    # Train
    trainer.train(num_epochs=2)
    
    return trainer


if __name__ == "__main__":
    train_demo()
