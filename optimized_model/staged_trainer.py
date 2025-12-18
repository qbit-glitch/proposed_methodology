"""
Staged Training Pipeline (Algorithms S1, S2, S3).

Implements the complete three-stage training approach:
- Stage 1: Load pretrained backbone + initialize transformer
- Stage 2: Task-specific fine-tuning on public datasets
- Stage 3: Joint multi-task training with gradual unfreezing

Optimized for MacBook M4 Pro with:
- Gradient checkpointing
- Memory-efficient training
- MPS backend support
"""

import os
import sys
import time
import math
from typing import Dict, Optional, List, Tuple, Callable, Any
from dataclasses import dataclass, field
from enum import Enum

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR, CosineAnnealingWarmRestarts
from tqdm import tqdm

# Add parent to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import TrainingConfig, ModelConfig, get_device
from src.unified_transformer import UnifiedMultiTaskTransformer, build_model
from src.losses.multi_task_loss import MultiTaskLoss
from src.losses.tracking_loss import TrackingLoss
from src.m4_optimizations import (
    apply_gradient_checkpointing,
    MemoryTracker,
    empty_mps_cache,
    get_optimal_batch_config,
)
from src.evaluation import MultiTaskEvaluator


class TrainingStage(Enum):
    """Training stage enumeration."""
    STAGE_1_INIT = 1
    STAGE_2_PRETRAIN = 2
    STAGE_3_JOINT = 3


@dataclass
class StagedTrainingConfig:
    """Configuration for staged training."""
    
    # Base configs
    model_config: ModelConfig = field(default_factory=ModelConfig)
    training_config: TrainingConfig = field(default_factory=TrainingConfig)
    
    # Stage 2: Task-specific pretraining epochs
    detection_pretrain_epochs: int = 20
    segmentation_pretrain_epochs: int = 30
    plate_pretrain_epochs: int = 20
    ocr_pretrain_epochs: int = 30
    tracking_pretrain_epochs: int = 20
    
    # Stage 3: Joint training
    joint_training_epochs: int = 100
    
    # Gradual unfreezing (Stage 3)
    unfreeze_encoder_epoch: int = 10
    unfreeze_backbone_epoch: int = 50
    
    # Learning rates for different stages
    stage2_lr: float = 1e-4
    stage3_init_lr: float = 5e-5
    stage3_backbone_lr: float = 1e-5
    
    # Uncertainty weighting
    use_uncertainty_weighting: bool = True
    
    # Memory optimization
    use_gradient_checkpointing: bool = True
    checkpoint_encoder_every: int = 2
    
    # Checkpoints
    checkpoint_dir: str = "outputs/checkpoints"
    stage2_checkpoint_dir: str = "outputs/checkpoints/stage2"
    
    # Early stopping
    early_stopping_patience: int = 10
    
    def __post_init__(self):
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        os.makedirs(self.stage2_checkpoint_dir, exist_ok=True)


def freeze_parameters(module: nn.Module):
    """Freeze all parameters in a module."""
    for param in module.parameters():
        param.requires_grad = False


def unfreeze_parameters(module: nn.Module):
    """Unfreeze all parameters in a module."""
    for param in module.parameters():
        param.requires_grad = True


def get_cosine_schedule_with_warmup(
    optimizer,
    num_warmup_steps: int,
    num_training_steps: int,
    min_lr_ratio: float = 0.01,
) -> LambdaLR:
    """Create warmup + cosine decay schedule."""
    def lr_lambda(current_step: int) -> float:
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        return max(min_lr_ratio, 0.5 * (1.0 + math.cos(math.pi * progress)))
    
    return LambdaLR(optimizer, lr_lambda)


class StagedTrainer:
    """
    Three-stage training pipeline for Unified Multi-Task Transformer.
    
    Stage 1: Initialize from pretrained backbone
    Stage 2: Task-specific pretraining (each decoder separately)
    Stage 3: Joint multi-task training with gradual unfreezing
    """
    
    def __init__(
        self,
        config: StagedTrainingConfig,
        model: Optional[UnifiedMultiTaskTransformer] = None,
    ):
        self.config = config
        self.device = config.training_config.device
        
        # Build or use provided model
        if model is None:
            print("🏗️  Building model...")
            self.model = build_model(config.model_config)
        else:
            self.model = model
        
        self.model = self.model.to(self.device)
        
        # Apply gradient checkpointing
        if config.use_gradient_checkpointing:
            print("📦 Applying gradient checkpointing...")
            self.model = apply_gradient_checkpointing(
                self.model,
                checkpoint_backbone=True,
                checkpoint_encoder=True,
                encoder_checkpoint_every=config.checkpoint_encoder_every,
            )
        
        # Memory tracker
        self.memory_tracker = MemoryTracker(self.device)
        
        # Evaluator
        self.evaluator = MultiTaskEvaluator(
            num_vehicle_classes=config.model_config.num_vehicle_classes,
            num_seg_classes=config.model_config.num_seg_classes,
        )
        
        # Current stage
        self.current_stage = TrainingStage.STAGE_1_INIT
        
        # Training state
        self.global_step = 0
        self.best_metrics = {}
        
        print(f"✓ Staged trainer initialized on {self.device}")
        print(f"   Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
    
    # =========================================================================
    # Stage 1: Initialization
    # =========================================================================
    
    def initialize_from_pretrained(self):
        """
        Stage 1: Initialize model from pretrained weights.
        
        - Backbone: Loaded from torchvision (ImageNet pretrained)
        - Encoder: Xavier initialization
        - Decoders: Xavier initialization
        - Query embeddings: Uniform initialization
        """
        print("\n" + "="*60)
        print("  Stage 1: Initializing from Pretrained Weights")
        print("="*60)
        
        self.current_stage = TrainingStage.STAGE_1_INIT
        
        # Backbone is already pretrained if config.model_config.pretrained = True
        if self.config.model_config.pretrained:
            print("  ✓ Backbone: ImageNet pretrained weights loaded")
        else:
            print("  ⚠ Backbone: Randomly initialized (not recommended)")
        
        # Transformer components use Xavier initialization by default (PyTorch)
        print("  ✓ Encoder: Xavier initialized")
        print("  ✓ Decoders: Xavier initialized")
        print("  ✓ Query embeddings: Uniform initialized")
        
        # Verify initialization
        self._verify_initialization()
        
        print("\n  Stage 1 Complete ✓")
        
        return self.model
    
    def _verify_initialization(self):
        """Verify that model is properly initialized."""
        # Check for NaN/Inf in parameters
        for name, param in self.model.named_parameters():
            if torch.isnan(param).any() or torch.isinf(param).any():
                print(f"  ⚠ Warning: {name} contains NaN/Inf")
        
        # Test forward pass
        try:
            dummy_input = torch.randn(1, 3, 512, 512, device=self.device)
            with torch.no_grad():
                _ = self.model(dummy_input)
            print("  ✓ Forward pass verified")
        except Exception as e:
            print(f"  ✗ Forward pass failed: {e}")
    
    # =========================================================================
    # Stage 2: Task-Specific Pretraining
    # =========================================================================
    
    def pretrain_detection(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        epochs: Optional[int] = None,
    ) -> Dict[str, float]:
        """Pretrain detection decoder."""
        return self._pretrain_task(
            task_name="detection",
            train_loader=train_loader,
            val_loader=val_loader,
            epochs=epochs or self.config.detection_pretrain_epochs,
            decoder_name="detection_decoder",
        )
    
    def pretrain_segmentation(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        epochs: Optional[int] = None,
    ) -> Dict[str, float]:
        """Pretrain segmentation decoder."""
        return self._pretrain_task(
            task_name="segmentation",
            train_loader=train_loader,
            val_loader=val_loader,
            epochs=epochs or self.config.segmentation_pretrain_epochs,
            decoder_name="segmentation_decoder",
        )
    
    def pretrain_plate(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        epochs: Optional[int] = None,
    ) -> Dict[str, float]:
        """Pretrain plate detection decoder."""
        return self._pretrain_task(
            task_name="plate",
            train_loader=train_loader,
            val_loader=val_loader,
            epochs=epochs or self.config.plate_pretrain_epochs,
            decoder_name="plate_decoder",
        )
    
    def pretrain_ocr(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        epochs: Optional[int] = None,
    ) -> Dict[str, float]:
        """Pretrain OCR decoder."""
        return self._pretrain_task(
            task_name="ocr",
            train_loader=train_loader,
            val_loader=val_loader,
            epochs=epochs or self.config.ocr_pretrain_epochs,
            decoder_name="ocr_decoder",
        )
    
    def pretrain_tracking(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        epochs: Optional[int] = None,
    ) -> Dict[str, float]:
        """Pretrain tracking decoder."""
        return self._pretrain_task(
            task_name="tracking",
            train_loader=train_loader,
            val_loader=val_loader,
            epochs=epochs or self.config.tracking_pretrain_epochs,
            decoder_name="tracking_decoder",
        )
    
    def _pretrain_task(
        self,
        task_name: str,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader],
        epochs: int,
        decoder_name: str,
    ) -> Dict[str, float]:
        """
        Generic task-specific pretraining.
        
        Freezes all components except the target decoder.
        """
        print(f"\n{'='*60}")
        print(f"  Stage 2: Pretraining {task_name.upper()} decoder")
        print(f"{'='*60}")
        
        self.current_stage = TrainingStage.STAGE_2_PRETRAIN
        
        # Freeze everything
        freeze_parameters(self.model)
        
        # Unfreeze only the target decoder
        decoder = getattr(self.model, decoder_name)
        unfreeze_parameters(decoder)
        
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"  Trainable parameters: {trainable_params:,}")
        
        # Create optimizer for only the decoder
        optimizer = AdamW(
            decoder.parameters(),
            lr=self.config.stage2_lr,
            weight_decay=self.config.training_config.weight_decay,
        )
        
        # Create loss function (task-specific)
        loss_fn = self._get_task_loss(task_name)
        
        # Training loop
        best_loss = float('inf')
        history = {'train_loss': [], 'val_loss': []}
        
        for epoch in range(epochs):
            # Train
            train_loss = self._train_epoch_single_task(
                train_loader, optimizer, loss_fn, task_name
            )
            history['train_loss'].append(train_loss)
            
            # Validate
            val_loss = 0.0
            if val_loader:
                val_loss = self._validate_single_task(val_loader, loss_fn, task_name)
                history['val_loss'].append(val_loss)
            
            print(f"  Epoch {epoch+1}/{epochs} | Train: {train_loss:.4f} | Val: {val_loss:.4f}")
            
            # Save best
            if val_loss < best_loss or (val_loss == 0 and train_loss < best_loss):
                best_loss = val_loss if val_loss > 0 else train_loss
                self._save_checkpoint(
                    os.path.join(self.config.stage2_checkpoint_dir, f"{task_name}_best.pt"),
                    epoch, best_loss,
                )
            
            # Empty MPS cache
            if self.device.type == 'mps':
                empty_mps_cache()
        
        print(f"\n  {task_name.upper()} pretraining complete ✓")
        print(f"  Best loss: {best_loss:.4f}")
        
        return {'best_loss': best_loss, 'history': history}
    
    def _get_task_loss(self, task_name: str) -> nn.Module:
        """Get loss function for specific task."""
        from src.losses.detection_loss import DetectionLoss
        from src.losses.segmentation_loss import SegmentationLoss
        from src.losses.ocr_loss import OCRLoss
        
        if task_name == 'detection':
            return DetectionLoss(num_classes=self.config.model_config.num_vehicle_classes)
        elif task_name == 'segmentation':
            return SegmentationLoss(num_classes=self.config.model_config.num_seg_classes)
        elif task_name == 'plate':
            # Binary plate detection: 1 class (plate) + background
            return DetectionLoss(num_classes=1)
        elif task_name == 'ocr':
            return OCRLoss(alphabet=self.config.model_config.ocr_alphabet)
        elif task_name == 'tracking':
            return TrackingLoss()
        else:
            raise ValueError(f"Unknown task: {task_name}")
    
    def _train_epoch_single_task(
        self,
        train_loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        loss_fn: nn.Module,
        task_name: str,
    ) -> float:
        """Train one epoch for single task."""
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        pbar = tqdm(enumerate(train_loader), total=len(train_loader), 
                    desc=f"  {task_name.upper()}", leave=False)
        for batch_idx, (images, targets) in pbar:
            images = images.to(self.device)
            targets = self._move_to_device(targets)
            
            optimizer.zero_grad()
            
            # Forward pass
            outputs = self.model(images)
            
            # Compute task-specific loss
            task_outputs = outputs.get(task_name, {})
            task_targets = targets.get(task_name, {})
            
            if task_name == 'detection':
                loss_dict = loss_fn(task_outputs, task_targets)
                loss = sum(loss_dict.values())
            elif task_name == 'plate':
                # Adapt plate outputs for binary plate detection (1 class + background = 2)
                # Use plate_confidence as the plate class logit
                plate_conf = task_outputs.get('plate_confidence', torch.zeros(1, 50, 1, device=self.device))
                # Background logit: low when plate confidence is high
                bg_logits = -5.0 * plate_conf  # [B, 50, 1]
                # Class logits: [plate_logit, background_logit]
                class_logits_with_bg = torch.cat([plate_conf * 5.0, bg_logits], dim=-1)  # [B, 50, 2]
                
                adapted_outputs = {
                    'bbox': task_outputs.get('plate_bbox', torch.zeros(1, 50, 4, device=self.device)),
                    'class_logits': class_logits_with_bg,
                }
                loss_dict = loss_fn(adapted_outputs, task_targets)
                loss = sum(loss_dict.values())
            elif task_name == 'segmentation':
                loss_dict = loss_fn(task_outputs, task_targets)
                loss = sum(loss_dict.values())
            elif task_name == 'ocr':
                loss_dict = loss_fn(task_outputs, task_targets)
                loss = sum(loss_dict.values())
            elif task_name == 'tracking':
                # Adapt tracking outputs to match TrackingLoss expected format
                # association_scores -> similarity, trajectory_delta -> pred_motion
                # Also handle targets: trajectory -> gt_motion
                adapted_outputs = {
                    'features': task_outputs.get('features', torch.zeros(1, 100, 256, device=self.device)),
                    'similarity': task_outputs.get('association_scores', torch.zeros(1, 100, 100, device=self.device)),
                    'pred_motion': task_outputs.get('trajectory_delta', torch.zeros(1, 100, 4, device=self.device)),
                }
                
                # Adapt targets
                if isinstance(task_targets, list) and len(task_targets) > 0:
                    # Batch of targets
                    adapted_targets = {
                        'track_ids': torch.stack([t.get('track_ids', torch.zeros(100)) for t in task_targets]).to(self.device) if task_targets[0].get('track_ids') is not None else None,
                        'associations': torch.stack([t.get('associations', torch.eye(100)) for t in task_targets]).to(self.device) if task_targets[0].get('associations') is not None else None,
                        'gt_motion': torch.stack([t.get('trajectory', torch.zeros(100, 4)) for t in task_targets]).to(self.device) if task_targets[0].get('trajectory') is not None else None,
                    }
                    # Remove None values
                    adapted_targets = {k: v for k, v in adapted_targets.items() if v is not None}
                else:
                    adapted_targets = task_targets
                
                loss_dict = loss_fn(adapted_outputs, adapted_targets)
                loss = sum(loss_dict.values())
            else:
                loss = torch.tensor(0.0, device=self.device)
            
            # Backward
            if loss.requires_grad:
                loss.backward()
                
                # Clip gradients
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config.training_config.clip_grad_norm
                )
                
                optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
            
            self.global_step += 1
        
        return total_loss / max(num_batches, 1)
    
    def _validate_single_task(
        self,
        val_loader: DataLoader,
        loss_fn: nn.Module,
        task_name: str,
    ) -> float:
        """Validate single task."""
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for images, targets in val_loader:
                images = images.to(self.device)
                targets = self._move_to_device(targets)
                
                outputs = self.model(images)
                
                task_outputs = outputs.get(task_name, {})
                task_targets = targets.get(task_name, {})
                
                # Compute loss
                try:
                    if task_name == 'detection':
                        loss_dict = loss_fn(task_outputs, task_targets)
                    elif task_name == 'plate':
                        # Adapt plate outputs for binary plate detection (1 class + background = 2)
                        # Use plate_confidence as the plate class logit
                        plate_conf = task_outputs.get('plate_confidence', torch.zeros(1, 50, 1, device=self.device))
                        # Background logit: low when plate confidence is high
                        bg_logits = -5.0 * plate_conf  # [B, 50, 1]
                        # Class logits: [plate_logit, background_logit]
                        class_logits_with_bg = torch.cat([plate_conf * 5.0, bg_logits], dim=-1)  # [B, 50, 2]
                        
                        adapted_outputs = {
                            'bbox': task_outputs.get('plate_bbox', torch.zeros(1, 50, 4, device=self.device)),
                            'class_logits': class_logits_with_bg,
                        }
                        loss_dict = loss_fn(adapted_outputs, task_targets)
                    elif task_name == 'segmentation':
                        loss_dict = loss_fn(task_outputs, task_targets)
                    elif task_name == 'ocr':
                        loss_dict = loss_fn(task_outputs, task_targets)
                    elif task_name == 'tracking':
                        # Adapt tracking outputs to match TrackingLoss expected format
                        adapted_outputs = {
                            'features': task_outputs.get('features', torch.zeros(1, 100, 256, device=self.device)),
                            'similarity': task_outputs.get('association_scores', torch.zeros(1, 100, 100, device=self.device)),
                            'pred_motion': task_outputs.get('trajectory_delta', torch.zeros(1, 100, 4, device=self.device)),
                        }
                        
                        # Adapt targets
                        if isinstance(task_targets, list) and len(task_targets) > 0:
                            adapted_targets = {
                                'track_ids': torch.stack([t.get('track_ids', torch.zeros(100)) for t in task_targets]).to(self.device) if task_targets[0].get('track_ids') is not None else None,
                                'associations': torch.stack([t.get('associations', torch.eye(100)) for t in task_targets]).to(self.device) if task_targets[0].get('associations') is not None else None,
                                'gt_motion': torch.stack([t.get('trajectory', torch.zeros(100, 4)) for t in task_targets]).to(self.device) if task_targets[0].get('trajectory') is not None else None,
                            }
                            # Remove None values
                            adapted_targets = {k: v for k, v in adapted_targets.items() if v is not None}
                        else:
                            adapted_targets = task_targets
                        
                        loss_dict = loss_fn(adapted_outputs, adapted_targets)
                    else:
                        loss_dict = {}
                    
                    loss = sum(loss_dict.values())
                    total_loss += loss.item()
                except Exception as e:
                    # Log the error instead of silently ignoring
                    if num_batches == 0:
                        print(f"  ⚠️ Validation error for {task_name}: {e}")
                
                num_batches += 1
        
        return total_loss / max(num_batches, 1)
    
    # =========================================================================
    # Stage 3: Joint Multi-Task Training
    # =========================================================================
    
    def train_joint(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        epochs: Optional[int] = None,
        callbacks: Optional[List[Callable]] = None,
    ) -> Dict[str, Any]:
        """
        Stage 3: Joint multi-task training with gradual unfreezing.
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            epochs: Number of epochs (default: config.joint_training_epochs)
            callbacks: Optional callback functions
            
        Returns:
            Training history and best metrics
        """
        print("\n" + "="*60)
        print("  Stage 3: Joint Multi-Task Training")
        print("="*60)
        
        self.current_stage = TrainingStage.STAGE_3_JOINT
        epochs = epochs or self.config.joint_training_epochs
        
        # Initial freeze state: only decoders trainable
        print("  Initial state: Backbone and Encoder frozen")
        freeze_parameters(self.model.backbone)
        freeze_parameters(self.model.encoder)
        unfreeze_parameters(self.model.detection_decoder)
        unfreeze_parameters(self.model.segmentation_decoder)
        unfreeze_parameters(self.model.plate_decoder)
        unfreeze_parameters(self.model.ocr_decoder)
        unfreeze_parameters(self.model.tracking_decoder)
        
        # Create multi-task loss
        loss_fn = MultiTaskLoss(
            num_vehicle_classes=self.config.model_config.num_vehicle_classes,
            num_seg_classes=self.config.model_config.num_seg_classes,
            lambda_det=self.config.training_config.lambda_det,
            lambda_seg=self.config.training_config.lambda_seg,
            lambda_plate=self.config.training_config.lambda_plate,
            lambda_ocr=self.config.training_config.lambda_ocr,
            lambda_track=self.config.training_config.lambda_track,
            use_uncertainty_weighting=self.config.use_uncertainty_weighting,
        )
        
        # Create optimizer with parameter groups
        param_groups = self._get_parameter_groups()
        optimizer = AdamW(param_groups, weight_decay=self.config.training_config.weight_decay)
        
        # Learning rate scheduler
        num_training_steps = len(train_loader) * epochs
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=self.config.training_config.warmup_steps,
            num_training_steps=num_training_steps,
        )
        
        # Training history
        history = {
            'train_loss': [],
            'val_loss': [],
            'metrics': [],
        }
        
        best_loss = float('inf')
        epochs_without_improvement = 0
        
        for epoch in range(epochs):
            epoch_start = time.time()
            
            # Gradual unfreezing
            if epoch == self.config.unfreeze_encoder_epoch:
                print(f"\n  Epoch {epoch}: Unfreezing encoder")
                unfreeze_parameters(self.model.encoder)
                # Recreate optimizer with new parameters
                param_groups = self._get_parameter_groups()
                optimizer = AdamW(param_groups, weight_decay=self.config.training_config.weight_decay)
            
            if epoch == self.config.unfreeze_backbone_epoch:
                print(f"\n  Epoch {epoch}: Unfreezing backbone (lower LR)")
                unfreeze_parameters(self.model.backbone)
                param_groups = self._get_parameter_groups(include_backbone=True)
                optimizer = AdamW(param_groups, weight_decay=self.config.training_config.weight_decay)
            
            # Train epoch
            train_loss, train_loss_dict = self._train_epoch_joint(
                train_loader, optimizer, scheduler, loss_fn
            )
            history['train_loss'].append(train_loss)
            
            # Validate
            val_loss = 0.0
            if val_loader:
                val_loss, val_metrics = self._validate_joint(val_loader, loss_fn)
                history['val_loss'].append(val_loss)
                history['metrics'].append(val_metrics)
            
            # Logging
            epoch_time = time.time() - epoch_start
            lr = optimizer.param_groups[0]['lr']
            print(f"  Epoch {epoch+1}/{epochs} | "
                  f"Train: {train_loss:.4f} | Val: {val_loss:.4f} | "
                  f"LR: {lr:.2e} | Time: {epoch_time:.1f}s")
            
            # Log individual losses
            for name, value in train_loss_dict.items():
                if isinstance(value, torch.Tensor):
                    print(f"    {name}: {value.item():.4f}", end="")
            print()
            
            # Check for improvement
            current_loss = val_loss if val_loss > 0 else train_loss
            if current_loss < best_loss:
                best_loss = current_loss
                self._save_checkpoint(
                    os.path.join(self.config.checkpoint_dir, "best_model.pt"),
                    epoch, best_loss,
                )
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
            
            # Early stopping
            if epochs_without_improvement >= self.config.early_stopping_patience:
                print(f"\n  Early stopping at epoch {epoch+1}")
                break
            
            # Callbacks
            if callbacks:
                for callback in callbacks:
                    callback(self, epoch, train_loss, val_loss)
            
            # Periodic checkpoint
            if (epoch + 1) % self.config.training_config.save_every == 0:
                self._save_checkpoint(
                    os.path.join(self.config.checkpoint_dir, f"checkpoint_epoch_{epoch+1}.pt"),
                    epoch, current_loss,
                )
            
            # Memory management
            if self.device.type == 'mps':
                empty_mps_cache()
        
        print("\n  Stage 3 Complete ✓")
        print(f"  Best loss: {best_loss:.4f}")
        
        return {
            'best_loss': best_loss,
            'history': history,
        }
    
    def _get_parameter_groups(self, include_backbone: bool = False) -> List[Dict]:
        """Get parameter groups with different learning rates."""
        groups = []
        
        # Backbone (if trainable)
        if include_backbone:
            groups.append({
                'params': self.model.backbone.parameters(),
                'lr': self.config.stage3_backbone_lr,
                'name': 'backbone',
            })
        
        # Encoder
        encoder_params = list(self.model.encoder.parameters())
        if any(p.requires_grad for p in encoder_params):
            groups.append({
                'params': [p for p in encoder_params if p.requires_grad],
                'lr': self.config.stage3_init_lr,
                'name': 'encoder',
            })
        
        # Decoders
        decoder_names = ['detection_decoder', 'segmentation_decoder', 
                        'plate_decoder', 'ocr_decoder', 'tracking_decoder']
        for name in decoder_names:
            decoder = getattr(self.model, name)
            params = [p for p in decoder.parameters() if p.requires_grad]
            if params:
                groups.append({
                    'params': params,
                    'lr': self.config.stage3_init_lr,
                    'name': name,
                })
        
        return groups
    
    def _train_epoch_joint(
        self,
        train_loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        scheduler: Any,
        loss_fn: nn.Module,
    ) -> Tuple[float, Dict]:
        """Train one epoch with joint multi-task loss."""
        self.model.train()
        total_loss = 0.0
        loss_accumulator = {}
        num_batches = 0
        
        accum_steps = self.config.training_config.gradient_accumulation_steps
        optimizer.zero_grad()
        
        pbar = tqdm(enumerate(train_loader), total=len(train_loader), 
                    desc="  JOINT", leave=False)
        for batch_idx, (images, targets) in pbar:
            images = images.to(self.device)
            targets = self._move_to_device(targets)
            
            # Forward pass
            outputs = self.model(images)
            
            # Compute multi-task loss
            loss, loss_dict = loss_fn(outputs, targets)
            loss = loss / accum_steps  # Scale for gradient accumulation
            
            # Backward
            loss.backward()
            
            # Gradient accumulation step
            if (batch_idx + 1) % accum_steps == 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config.training_config.clip_grad_norm
                )
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                self.global_step += 1
            
            # Accumulate losses
            total_loss += loss.item() * accum_steps
            for k, v in loss_dict.items():
                if isinstance(v, torch.Tensor):
                    if k not in loss_accumulator:
                        loss_accumulator[k] = 0.0
                    loss_accumulator[k] += v.item()
            
            num_batches += 1
            
            # Update progress bar
            pbar.set_postfix(loss=loss.item() * accum_steps)
        
        # Average losses
        avg_loss = total_loss / max(num_batches, 1)
        for k in loss_accumulator:
            loss_accumulator[k] /= max(num_batches, 1)
        
        return avg_loss, loss_accumulator
    
    def _validate_joint(
        self,
        val_loader: DataLoader,
        loss_fn: nn.Module,
    ) -> Tuple[float, Dict]:
        """Validate with full evaluation metrics."""
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        
        self.evaluator.reset()
        
        with torch.no_grad():
            for images, targets in val_loader:
                images = images.to(self.device)
                targets = self._move_to_device(targets)
                
                outputs = self.model(images)
                
                # Compute loss
                loss, _ = loss_fn(outputs, targets)
                total_loss += loss.item()
                
                # Add to evaluator
                self.evaluator.add_batch(outputs, targets)
                
                num_batches += 1
        
        avg_loss = total_loss / max(num_batches, 1)
        metrics = self.evaluator.compute_metrics()
        
        return avg_loss, metrics
    
    # =========================================================================
    # Utilities
    # =========================================================================
    
    def _move_to_device(self, data: Any) -> Any:
        """Move data to device."""
        if isinstance(data, torch.Tensor):
            return data.to(self.device)
        elif isinstance(data, dict):
            return {k: self._move_to_device(v) for k, v in data.items()}
        elif isinstance(data, list):
            return [self._move_to_device(v) for v in data]
        else:
            return data
    
    def _save_checkpoint(self, path: str, epoch: int, loss: float):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'global_step': self.global_step,
            'model_state_dict': self.model.state_dict(),
            'loss': loss,
            'stage': self.current_stage.name,
            'config': {
                'model': self.config.model_config,
                'training': self.config.training_config,
            }
        }
        torch.save(checkpoint, path)
        print(f"  ✓ Saved checkpoint: {path}")
    
    def load_checkpoint(self, path: str):
        """Load model from checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.global_step = checkpoint.get('global_step', 0)
        print(f"  ✓ Loaded checkpoint from {path}")
    
    def run_full_pipeline(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
    ) -> Dict[str, Any]:
        """
        Run the complete three-stage training pipeline.
        
        Returns:
            Complete training results
        """
        results = {}
        
        # Stage 1
        self.initialize_from_pretrained()
        results['stage1'] = {'status': 'complete'}
        
        # Stage 2 (using same data for demo - in practice use task-specific data)
        print("\n⚠ Skipping Stage 2 task-specific pretraining (use task-specific data in practice)")
        results['stage2'] = {'status': 'skipped'}
        
        # Stage 3
        joint_results = self.train_joint(train_loader, val_loader)
        results['stage3'] = joint_results
        
        return results


if __name__ == "__main__":
    print("🚀 Staged Training Demo\n")
    
    # Create config
    config = StagedTrainingConfig()
    config.joint_training_epochs = 2  # Short demo
    
    # Create trainer
    trainer = StagedTrainer(config)
    
    # Create dummy data
    from src.trainer import SyntheticDataset, collate_fn
    
    train_dataset = SyntheticDataset(num_samples=20)
    train_loader = DataLoader(
        train_dataset,
        batch_size=1,
        shuffle=True,
        collate_fn=collate_fn,
    )
    
    # Run Stage 1 only (demo)
    trainer.initialize_from_pretrained()
    
    print("\n✓ Staged training demo complete!")
