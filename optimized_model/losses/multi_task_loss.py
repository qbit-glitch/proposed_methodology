"""
Multi-Task Loss combining all task-specific losses.

Implements Algorithm 10: Training Procedure
L_total = λ₁L_det + λ₂L_seg + λ₃L_plate + λ₄L_ocr + λ₅L_track
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple

from .detection_loss import DetectionLoss
from .segmentation_loss import SegmentationLoss
from .ocr_loss import OCRLoss
from .tracking_loss import TrackingLoss


class MultiTaskLoss(nn.Module):
    """
    Combined multi-task loss with learnable or fixed weights.
    
    L_total = λ₁L_det + λ₂L_seg + λ₃L_plate + λ₄L_ocr + λ₅L_track
    
    Supports:
    - Fixed weights (specified in config)
    - Uncertainty weighting (learned σ per task)
    """
    
    def __init__(
        self,
        num_vehicle_classes: int = 6,
        num_seg_classes: int = 3,
        # Loss weights (lambdas)
        lambda_det: float = 1.0,
        lambda_seg: float = 1.0,
        lambda_plate: float = 2.0,
        lambda_ocr: float = 1.5,
        lambda_track: float = 1.0,
        # Uncertainty weighting
        use_uncertainty_weighting: bool = False,
    ):
        super().__init__()
        
        # Store weights
        self.lambda_det = lambda_det
        self.lambda_seg = lambda_seg
        self.lambda_plate = lambda_plate
        self.lambda_ocr = lambda_ocr
        self.lambda_track = lambda_track
        
        # Task-specific losses
        self.detection_loss = DetectionLoss(num_classes=num_vehicle_classes)
        self.segmentation_loss = SegmentationLoss(num_classes=num_seg_classes)
        self.plate_loss = DetectionLoss(num_classes=1)  # Binary: plate or not
        self.ocr_loss = OCRLoss()
        self.tracking_loss = TrackingLoss()
        
        # Uncertainty weighting (if enabled)
        self.use_uncertainty_weighting = use_uncertainty_weighting
        if use_uncertainty_weighting:
            # Learned log(σ²) for each task
            self.log_vars = nn.Parameter(torch.zeros(5))
    
    def forward(
        self,
        outputs: Dict[str, Dict],
        targets: Dict[str, any],
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Compute combined multi-task loss.
        
        Args:
            outputs: Dict with outputs from each decoder
                - 'detection': {'bbox', 'class_logits', ...}
                - 'segmentation': {'masks', 'class_logits'}
                - 'plate': {'plate_bbox', 'plate_confidence'}
                - 'ocr': {'char_logits'}
                - 'tracking': {'features', 'trajectory_delta', ...}
            targets: Dict with targets for each task
                - 'detection': List[{'labels', 'boxes'}]
                - 'segmentation': [B, H, W] tensor
                - 'plate': List[{'labels', 'boxes'}]
                - 'ocr': List[str]
                - 'tracking': Optional[Dict]
                
        Returns:
            - total_loss: Combined weighted loss
            - loss_dict: Dict with individual losses
        """
        loss_dict = {}
        
        # Detection loss
        if 'detection' in outputs and 'detection' in targets:
            det_losses = self.detection_loss(outputs['detection'], targets['detection'])
            for k, v in det_losses.items():
                loss_dict[k] = v
        
        # Segmentation loss
        if 'segmentation' in outputs and 'segmentation' in targets:
            seg_losses = self.segmentation_loss(outputs['segmentation'], targets['segmentation'])
            for k, v in seg_losses.items():
                loss_dict[k] = v
        
        # Plate detection loss (simplified - just bbox L1)
        if 'plate' in outputs and 'plate' in targets:
            device = outputs['plate']['plate_bbox'].device
            # Simple L1 loss for plate bboxes (placeholder)
            loss_dict['loss_plate_bbox'] = torch.tensor(0.0, device=device)
        
        # OCR loss
        if 'ocr' in outputs and 'ocr' in targets:
            ocr_losses = self.ocr_loss(outputs['ocr'], targets['ocr'])
            for k, v in ocr_losses.items():
                loss_dict[k] = v
        
        # Tracking loss
        if 'tracking' in outputs:
            track_targets = targets.get('tracking', None)
            track_losses = self.tracking_loss(outputs['tracking'], track_targets)
            for k, v in track_losses.items():
                loss_dict[k] = v
        
        # Compute total loss
        if self.use_uncertainty_weighting:
            total_loss = self._uncertainty_weighted_loss(loss_dict)
        else:
            total_loss = self._fixed_weighted_loss(loss_dict)
        
        loss_dict['total_loss'] = total_loss
        
        return total_loss, loss_dict
    
    def _fixed_weighted_loss(self, loss_dict: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Compute total loss with fixed weights."""
        total = 0.0
        
        # Detection
        for k in ['loss_det_cls', 'loss_det_bbox', 'loss_det_giou']:
            if k in loss_dict:
                total = total + self.lambda_det * loss_dict[k]
        
        # Segmentation
        for k in ['loss_seg_ce', 'loss_seg_dice']:
            if k in loss_dict:
                total = total + self.lambda_seg * loss_dict[k]
        
        # Plate
        for k in ['loss_plate_cls', 'loss_plate_bbox', 'loss_plate_giou']:
            if k in loss_dict:
                total = total + self.lambda_plate * loss_dict[k]
        
        # OCR
        if 'loss_ocr_ctc' in loss_dict:
            total = total + self.lambda_ocr * loss_dict['loss_ocr_ctc']
        
        # Tracking
        for k in ['loss_track_assoc', 'loss_track_traj']:
            if k in loss_dict:
                total = total + self.lambda_track * loss_dict[k]
        
        return total
    
    def _uncertainty_weighted_loss(self, loss_dict: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Compute total loss with learned uncertainty weighting.
        
        L = Σ (1/(2σᵢ²)) * Lᵢ + log(σᵢ)
        """
        losses = [
            sum(loss_dict.get(k, 0) for k in ['loss_det_cls', 'loss_det_bbox', 'loss_det_giou']),
            sum(loss_dict.get(k, 0) for k in ['loss_seg_ce', 'loss_seg_dice']),
            sum(loss_dict.get(k, 0) for k in ['loss_plate_cls', 'loss_plate_bbox', 'loss_plate_giou']),
            loss_dict.get('loss_ocr_ctc', 0),
            sum(loss_dict.get(k, 0) for k in ['loss_track_assoc', 'loss_track_traj']),
        ]
        
        total = 0.0
        for i, loss in enumerate(losses):
            if isinstance(loss, torch.Tensor):
                precision = torch.exp(-self.log_vars[i])
                total = total + precision * loss + self.log_vars[i]
        
        return total


if __name__ == "__main__":
    # Test multi-task loss
    loss_fn = MultiTaskLoss(
        num_vehicle_classes=6,
        num_seg_classes=3,
    )
    
    # Dummy outputs
    outputs = {
        'detection': {
            'class_logits': torch.randn(2, 100, 7),
            'bbox': torch.rand(2, 100, 4),
        },
        'segmentation': {
            'masks': torch.randn(2, 50, 64, 64),
            'class_logits': torch.randn(2, 50, 3),
        },
        'ocr': {
            'char_logits': torch.randn(2, 20, 37),
        },
        'tracking': {
            'features': torch.randn(2, 100, 256),
            'trajectory_delta': torch.randn(2, 100, 4),
            'association_scores': torch.randn(2, 100, 100),
        },
    }
    
    # Dummy targets
    targets = {
        'detection': [
            {'labels': torch.tensor([0, 1]), 'boxes': torch.rand(2, 4)},
            {'labels': torch.tensor([2]), 'boxes': torch.rand(1, 4)},
        ],
        'segmentation': torch.randint(0, 3, (2, 256, 256)),
        'ocr': ["DL01AB1234", "MH12XY5678"],
    }
    
    total_loss, loss_dict = loss_fn(outputs, targets)
    
    print("📊 Multi-Task Loss Test:")
    for name, value in loss_dict.items():
        if isinstance(value, torch.Tensor):
            print(f"   {name}: {value.item():.4f}")
    print(f"\n   TOTAL: {total_loss.item():.4f}")
