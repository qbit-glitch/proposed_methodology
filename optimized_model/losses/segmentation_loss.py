"""
Segmentation Loss Functions.

Implements losses for semantic segmentation:
- Cross-Entropy Loss for pixel classification
- Dice Loss for mask overlap
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple


class DiceLoss(nn.Module):
    """
    Dice Loss for segmentation.
    
    Dice = 2 * |A ∩ B| / (|A| + |B|)
    Loss = 1 - Dice
    """
    
    def __init__(self, smooth: float = 1e-6):
        super().__init__()
        self.smooth = smooth
    
    def forward(
        self,
        inputs: torch.Tensor,  # [B, N_q, H, W] logits
        targets: torch.Tensor,  # [B, H, W] class indices or [B, C, H, W] one-hot
    ) -> torch.Tensor:
        """Compute dice loss."""
        # Apply sigmoid to get probabilities
        inputs = inputs.sigmoid()
        
        # Flatten spatial dimensions
        inputs_flat = inputs.flatten(2)  # [B, N_q, H*W]
        
        if targets.dim() == 3:
            # Convert to one-hot if needed
            targets_flat = targets.flatten(1)  # [B, H*W]
        else:
            targets_flat = targets.flatten(2)  # [B, C, H*W]
        
        # Compute dice per query (average across queries)
        numerator = 2 * (inputs_flat * targets_flat.unsqueeze(1)).sum(-1)
        denominator = inputs_flat.sum(-1) + targets_flat.unsqueeze(1).sum(-1)
        
        dice = (numerator + self.smooth) / (denominator + self.smooth)
        loss = 1 - dice.mean()
        
        return loss


class SegmentationLoss(nn.Module):
    """
    Combined segmentation loss.
    
    L_seg = L_CE + L_Dice
    """
    
    def __init__(
        self,
        num_classes: int = 3,
        dice_weight: float = 1.0,
        ce_weight: float = 1.0,
        ignore_index: int = -100,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.dice_weight = dice_weight
        self.ce_weight = ce_weight
        
        self.ce_loss = nn.CrossEntropyLoss(ignore_index=ignore_index)
        self.dice_loss = DiceLoss()
    
    def forward(
        self,
        outputs: Dict[str, torch.Tensor],
        targets: torch.Tensor,  # [B, H, W] pixel-wise labels
    ) -> Dict[str, torch.Tensor]:
        """
        Compute segmentation losses.
        
        Args:
            outputs:
                - 'masks': [B, N_q, H, W] logits
                - 'class_logits': [B, N_q, num_classes]
            targets: [B, H, W] ground truth labels
            
        Returns:
            Dict with 'loss_seg_ce', 'loss_seg_dice'
        """
        masks = outputs['masks']  # [B, N_q, H_m, W_m]
        class_logits = outputs['class_logits']  # [B, N_q, C]
        
        B, N_q, H_m, W_m = masks.shape
        
        # Handle different target tensor shapes - need [B, H, W] for CE loss
        while targets.dim() > 3:
            # Keep squeezing dimensions until we get 3D [B, H, W]
            targets = targets.squeeze(1)
        
        if targets.dim() == 2:
            # [B, H*W] -> [B, H, W] (assuming square)
            side = int(targets.shape[1] ** 0.5)
            targets = targets.view(B, side, side)
        
        # Ensure targets is 3D [B, H, W] and long dtype
        targets = targets.long()
        H, W = targets.shape[-2:]
        
        # Upsample masks to target resolution
        masks_up = F.interpolate(masks, size=(H, W), mode='bilinear', align_corners=False)
        
        # Compute class probabilities per query
        class_probs = F.softmax(class_logits, dim=-1)  # [B, N_q, C]
        
        # Combine masks weighted by class probabilities
        # Result: [B, C, H, W]
        combined_masks = torch.zeros(B, self.num_classes, H, W, device=masks.device)
        
        for c in range(self.num_classes):
            # Weight masks by probability of class c
            weights = class_probs[:, :, c:c+1]  # [B, N_q, 1]
            weighted_masks = masks_up.sigmoid() * weights.unsqueeze(-1)  # [B, N_q, H, W]
            combined_masks[:, c] = weighted_masks.sum(dim=1)  # [B, H, W]
        
        # Cross-entropy loss
        loss_ce = self.ce_loss(combined_masks, targets)
        
        # Dice loss (per class)
        targets_onehot = F.one_hot(targets.clamp(0, self.num_classes - 1), self.num_classes)
        targets_onehot = targets_onehot.permute(0, 3, 1, 2).float()  # [B, C, H, W]
        
        loss_dice = 0
        for c in range(self.num_classes):
            pred_c = combined_masks[:, c:c+1]
            tgt_c = targets_onehot[:, c:c+1]
            loss_dice += self.dice_loss(pred_c, tgt_c)
        loss_dice = loss_dice / self.num_classes
        
        return {
            'loss_seg_ce': self.ce_weight * loss_ce,
            'loss_seg_dice': self.dice_weight * loss_dice,
        }


if __name__ == "__main__":
    # Test segmentation loss
    loss_fn = SegmentationLoss(num_classes=3)
    
    # Dummy outputs and targets
    outputs = {
        'masks': torch.randn(2, 50, 64, 64),
        'class_logits': torch.randn(2, 50, 3),
    }
    targets = torch.randint(0, 3, (2, 256, 256))
    
    losses = loss_fn(outputs, targets)
    
    print("📊 Segmentation Loss Test:")
    for name, value in losses.items():
        print(f"   {name}: {value.item():.4f}")
