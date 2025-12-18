"""
Detection Loss Functions.

Implements losses for object detection:
- Focal Loss for classification (handles class imbalance)
- GIoU Loss for bounding box regression
- L1 Loss for bounding box regression
- Hungarian matching for bipartite assignment
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, List, Optional
from scipy.optimize import linear_sum_assignment


def box_cxcywh_to_xyxy(boxes: torch.Tensor) -> torch.Tensor:
    """
    Convert boxes from (cx, cy, w, h) to (x1, y1, x2, y2) format.
    
    Args:
        boxes: [N, 4] in (cx, cy, w, h) format
        
    Returns:
        boxes: [N, 4] in (x1, y1, x2, y2) format
    """
    cx, cy, w, h = boxes.unbind(-1)
    x1 = cx - 0.5 * w
    y1 = cy - 0.5 * h
    x2 = cx + 0.5 * w
    y2 = cy + 0.5 * h
    return torch.stack([x1, y1, x2, y2], dim=-1)


def box_iou(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    """
    Compute IoU between two sets of boxes.
    
    Args:
        boxes1: [N, 4] in (x1, y1, x2, y2) format
        boxes2: [M, 4] in (x1, y1, x2, y2) format
        
    Returns:
        iou: [N, M] IoU matrix
    """
    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])
    
    # Intersection
    lt = torch.max(boxes1[:, None, :2], boxes2[None, :, :2])  # [N, M, 2]
    rb = torch.min(boxes1[:, None, 2:], boxes2[None, :, 2:])  # [N, M, 2]
    wh = (rb - lt).clamp(min=0)  # [N, M, 2]
    inter = wh[:, :, 0] * wh[:, :, 1]  # [N, M]
    
    union = area1[:, None] + area2[None, :] - inter
    iou = inter / (union + 1e-6)
    
    return iou


def generalized_box_iou(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    """
    Compute Generalized IoU between two sets of boxes.
    
    GIoU = IoU - |C \ (A ∪ B)| / |C|
    where C is the smallest enclosing box.
    
    Args:
        boxes1: [N, 4] in (x1, y1, x2, y2) format
        boxes2: [M, 4] in (x1, y1, x2, y2) format
        
    Returns:
        giou: [N, M] GIoU matrix
    """
    iou = box_iou(boxes1, boxes2)
    
    # Smallest enclosing box
    lt = torch.min(boxes1[:, None, :2], boxes2[None, :, :2])
    rb = torch.max(boxes1[:, None, 2:], boxes2[None, :, 2:])
    wh = (rb - lt).clamp(min=0)
    area_c = wh[:, :, 0] * wh[:, :, 1]
    
    # Areas
    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])
    
    # Intersection
    lt_i = torch.max(boxes1[:, None, :2], boxes2[None, :, :2])
    rb_i = torch.min(boxes1[:, None, 2:], boxes2[None, :, 2:])
    wh_i = (rb_i - lt_i).clamp(min=0)
    inter = wh_i[:, :, 0] * wh_i[:, :, 1]
    
    union = area1[:, None] + area2[None, :] - inter
    
    giou = iou - (area_c - union) / (area_c + 1e-6)
    
    return giou


class FocalLoss(nn.Module):
    """
    Focal Loss for classification.
    
    FL(p_t) = -α_t * (1 - p_t)^γ * log(p_t)
    
    Focuses on hard examples and handles class imbalance.
    """
    
    def __init__(
        self,
        alpha: float = 0.25,
        gamma: float = 2.0,
        reduction: str = 'mean'
    ):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(
        self,
        inputs: torch.Tensor,  # [N, C] logits
        targets: torch.Tensor,  # [N] class indices
    ) -> torch.Tensor:
        """Compute focal loss."""
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        p_t = torch.exp(-ce_loss)
        
        # Focal weight
        focal_weight = (1 - p_t) ** self.gamma
        
        # Alpha weighting
        if self.alpha is not None:
            alpha_t = self.alpha * targets.float() + (1 - self.alpha) * (1 - targets.float())
            focal_weight = alpha_t * focal_weight
        
        loss = focal_weight * ce_loss
        
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        return loss


class HungarianMatcher(nn.Module):
    """
    Hungarian Matcher for bipartite matching between predictions and targets.
    
    Finds optimal one-to-one assignment minimizing:
    cost = λ_cls * cost_cls + λ_bbox * cost_bbox + λ_giou * cost_giou
    """
    
    def __init__(
        self,
        cost_class: float = 1.0,
        cost_bbox: float = 5.0,
        cost_giou: float = 2.0,
    ):
        super().__init__()
        self.cost_class = cost_class
        self.cost_bbox = cost_bbox
        self.cost_giou = cost_giou
    
    @torch.no_grad()
    def forward(
        self,
        outputs: Dict[str, torch.Tensor],
        targets: List[Dict[str, torch.Tensor]],
    ) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        """
        Perform Hungarian matching.
        
        Args:
            outputs:
                - 'pred_logits': [B, num_queries, num_classes]
                - 'pred_boxes': [B, num_queries, 4]
            targets: List of dicts with:
                - 'labels': [num_targets]
                - 'boxes': [num_targets, 4]
                
        Returns:
            List of (pred_indices, target_indices) tuples for each batch
        """
        B, num_queries = outputs['pred_logits'].shape[:2]
        
        # Flatten predictions
        out_probs = outputs['pred_logits'].softmax(-1)  # [B, Q, C]
        out_boxes = outputs['pred_boxes']  # [B, Q, 4]
        
        indices = []
        
        for b in range(B):
            tgt_labels = targets[b]['labels']  # [T]
            tgt_boxes = targets[b]['boxes']    # [T, 4]
            
            if len(tgt_labels) == 0:
                indices.append((torch.tensor([], dtype=torch.long),
                               torch.tensor([], dtype=torch.long)))
                continue
            
            # Classification cost: -log(p(target_class))
            cost_class = -out_probs[b, :, tgt_labels]  # [Q, T]
            
            # L1 cost for bboxes
            cost_bbox = torch.cdist(out_boxes[b], tgt_boxes, p=1)  # [Q, T]
            
            # GIoU cost
            out_boxes_xyxy = box_cxcywh_to_xyxy(out_boxes[b])
            tgt_boxes_xyxy = box_cxcywh_to_xyxy(tgt_boxes)
            cost_giou = -generalized_box_iou(out_boxes_xyxy, tgt_boxes_xyxy)  # [Q, T]
            
            # Combined cost
            cost = (
                self.cost_class * cost_class +
                self.cost_bbox * cost_bbox +
                self.cost_giou * cost_giou
            )
            
            # Hungarian algorithm
            cost_np = cost.cpu().numpy()
            row_ind, col_ind = linear_sum_assignment(cost_np)
            
            indices.append((
                torch.tensor(row_ind, dtype=torch.long),
                torch.tensor(col_ind, dtype=torch.long)
            ))
        
        return indices


class DetectionLoss(nn.Module):
    """
    Combined detection loss with Hungarian matching.
    
    L_det = λ_cls * L_focal + λ_bbox * L_L1 + λ_giou * L_giou
    """
    
    def __init__(
        self,
        num_classes: int,
        focal_alpha: float = 0.25,
        focal_gamma: float = 2.0,
        bbox_loss_coef: float = 5.0,
        giou_loss_coef: float = 2.0,
        eos_coef: float = 0.1,  # Weight for "no object" class
    ):
        super().__init__()
        self.num_classes = num_classes
        self.bbox_loss_coef = bbox_loss_coef
        self.giou_loss_coef = giou_loss_coef
        
        # Matcher
        self.matcher = HungarianMatcher(
            cost_class=1.0,
            cost_bbox=bbox_loss_coef,
            cost_giou=giou_loss_coef,
        )
        
        # Focal loss
        self.focal_loss = FocalLoss(alpha=focal_alpha, gamma=focal_gamma)
        
        # Background weight
        empty_weight = torch.ones(num_classes + 1)
        empty_weight[-1] = eos_coef  # Lower weight for "no object"
        self.register_buffer('empty_weight', empty_weight)
    
    def loss_labels(
        self,
        outputs: Dict,
        targets: List[Dict],
        indices: List[Tuple],
        num_boxes: int,
    ) -> torch.Tensor:
        """Compute classification loss."""
        pred_logits = outputs['pred_logits']  # [B, Q, C]
        device = pred_logits.device
        
        # All predictions default to "no object" (last class)
        target_classes = torch.full(
            pred_logits.shape[:2],
            self.num_classes,  # Background class
            dtype=torch.long,
            device=device
        )
        
        # Assign matched targets
        for b, (pred_idx, tgt_idx) in enumerate(indices):
            if len(pred_idx) > 0:
                # Ensure indices and labels are on correct device
                pred_idx = pred_idx.to(device)
                tgt_idx = tgt_idx.to(device)
                labels = targets[b]['labels'].to(device)
                target_classes[b, pred_idx] = labels[tgt_idx]
        
        # Ensure weight is on correct device
        weight = self.empty_weight.to(device)
        
        # Cross entropy loss
        loss = F.cross_entropy(
            pred_logits.transpose(1, 2),  # [B, C, Q]
            target_classes,
            weight=weight,
            reduction='mean'
        )
        
        return loss
    
    def loss_boxes(
        self,
        outputs: Dict,
        targets: List[Dict],
        indices: List[Tuple],
        num_boxes: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute bbox L1 and GIoU losses."""
        device = outputs['pred_boxes'].device
        
        # Gather matched predictions and targets
        pred_boxes_list = []
        tgt_boxes_list = []
        
        for b, (pred_idx, tgt_idx) in enumerate(indices):
            if len(pred_idx) > 0:
                pred_boxes_list.append(outputs['pred_boxes'][b, pred_idx])
                tgt_boxes_list.append(targets[b]['boxes'][tgt_idx])
        
        if len(pred_boxes_list) == 0:
            return torch.tensor(0.0, device=device), torch.tensor(0.0, device=device)
        
        pred_boxes = torch.cat(pred_boxes_list)
        tgt_boxes = torch.cat(tgt_boxes_list).to(device)
        
        # L1 loss
        loss_bbox = F.l1_loss(pred_boxes, tgt_boxes, reduction='sum') / num_boxes
        
        # GIoU loss
        pred_xyxy = box_cxcywh_to_xyxy(pred_boxes)
        tgt_xyxy = box_cxcywh_to_xyxy(tgt_boxes)
        giou = generalized_box_iou(pred_xyxy, tgt_xyxy)
        loss_giou = (1 - giou.diag()).sum() / num_boxes
        
        return loss_bbox, loss_giou
    
    def forward(
        self,
        outputs: Dict,
        targets: List[Dict],
    ) -> Dict[str, torch.Tensor]:
        """
        Compute detection losses.
        
        Args:
            outputs:
                - 'pred_logits': [B, Q, C]
                - 'pred_boxes': [B, Q, 4]
            targets: List of dicts per batch with 'labels' and 'boxes'
            
        Returns:
            Dict with 'loss_cls', 'loss_bbox', 'loss_giou'
        """
        # Rename outputs for matcher
        outputs_for_matcher = {
            'pred_logits': outputs.get('class_logits', outputs.get('pred_logits')),
            'pred_boxes': outputs.get('bbox', outputs.get('pred_boxes')),
        }
        
        # Hungarian matching
        indices = self.matcher(outputs_for_matcher, targets)
        
        # Number of positive samples
        num_boxes = sum(len(t['labels']) for t in targets)
        num_boxes = max(num_boxes, 1)
        
        # Compute losses
        loss_cls = self.loss_labels(outputs_for_matcher, targets, indices, num_boxes)
        loss_bbox, loss_giou = self.loss_boxes(outputs_for_matcher, targets, indices, num_boxes)
        
        return {
            'loss_det_cls': loss_cls,
            'loss_det_bbox': self.bbox_loss_coef * loss_bbox,
            'loss_det_giou': self.giou_loss_coef * loss_giou,
        }


if __name__ == "__main__":
    # Test detection loss
    loss_fn = DetectionLoss(num_classes=6)
    
    # Dummy outputs and targets
    outputs = {
        'class_logits': torch.randn(2, 100, 7),
        'bbox': torch.rand(2, 100, 4),
    }
    
    targets = [
        {'labels': torch.tensor([0, 1, 2]), 'boxes': torch.rand(3, 4)},
        {'labels': torch.tensor([3, 4]), 'boxes': torch.rand(2, 4)},
    ]
    
    losses = loss_fn(outputs, targets)
    
    print("📊 Detection Loss Test:")
    for name, value in losses.items():
        print(f"   {name}: {value.item():.4f}")
