"""
Tracking Loss Functions.

Implements losses for multi-object tracking:
- Association loss (cross-entropy for track matching)
- Trajectory loss (Smooth L1 for motion prediction)
- Identity embedding loss (contrastive)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple
from scipy.optimize import linear_sum_assignment


class AssociationLoss(nn.Module):
    """
    Association loss for track-to-detection matching.
    
    Uses cross-entropy loss on the affinity/similarity matrix.
    """
    
    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature
        self.ce_loss = nn.CrossEntropyLoss(reduction='mean')
    
    def forward(
        self,
        similarity_matrix: torch.Tensor,  # [N_tracks, N_detections]
        gt_assignment: torch.Tensor,  # [N_tracks] indices or [N_tracks, N_detections] one-hot
    ) -> torch.Tensor:
        """
        Compute association loss.
        
        Args:
            similarity_matrix: Predicted similarity scores [N_tracks, N_detections]
            gt_assignment: Ground truth assignment indices [N_tracks]
            
        Returns:
            Association loss scalar
        """
        if similarity_matrix.numel() == 0 or gt_assignment.numel() == 0:
            return torch.tensor(0.0, device=similarity_matrix.device)
        
        # Scale by temperature
        logits = similarity_matrix / self.temperature
        
        # If gt_assignment is one-hot, convert to indices
        if gt_assignment.dim() == 2:
            gt_indices = gt_assignment.argmax(dim=-1)
        else:
            gt_indices = gt_assignment
        
        # Filter valid assignments (ignore -1 or out of range)
        valid_mask = (gt_indices >= 0) & (gt_indices < similarity_matrix.size(1))
        
        if valid_mask.sum() == 0:
            return torch.tensor(0.0, device=similarity_matrix.device)
        
        loss = self.ce_loss(logits[valid_mask], gt_indices[valid_mask])
        
        return loss


class TrajectoryLoss(nn.Module):
    """
    Trajectory prediction loss using Smooth L1.
    
    Predicts motion (velocity/displacement) between frames.
    """
    
    def __init__(self, beta: float = 1.0):
        super().__init__()
        self.smooth_l1 = nn.SmoothL1Loss(reduction='mean', beta=beta)
    
    def forward(
        self,
        pred_motion: torch.Tensor,  # [N, 4] predicted (dx, dy, dw, dh)
        gt_motion: torch.Tensor,  # [N, 4] ground truth motion
        mask: Optional[torch.Tensor] = None,  # [N] validity mask
    ) -> torch.Tensor:
        """
        Compute trajectory prediction loss.
        
        Args:
            pred_motion: Predicted motion vectors [N, 4]
            gt_motion: Ground truth motion [N, 4]
            mask: Optional mask for valid predictions
            
        Returns:
            Trajectory loss scalar
        """
        if pred_motion.numel() == 0:
            return torch.tensor(0.0, device=pred_motion.device)
        
        if mask is not None:
            pred_motion = pred_motion[mask]
            gt_motion = gt_motion[mask]
        
        if pred_motion.numel() == 0:
            return torch.tensor(0.0, device=pred_motion.device)
        
        return self.smooth_l1(pred_motion, gt_motion)


class IdentityEmbeddingLoss(nn.Module):
    """
    Contrastive loss for learning identity embeddings.
    
    Encourages embeddings of same track to be similar,
    different tracks to be dissimilar.
    """
    
    def __init__(self, margin: float = 0.5, temperature: float = 0.1):
        super().__init__()
        self.margin = margin
        self.temperature = temperature
    
    def forward(
        self,
        embeddings: torch.Tensor,  # [N, D] track embeddings
        track_ids: torch.Tensor,  # [N] track ID labels
    ) -> torch.Tensor:
        """
        Compute identity embedding loss.
        
        Args:
            embeddings: Track embeddings [N, D]
            track_ids: Track ID labels [N]
            
        Returns:
            Contrastive loss scalar
        """
        if embeddings.size(0) < 2:
            return torch.tensor(0.0, device=embeddings.device)
        
        # Normalize embeddings
        embeddings = F.normalize(embeddings, dim=-1)
        
        # Compute similarity matrix
        similarity = torch.matmul(embeddings, embeddings.T)  # [N, N]
        
        # Create positive/negative masks
        positive_mask = (track_ids.unsqueeze(0) == track_ids.unsqueeze(1)).float()
        negative_mask = 1 - positive_mask
        
        # Remove diagonal
        identity = torch.eye(embeddings.size(0), device=embeddings.device)
        positive_mask = positive_mask - identity
        
        # InfoNCE-style loss
        logits = similarity / self.temperature
        
        # For each sample, positive samples in numerator, all in denominator
        exp_logits = torch.exp(logits)
        
        # Positive sum (excluding self)
        pos_sum = (exp_logits * positive_mask).sum(dim=1)
        
        # All sum (excluding self)
        all_sum = (exp_logits * (1 - identity)).sum(dim=1)
        
        # Avoid log(0)
        loss = -torch.log((pos_sum + 1e-8) / (all_sum + 1e-8))
        
        # Only compute for samples with at least one positive
        has_positive = positive_mask.sum(dim=1) > 0
        
        if has_positive.sum() == 0:
            return torch.tensor(0.0, device=embeddings.device)
        
        return loss[has_positive].mean()


def compute_motion_from_boxes(
    boxes_t: torch.Tensor,  # [N, 4] boxes at time t (cx, cy, w, h)
    boxes_t1: torch.Tensor,  # [N, 4] boxes at time t+1 (cx, cy, w, h)
) -> torch.Tensor:
    """
    Compute motion vectors from box pairs.
    
    Returns:
        Motion vectors [N, 4] as (dx, dy, dw, dh)
    """
    motion = boxes_t1 - boxes_t
    return motion


class TrackingLoss(nn.Module):
    """
    Combined tracking loss for multi-object tracking.
    
    L_track = λ_assoc * L_assoc + λ_traj * L_traj + λ_id * L_id
    """
    
    def __init__(
        self,
        assoc_weight: float = 1.0,
        traj_weight: float = 1.0,
        id_weight: float = 0.5,
        temperature: float = 0.07,
    ):
        super().__init__()
        self.assoc_weight = assoc_weight
        self.traj_weight = traj_weight
        self.id_weight = id_weight
        
        self.assoc_loss = AssociationLoss(temperature=temperature)
        self.traj_loss = TrajectoryLoss()
        self.id_loss = IdentityEmbeddingLoss(temperature=0.1)
    
    def forward(
        self,
        outputs: Dict[str, torch.Tensor],
        targets: Optional[Dict] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute tracking losses.
        
        Args:
            outputs:
                - 'features': Track features [B, N_q, D]
                - 'similarity': Similarity matrix [B, N_tracks, N_dets] (optional)
                - 'pred_motion': Predicted motion [B, N_q, 4] (optional)
            targets:
                - 'track_ids': [B, N_gt] track IDs
                - 'associations': [B, N_tracks, N_dets] assignment matrix
                - 'gt_motion': [B, N_gt, 4] ground truth motion
                
        Returns:
            Dict with 'loss_track_assoc', 'loss_track_traj', 'loss_track_id'
        """
        device = outputs['features'].device
        losses = {}
        
        # Initialize all losses to zero
        loss_assoc = torch.tensor(0.0, device=device)
        loss_traj = torch.tensor(0.0, device=device)
        loss_id = torch.tensor(0.0, device=device)
        
        if targets is not None:
            B = outputs['features'].size(0)
            
            # Association loss
            if 'similarity' in outputs and 'associations' in targets:
                for b in range(B):
                    sim = outputs['similarity'][b]  # [N_tracks, N_dets]
                    assoc = targets['associations'][b]  # Assignment
                    loss_assoc = loss_assoc + self.assoc_loss(sim, assoc)
                loss_assoc = loss_assoc / max(B, 1)
            
            # Trajectory loss
            if 'pred_motion' in outputs and 'gt_motion' in targets:
                pred_motion = outputs['pred_motion']  # [B, N, 4]
                gt_motion = targets['gt_motion']  # [B, N, 4]
                
                # Flatten batch dimension
                pred_flat = pred_motion.view(-1, 4)
                gt_flat = gt_motion.view(-1, 4)
                
                loss_traj = self.traj_loss(pred_flat, gt_flat)
            
            # Identity embedding loss
            if 'features' in outputs and 'track_ids' in targets:
                features = outputs['features']  # [B, N, D]
                track_ids = targets['track_ids']  # [B, N]
                
                for b in range(B):
                    feat = features[b]  # [N, D]
                    ids = track_ids[b]  # [N]
                    
                    # Filter valid tracks (id >= 0)
                    valid = ids >= 0
                    if valid.sum() > 1:
                        loss_id = loss_id + self.id_loss(feat[valid], ids[valid])
                
                loss_id = loss_id / max(B, 1)
        
        losses['loss_track_assoc'] = self.assoc_weight * loss_assoc
        losses['loss_track_traj'] = self.traj_weight * loss_traj
        losses['loss_track_id'] = self.id_weight * loss_id
        
        return losses


class HungarianTrackMatcher:
    """
    Hungarian matching for track-to-detection association.
    
    Used during inference to associate tracks across frames.
    """
    
    def __init__(
        self,
        cost_appearance: float = 1.0,
        cost_motion: float = 1.0,
        cost_iou: float = 1.0,
        match_threshold: float = 0.3,
    ):
        self.cost_appearance = cost_appearance
        self.cost_motion = cost_motion
        self.cost_iou = cost_iou
        self.match_threshold = match_threshold
    
    def match(
        self,
        track_features: torch.Tensor,  # [N_tracks, D]
        det_features: torch.Tensor,  # [N_dets, D]
        track_boxes: torch.Tensor,  # [N_tracks, 4]
        det_boxes: torch.Tensor,  # [N_dets, 4]
        pred_boxes: Optional[torch.Tensor] = None,  # [N_tracks, 4] predicted positions
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Perform Hungarian matching.
        
        Returns:
            - matched_track_idx: Indices of matched tracks
            - matched_det_idx: Indices of matched detections
            - unmatched_track_idx: Indices of unmatched tracks
        """
        N_tracks = track_features.size(0)
        N_dets = det_features.size(0)
        
        if N_tracks == 0 or N_dets == 0:
            return (
                torch.tensor([], dtype=torch.long),
                torch.tensor([], dtype=torch.long),
                torch.arange(N_tracks, dtype=torch.long),
            )
        
        device = track_features.device
        
        # Appearance cost (cosine distance)
        track_norm = F.normalize(track_features, dim=-1)
        det_norm = F.normalize(det_features, dim=-1)
        appearance_sim = torch.matmul(track_norm, det_norm.T)  # [N_tracks, N_dets]
        appearance_cost = 1 - appearance_sim
        
        # IoU cost
        iou_cost = 1 - self._compute_iou(track_boxes, det_boxes)
        
        # Motion cost (if predicted boxes available)
        if pred_boxes is not None:
            motion_cost = 1 - self._compute_iou(pred_boxes, det_boxes)
        else:
            motion_cost = torch.zeros_like(iou_cost)
        
        # Combined cost matrix
        cost_matrix = (
            self.cost_appearance * appearance_cost +
            self.cost_motion * motion_cost +
            self.cost_iou * iou_cost
        )
        
        # Convert to numpy for scipy
        cost_np = cost_matrix.detach().cpu().numpy()
        
        # Hungarian matching
        row_indices, col_indices = linear_sum_assignment(cost_np)
        
        # Filter by threshold
        matched_rows = []
        matched_cols = []
        for r, c in zip(row_indices, col_indices):
            if cost_np[r, c] < self.match_threshold:
                matched_rows.append(r)
                matched_cols.append(c)
        
        matched_track_idx = torch.tensor(matched_rows, dtype=torch.long, device=device)
        matched_det_idx = torch.tensor(matched_cols, dtype=torch.long, device=device)
        
        all_tracks = set(range(N_tracks))
        unmatched = all_tracks - set(matched_rows)
        unmatched_track_idx = torch.tensor(list(unmatched), dtype=torch.long, device=device)
        
        return matched_track_idx, matched_det_idx, unmatched_track_idx
    
    def _compute_iou(self, boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
        """Compute IoU between two sets of boxes in (cx, cy, w, h) format."""
        # Convert to (x1, y1, x2, y2)
        b1_x1 = boxes1[:, 0:1] - boxes1[:, 2:3] / 2
        b1_y1 = boxes1[:, 1:2] - boxes1[:, 3:4] / 2
        b1_x2 = boxes1[:, 0:1] + boxes1[:, 2:3] / 2
        b1_y2 = boxes1[:, 1:2] + boxes1[:, 3:4] / 2
        
        b2_x1 = boxes2[:, 0:1] - boxes2[:, 2:3] / 2
        b2_y1 = boxes2[:, 1:2] - boxes2[:, 3:4] / 2
        b2_x2 = boxes2[:, 0:1] + boxes2[:, 2:3] / 2
        b2_y2 = boxes2[:, 1:2] + boxes2[:, 3:4] / 2
        
        # Intersection
        inter_x1 = torch.max(b1_x1, b2_x1.T)
        inter_y1 = torch.max(b1_y1, b2_y1.T)
        inter_x2 = torch.min(b1_x2, b2_x2.T)
        inter_y2 = torch.min(b1_y2, b2_y2.T)
        
        inter_area = torch.clamp(inter_x2 - inter_x1, min=0) * torch.clamp(inter_y2 - inter_y1, min=0)
        
        # Union
        area1 = (b1_x2 - b1_x1) * (b1_y2 - b1_y1)
        area2 = (b2_x2 - b2_x1) * (b2_y2 - b2_y1)
        union_area = area1 + area2.T - inter_area
        
        iou = inter_area / (union_area + 1e-8)
        
        return iou


if __name__ == "__main__":
    print("📊 Tracking Loss Test\n")
    
    # Test tracking loss
    loss_fn = TrackingLoss()
    
    # Dummy outputs
    B, N_q, D = 2, 100, 256
    outputs = {
        'features': torch.randn(B, N_q, D),
        'similarity': torch.randn(B, 50, 100),  # 50 tracks, 100 detections
        'pred_motion': torch.randn(B, N_q, 4),
    }
    
    # Dummy targets
    targets = {
        'track_ids': torch.randint(-1, 10, (B, N_q)),  # -1 for no track
        'associations': torch.randint(0, 100, (B, 50)),  # Assignment indices
        'gt_motion': torch.randn(B, N_q, 4),
    }
    
    losses = loss_fn(outputs, targets)
    
    print("Tracking Losses:")
    for name, value in losses.items():
        print(f"   {name}: {value.item():.4f}")
    
    # Test Hungarian matcher
    print("\n📊 Hungarian Matcher Test")
    matcher = HungarianTrackMatcher()
    
    track_features = torch.randn(10, 256)
    det_features = torch.randn(15, 256)
    track_boxes = torch.rand(10, 4)
    det_boxes = torch.rand(15, 4)
    
    matched_t, matched_d, unmatched = matcher.match(
        track_features, det_features, track_boxes, det_boxes
    )
    
    print(f"   Matched tracks: {len(matched_t)}")
    print(f"   Unmatched tracks: {len(unmatched)}")
    
    print("\n✓ Tracking loss tests passed!")
