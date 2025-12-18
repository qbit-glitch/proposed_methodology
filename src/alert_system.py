"""
Alert System - Algorithm 8.

Implements parking violation detection:
- IoU computation between vehicle bboxes and segmentation masks
- Alert generation based on stopped time and overlap thresholds
- Visualization with red bounding boxes for violations
"""

import torch
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass


@dataclass
class Alert:
    """Represents a parking violation alert."""
    track_id: int
    bbox: Tuple[float, float, float, float]  # x, y, w, h
    stopped_time: float  # seconds
    overlap: float  # fraction overlapping with driveway
    alert_type: str  # 'RED' or 'NORMAL'
    plate_text: Optional[str] = None
    vehicle_class: Optional[str] = None


class AlertSystem:
    """
    Alert generation system for parking violations.
    
    Conditions for RED alert:
    1. Vehicle stopped for > 2 minutes (120 seconds)
    2. Vehicle overlaps with driveway by > 30%
    """
    
    def __init__(
        self,
        stopped_time_threshold: float = 120.0,  # seconds
        overlap_threshold: float = 0.3,  # 30%
    ):
        self.stopped_time_threshold = stopped_time_threshold
        self.overlap_threshold = overlap_threshold
    
    def compute_bbox_mask_overlap(
        self,
        bbox: torch.Tensor,  # [4] normalized (x_center, y_center, w, h)
        mask: torch.Tensor,  # [H, W] probability mask
        image_size: Tuple[int, int]
    ) -> float:
        """
        Compute overlap between bounding box and segmentation mask.
        
        Args:
            bbox: Normalized bbox [x_center, y_center, w, h]
            mask: Segmentation mask [H, W] with probabilities
            image_size: (H, W) of the image
            
        Returns:
            Overlap fraction (0 to 1)
        """
        H, W = image_size
        
        # Convert normalized bbox to pixel coordinates
        x_center, y_center, w, h = bbox.tolist()
        x1 = int((x_center - w / 2) * W)
        y1 = int((y_center - h / 2) * H)
        x2 = int((x_center + w / 2) * W)
        y2 = int((y_center + h / 2) * H)
        
        # Clamp to image bounds
        x1 = max(0, min(x1, W - 1))
        y1 = max(0, min(y1, H - 1))
        x2 = max(0, min(x2, W))
        y2 = max(0, min(y2, H))
        
        if x2 <= x1 or y2 <= y1:
            return 0.0
        
        # Create bbox mask
        bbox_mask = torch.zeros_like(mask)
        bbox_mask[y1:y2, x1:x2] = 1.0
        
        # Compute overlap
        # Intersection: bbox area that overlaps with driveway
        intersection = (bbox_mask * mask).sum()
        # Bbox area
        bbox_area = bbox_mask.sum()
        
        if bbox_area == 0:
            return 0.0
        
        overlap = (intersection / bbox_area).item()
        return overlap
    
    def compute_iou(
        self,
        bbox: torch.Tensor,  # [4]
        mask: torch.Tensor,  # [H, W]
        image_size: Tuple[int, int]
    ) -> float:
        """Compute IoU between bbox and mask (alternative metric)."""
        H, W = image_size
        
        x_center, y_center, w, h = bbox.tolist()
        x1 = int((x_center - w / 2) * W)
        y1 = int((y_center - h / 2) * H)
        x2 = int((x_center + w / 2) * W)
        y2 = int((y_center + h / 2) * H)
        
        x1 = max(0, min(x1, W - 1))
        y1 = max(0, min(y1, H - 1))
        x2 = max(0, min(x2, W))
        y2 = max(0, min(y2, H))
        
        if x2 <= x1 or y2 <= y1:
            return 0.0
        
        bbox_mask = torch.zeros_like(mask)
        bbox_mask[y1:y2, x1:x2] = 1.0
        
        mask_binary = (mask > 0.5).float()
        
        intersection = (bbox_mask * mask_binary).sum()
        union = ((bbox_mask + mask_binary) > 0).float().sum()
        
        if union == 0:
            return 0.0
        
        return (intersection / union).item()
    
    def generate_alerts(
        self,
        detection_bboxes: torch.Tensor,  # [B, N, 4]
        detection_classes: torch.Tensor,  # [B, N, num_classes]
        driveway_mask: torch.Tensor,  # [B, H, W]
        stopped_times: torch.Tensor,  # [B, N]
        track_ids: Optional[List[List[int]]] = None,
        plate_texts: Optional[List[List[str]]] = None,
        vehicle_classes: Optional[List[str]] = None,
    ) -> List[List[Alert]]:
        """
        Generate alerts for all detected vehicles.
        
        Args:
            detection_bboxes: [B, N, 4] normalized bboxes
            detection_classes: [B, N, num_classes] class logits
            driveway_mask: [B, H, W] driveway segmentation mask
            stopped_times: [B, N] stopped time per vehicle in seconds
            track_ids: Optional track IDs per vehicle
            plate_texts: Optional recognized plate texts
            
        Returns:
            List of lists of Alert objects (one list per batch)
        """
        B, N, _ = detection_bboxes.shape
        H, W = driveway_mask.shape[1:]
        image_size = (H, W)
        
        if vehicle_classes is None:
            vehicle_classes = ["car", "scooty", "bike", "bus", "truck", "auto"]
        
        all_alerts = []
        
        for b in range(B):
            batch_alerts = []
            
            for i in range(N):
                bbox = detection_bboxes[b, i]
                class_logits = detection_classes[b, i]
                stopped_time = stopped_times[b, i].item()
                mask = driveway_mask[b]
                
                # Skip background detections
                class_idx = class_logits.argmax().item()
                if class_idx == len(vehicle_classes):  # Background
                    continue
                
                # Skip low confidence detections
                class_prob = F.softmax(class_logits, dim=-1)[class_idx].item()
                if class_prob < 0.5:
                    continue
                
                # Compute overlap with driveway
                overlap = self.compute_bbox_mask_overlap(bbox, mask, image_size)
                
                # Determine alert type
                is_stopped_long = stopped_time > self.stopped_time_threshold
                is_overlapping = overlap > self.overlap_threshold
                
                if is_stopped_long and is_overlapping:
                    alert_type = "RED"
                else:
                    alert_type = "NORMAL"
                
                # Create alert
                alert = Alert(
                    track_id=track_ids[b][i] if track_ids else i,
                    bbox=tuple(bbox.tolist()),
                    stopped_time=stopped_time,
                    overlap=overlap,
                    alert_type=alert_type,
                    plate_text=plate_texts[b][i] if plate_texts else None,
                    vehicle_class=vehicle_classes[class_idx]
                )
                
                batch_alerts.append(alert)
            
            all_alerts.append(batch_alerts)
        
        return all_alerts
    
    def get_violation_summary(self, alerts: List[Alert]) -> Dict:
        """Get summary statistics for violations."""
        total = len(alerts)
        violations = [a for a in alerts if a.alert_type == "RED"]
        
        return {
            'total_vehicles': total,
            'violations': len(violations),
            'violation_rate': len(violations) / total if total > 0 else 0,
            'avg_stopped_time': sum(a.stopped_time for a in violations) / len(violations) if violations else 0,
            'avg_overlap': sum(a.overlap for a in violations) / len(violations) if violations else 0,
        }


def generate_alerts_from_outputs(
    outputs: Dict,
    alert_system: AlertSystem,
    image_size: Tuple[int, int] = (512, 512)
) -> List[List[Alert]]:
    """
    Convenience function to generate alerts from model outputs.
    
    Args:
        outputs: Dictionary from UnifiedMultiTaskTransformer forward pass
        alert_system: AlertSystem instance
        image_size: (H, W) for mask upsampling
        
    Returns:
        List of lists of Alert objects
    """
    # Get detection bboxes and classes
    detection_bboxes = outputs['detection']['bbox']
    detection_classes = outputs['detection']['class_logits']
    
    # Get stopped times from tracking
    stopped_times = outputs['tracking_state'].get(
        'stopped_times',
        torch.zeros_like(detection_bboxes[:, :, 0])
    )
    
    # Get driveway mask from segmentation (upsample to image size)
    seg_masks = outputs['segmentation']['masks']
    seg_class_logits = outputs['segmentation']['class_logits']
    
    # Simple driveway mask (weighted by class probability)
    B, N_q, H_m, W_m = seg_masks.shape
    seg_probs = F.softmax(seg_class_logits, dim=-1)  # [B, N_q, 3]
    
    # Upsample masks
    masks_up = F.interpolate(
        seg_masks, size=image_size, mode='bilinear', align_corners=False
    )  # [B, N_q, H, W]
    masks_up = masks_up.sigmoid()
    
    # Weighted sum for driveway (class 1)
    driveway_mask = (masks_up * seg_probs[:, :, 1:2].unsqueeze(-1)).sum(dim=1)  # [B, H, W]
    
    # Generate alerts
    alerts = alert_system.generate_alerts(
        detection_bboxes=detection_bboxes,
        detection_classes=detection_classes,
        driveway_mask=driveway_mask.squeeze(1),
        stopped_times=stopped_times,
    )
    
    return alerts


if __name__ == "__main__":
    # Test alert system
    alert_system = AlertSystem(
        stopped_time_threshold=120.0,
        overlap_threshold=0.3
    )
    
    # Create dummy data
    B, N = 2, 10
    detection_bboxes = torch.rand(B, N, 4)
    detection_classes = torch.randn(B, N, 7)  # 6 classes + background
    driveway_mask = torch.rand(B, 512, 512)
    stopped_times = torch.rand(B, N) * 200  # 0-200 seconds
    
    # Generate alerts
    alerts = alert_system.generate_alerts(
        detection_bboxes=detection_bboxes,
        detection_classes=detection_classes,
        driveway_mask=driveway_mask,
        stopped_times=stopped_times,
    )
    
    print(f"📊 Alert Generation Test:")
    for b, batch_alerts in enumerate(alerts):
        print(f"\n   Batch {b}:")
        violations = [a for a in batch_alerts if a.alert_type == "RED"]
        print(f"   - Total vehicles: {len(batch_alerts)}")
        print(f"   - Violations (RED): {len(violations)}")
        
        if violations:
            v = violations[0]
            print(f"   - Sample violation:")
            print(f"     - Track ID: {v.track_id}")
            print(f"     - Stopped time: {v.stopped_time:.1f}s")
            print(f"     - Overlap: {v.overlap:.2%}")
    
    # Get summary
    if alerts[0]:
        summary = alert_system.get_violation_summary(alerts[0])
        print(f"\n📈 Summary:")
        print(f"   - Violation rate: {summary['violation_rate']:.2%}")
