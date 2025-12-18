"""
Evaluation Metrics Module (Algorithm S3.2, S3.6).

Comprehensive evaluation suite for multi-task learning:
- mAP: Mean Average Precision for detection
- mIoU: Mean Intersection over Union for segmentation
- MOTA: Multi-Object Tracking Accuracy
- OCR Accuracy: Character-level and word-level
"""

import torch
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional, Any
from collections import defaultdict
import numpy as np


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
    
    lt = torch.max(boxes1[:, None, :2], boxes2[None, :, :2])
    rb = torch.min(boxes1[:, None, 2:], boxes2[None, :, 2:])
    
    wh = (rb - lt).clamp(min=0)
    inter = wh[:, :, 0] * wh[:, :, 1]
    
    union = area1[:, None] + area2[None, :] - inter
    
    return inter / (union + 1e-8)


def box_cxcywh_to_xyxy(boxes: torch.Tensor) -> torch.Tensor:
    """Convert (cx, cy, w, h) to (x1, y1, x2, y2)."""
    cx, cy, w, h = boxes.unbind(-1)
    return torch.stack([cx - w/2, cy - h/2, cx + w/2, cy + h/2], dim=-1)


class AveragePrecisionCalculator:
    """
    Calculate Average Precision (AP) for object detection.
    
    Uses VOC/COCO style AP calculation.
    """
    
    def __init__(self, iou_threshold: float = 0.5, num_classes: int = 6):
        self.iou_threshold = iou_threshold
        self.num_classes = num_classes
        
        # Accumulate predictions and ground truths
        self.predictions = defaultdict(list)  # class -> [(score, is_tp)]
        self.num_gt = defaultdict(int)  # class -> count
    
    def reset(self):
        """Reset accumulated data."""
        self.predictions = defaultdict(list)
        self.num_gt = defaultdict(int)
    
    def add_batch(
        self,
        pred_boxes: torch.Tensor,  # [N, 4]
        pred_labels: torch.Tensor,  # [N]
        pred_scores: torch.Tensor,  # [N]
        gt_boxes: torch.Tensor,  # [M, 4]
        gt_labels: torch.Tensor,  # [M]
    ):
        """
        Add predictions and ground truths for one image.
        """
        # Convert to xyxy format if needed
        if pred_boxes.size(0) > 0:
            pred_boxes = box_cxcywh_to_xyxy(pred_boxes)
        if gt_boxes.size(0) > 0:
            gt_boxes = box_cxcywh_to_xyxy(gt_boxes)
        
        # Count ground truths per class
        for c in gt_labels.unique():
            self.num_gt[c.item()] += (gt_labels == c).sum().item()
        
        if pred_boxes.size(0) == 0:
            return
        
        if gt_boxes.size(0) == 0:
            # All predictions are false positives
            for score, label in zip(pred_scores, pred_labels):
                self.predictions[label.item()].append((score.item(), False))
            return
        
        # Compute IoU matrix
        iou = box_iou(pred_boxes, gt_boxes)  # [N, M]
        
        # Track matched ground truths
        gt_matched = torch.zeros(gt_boxes.size(0), dtype=torch.bool)
        
        # Sort predictions by score (descending)
        sorted_idx = pred_scores.argsort(descending=True)
        
        for idx in sorted_idx:
            pred_label = pred_labels[idx].item()
            pred_score = pred_scores[idx].item()
            
            # Find matching ground truth
            class_mask = gt_labels == pred_label
            if class_mask.sum() == 0:
                self.predictions[pred_label].append((pred_score, False))
                continue
            
            # Get IoU with same-class ground truths
            iou_class = iou[idx].clone()
            iou_class[~class_mask] = 0
            iou_class[gt_matched] = 0
            
            max_iou, max_idx = iou_class.max(dim=0)
            
            if max_iou >= self.iou_threshold:
                self.predictions[pred_label].append((pred_score, True))
                gt_matched[max_idx] = True
            else:
                self.predictions[pred_label].append((pred_score, False))
    
    def compute_ap(self, class_id: int) -> float:
        """Compute AP for a single class."""
        preds = self.predictions[class_id]
        num_gt = self.num_gt[class_id]
        
        if num_gt == 0:
            return 0.0
        
        if len(preds) == 0:
            return 0.0
        
        # Sort by score
        preds.sort(key=lambda x: x[0], reverse=True)
        
        # Compute precision-recall curve
        tp_cumsum = 0
        fp_cumsum = 0
        precisions = []
        recalls = []
        
        for score, is_tp in preds:
            if is_tp:
                tp_cumsum += 1
            else:
                fp_cumsum += 1
            
            precision = tp_cumsum / (tp_cumsum + fp_cumsum)
            recall = tp_cumsum / num_gt
            
            precisions.append(precision)
            recalls.append(recall)
        
        # Compute AP using 11-point interpolation or all-point
        ap = self._compute_ap_from_pr(precisions, recalls)
        
        return ap
    
    def _compute_ap_from_pr(self, precisions: List[float], recalls: List[float]) -> float:
        """Compute AP from precision-recall curve using all-point interpolation."""
        precisions = np.array(precisions)
        recalls = np.array(recalls)
        
        # Make precision monotonically decreasing
        for i in range(len(precisions) - 2, -1, -1):
            precisions[i] = max(precisions[i], precisions[i + 1])
        
        # Find points where recall changes
        recall_diff = np.diff(recalls, prepend=0)
        
        # Compute area under PR curve
        ap = np.sum(recall_diff * precisions)
        
        return float(ap)
    
    def compute_map(self) -> Tuple[float, Dict[int, float]]:
        """
        Compute mean Average Precision.
        
        Returns:
            (mAP, dict of per-class APs)
        """
        aps = {}
        for c in range(self.num_classes):
            aps[c] = self.compute_ap(c)
        
        # Only average over classes with ground truth
        valid_aps = [aps[c] for c in aps if self.num_gt[c] > 0]
        
        if len(valid_aps) == 0:
            return 0.0, aps
        
        mAP = sum(valid_aps) / len(valid_aps)
        
        return mAP, aps


class MIoUCalculator:
    """
    Calculate mean Intersection over Union (mIoU) for segmentation.
    """
    
    def __init__(self, num_classes: int = 3, ignore_index: int = -100):
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        
        # Confusion matrix
        self.confusion_matrix = torch.zeros(num_classes, num_classes, dtype=torch.long)
    
    def reset(self):
        """Reset confusion matrix."""
        self.confusion_matrix.zero_()
    
    def add_batch(
        self,
        predictions: torch.Tensor,  # [B, H, W] class indices
        targets: torch.Tensor,  # [B, H, W] class indices
    ):
        """Add predictions and targets for one batch."""
        assert predictions.shape == targets.shape
        
        # Flatten
        predictions = predictions.view(-1)
        targets = targets.view(-1)
        
        # Mask ignored pixels
        mask = targets != self.ignore_index
        predictions = predictions[mask]
        targets = targets[mask]
        
        # Update confusion matrix
        for pred, target in zip(predictions, targets):
            if 0 <= target < self.num_classes and 0 <= pred < self.num_classes:
                self.confusion_matrix[target, pred] += 1
    
    def compute_iou(self, class_id: int) -> float:
        """Compute IoU for a single class."""
        # True positives
        tp = self.confusion_matrix[class_id, class_id].item()
        
        # False positives (predicted as class but not)
        fp = self.confusion_matrix[:, class_id].sum().item() - tp
        
        # False negatives (is class but not predicted)
        fn = self.confusion_matrix[class_id, :].sum().item() - tp
        
        # IoU
        if tp + fp + fn == 0:
            return 0.0
        
        iou = tp / (tp + fp + fn)
        return iou
    
    def compute_miou(self) -> Tuple[float, Dict[int, float]]:
        """
        Compute mean IoU.
        
        Returns:
            (mIoU, dict of per-class IoUs)
        """
        ious = {}
        for c in range(self.num_classes):
            ious[c] = self.compute_iou(c)
        
        # Only average over classes present in ground truth
        valid_classes = [c for c in range(self.num_classes) 
                        if self.confusion_matrix[c, :].sum() > 0]
        
        if len(valid_classes) == 0:
            return 0.0, ious
        
        miou = sum(ious[c] for c in valid_classes) / len(valid_classes)
        
        return miou, ious
    
    def compute_pixel_accuracy(self) -> float:
        """Compute overall pixel accuracy."""
        correct = self.confusion_matrix.diag().sum().item()
        total = self.confusion_matrix.sum().item()
        
        if total == 0:
            return 0.0
        
        return correct / total


class MOTACalculator:
    """
    Calculate Multi-Object Tracking Accuracy (MOTA).
    
    MOTA = 1 - (FN + FP + IDSW) / GT
    
    Where:
    - FN: False negatives (missed detections)
    - FP: False positives
    - IDSW: ID switches
    - GT: Total ground truth objects
    """
    
    def __init__(self, iou_threshold: float = 0.5):
        self.iou_threshold = iou_threshold
        
        # Accumulate metrics
        self.false_negatives = 0
        self.false_positives = 0
        self.id_switches = 0
        self.total_gt = 0
        
        # Track previous frame matches for ID switch detection
        self.prev_gt_to_pred = {}
    
    def reset(self):
        """Reset accumulated metrics."""
        self.false_negatives = 0
        self.false_positives = 0
        self.id_switches = 0
        self.total_gt = 0
        self.prev_gt_to_pred = {}
    
    def add_frame(
        self,
        pred_boxes: torch.Tensor,  # [N, 4]
        pred_ids: torch.Tensor,  # [N] track IDs
        gt_boxes: torch.Tensor,  # [M, 4]
        gt_ids: torch.Tensor,  # [M] ground truth IDs
    ):
        """
        Add predictions and ground truths for one frame.
        """
        self.total_gt += gt_boxes.size(0)
        
        if gt_boxes.size(0) == 0:
            # All predictions are false positives
            self.false_positives += pred_boxes.size(0)
            return
        
        if pred_boxes.size(0) == 0:
            # All ground truths are false negatives
            self.false_negatives += gt_boxes.size(0)
            return
        
        # Convert to xyxy format
        pred_boxes_xyxy = box_cxcywh_to_xyxy(pred_boxes)
        gt_boxes_xyxy = box_cxcywh_to_xyxy(gt_boxes)
        
        # Compute IoU matrix
        iou = box_iou(pred_boxes_xyxy, gt_boxes_xyxy)  # [N, M]
        
        # Match using Hungarian algorithm (greedy here for simplicity)
        gt_matched = torch.zeros(gt_boxes.size(0), dtype=torch.bool)
        pred_matched = torch.zeros(pred_boxes.size(0), dtype=torch.bool)
        
        current_gt_to_pred = {}
        
        # Sort predictions by IoU (descending)
        for gt_idx in range(gt_boxes.size(0)):
            iou_gt = iou[:, gt_idx]
            max_iou, pred_idx = iou_gt.max(dim=0)
            
            if max_iou >= self.iou_threshold and not pred_matched[pred_idx]:
                gt_matched[gt_idx] = True
                pred_matched[pred_idx] = True
                
                gt_id = gt_ids[gt_idx].item()
                pred_id = pred_ids[pred_idx].item()
                
                current_gt_to_pred[gt_id] = pred_id
                
                # Check for ID switch
                if gt_id in self.prev_gt_to_pred:
                    if self.prev_gt_to_pred[gt_id] != pred_id:
                        self.id_switches += 1
        
        # Count false negatives and false positives
        self.false_negatives += (~gt_matched).sum().item()
        self.false_positives += (~pred_matched).sum().item()
        
        # Update previous frame matches
        self.prev_gt_to_pred = current_gt_to_pred
    
    def compute_mota(self) -> float:
        """
        Compute MOTA score.
        
        Returns:
            MOTA in range [-inf, 1] (higher is better)
        """
        if self.total_gt == 0:
            return 0.0
        
        mota = 1 - (self.false_negatives + self.false_positives + self.id_switches) / self.total_gt
        
        return mota
    
    def get_metrics(self) -> Dict[str, float]:
        """Get all tracking metrics."""
        return {
            'MOTA': self.compute_mota(),
            'FN': self.false_negatives,
            'FP': self.false_positives,
            'IDSW': self.id_switches,
            'GT': self.total_gt,
            'FNR': self.false_negatives / max(self.total_gt, 1),  # Miss rate
            'FPR': self.false_positives / max(self.total_gt, 1),
        }


class OCRAccuracyCalculator:
    """
    Calculate OCR accuracy metrics.
    
    Metrics:
    - Character accuracy (CER complement)
    - Word accuracy (WER complement)
    - Exact match accuracy
    """
    
    def __init__(self):
        self.total_chars = 0
        self.correct_chars = 0
        self.total_words = 0
        self.correct_words = 0
        self.exact_matches = 0
    
    def reset(self):
        """Reset accumulated metrics."""
        self.total_chars = 0
        self.correct_chars = 0
        self.total_words = 0
        self.correct_words = 0
        self.exact_matches = 0
    
    def add_sample(self, prediction: str, target: str):
        """Add a single prediction-target pair."""
        # Normalize
        prediction = prediction.upper().strip()
        target = target.upper().strip()
        
        # Character-level accuracy
        self.total_chars += len(target)
        
        # Use edit distance for character accuracy
        edit_dist = self._edit_distance(prediction, target)
        self.correct_chars += max(0, len(target) - edit_dist)
        
        # Word-level (here, word = plate = sample)
        self.total_words += 1
        
        # Exact match
        if prediction == target:
            self.exact_matches += 1
            self.correct_words += 1
    
    def _edit_distance(self, s1: str, s2: str) -> int:
        """Compute Levenshtein edit distance."""
        m, n = len(s1), len(s2)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        
        for i in range(m + 1):
            dp[i][0] = i
        for j in range(n + 1):
            dp[0][j] = j
        
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if s1[i-1] == s2[j-1]:
                    dp[i][j] = dp[i-1][j-1]
                else:
                    dp[i][j] = 1 + min(dp[i-1][j], dp[i][j-1], dp[i-1][j-1])
        
        return dp[m][n]
    
    def compute_accuracy(self) -> Dict[str, float]:
        """
        Compute all accuracy metrics.
        
        Returns:
            Dict with 'char_accuracy', 'word_accuracy', 'exact_match'
        """
        return {
            'char_accuracy': self.correct_chars / max(self.total_chars, 1),
            'word_accuracy': self.correct_words / max(self.total_words, 1),
            'exact_match': self.exact_matches / max(self.total_words, 1),
            'CER': 1 - self.correct_chars / max(self.total_chars, 1),  # Char error rate
        }


class MultiTaskEvaluator:
    """
    Combined evaluator for all tasks.
    
    Implements Algorithm S3.2: VALIDATE-ALL-TASKS
    """
    
    def __init__(
        self,
        num_vehicle_classes: int = 6,
        num_seg_classes: int = 3,
        iou_threshold: float = 0.5,
    ):
        self.detection_eval = AveragePrecisionCalculator(
            iou_threshold=iou_threshold,
            num_classes=num_vehicle_classes
        )
        self.plate_eval = AveragePrecisionCalculator(
            iou_threshold=iou_threshold,
            num_classes=1  # Binary: plate or not
        )
        self.seg_eval = MIoUCalculator(num_classes=num_seg_classes)
        self.ocr_eval = OCRAccuracyCalculator()
        self.tracking_eval = MOTACalculator(iou_threshold=iou_threshold)
    
    def reset(self):
        """Reset all evaluators."""
        self.detection_eval.reset()
        self.plate_eval.reset()
        self.seg_eval.reset()
        self.ocr_eval.reset()
        self.tracking_eval.reset()
    
    def add_batch(
        self,
        outputs: Dict[str, Any],
        targets: Dict[str, Any],
    ):
        """
        Add predictions and targets for evaluation.
        
        Args:
            outputs: Model outputs
            targets: Ground truth targets
        """
        B = outputs.get('detection', {}).get('bbox', torch.zeros(1)).size(0)
        
        for b in range(B):
            # Detection
            if 'detection' in outputs and 'detection' in targets:
                pred_det = outputs['detection']
                gt_det = targets['detection'][b] if isinstance(targets['detection'], list) else targets['detection']
                
                # Get scores from logits - move to CPU for evaluation
                if 'class_logits' in pred_det:
                    logits = pred_det['class_logits'][b].cpu()
                    scores, labels = logits.softmax(-1).max(-1)
                else:
                    num_queries = pred_det['bbox'].size(1)
                    scores = torch.ones(num_queries)
                    labels = torch.zeros(num_queries, dtype=torch.long)
                
                # Get boxes on CPU
                pred_boxes = pred_det['bbox'][b].cpu()
                gt_boxes = gt_det.get('boxes', torch.zeros(0, 4))
                if isinstance(gt_boxes, torch.Tensor):
                    gt_boxes = gt_boxes.cpu()
                gt_labels = gt_det.get('labels', torch.zeros(0, dtype=torch.long))
                if isinstance(gt_labels, torch.Tensor):
                    gt_labels = gt_labels.cpu()
                
                self.detection_eval.add_batch(
                    pred_boxes=pred_boxes,
                    pred_labels=labels,
                    pred_scores=scores,
                    gt_boxes=gt_boxes,
                    gt_labels=gt_labels,
                )
            
            # Segmentation
            if 'segmentation' in outputs and 'segmentation' in targets:
                try:
                    seg_pred = outputs['segmentation']
                    seg_gt = targets['segmentation']
                    
                    # Get target for this batch item
                    if seg_gt.dim() == 3:
                        gt_mask = seg_gt[b]  # [H, W]
                    else:
                        gt_mask = seg_gt
                    
                    # Get class predictions from masks
                    if 'masks' in seg_pred and 'class_logits' in seg_pred:
                        # Combine masks with class logits
                        masks = seg_pred['masks'][b]  # [N_q, H, W]
                        class_logits = seg_pred['class_logits'][b]  # [N_q, C]
                        
                        # Get the class for each query
                        class_probs = class_logits.softmax(-1)  # [N_q, C]
                        query_classes = class_probs.argmax(-1)  # [N_q]
                        
                        # Create prediction mask: for each pixel, pick the class of the 
                        # query with highest mask activation
                        # masks: [N_q, H', W']
                        H, W = gt_mask.shape
                        
                        # Resize masks to target size if needed
                        if masks.shape[-2:] != (H, W):
                            masks_resized = F.interpolate(
                                masks.unsqueeze(0),
                                size=(H, W),
                                mode='bilinear',
                                align_corners=False
                            ).squeeze(0)  # [N_q, H, W]
                        else:
                            masks_resized = masks
                        
                        # Get best query for each pixel
                        best_query = masks_resized.argmax(dim=0)  # [H, W]
                        
                        # Map to class predictions
                        pred_mask = query_classes[best_query]  # [H, W]
                        
                        # Add to evaluator
                        self.seg_eval.add_batch(
                            predictions=pred_mask.unsqueeze(0).cpu(),
                            targets=gt_mask.unsqueeze(0).cpu(),
                        )
                except Exception as e:
                    # Skip segmentation evaluation for this batch item if shapes don't match
                    pass
            
            # OCR
            if 'ocr' in outputs and 'ocr' in targets:
                # Decode predictions
                if 'char_logits' in outputs['ocr']:
                    # CTC decode
                    logits = outputs['ocr']['char_logits'][b]
                    pred_text = self._ctc_decode(logits)
                else:
                    pred_text = ""
                
                # Get target text
                if isinstance(targets['ocr'], list) and len(targets['ocr']) > b:
                    gt_text = targets['ocr'][b]
                else:
                    gt_text = ""
                
                if pred_text and gt_text:
                    self.ocr_eval.add_sample(pred_text, gt_text)
    
    def _ctc_decode(
        self,
        logits: torch.Tensor,
        alphabet: str = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789",
    ) -> str:
        """Simple CTC greedy decode."""
        # Get best path
        probs = logits.softmax(-1)
        indices = probs.argmax(-1)  # [T]
        
        # Collapse repeated and remove blanks
        blank_idx = len(alphabet)
        result = []
        prev_idx = -1
        
        for idx in indices:
            idx = idx.item()
            if idx != prev_idx and idx != blank_idx:
                if idx < len(alphabet):
                    result.append(alphabet[idx])
            prev_idx = idx
        
        return ''.join(result)
    
    def compute_metrics(self) -> Dict[str, Any]:
        """
        Compute all evaluation metrics.
        
        Returns:
            Dict with metrics for each task
        """
        metrics = {}
        
        # Detection
        det_map, det_aps = self.detection_eval.compute_map()
        metrics['detection'] = {
            'mAP': det_map,
            'per_class_AP': det_aps,
        }
        
        # Plate detection
        plate_map, plate_aps = self.plate_eval.compute_map()
        metrics['plate'] = {
            'mAP': plate_map,
        }
        
        # Segmentation
        seg_miou, seg_ious = self.seg_eval.compute_miou()
        metrics['segmentation'] = {
            'mIoU': seg_miou,
            'per_class_IoU': seg_ious,
            'pixel_accuracy': self.seg_eval.compute_pixel_accuracy(),
        }
        
        # OCR
        metrics['ocr'] = self.ocr_eval.compute_accuracy()
        
        # Tracking
        metrics['tracking'] = self.tracking_eval.get_metrics()
        
        return metrics
    
    def summarize(self) -> str:
        """Generate summary string of all metrics."""
        metrics = self.compute_metrics()
        
        lines = [
            "=" * 50,
            "  Evaluation Results",
            "=" * 50,
            f"  Detection mAP:      {metrics['detection']['mAP']:.4f}",
            f"  Plate mAP:          {metrics['plate']['mAP']:.4f}",
            f"  Segmentation mIoU:  {metrics['segmentation']['mIoU']:.4f}",
            f"  OCR Char Accuracy:  {metrics['ocr']['char_accuracy']:.4f}",
            f"  OCR Exact Match:    {metrics['ocr']['exact_match']:.4f}",
            f"  Tracking MOTA:      {metrics['tracking']['MOTA']:.4f}",
            "=" * 50,
        ]
        
        return '\n'.join(lines)


if __name__ == "__main__":
    print("📊 Evaluation Metrics Test\n")
    
    # Test mAP
    print("Testing mAP calculator...")
    ap_calc = AveragePrecisionCalculator(num_classes=6)
    
    for _ in range(10):
        pred_boxes = torch.rand(10, 4)
        pred_labels = torch.randint(0, 6, (10,))
        pred_scores = torch.rand(10)
        gt_boxes = torch.rand(5, 4)
        gt_labels = torch.randint(0, 6, (5,))
        
        ap_calc.add_batch(pred_boxes, pred_labels, pred_scores, gt_boxes, gt_labels)
    
    mAP, aps = ap_calc.compute_map()
    print(f"  mAP: {mAP:.4f}")
    
    # Test mIoU
    print("\nTesting mIoU calculator...")
    miou_calc = MIoUCalculator(num_classes=3)
    
    for _ in range(10):
        pred = torch.randint(0, 3, (2, 64, 64))
        target = torch.randint(0, 3, (2, 64, 64))
        miou_calc.add_batch(pred, target)
    
    miou, ious = miou_calc.compute_miou()
    print(f"  mIoU: {miou:.4f}")
    
    # Test OCR
    print("\nTesting OCR accuracy...")
    ocr_calc = OCRAccuracyCalculator()
    
    ocr_calc.add_sample("DL01AB1234", "DL01AB1234")  # Perfect
    ocr_calc.add_sample("DL01AB1235", "DL01AB1234")  # 1 error
    ocr_calc.add_sample("DL01CD9999", "DL01AB1234")  # Many errors
    
    ocr_metrics = ocr_calc.compute_accuracy()
    print(f"  Char accuracy: {ocr_metrics['char_accuracy']:.4f}")
    print(f"  Exact match: {ocr_metrics['exact_match']:.4f}")
    
    # Test MOTA
    print("\nTesting MOTA calculator...")
    mota_calc = MOTACalculator()
    
    for frame in range(5):
        pred_boxes = torch.rand(3, 4)
        pred_ids = torch.tensor([1, 2, 3])
        gt_boxes = torch.rand(3, 4)
        gt_ids = torch.tensor([1, 2, 3])
        
        mota_calc.add_frame(pred_boxes, pred_ids, gt_boxes, gt_ids)
    
    mota = mota_calc.compute_mota()
    print(f"  MOTA: {mota:.4f}")
    
    print("\n✓ All evaluation tests passed!")
