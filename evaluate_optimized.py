#!/usr/bin/env python3
"""
Evaluate and Visualize the Trained Optimized Model.

Generates:
- Detection mAP
- Segmentation mIoU
- Tracking MOTA (if applicable)
- OCR Accuracy (if applicable)
- Visualization samples
"""

import os
import sys
import argparse
from pathlib import Path
from tqdm import tqdm
from datetime import datetime

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from PIL import Image
import numpy as np

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from optimized_model.config import OptimizedModelConfig
from optimized_model.optimized_unified_transformer import OptimizedUnifiedTransformer
from optimized_model.visualizer import Visualizer, create_visualization_grid
from optimized_model.evaluation import (
    AveragePrecisionCalculator,
    MIoUCalculator,
    MOTACalculator,
)

from src.datasets import (
    create_coco_loaders,
    create_cityscapes_loaders,
    create_ccpd_loaders,
    create_mot17_loaders,
    MultiTaskDataset,
    multi_task_collate_fn,
)


def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate Optimized Model')
    parser.add_argument('--checkpoint', type=str, 
                       default='outputs_optimized/checkpoints/best_model.pt',
                       help='Path to model checkpoint')
    parser.add_argument('--data-dir', type=str, default='datasets',
                       help='Path to datasets')
    parser.add_argument('--output-dir', type=str, default='outputs_optimized/evaluation',
                       help='Output directory for results')
    parser.add_argument('--num-samples', type=int, default=10,
                       help='Number of samples to visualize')
    parser.add_argument('--subset-ratio', type=float, default=0.01,
                       help='Dataset subset ratio for evaluation')
    return parser.parse_args()


def load_model(checkpoint_path: str, device: torch.device):
    """Load trained model from checkpoint."""
    print(f"📂 Loading model from {checkpoint_path}")
    
    config = OptimizedModelConfig(
        use_deformable_attention=False,
        use_hierarchical_decoder=True,
        use_query_refinement=False,
        use_bifpn=True,
    )
    
    model = OptimizedUnifiedTransformer(config)
    
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    print(f"  ✓ Loaded checkpoint from epoch {checkpoint.get('epoch', 'unknown')}")
    print(f"  ✓ Best loss: {checkpoint.get('best_loss', 'unknown')}")
    
    return model, config


def evaluate_detection(model, dataloader, device, num_classes=7):
    """Evaluate detection performance (mAP)."""
    print("\n📊 Evaluating Detection (mAP)...")
    
    ap_calculator = AveragePrecisionCalculator(num_classes=num_classes)
    
    for images, targets in tqdm(dataloader, desc="Detection"):
        images = images.to(device)
        
        with torch.no_grad():
            outputs = model(images)
        
        # Get predictions
        pred_boxes = outputs['detection']['bbox']  # [B, N, 4]
        pred_logits = outputs['detection']['class_logits']  # [B, N, C]
        pred_scores, pred_labels = pred_logits.softmax(-1).max(-1)
        
        # Process each image in batch
        for b in range(images.shape[0]):
            # Get predictions for this image
            p_boxes = pred_boxes[b]  # [N, 4]
            p_labels = pred_labels[b]  # [N]
            p_scores = pred_scores[b]  # [N]
            
            # Filter background
            fg_mask = p_labels < (num_classes - 1)
            p_boxes = p_boxes[fg_mask]
            p_labels = p_labels[fg_mask]
            p_scores = p_scores[fg_mask]
            
            # Convert to xyxy format
            if p_boxes.shape[0] > 0:
                x_c, y_c, w, h = p_boxes.unbind(-1)
                p_boxes_xyxy = torch.stack([
                    x_c - w/2, y_c - h/2,
                    x_c + w/2, y_c + h/2
                ], dim=-1)
            else:
                p_boxes_xyxy = torch.zeros(0, 4, device=device)
            
            # Get ground truth
            if targets['detection'] and b < len(targets['detection']):
                gt = targets['detection'][b]
                if gt and 'boxes' in gt and gt['boxes'].numel() > 0:
                    gt_boxes = gt['boxes'].to(device)
                    gt_labels = gt['labels'].to(device)
                    
                    # Convert gt boxes to xyxy if needed
                    if gt_boxes.shape[-1] == 4:
                        gt_boxes_xyxy = gt_boxes  # Assume already xyxy
                    else:
                        gt_boxes_xyxy = torch.zeros(0, 4, device=device)
                        gt_labels = torch.zeros(0, dtype=torch.long, device=device)
                else:
                    gt_boxes_xyxy = torch.zeros(0, 4, device=device)
                    gt_labels = torch.zeros(0, dtype=torch.long, device=device)
            else:
                gt_boxes_xyxy = torch.zeros(0, 4, device=device)
                gt_labels = torch.zeros(0, dtype=torch.long, device=device)
            
            # Add to calculator
            if p_boxes_xyxy.shape[0] > 0 or gt_boxes_xyxy.shape[0] > 0:
                ap_calculator.add_batch(
                    p_boxes_xyxy, p_labels, p_scores,
                    gt_boxes_xyxy, gt_labels
                )
    
    mAP, per_class_ap = ap_calculator.compute_map()
    return mAP, per_class_ap


def evaluate_segmentation(model, dataloader, device, num_classes=3):
    """Evaluate segmentation performance (mIoU)."""
    print("\n📊 Evaluating Segmentation (mIoU)...")
    
    miou_calculator = MIoUCalculator(num_classes=num_classes)
    
    for images, targets in tqdm(dataloader, desc="Segmentation"):
        images = images.to(device)
        
        with torch.no_grad():
            outputs = model(images)
        
        # Get segmentation predictions
        seg_masks = outputs['segmentation']['masks']  # [B, N, H, W]
        seg_logits = outputs['segmentation']['class_logits']  # [B, N, C]
        
        # Aggregate masks per class
        B = images.shape[0]
        for b in range(B):
            if targets['segmentation'] is not None:
                gt_mask = targets['segmentation']
                if isinstance(gt_mask, torch.Tensor) and gt_mask.numel() > 0:
                    gt_mask = gt_mask[b] if gt_mask.dim() > 2 else gt_mask
                    gt_mask = gt_mask.to(device)
                    
                    # Get predicted class per query
                    query_classes = seg_logits[b].argmax(-1)  # [N]
                    masks = seg_masks[b]  # [N, H, W]
                    
                    # Combine masks by class
                    H, W = masks.shape[1:]
                    pred_mask = torch.zeros(H, W, device=device, dtype=torch.long)
                    
                    for c in range(num_classes):
                        class_masks = masks[query_classes == c]
                        if class_masks.shape[0] > 0:
                            combined = class_masks.max(0)[0]
                            pred_mask[combined > 0.5] = c
                    
                    # Resize gt_mask if needed
                    if gt_mask.shape != pred_mask.shape:
                        gt_mask = F.interpolate(
                            gt_mask.unsqueeze(0).unsqueeze(0).float(),
                            size=(H, W), mode='nearest'
                        ).squeeze().long()
                    
                    miou_calculator.add_batch(pred_mask.unsqueeze(0), gt_mask.unsqueeze(0))
    
    mIoU, per_class_iou = miou_calculator.compute_miou()
    pixel_acc = miou_calculator.compute_pixel_accuracy()
    
    return mIoU, per_class_iou, pixel_acc


def generate_visualizations(model, dataloader, visualizer, output_dir, num_samples=10, device=None):
    """Generate visualization samples."""
    print(f"\n🎨 Generating {num_samples} visualization samples...")
    
    os.makedirs(output_dir, exist_ok=True)
    
    sample_count = 0
    all_visualizations = []
    
    for images, targets in dataloader:
        if sample_count >= num_samples:
            break
            
        images = images.to(device)
        
        with torch.no_grad():
            outputs = model(images)
        
        # Visualize each image in batch
        for b in range(images.shape[0]):
            if sample_count >= num_samples:
                break
            
            try:
                # Simple visualization: just save the input with detections
                img_tensor = images[b].cpu()
                
                # Convert to PIL
                img = img_tensor.permute(1, 2, 0).numpy()
                img = (img * 255).clip(0, 255).astype(np.uint8)
                pil_img = Image.fromarray(img)
                
                # Draw detection boxes
                det_boxes = outputs['detection']['bbox'][b].cpu()
                det_logits = outputs['detection']['class_logits'][b].cpu()
                det_scores, det_labels = det_logits.softmax(-1).max(-1)
                
                from PIL import ImageDraw
                draw = ImageDraw.Draw(pil_img)
                H, W = pil_img.size[1], pil_img.size[0]
                
                for i in range(min(20, det_boxes.shape[0])):
                    if det_scores[i] > 0.3 and det_labels[i] < 6:
                        cx, cy, w, h = det_boxes[i]
                        x1 = int((cx - w/2) * W)
                        y1 = int((cy - h/2) * H)
                        x2 = int((cx + w/2) * W)
                        y2 = int((cy + h/2) * H)
                        draw.rectangle([x1, y1, x2, y2], outline='red', width=2)
                        draw.text((x1, y1-10), f'{det_labels[i].item()}', fill='red')
                
                save_path = os.path.join(output_dir, f'sample_{sample_count:03d}.png')
                pil_img.save(save_path)
                all_visualizations.append(pil_img)
                sample_count += 1
                
            except Exception as e:
                print(f"    ⚠️ Skipping sample {sample_count}: {e}")
                sample_count += 1
    
    print(f"  ✓ Saved {len(all_visualizations)} visualizations")
    return all_visualizations


def main():
    args = parse_args()
    
    print("=" * 60)
    print("  📊 Optimized Model Evaluation & Visualization")
    print("=" * 60)
    
    # Device
    device = torch.device('mps' if torch.backends.mps.is_available() else 
                         'cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n🖥️  Device: {device}")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load model
    if not os.path.exists(args.checkpoint):
        print(f"❌ Checkpoint not found: {args.checkpoint}")
        return
    
    model, config = load_model(args.checkpoint, device)
    
    # Load datasets
    print(f"\n📦 Loading evaluation datasets (subset_ratio={args.subset_ratio})...")
    
    coco_train, coco_val = create_coco_loaders(
        os.path.join(args.data_dir, 'coco2017'),
        batch_size=2, num_workers=0, image_size=(512, 512),
        subset_ratio=args.subset_ratio
    )
    city_train, city_val = create_cityscapes_loaders(
        os.path.join(args.data_dir, 'cityscapes'),
        batch_size=2, num_workers=0, image_size=(512, 512),
        subset_ratio=args.subset_ratio
    )
    
    print(f"   ✓ COCO val: {len(coco_val.dataset)} images")
    print(f"   ✓ Cityscapes val: {len(city_val.dataset)} images")
    
    # Create multi-task dataset for visualization
    val_dataset = MultiTaskDataset(
        datasets={
            'detection': coco_val.dataset,
            'segmentation': city_val.dataset,
        },
        primary_task='detection',
    )
    
    combined_loader = DataLoader(
        val_dataset, batch_size=2, shuffle=False,
        num_workers=0, collate_fn=multi_task_collate_fn
    )
    
    # Evaluate detection
    det_mAP, det_per_class = evaluate_detection(model, coco_val, device)
    
    # Evaluate segmentation
    seg_mIoU, seg_per_class, pixel_acc = evaluate_segmentation(model, city_val, device)
    
    # Generate visualizations
    visualizer = Visualizer()
    visualizations = generate_visualizations(
        model, combined_loader, visualizer,
        os.path.join(args.output_dir, 'visualizations'),
        num_samples=args.num_samples,
        device=device
    )
    
    # Print results
    print("\n" + "=" * 60)
    print("  📊 EVALUATION RESULTS")
    print("=" * 60)
    
    print(f"\n🎯 Detection:")
    print(f"   mAP@0.5: {det_mAP:.4f}")
    
    print(f"\n🗺️  Segmentation:")
    print(f"   mIoU: {seg_mIoU:.4f}")
    print(f"   Pixel Accuracy: {pixel_acc:.4f}")
    
    # Save results to file
    results_path = os.path.join(args.output_dir, 'evaluation_results.txt')
    with open(results_path, 'w') as f:
        f.write(f"Optimized Model Evaluation Results\n")
        f.write(f"Generated: {datetime.now().isoformat()}\n")
        f.write(f"Checkpoint: {args.checkpoint}\n")
        f.write(f"\n{'='*40}\n\n")
        f.write(f"Detection:\n")
        f.write(f"  mAP@0.5: {det_mAP:.4f}\n")
        f.write(f"\nSegmentation:\n")
        f.write(f"  mIoU: {seg_mIoU:.4f}\n")
        f.write(f"  Pixel Accuracy: {pixel_acc:.4f}\n")
    
    print(f"\n💾 Results saved to: {results_path}")
    print(f"   Visualizations: {args.output_dir}/visualizations/")
    
    print("\n✅ Evaluation complete!")


if __name__ == '__main__':
    main()
