#!/usr/bin/env python3
"""
Demo script for Unified Multi-Task Vision Transformer.

Tests the complete pipeline step-by-step with visualizations:
1. Backbone feature extraction
2. Transformer encoding
3. Detection decoding
4. Segmentation decoding
5. Plate detection
6. OCR decoding
7. Tracking
8. Alert generation

Usage:
    python demo.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
import time

# Add src to path
from src.config import ModelConfig, get_device, get_config
from src.backbone import ResNet50Backbone, FeaturePyramidFlattener
from src.positional_encoding import MultiScalePositionalEncoding
from src.transformer_encoder import TransformerEncoder
from src.decoders.detection_decoder import DetectionDecoder
from src.decoders.segmentation_decoder import SegmentationDecoder
from src.decoders.plate_decoder import PlateDecoder
from src.decoders.ocr_decoder import OCRDecoder
from src.decoders.tracking_decoder import TrackingDecoder
from src.unified_transformer import UnifiedMultiTaskTransformer, build_model
from src.alert_system import AlertSystem, generate_alerts_from_outputs
from src.visualizer import Visualizer, create_visualization_grid


def create_synthetic_image(height: int = 512, width: int = 512) -> torch.Tensor:
    """Create a synthetic test image with some structure."""
    # Create gradient background (simulating road scene)
    y = torch.linspace(0, 1, height).unsqueeze(1).repeat(1, width)
    x = torch.linspace(0, 1, width).unsqueeze(0).repeat(height, 1)
    
    # Sky (blue gradient at top)
    sky = torch.stack([
        0.5 * (1 - y),  # R
        0.7 * (1 - y),  # G
        0.9 * (1 - y),  # B
    ])
    
    # Road (gray at bottom)
    road = torch.stack([
        0.3 * y,  # R
        0.3 * y,  # G
        0.35 * y,  # B
    ])
    
    # Combine
    image = sky + road
    
    # Add some rectangular "vehicles"
    # Vehicle 1
    image[:, 200:280, 100:180] = torch.tensor([0.2, 0.4, 0.8]).view(3, 1, 1)
    # Vehicle 2
    image[:, 300:380, 300:400] = torch.tensor([0.8, 0.2, 0.2]).view(3, 1, 1)
    # Vehicle 3
    image[:, 350:420, 150:220] = torch.tensor([0.9, 0.9, 0.9]).view(3, 1, 1)
    
    # Normalize to [0, 1]
    image = torch.clamp(image, 0, 1)
    
    return image


def print_separator(title: str):
    """Print a formatted section separator."""
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


def test_step_by_step():
    """Test each component step by step."""
    
    print_separator("🚀 UNIFIED MULTI-TASK TRANSFORMER DEMO")
    
    # Get device
    device = get_device()
    print(f"\n📱 Device: {device}")
    
    # Create config
    config = ModelConfig()
    print(f"📐 Image size: {config.image_size}")
    print(f"📊 Hidden dim: {config.hidden_dim}")
    
    # Create synthetic input
    print_separator("Step 1: Input Image")
    image = create_synthetic_image(512, 512)
    image_batch = image.unsqueeze(0).to(device)  # [1, 3, 512, 512]
    print(f"   Input shape: {image_batch.shape}")
    
    # Save input image
    os.makedirs("outputs", exist_ok=True)
    visualizer = Visualizer()
    input_pil = visualizer.tensor_to_pil(image)
    input_pil.save("outputs/01_input_image.png")
    print(f"   ✓ Saved: outputs/01_input_image.png")
    
    # Step 2: Backbone
    print_separator("Step 2: Backbone Feature Extraction")
    backbone = ResNet50Backbone(config).to(device)
    with torch.no_grad():
        backbone_features = backbone(image_batch)
    
    print(f"   C3 shape: {backbone_features['c3'].shape}")
    print(f"   C4 shape: {backbone_features['c4'].shape}")
    print(f"   C5 shape: {backbone_features['c5'].shape}")
    print(f"   C3_flat shape: {backbone_features['c3_flat'].shape}")
    
    # Step 3: Feature Pyramid Flattening
    print_separator("Step 3: Feature Pyramid Flattening")
    flattener = FeaturePyramidFlattener(config)
    with torch.no_grad():
        features_flat, scale_lengths = flattener(backbone_features)
    
    print(f"   Flattened shape: {features_flat.shape}")
    print(f"   Scale lengths: {scale_lengths}")
    print(f"   Total tokens: {sum(scale_lengths)}")
    
    # Step 4: Positional Encoding
    print_separator("Step 4: Positional Encoding")
    pos_encoder = MultiScalePositionalEncoding(config.hidden_dim).to(device)
    scale_shapes = [
        backbone_features['c3_shape'],
        backbone_features['c4_shape'],
        backbone_features['c5_shape']
    ]
    with torch.no_grad():
        features_pos = pos_encoder(features_flat, scale_shapes, scale_lengths)
    
    print(f"   Output shape: {features_pos.shape}")
    
    # Step 5: Transformer Encoder
    print_separator("Step 5: Transformer Encoder")
    encoder = TransformerEncoder(config).to(device)
    start_time = time.time()
    with torch.no_grad():
        memory = encoder(features_pos)
    encoder_time = time.time() - start_time
    
    print(f"   Memory shape: {memory.shape}")
    print(f"   Encoding time: {encoder_time:.3f}s")
    
    # Step 6: Detection Decoder
    print_separator("Step 6: Detection Decoder")
    det_decoder = DetectionDecoder(config).to(device)
    start_time = time.time()
    with torch.no_grad():
        det_features, det_outputs = det_decoder(memory)
    det_time = time.time() - start_time
    
    print(f"   Detection features: {det_features.shape}")
    print(f"   BBox: {det_outputs['bbox'].shape}")
    print(f"   Class logits: {det_outputs['class_logits'].shape}")
    print(f"   Decoding time: {det_time:.3f}s")
    
    # Visualize detections
    det_image = input_pil.copy()
    det_image = visualizer.draw_bboxes(
        det_image, 
        det_outputs['bbox'][0].cpu(), 
        det_outputs['class_logits'][0].cpu(),
        scores_threshold=0.3
    )
    det_image.save("outputs/02_detection.png")
    print(f"   ✓ Saved: outputs/02_detection.png")
    
    # Step 7: Segmentation Decoder
    print_separator("Step 7: Segmentation Decoder")
    seg_decoder = SegmentationDecoder(config).to(device)
    start_time = time.time()
    with torch.no_grad():
        seg_features, seg_outputs = seg_decoder(
            memory, det_features, backbone_features
        )
    seg_time = time.time() - start_time
    
    print(f"   Segmentation features: {seg_features.shape}")
    print(f"   Masks: {seg_outputs['masks'].shape}")
    print(f"   Class logits: {seg_outputs['class_logits'].shape}")
    print(f"   Decoding time: {seg_time:.3f}s")
    
    # Visualize segmentation
    seg_mask = seg_outputs['masks'][0].mean(dim=0).cpu()
    seg_mask = (seg_mask > 0).long()
    seg_image = visualizer.draw_segmentation(input_pil.copy(), seg_mask)
    seg_image.save("outputs/03_segmentation.png")
    print(f"   ✓ Saved: outputs/03_segmentation.png")
    
    # Step 8: Plate Decoder
    print_separator("Step 8: Plate Detection Decoder")
    plate_decoder = PlateDecoder(config).to(device)
    start_time = time.time()
    with torch.no_grad():
        plate_features, plate_outputs = plate_decoder(memory, det_features)
    plate_time = time.time() - start_time
    
    print(f"   Plate features: {plate_features.shape}")
    print(f"   Plate BBox: {plate_outputs['plate_bbox'].shape}")
    print(f"   Confidence: {plate_outputs['plate_confidence'].shape}")
    print(f"   Decoding time: {plate_time:.3f}s")
    
    # Visualize plates
    plate_image = visualizer.draw_plates(
        input_pil.copy(),
        plate_outputs['plate_bbox'][0].cpu(),
        plate_outputs['plate_confidence'][0].cpu(),
        threshold=0.2
    )
    plate_image.save("outputs/04_plates.png")
    print(f"   ✓ Saved: outputs/04_plates.png")
    
    # Step 9: OCR Decoder
    print_separator("Step 9: OCR Decoder")
    ocr_decoder = OCRDecoder(config).to(device)
    start_time = time.time()
    with torch.no_grad():
        ocr_outputs = ocr_decoder(memory, plate_features)
    ocr_time = time.time() - start_time
    
    print(f"   OCR features: {ocr_outputs['features'].shape}")
    print(f"   Char logits: {ocr_outputs['char_logits'].shape}")
    print(f"   Decoding time: {ocr_time:.3f}s")
    
    # Decode text (random for demo)
    decoded_texts = ocr_decoder.head.decode_greedy(ocr_outputs['char_logits'])
    print(f"   Decoded texts (random): {decoded_texts}")
    
    # Step 10: Tracking Decoder
    print_separator("Step 10: Tracking Decoder")
    track_decoder = TrackingDecoder(config).to(device)
    start_time = time.time()
    with torch.no_grad():
        track_outputs, track_state = track_decoder(
            memory, det_features, seg_features, det_outputs['bbox']
        )
    track_time = time.time() - start_time
    
    print(f"   Track features: {track_outputs['features'].shape}")
    print(f"   Trajectory delta: {track_outputs['trajectory_delta'].shape}")
    print(f"   Velocity: {track_outputs['velocity'].shape}")
    print(f"   Association scores: {track_outputs['association_scores'].shape}")
    print(f"   Matches: {len(track_outputs['matches'][0])} tracks")
    print(f"   Decoding time: {track_time:.3f}s")
    
    # Step 11: Alert Generation
    print_separator("Step 11: Alert Generation")
    alert_system = AlertSystem()
    
    # Create fake stopped times (some vehicles stopped)
    stopped_times = torch.zeros_like(det_outputs['bbox'][:, :, 0])
    stopped_times[0, 0] = 150  # Vehicle 0: stopped 150s (VIOLATION)
    stopped_times[0, 1] = 60   # Vehicle 1: stopped 60s (OK)
    stopped_times[0, 2] = 180  # Vehicle 2: stopped 180s (depends on overlap)
    
    # Get driveway mask
    driveway_mask = seg_outputs['masks'][0].mean(dim=0).cpu().sigmoid()
    driveway_mask = F.interpolate(
        driveway_mask.unsqueeze(0).unsqueeze(0),
        size=(512, 512),
        mode='bilinear'
    ).squeeze()
    
    alerts = alert_system.generate_alerts(
        detection_bboxes=det_outputs['bbox'].cpu(),
        detection_classes=det_outputs['class_logits'].cpu(),
        driveway_mask=driveway_mask.unsqueeze(0),
        stopped_times=stopped_times.cpu(),
    )
    
    violations = [a for a in alerts[0] if a.alert_type == "RED"]
    print(f"   Total detections: {len(alerts[0])}")
    print(f"   Violations: {len(violations)}")
    
    # Visualize alerts
    alert_image = visualizer.draw_bboxes(
        input_pil.copy(),
        det_outputs['bbox'][0].cpu(),
        det_outputs['class_logits'][0].cpu(),
        scores_threshold=0.3,
        alerts=alerts[0]
    )
    alert_image.save("outputs/05_alerts.png")
    print(f"   ✓ Saved: outputs/05_alerts.png")
    
    # Summary
    print_separator("📊 TIMING SUMMARY")
    total_time = encoder_time + det_time + seg_time + plate_time + ocr_time + track_time
    print(f"   Encoder:      {encoder_time:.3f}s")
    print(f"   Detection:    {det_time:.3f}s")
    print(f"   Segmentation: {seg_time:.3f}s")
    print(f"   Plate:        {plate_time:.3f}s")
    print(f"   OCR:          {ocr_time:.3f}s")
    print(f"   Tracking:     {track_time:.3f}s")
    print(f"   ─────────────────────")
    print(f"   TOTAL:        {total_time:.3f}s")
    print(f"   FPS:          {1/total_time:.1f}")
    
    return True


def test_unified_model():
    """Test the complete unified model."""
    print_separator("🔄 UNIFIED MODEL TEST")
    
    device = get_device()
    config = ModelConfig()
    
    # Build model
    print("   Building model...")
    model = build_model(config).to(device)
    
    # Print parameters
    print(f"\n   📊 Model Parameters:")
    for name, count in model.num_parameters_by_component.items():
        print(f"      {name}: {count:,}")
    print(f"      {'─'*30}")
    print(f"      TOTAL: {model.num_parameters:,}")
    
    # Test forward pass
    print("\n   Testing forward pass...")
    image = create_synthetic_image(512, 512)
    image_batch = image.unsqueeze(0).to(device)
    
    start_time = time.time()
    with torch.no_grad():
        outputs = model(image_batch)
    forward_time = time.time() - start_time
    
    print(f"\n   ✓ Forward pass successful!")
    print(f"   ⏱️  Time: {forward_time:.3f}s")
    print(f"   🎯 FPS: {1/forward_time:.1f}")
    
    # Print output shapes
    print(f"\n   📦 Output shapes:")
    print(f"      Memory: {outputs['memory'].shape}")
    print(f"      Detection bbox: {outputs['detection']['bbox'].shape}")
    print(f"      Segmentation masks: {outputs['segmentation']['masks'].shape}")
    print(f"      Plate bbox: {outputs['plate']['plate_bbox'].shape}")
    print(f"      OCR logits: {outputs['ocr']['char_logits'].shape}")
    print(f"      Track features: {outputs['tracking']['features'].shape}")
    
    # Create combined visualization
    visualizer = Visualizer()
    
    # Save combined visualization
    combined_image = visualizer.visualize_all(
        image,
        {k: {kk: vv.cpu() if isinstance(vv, torch.Tensor) else vv 
             for kk, vv in v.items()} if isinstance(v, dict) else v.cpu() if isinstance(v, torch.Tensor) else v
         for k, v in outputs.items()},
        save_path="outputs/06_combined.png"
    )
    
    return True


def main():
    """Main demo function."""
    try:
        # Step by step test
        success = test_step_by_step()
        
        if success:
            # Unified model test
            test_unified_model()
        
        print_separator("✅ DEMO COMPLETE")
        print("\n   All visualizations saved to: outputs/")
        print("   Files:")
        for f in sorted(os.listdir("outputs")):
            if f.endswith(".png"):
                print(f"      - {f}")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
