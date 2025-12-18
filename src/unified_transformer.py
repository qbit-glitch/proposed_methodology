"""
Unified Multi-Task Vision Transformer - Algorithm 1.

Combines all components into a single end-to-end model:
1. ResNet50 Backbone for feature extraction
2. Transformer Encoder for global context
3. Five parallel decoders with hierarchical information flow:
   - Detection Decoder (independent)
   - Segmentation Decoder (uses detection)
   - Plate Decoder (uses detection)
   - OCR Decoder (uses plate)
   - Tracking Decoder (uses detection + segmentation)
4. Alert Generation System
"""

import torch
import torch.nn as nn
from typing import Dict, Tuple, Optional, List

from .config import ModelConfig, get_device
from .backbone import ResNet50Backbone, FeaturePyramidFlattener
from .positional_encoding import MultiScalePositionalEncoding
from .transformer_encoder import TransformerEncoder
from .vit_encoder import ViTEncoder, ViTFeatureFusion
from .decoders.detection_decoder import DetectionDecoder
from .decoders.segmentation_decoder import SegmentationDecoder
from .decoders.plate_decoder import PlateDecoder
from .decoders.ocr_decoder import OCRDecoder
from .decoders.tracking_decoder import TrackingDecoder


class UnifiedMultiTaskTransformer(nn.Module):
    """
    Unified Multi-Task Vision Transformer for Parking Violation Detection.
    
    A single end-to-end model that performs:
    - Vehicle detection (6 classes + attributes)
    - Scene segmentation (driveway/footpath)
    - License plate detection
    - OCR text recognition
    - Multi-object tracking
    - Parking violation alert generation
    """
    
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        
        # ===== Stage 1: Feature Extraction =====
        self.backbone = ResNet50Backbone(config)
        self.flattener = FeaturePyramidFlattener(config)
        self.pos_encoder = MultiScalePositionalEncoding(
            hidden_dim=config.hidden_dim,
            max_size=128
        )
        
        # ===== Stage 1b: ViT for Global/Temporal Features =====
        self.use_vit = getattr(config, 'use_pretrained_vit', True)
        if self.use_vit:
            self.vit_encoder = ViTEncoder(config)
            self.vit_fusion = ViTFeatureFusion(config)
        
        # ===== Stage 2: Transformer Encoder =====
        self.encoder = TransformerEncoder(config)
        
        # ===== Stage 3: Parallel Decoders =====
        # Detection decoder (independent)
        self.detection_decoder = DetectionDecoder(config)
        
        # Segmentation decoder (uses detection features)
        self.segmentation_decoder = SegmentationDecoder(config)
        
        # Plate decoder (uses detection features)
        self.plate_decoder = PlateDecoder(config)
        
        # OCR decoder (uses plate features)
        self.ocr_decoder = OCRDecoder(config)
        
        # Tracking decoder (uses detection + segmentation features)
        self.tracking_decoder = TrackingDecoder(config)
        
        print(f"✓ Unified Multi-Task Transformer initialized")
        print(f"  - Backbone: ResNet50 (pretrained={config.pretrained})")
        print(f"  - ViT Encoder: {self.use_vit} (pretrained={getattr(config, 'use_pretrained_vit', True)})")
        print(f"  - Encoder: {config.num_encoder_layers} layers")
        print(f"  - Decoder: {config.num_decoder_layers} layers")
        print(f"  - Hidden dim: {config.hidden_dim}")
    
    def forward(
        self,
        images: torch.Tensor,
        prev_tracking_state: Optional[Dict] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through the entire model.
        
        Args:
            images: Input images [B, 3, H, W]
            prev_tracking_state: Previous frame tracking state for temporal association
            
        Returns:
            Dictionary containing all outputs from all decoders
        """
        B = images.shape[0]
        
        # ===== Stage 1: Feature Extraction =====
        # Extract multi-scale features from backbone (ResNet50)
        backbone_features = self.backbone(images)
        
        # Flatten and concatenate features
        features_flat, scale_lengths = self.flattener(backbone_features)
        
        # Get scale shapes for positional encoding
        scale_shapes = [
            backbone_features['c3_shape'],
            backbone_features['c4_shape'],
            backbone_features['c5_shape']
        ]
        
        # Add positional encoding
        features_pos = self.pos_encoder(features_flat, scale_shapes, scale_lengths)
        
        # ===== Stage 1b: ViT Global Features =====
        if self.use_vit:
            # Extract global features from pretrained ViT
            vit_outputs = self.vit_encoder(images, return_patch_features=False)
            global_features = vit_outputs['global']  # [B, D]
            
            # Fuse local (CNN) and global (ViT) features
            features_pos = self.vit_fusion(features_pos, global_features)
        
        # ===== Stage 2: Transformer Encoder =====
        memory = self.encoder(features_pos)
        
        # ===== Stage 3: Hierarchical Parallel Decoding =====
        
        # Stage 3a: Detection Decoder (independent)
        detection_features, detection_outputs = self.detection_decoder(memory)
        
        # Stage 3b: Segmentation Decoder (uses detection features)
        segmentation_features, segmentation_outputs = self.segmentation_decoder(
            memory=memory,
            detection_features=detection_features,
            backbone_features=backbone_features
        )
        
        # Stage 3c: Plate Decoder (uses detection features)
        plate_features, plate_outputs = self.plate_decoder(
            memory=memory,
            detection_features=detection_features
        )
        
        # Stage 3d: OCR Decoder (uses plate features)
        ocr_outputs = self.ocr_decoder(
            memory=memory,
            plate_features=plate_features
        )
        
        # Stage 3e: Tracking Decoder (uses detection + segmentation features)
        tracking_outputs, new_tracking_state = self.tracking_decoder(
            memory=memory,
            detection_features=detection_features,
            segmentation_features=segmentation_features,
            detection_boxes=detection_outputs['bbox'],
            prev_state=prev_tracking_state
        )
        
        # ===== Combine all outputs =====
        outputs = {
            # Encoder memory for potential extensions
            'memory': memory,
            'scale_lengths': scale_lengths,
            
            # Detection outputs
            'detection': detection_outputs,
            'detection_features': detection_features,
            
            # Segmentation outputs
            'segmentation': segmentation_outputs,
            'segmentation_features': segmentation_features,
            
            # Plate outputs
            'plate': plate_outputs,
            'plate_features': plate_features,
            
            # OCR outputs
            'ocr': ocr_outputs,
            
            # Tracking outputs
            'tracking': tracking_outputs,
            'tracking_state': new_tracking_state,
            
            # Backbone features (for visualization)
            'backbone_features': backbone_features,
        }
        
        return outputs
    
    def get_semantic_segmentation(
        self,
        outputs: Dict,
        image_size: Tuple[int, int]
    ) -> torch.Tensor:
        """Get semantic segmentation mask from outputs."""
        masks = outputs['segmentation']['masks']
        class_logits = outputs['segmentation']['class_logits']
        semantic = self.segmentation_decoder.get_semantic_masks(
            masks, class_logits, image_size
        )
        return semantic
    
    def decode_plates(self, outputs: Dict) -> List[Dict]:
        """Decode license plate texts from OCR outputs."""
        return self.ocr_decoder.decode_plates(
            char_logits=outputs['ocr']['char_logits'],
            plate_confidence=outputs['plate']['plate_confidence']
        )
    
    @property
    def num_parameters(self) -> int:
        """Total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    @property
    def num_parameters_by_component(self) -> Dict[str, int]:
        """Number of parameters per component."""
        return {
            'backbone': sum(p.numel() for p in self.backbone.parameters()),
            'encoder': sum(p.numel() for p in self.encoder.parameters()),
            'detection_decoder': sum(p.numel() for p in self.detection_decoder.parameters()),
            'segmentation_decoder': sum(p.numel() for p in self.segmentation_decoder.parameters()),
            'plate_decoder': sum(p.numel() for p in self.plate_decoder.parameters()),
            'ocr_decoder': sum(p.numel() for p in self.ocr_decoder.parameters()),
            'tracking_decoder': sum(p.numel() for p in self.tracking_decoder.parameters()),
        }


def build_model(config: Optional[ModelConfig] = None) -> UnifiedMultiTaskTransformer:
    """Factory function to build the unified model."""
    if config is None:
        config = ModelConfig()
    return UnifiedMultiTaskTransformer(config)


if __name__ == "__main__":
    from config import ModelConfig, get_device
    
    # Build model
    config = ModelConfig()
    model = build_model(config)
    
    # Move to device
    device = get_device()
    model = model.to(device)
    
    # Print parameter counts
    print(f"\n📊 Model Parameters:")
    for name, count in model.num_parameters_by_component.items():
        print(f"   {name}: {count:,}")
    print(f"   TOTAL: {model.num_parameters:,}")
    
    # Test forward pass
    print(f"\n🔄 Testing forward pass...")
    images = torch.randn(2, 3, 512, 512, device=device)
    
    with torch.no_grad():
        outputs = model(images)
    
    print(f"\n📦 Output shapes:")
    print(f"   Memory: {outputs['memory'].shape}")
    print(f"   Detection bbox: {outputs['detection']['bbox'].shape}")
    print(f"   Detection class: {outputs['detection']['class_logits'].shape}")
    print(f"   Segmentation masks: {outputs['segmentation']['masks'].shape}")
    print(f"   Plate bbox: {outputs['plate']['plate_bbox'].shape}")
    print(f"   OCR char logits: {outputs['ocr']['char_logits'].shape}")
    print(f"   Tracking features: {outputs['tracking']['features'].shape}")
    
    print(f"\n✓ Forward pass successful!")
