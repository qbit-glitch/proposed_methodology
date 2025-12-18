"""
Segmentation Decoder - Algorithm 4.

Implements segmentation with:
- 50 segmentation queries
- Inter-decoder cross-attention with detection features
- Mask head with FPN-style upsampling for driveway/footpath segmentation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional, List

from .base_decoder import BaseDecoder

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import ModelConfig


class FPNPixelDecoder(nn.Module):
    """
    Feature Pyramid Network-style pixel decoder for mask generation.
    
    Upsamples encoder features back to image resolution using
    multi-scale feature fusion.
    """
    
    def __init__(self, hidden_dim: int, mask_dim: int = 256):
        super().__init__()
        
        # Lateral connections for each scale
        self.lateral_c5 = nn.Conv2d(hidden_dim, mask_dim, 1)
        self.lateral_c4 = nn.Conv2d(hidden_dim, mask_dim, 1)
        self.lateral_c3 = nn.Conv2d(hidden_dim, mask_dim, 1)
        
        # Output convolutions after fusion
        self.output_c5 = nn.Conv2d(mask_dim, mask_dim, 3, padding=1)
        self.output_c4 = nn.Conv2d(mask_dim, mask_dim, 3, padding=1)
        self.output_c3 = nn.Conv2d(mask_dim, mask_dim, 3, padding=1)
        
        # Final upsample to H/4 resolution
        self.final_conv = nn.Sequential(
            nn.Conv2d(mask_dim, mask_dim, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(mask_dim, mask_dim, 3, padding=1),
        )
        
        self.mask_dim = mask_dim
    
    def forward(
        self,
        c3: torch.Tensor,  # [B, H/8, W/8, D]
        c4: torch.Tensor,  # [B, H/16, W/16, D]
        c5: torch.Tensor,  # [B, H/32, W/32, D]
    ) -> torch.Tensor:
        """
        Forward pass to generate pixel features for mask prediction.
        
        Returns:
            mask_features: [B, mask_dim, H/4, W/4]
        """
        # Convert from [B, H, W, D] to [B, D, H, W]
        c3 = c3.permute(0, 3, 1, 2)
        c4 = c4.permute(0, 3, 1, 2)
        c5 = c5.permute(0, 3, 1, 2)
        
        # Build FPN top-down
        p5 = self.lateral_c5(c5)  # [B, mask_dim, H/32, W/32]
        p5 = self.output_c5(p5)
        
        # Upsample P5 and add to C4
        p5_up = F.interpolate(p5, size=c4.shape[2:], mode='bilinear', align_corners=False)
        p4 = self.lateral_c4(c4) + p5_up
        p4 = self.output_c4(p4)  # [B, mask_dim, H/16, W/16]
        
        # Upsample P4 and add to C3
        p4_up = F.interpolate(p4, size=c3.shape[2:], mode='bilinear', align_corners=False)
        p3 = self.lateral_c3(c3) + p4_up
        p3 = self.output_c3(p3)  # [B, mask_dim, H/8, W/8]
        
        # Upsample to H/4 resolution
        mask_features = F.interpolate(p3, scale_factor=2, mode='bilinear', align_corners=False)
        mask_features = self.final_conv(mask_features)  # [B, mask_dim, H/4, W/4]
        
        return mask_features


class MaskHead(nn.Module):
    """
    Mask prediction head using dot product between queries and pixel features.
    """
    
    def __init__(self, hidden_dim: int, mask_dim: int, num_classes: int = 3):
        super().__init__()
        
        # Project query features to mask dimension
        self.query_proj = nn.Linear(hidden_dim, mask_dim)
        
        # Classification head for segmentation classes
        self.class_head = nn.Linear(hidden_dim, num_classes)
        
        self.num_classes = num_classes
    
    def forward(
        self,
        query_features: torch.Tensor,  # [B, num_queries, hidden_dim]
        mask_features: torch.Tensor,   # [B, mask_dim, H/4, W/4]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate per-query masks and class predictions.
        
        Returns:
            - masks: [B, num_queries, H/4, W/4]
            - class_logits: [B, num_queries, num_classes]
        """
        B, N_q, D = query_features.shape
        _, C, H, W = mask_features.shape
        
        # Project queries to mask dimension
        query_proj = self.query_proj(query_features)  # [B, N_q, mask_dim]
        
        # Compute masks via dot product
        # [B, N_q, mask_dim] @ [B, mask_dim, H*W] -> [B, N_q, H*W]
        mask_features_flat = mask_features.view(B, C, -1)  # [B, mask_dim, H*W]
        masks = torch.bmm(query_proj, mask_features_flat)  # [B, N_q, H*W]
        masks = masks.view(B, N_q, H, W)  # [B, N_q, H, W]
        
        # Class predictions
        class_logits = self.class_head(query_features)  # [B, N_q, num_classes]
        
        return masks, class_logits


class SegmentationDecoder(nn.Module):
    """
    Segmentation Decoder for driveway/footpath segmentation.
    
    50 queries → 6 decoder layers (with cross-attention to detection) → Mask head
    """
    
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        
        # Base decoder WITH inter-decoder attention (uses detection features)
        self.decoder = BaseDecoder(
            hidden_dim=config.hidden_dim,
            num_heads=config.num_heads,
            feedforward_dim=config.dim_feedforward,
            num_layers=config.num_decoder_layers,
            dropout=config.dropout,
            has_inter_decoder_attn=True,  # Uses detection features
            num_queries=config.num_segmentation_queries
        )
        
        # FPN pixel decoder
        self.pixel_decoder = FPNPixelDecoder(
            hidden_dim=config.hidden_dim,
            mask_dim=config.hidden_dim
        )
        
        # Mask head
        self.mask_head = MaskHead(
            hidden_dim=config.hidden_dim,
            mask_dim=config.hidden_dim,
            num_classes=config.num_seg_classes
        )
    
    def forward(
        self,
        memory: torch.Tensor,
        detection_features: torch.Tensor,
        backbone_features: Dict[str, torch.Tensor],
        memory_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Forward pass of segmentation decoder.
        
        Args:
            memory: Encoder memory [B, N, D]
            detection_features: Features from detection decoder [B, 100, D]
            backbone_features: Dictionary with 'c3', 'c4', 'c5' from backbone
            memory_mask: Optional mask [B, N]
            
        Returns:
            - seg_features: [B, 50, D] for use by tracking decoder
            - outputs: Dictionary with masks and class predictions
        """
        # Decode with inter-decoder attention to detection features
        seg_features = self.decoder(
            memory=memory,
            decoder_context=detection_features,
            memory_mask=memory_mask
        )
        
        # Generate pixel features from backbone
        mask_features = self.pixel_decoder(
            c3=backbone_features['c3'],
            c4=backbone_features['c4'],
            c5=backbone_features['c5']
        )
        
        # Generate masks
        masks, class_logits = self.mask_head(seg_features, mask_features)
        
        outputs = {
            'masks': masks,  # [B, 50, H/4, W/4]
            'class_logits': class_logits,  # [B, 50, 3]
        }
        
        return seg_features, outputs
    
    def get_semantic_masks(
        self,
        masks: torch.Tensor,
        class_logits: torch.Tensor,
        image_size: Tuple[int, int]
    ) -> Dict[str, torch.Tensor]:
        """
        Convert per-query masks to semantic segmentation.
        
        Args:
            masks: [B, num_queries, H/4, W/4]
            class_logits: [B, num_queries, num_classes]
            image_size: (H, W) for upsampling
            
        Returns:
            Dictionary with 'driveway', 'footpath', 'combined' masks
        """
        B = masks.shape[0]
        H, W = image_size
        
        # Upsample masks to full resolution
        masks_up = F.interpolate(masks, size=(H, W), mode='bilinear', align_corners=False)
        masks_up = masks_up.sigmoid()  # [B, N_q, H, W]
        
        # Get class probabilities
        class_probs = F.softmax(class_logits, dim=-1)  # [B, N_q, 3]
        
        # Weighted sum of masks per class
        # driveway (class 1), footpath (class 2)
        driveway_mask = (masks_up * class_probs[:, :, 1:2].unsqueeze(-1)).sum(dim=1)
        footpath_mask = (masks_up * class_probs[:, :, 2:3].unsqueeze(-1)).sum(dim=1)
        
        # Combined semantic segmentation
        combined = torch.zeros(B, self.config.num_seg_classes, H, W, device=masks.device)
        combined[:, 0] = 1.0  # Background
        combined[:, 1] = driveway_mask.squeeze(1)
        combined[:, 2] = footpath_mask.squeeze(1)
        combined = F.softmax(combined, dim=1)
        
        return {
            'driveway': driveway_mask,
            'footpath': footpath_mask,
            'combined': combined,
            'semantic': combined.argmax(dim=1)  # [B, H, W]
        }


if __name__ == "__main__":
    from config import ModelConfig
    
    config = ModelConfig()
    decoder = SegmentationDecoder(config)
    
    # Dummy inputs
    memory = torch.randn(2, 5376, config.hidden_dim)
    det_features = torch.randn(2, 100, config.hidden_dim)
    backbone_features = {
        'c3': torch.randn(2, 64, 64, config.hidden_dim),
        'c4': torch.randn(2, 32, 32, config.hidden_dim),
        'c5': torch.randn(2, 16, 16, config.hidden_dim),
    }
    
    # Forward pass
    seg_features, outputs = decoder(memory, det_features, backbone_features)
    
    print(f"📊 Segmentation Decoder Output:")
    print(f"   Features: {seg_features.shape}")
    print(f"   Masks: {outputs['masks'].shape}")
    print(f"   Class logits: {outputs['class_logits'].shape}")
    
    # Test semantic mask generation
    semantic = decoder.get_semantic_masks(
        outputs['masks'], 
        outputs['class_logits'],
        image_size=(512, 512)
    )
    print(f"   Semantic: {semantic['semantic'].shape}")
    print(f"   Parameters: {sum(p.numel() for p in decoder.parameters()):,}")
