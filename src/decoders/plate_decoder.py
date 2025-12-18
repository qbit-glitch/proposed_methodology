"""
Plate Detection Decoder - Algorithm 5.

Implements license plate detection with:
- 50 plate detection queries
- Cross-attention to detection features (constrained search near vehicles)
- Plate bounding box regression
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional

from .base_decoder import BaseDecoder

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import ModelConfig


class PlateHead(nn.Module):
    """Output head for plate detection."""
    
    def __init__(self, hidden_dim: int):
        super().__init__()
        
        # Plate BBox regression (x, y, w, h in [0, 1])
        self.bbox_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 4),
            nn.Sigmoid()
        )
        
        # Confidence score (is this a valid plate?)
        self.conf_head = nn.Linear(hidden_dim, 1)
        
        # Plate type classification (for different plate formats)
        self.type_head = nn.Linear(hidden_dim, 5)  # Standard, Commercial, etc.
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Args:
            x: Decoder output [B, num_queries, D]
            
        Returns:
            - 'bbox': [B, num_queries, 4]
            - 'confidence': [B, num_queries, 1]
            - 'type_logits': [B, num_queries, 5]
        """
        return {
            'plate_bbox': self.bbox_head(x),
            'plate_confidence': self.conf_head(x).sigmoid(),
            'plate_type_logits': self.type_head(x),
        }


class PlateDecoder(nn.Module):
    """
    Plate Detection Decoder.
    
    Uses cross-attention to detection features to constrain plate search
    to regions near detected vehicles.
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
            has_inter_decoder_attn=True,  # Cross-attend to detection
            num_queries=config.num_plate_queries
        )
        
        # Plate output head
        self.head = PlateHead(config.hidden_dim)
    
    def forward(
        self,
        memory: torch.Tensor,
        detection_features: torch.Tensor,
        memory_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Forward pass of plate decoder.
        
        Args:
            memory: Encoder memory [B, N, D]
            detection_features: Features from detection decoder [B, 100, D]
            memory_mask: Optional mask [B, N]
            
        Returns:
            - plate_features: [B, 50, D] for use by OCR decoder
            - outputs: Dictionary with plate bbox and confidence
        """
        # Decode with cross-attention to detection features
        plate_features = self.decoder(
            memory=memory,
            decoder_context=detection_features,
            memory_mask=memory_mask
        )
        
        # Apply output head
        outputs = self.head(plate_features)
        
        return plate_features, outputs


if __name__ == "__main__":
    from config import ModelConfig
    
    config = ModelConfig()
    decoder = PlateDecoder(config)
    
    # Dummy inputs
    memory = torch.randn(2, 5376, config.hidden_dim)
    det_features = torch.randn(2, 100, config.hidden_dim)
    
    # Forward pass
    plate_features, outputs = decoder(memory, det_features)
    
    print(f"📊 Plate Decoder Output:")
    print(f"   Features: {plate_features.shape}")
    print(f"   Plate BBox: {outputs['plate_bbox'].shape}")
    print(f"   Confidence: {outputs['plate_confidence'].shape}")
    print(f"   Type logits: {outputs['plate_type_logits'].shape}")
    print(f"   Parameters: {sum(p.numel() for p in decoder.parameters()):,}")
