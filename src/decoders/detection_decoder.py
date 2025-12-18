"""
Detection Decoder - Algorithm 3.

Implements detection of vehicles with:
- 100 learnable detection queries
- 6 decoder layers with self-attention and cross-attention
- Output heads: BBox regression, Classification, Color, Type
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


class DetectionHead(nn.Module):
    """Output heads for detection decoder."""
    
    def __init__(self, hidden_dim: int, config: ModelConfig):
        super().__init__()
        
        # BBox regression head (predicts x, y, w, h in [0, 1])
        self.bbox_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 4),
            nn.Sigmoid()  # Normalize to [0, 1]
        )
        
        # Classification head (6 vehicle classes + background)
        self.class_head = nn.Linear(hidden_dim, config.num_vehicle_classes + 1)
        
        # Color head
        self.color_head = nn.Linear(hidden_dim, config.num_colors)
        
        # Type head
        self.type_head = nn.Linear(hidden_dim, config.num_types)
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass through detection heads.
        
        Args:
            x: Decoder output [B, num_queries, D]
            
        Returns:
            Dictionary with:
            - 'bbox': [B, num_queries, 4] (x_center, y_center, width, height)
            - 'class_logits': [B, num_queries, num_classes+1]
            - 'color_logits': [B, num_queries, num_colors]
            - 'type_logits': [B, num_queries, num_types]
        """
        return {
            'bbox': self.bbox_head(x),
            'class_logits': self.class_head(x),
            'color_logits': self.color_head(x),
            'type_logits': self.type_head(x),
        }


class DetectionDecoder(nn.Module):
    """
    Detection Decoder for vehicle detection.
    
    100 queries → 6 decoder layers → Detection heads
    """
    
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        
        # Base decoder (no inter-decoder attention for detection)
        self.decoder = BaseDecoder(
            hidden_dim=config.hidden_dim,
            num_heads=config.num_heads,
            feedforward_dim=config.dim_feedforward,
            num_layers=config.num_decoder_layers,
            dropout=config.dropout,
            has_inter_decoder_attn=False,  # Detection is independent
            num_queries=config.num_detection_queries
        )
        
        # Detection output heads
        self.head = DetectionHead(config.hidden_dim, config)
    
    def forward(
        self,
        memory: torch.Tensor,
        memory_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Forward pass of detection decoder.
        
        Args:
            memory: Encoder memory [B, N, D]
            memory_mask: Optional mask [B, N]
            
        Returns:
            - decoder_features: [B, 100, D] for use by other decoders
            - outputs: Dictionary with bbox, class, color, type predictions
        """
        # Decode
        decoder_features = self.decoder(memory, memory_mask=memory_mask)
        
        # Apply heads
        outputs = self.head(decoder_features)
        
        return decoder_features, outputs


if __name__ == "__main__":
    from config import ModelConfig
    
    config = ModelConfig()
    decoder = DetectionDecoder(config)
    
    # Dummy input
    memory = torch.randn(2, 5376, config.hidden_dim)
    
    # Forward pass
    features, outputs = decoder(memory)
    
    print(f"📊 Detection Decoder Output:")
    print(f"   Features: {features.shape}")
    print(f"   BBox: {outputs['bbox'].shape}")
    print(f"   Class logits: {outputs['class_logits'].shape}")
    print(f"   Color logits: {outputs['color_logits'].shape}")
    print(f"   Type logits: {outputs['type_logits'].shape}")
    print(f"   Parameters: {sum(p.numel() for p in decoder.parameters()):,}")
