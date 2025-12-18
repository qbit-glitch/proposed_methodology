"""
ResNet50 Backbone with Multi-Scale Feature Extraction.

Implements Algorithm 1: CNN-BACKBONE (Feature Extraction)
Extracts features at scales C3 (H/8), C4 (H/16), C5 (H/32).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50, ResNet50_Weights
from typing import Dict, Tuple, List

from .config import ModelConfig


class FeatureProjection(nn.Module):
    """Project features from different scales to common dimension."""
    
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.norm = nn.BatchNorm2d(out_channels)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(self.conv(x))


class ResNet50Backbone(nn.Module):
    """
    ResNet50 backbone for multi-scale feature extraction.
    
    Outputs feature maps at three scales:
    - C3: H/8 x W/8 x 512
    - C4: H/16 x W/16 x 1024  
    - C5: H/32 x W/32 x 2048
    
    All features are projected to common dimension D=hidden_dim.
    """
    
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        
        # Load pretrained ResNet50
        if config.pretrained:
            weights = ResNet50_Weights.IMAGENET1K_V2
            resnet = resnet50(weights=weights)
            print("✓ Loaded pretrained ResNet50 weights (ImageNet1K V2)")
        else:
            resnet = resnet50(weights=None)
            print("✓ Initialized ResNet50 from scratch")
        
        # Extract layers up to each scale
        self.layer0 = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool
        )
        self.layer1 = resnet.layer1  # 256 channels, H/4
        self.layer2 = resnet.layer2  # 512 channels, H/8 (C3)
        self.layer3 = resnet.layer3  # 1024 channels, H/16 (C4)
        self.layer4 = resnet.layer4  # 2048 channels, H/32 (C5)
        
        # Channel dimensions at each scale
        self.c3_channels = 512
        self.c4_channels = 1024
        self.c5_channels = 2048
        
        # Project to common dimension D
        hidden_dim = config.hidden_dim
        self.proj_c3 = FeatureProjection(self.c3_channels, hidden_dim)
        self.proj_c4 = FeatureProjection(self.c4_channels, hidden_dim)
        self.proj_c5 = FeatureProjection(self.c5_channels, hidden_dim)
        
        # Optionally freeze backbone
        if config.freeze_backbone:
            self._freeze_backbone()
    
    def _freeze_backbone(self):
        """Freeze backbone parameters for transfer learning."""
        for param in self.layer0.parameters():
            param.requires_grad = False
        for param in self.layer1.parameters():
            param.requires_grad = False
        for param in self.layer2.parameters():
            param.requires_grad = False
        for param in self.layer3.parameters():
            param.requires_grad = False
        for param in self.layer4.parameters():
            param.requires_grad = False
        print("✓ Backbone parameters frozen")
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass to extract multi-scale features.
        
        Args:
            x: Input image tensor [B, 3, H, W]
            
        Returns:
            Dictionary with projected features:
            - 'c3': [B, H/8, W/8, D]
            - 'c4': [B, H/16, W/16, D]
            - 'c5': [B, H/32, W/32, D]
            - 'c3_flat': [B, N3, D] where N3 = (H/8) * (W/8)
            - 'c4_flat': [B, N4, D]
            - 'c5_flat': [B, N5, D]
        """
        # Extract features at each layer
        x = self.layer0(x)
        x = self.layer1(x)
        c3 = self.layer2(x)   # [B, 512, H/8, W/8]
        c4 = self.layer3(c3)  # [B, 1024, H/16, W/16]
        c5 = self.layer4(c4)  # [B, 2048, H/32, W/32]
        
        # Project to common dimension
        c3_proj = self.proj_c3(c3)  # [B, D, H/8, W/8]
        c4_proj = self.proj_c4(c4)  # [B, D, H/16, W/16]
        c5_proj = self.proj_c5(c5)  # [B, D, H/32, W/32]
        
        # Flatten spatial dimensions for transformer
        B = x.shape[0]
        
        # Reshape from [B, D, H, W] to [B, H*W, D]
        c3_flat = c3_proj.flatten(2).transpose(1, 2)  # [B, N3, D]
        c4_flat = c4_proj.flatten(2).transpose(1, 2)  # [B, N4, D]
        c5_flat = c5_proj.flatten(2).transpose(1, 2)  # [B, N5, D]
        
        return {
            # Spatial format (for mask generation)
            'c3': c3_proj.permute(0, 2, 3, 1),  # [B, H/8, W/8, D]
            'c4': c4_proj.permute(0, 2, 3, 1),  # [B, H/16, W/16, D]
            'c5': c5_proj.permute(0, 2, 3, 1),  # [B, H/32, W/32, D]
            # Flattened format (for transformer)
            'c3_flat': c3_flat,
            'c4_flat': c4_flat,
            'c5_flat': c5_flat,
            # Original spatial shapes (for reconstruction)
            'c3_shape': c3_proj.shape[2:],  # (H/8, W/8)
            'c4_shape': c4_proj.shape[2:],  # (H/16, W/16)
            'c5_shape': c5_proj.shape[2:],  # (H/32, W/32)
        }


class FeaturePyramidFlattener(nn.Module):
    """
    Flatten and concatenate multi-scale features for transformer input.
    
    Concatenates C3, C4, C5 features into single sequence:
    F_flat = [C3_flat; C4_flat; C5_flat] ∈ R^{N x D}
    where N = N3 + N4 + N5
    """
    
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
    
    def forward(
        self, 
        features: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, List[int]]:
        """
        Concatenate multi-scale flattened features.
        
        Args:
            features: Dictionary from ResNet50Backbone containing 'c3_flat', 'c4_flat', 'c5_flat'
            
        Returns:
            - concatenated: [B, N, D] where N = N3 + N4 + N5
            - scale_lengths: [N3, N4, N5] for later reconstruction
        """
        c3_flat = features['c3_flat']  # [B, N3, D]
        c4_flat = features['c4_flat']  # [B, N4, D]
        c5_flat = features['c5_flat']  # [B, N5, D]
        
        # Record lengths for later use
        scale_lengths = [c3_flat.shape[1], c4_flat.shape[1], c5_flat.shape[1]]
        
        # Concatenate along sequence dimension
        concatenated = torch.cat([c3_flat, c4_flat, c5_flat], dim=1)  # [B, N, D]
        
        return concatenated, scale_lengths


if __name__ == "__main__":
    # Test backbone
    from .config import ModelConfig
    
    config = ModelConfig()
    backbone = ResNet50Backbone(config)
    flattener = FeaturePyramidFlattener(config)
    
    # Create dummy input
    x = torch.randn(2, 3, 512, 512)
    
    # Forward pass
    features = backbone(x)
    print(f"\n📐 Feature Shapes:")
    print(f"   C3: {features['c3'].shape}")
    print(f"   C4: {features['c4'].shape}")
    print(f"   C5: {features['c5'].shape}")
    print(f"   C3_flat: {features['c3_flat'].shape}")
    print(f"   C4_flat: {features['c4_flat'].shape}")
    print(f"   C5_flat: {features['c5_flat'].shape}")
    
    # Concatenate
    concat_features, scale_lengths = flattener(features)
    print(f"\n📊 Concatenated Features:")
    print(f"   Shape: {concat_features.shape}")
    print(f"   Scale lengths: {scale_lengths}")
    print(f"   Total tokens: {sum(scale_lengths)}")
