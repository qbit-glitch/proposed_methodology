"""
Vision Transformer (ViT) Encoder with Pretrained Weights.

Uses pretrained ViT from torchvision to extract global/temporal features.
Complements ResNet50 backbone's local spatial features by providing
global context through self-attention across image patches.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import vit_b_16, ViT_B_16_Weights
from typing import Dict, Optional, Tuple

from .config import ModelConfig


class ViTEncoder(nn.Module):
    """
    Pretrained Vision Transformer for global feature extraction.
    
    ViT processes the image as patches and applies self-attention,
    capturing long-range dependencies that CNNs miss.
    
    Output:
        - Global feature: [B, D] - CLS token representation
        - Patch features: [B, N_patches, D] - All patch embeddings
    """
    
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.hidden_dim = config.hidden_dim
        
        # Load pretrained ViT
        use_pretrained = getattr(config, 'use_pretrained_vit', True)
        if use_pretrained:
            weights = ViT_B_16_Weights.IMAGENET1K_V1
            self.vit = vit_b_16(weights=weights)
            print("✓ Loaded pretrained ViT-B/16 weights (ImageNet1K V1)")
        else:
            self.vit = vit_b_16(weights=None)
            print("✓ Initialized ViT-B/16 from scratch")
        
        # ViT-B/16 has hidden_dim=768, project to our hidden_dim
        self.vit_hidden_dim = 768
        self.proj = nn.Linear(self.vit_hidden_dim, self.hidden_dim)
        self.norm = nn.LayerNorm(self.hidden_dim)
        
        # Remove the classification head (we don't need it)
        self.vit.heads = nn.Identity()
        
        # Optionally freeze ViT
        freeze_vit = getattr(config, 'freeze_vit', False)
        if freeze_vit:
            self._freeze_vit()
    
    def _freeze_vit(self):
        """Freeze ViT parameters for transfer learning."""
        for param in self.vit.parameters():
            param.requires_grad = False
        print("✓ ViT parameters frozen")
    
    def forward(
        self, 
        x: torch.Tensor,
        return_patch_features: bool = True
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through ViT.
        
        Args:
            x: Input images [B, 3, H, W]
            return_patch_features: Whether to return all patch embeddings
            
        Returns:
            Dictionary containing:
            - 'global': [B, D] - Global CLS token feature
            - 'patches': [B, N_patches, D] - Patch features (if requested)
        """
        B = x.shape[0]
        
        # ViT expects 224x224 input, resize if needed
        if x.shape[2] != 224 or x.shape[3] != 224:
            x = F.interpolate(x, size=(224, 224), mode='bilinear', align_corners=False)
        
        # Get patch embeddings from ViT encoder
        # ViT forward: conv_proj -> add class token -> pos embed -> encoder blocks
        x = self.vit._process_input(x)  # [B, N_patches, 768]
        n = x.shape[0]
        
        # Add class token
        batch_class_token = self.vit.class_token.expand(n, -1, -1)
        x = torch.cat([batch_class_token, x], dim=1)  # [B, 1 + N_patches, 768]
        
        # Add positional embedding
        x = x + self.vit.encoder.pos_embedding
        
        # Pass through transformer encoder blocks
        x = self.vit.encoder.ln(self.vit.encoder.layers(self.vit.encoder.dropout(x)))
        
        # Extract CLS token (global feature) and patch features
        cls_token = x[:, 0]  # [B, 768]
        patch_features = x[:, 1:]  # [B, N_patches, 768]
        
        # Project to our hidden dimension
        global_feat = self.norm(self.proj(cls_token))  # [B, D]
        
        result = {'global': global_feat}
        
        if return_patch_features:
            # Project patch features
            patch_feat = self.norm(self.proj(patch_features))  # [B, N_patches, D]
            result['patches'] = patch_feat
        
        return result


class ViTFeatureFusion(nn.Module):
    """
    Fuse ViT global features with ResNet50 multi-scale features.
    
    Combines local (CNN) and global (ViT) representations for
    richer feature representation.
    """
    
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.hidden_dim = config.hidden_dim
        
        # Learnable fusion weights
        self.local_weight = nn.Parameter(torch.ones(1))
        self.global_weight = nn.Parameter(torch.ones(1))
        
        # Cross-attention: local features attend to global
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=config.hidden_dim,
            num_heads=config.num_heads,
            dropout=config.dropout,
            batch_first=True
        )
        self.norm = nn.LayerNorm(config.hidden_dim)
    
    def forward(
        self,
        local_features: torch.Tensor,  # [B, N, D] from ResNet50+Encoder
        global_features: torch.Tensor,  # [B, D] from ViT CLS token
    ) -> torch.Tensor:
        """
        Fuse local and global features.
        
        Args:
            local_features: Multi-scale features from CNN backbone [B, N, D]
            global_features: Global context from ViT [B, D]
            
        Returns:
            Fused features [B, N, D]
        """
        B, N, D = local_features.shape
        
        # Expand global features for cross-attention
        global_expanded = global_features.unsqueeze(1)  # [B, 1, D]
        
        # Cross-attention: local queries, global key/value
        attn_out, _ = self.cross_attn(
            query=local_features,
            key=global_expanded,
            value=global_expanded
        )
        
        # Normalize fusion weights
        weights = F.softmax(torch.stack([self.local_weight, self.global_weight]), dim=0)
        
        # Weighted combination
        fused = weights[0] * local_features + weights[1] * attn_out
        fused = self.norm(fused)
        
        return fused


if __name__ == "__main__":
    from .config import ModelConfig
    
    # Test ViT encoder
    config = ModelConfig()
    vit_encoder = ViTEncoder(config)
    
    # Test forward pass
    x = torch.randn(2, 3, 512, 512)
    outputs = vit_encoder(x)
    
    print(f"\n📐 ViT Encoder Output Shapes:")
    print(f"   Global feature: {outputs['global'].shape}")
    print(f"   Patch features: {outputs['patches'].shape}")
    
    # Test fusion
    fusion = ViTFeatureFusion(config)
    local_feat = torch.randn(2, 1000, config.hidden_dim)
    fused = fusion(local_feat, outputs['global'])
    print(f"   Fused features: {fused.shape}")
