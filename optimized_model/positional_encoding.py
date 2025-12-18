"""
Sinusoidal Positional Encoding for Transformer.

Implements 2D positional encoding for spatial feature maps.
PE(pos, 2i) = sin(pos / 10000^(2i/D))
PE(pos, 2i+1) = cos(pos / 10000^(2i/D))
"""

import torch
import torch.nn as nn
import math
from typing import Tuple


class PositionalEncoding2D(nn.Module):
    """
    2D Sinusoidal Positional Encoding.
    
    Generates positional encodings for 2D spatial positions,
    suitable for image features from CNN backbone.
    """
    
    def __init__(self, hidden_dim: int, max_h: int = 128, max_w: int = 128):
        """
        Args:
            hidden_dim: Dimension of the model (D)
            max_h: Maximum height for positional encoding
            max_w: Maximum width for positional encoding
        """
        super().__init__()
        self.hidden_dim = hidden_dim
        
        # Generate position encoding table
        pe = self._generate_2d_encoding(max_h, max_w, hidden_dim)
        self.register_buffer('pe', pe)
    
    def _generate_2d_encoding(
        self, 
        max_h: int, 
        max_w: int, 
        hidden_dim: int
    ) -> torch.Tensor:
        """Generate 2D positional encoding table."""
        # Use half dimensions for x and half for y
        d_half = hidden_dim // 2
        
        # Create position indices
        y_pos = torch.arange(max_h).unsqueeze(1).float()
        x_pos = torch.arange(max_w).unsqueeze(1).float()
        
        # Create dimension indices
        div_term = torch.exp(
            torch.arange(0, d_half, 2).float() * 
            (-math.log(10000.0) / d_half)
        )
        
        # Compute encodings for y dimension
        pe_y = torch.zeros(max_h, d_half)
        pe_y[:, 0::2] = torch.sin(y_pos * div_term)
        pe_y[:, 1::2] = torch.cos(y_pos * div_term)
        
        # Compute encodings for x dimension
        pe_x = torch.zeros(max_w, d_half)
        pe_x[:, 0::2] = torch.sin(x_pos * div_term)
        pe_x[:, 1::2] = torch.cos(x_pos * div_term)
        
        # Combine: [max_h, max_w, hidden_dim]
        pe_y = pe_y.unsqueeze(1).repeat(1, max_w, 1)  # [H, W, D/2]
        pe_x = pe_x.unsqueeze(0).repeat(max_h, 1, 1)  # [H, W, D/2]
        pe = torch.cat([pe_y, pe_x], dim=-1)  # [H, W, D]
        
        return pe
    
    def forward(self, x: torch.Tensor, h: int, w: int) -> torch.Tensor:
        """
        Add positional encoding to input features.
        
        Args:
            x: Input features [B, N, D] where N = H * W
            h: Height of feature map
            w: Width of feature map
            
        Returns:
            Features with positional encoding added [B, N, D]
        """
        # Get positional encoding for this size
        pe = self.pe[:h, :w, :].reshape(-1, self.hidden_dim)  # [H*W, D]
        
        # Add to input (broadcasting over batch)
        return x + pe.unsqueeze(0)


class MultiScalePositionalEncoding(nn.Module):
    """
    Positional encoding for multi-scale concatenated features.
    
    Handles features from C3 (H/8), C4 (H/16), C5 (H/32) scales
    concatenated into single sequence.
    """
    
    def __init__(self, hidden_dim: int, max_size: int = 128):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.pe_2d = PositionalEncoding2D(hidden_dim, max_size, max_size)
        
        # Learnable scale embeddings to distinguish C3, C4, C5
        self.scale_embed = nn.Embedding(3, hidden_dim)
    
    def forward(
        self,
        features: torch.Tensor,
        scale_shapes: list,  # [(h3, w3), (h4, w4), (h5, w5)]
        scale_lengths: list  # [N3, N4, N5]
    ) -> torch.Tensor:
        """
        Add positional encoding to concatenated multi-scale features.
        
        Args:
            features: Concatenated features [B, N, D] where N = N3 + N4 + N5
            scale_shapes: List of (H, W) tuples for each scale
            scale_lengths: List of lengths [N3, N4, N5]
            
        Returns:
            Features with positional encoding [B, N, D]
        """
        B, N, D = features.shape
        device = features.device
        
        # Initialize output
        output = features.clone()
        
        # Add positional encoding for each scale
        start_idx = 0
        for scale_idx, (shape, length) in enumerate(zip(scale_shapes, scale_lengths)):
            h, w = shape
            end_idx = start_idx + length
            
            # Get 2D positional encoding
            pe = self.pe_2d.pe[:h, :w, :].reshape(-1, D).to(device)  # [length, D]
            
            # Add scale embedding
            scale_emb = self.scale_embed.weight[scale_idx]  # [D]
            
            # Add both encodings
            output[:, start_idx:end_idx, :] += pe.unsqueeze(0) + scale_emb.unsqueeze(0).unsqueeze(0)
            
            start_idx = end_idx
        
        return output


class LearnablePositionalEncoding(nn.Module):
    """
    Learnable positional encoding as alternative to sinusoidal.
    """
    
    def __init__(self, max_len: int, hidden_dim: int):
        super().__init__()
        self.pe = nn.Parameter(torch.randn(1, max_len, hidden_dim) * 0.02)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add learnable positional encoding."""
        return x + self.pe[:, :x.size(1), :]


if __name__ == "__main__":
    # Test positional encoding
    hidden_dim = 256
    
    # Test 2D encoding
    pe_2d = PositionalEncoding2D(hidden_dim)
    x = torch.randn(2, 64 * 64, hidden_dim)
    y = pe_2d(x, h=64, w=64)
    print(f"2D PE Input: {x.shape} -> Output: {y.shape}")
    
    # Test multi-scale encoding
    ms_pe = MultiScalePositionalEncoding(hidden_dim)
    concat_feat = torch.randn(2, 64*64 + 32*32 + 16*16, hidden_dim)
    scale_shapes = [(64, 64), (32, 32), (16, 16)]
    scale_lengths = [64*64, 32*32, 16*16]
    
    y = ms_pe(concat_feat, scale_shapes, scale_lengths)
    print(f"Multi-scale PE Input: {concat_feat.shape} -> Output: {y.shape}")
