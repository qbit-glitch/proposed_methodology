"""
Transformer Encoder with Multi-Head Self-Attention.

Implements Algorithm 2: TRANSFORMER-ENCODER
6-layer transformer encoder with:
- Multi-Head Self-Attention (8 heads)
- Feed-Forward Network (D -> 4D -> D)
- Pre-LayerNorm architecture
- Residual connections
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

from .config import ModelConfig


class MultiHeadSelfAttention(nn.Module):
    """
    Multi-Head Self-Attention Layer.
    
    Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) V
    """
    
    def __init__(
        self, 
        hidden_dim: int, 
        num_heads: int, 
        dropout: float = 0.1
    ):
        super().__init__()
        assert hidden_dim % num_heads == 0, "hidden_dim must be divisible by num_heads"
        
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        # Linear projections for Q, K, V
        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)
        
        # Output projection
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(
        self, 
        x: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
        attn_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass of self-attention.
        
        Args:
            x: Input tensor [B, N, D]
            key_padding_mask: Mask for padding [B, N]
            attn_mask: Attention mask [N, N]
            
        Returns:
            Output tensor [B, N, D]
        """
        B, N, D = x.shape
        
        # Compute Q, K, V
        Q = self.q_proj(x)  # [B, N, D]
        K = self.k_proj(x)  # [B, N, D]
        V = self.v_proj(x)  # [B, N, D]
        
        # Reshape for multi-head attention
        # [B, N, D] -> [B, N, h, d_k] -> [B, h, N, d_k]
        Q = Q.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Compute attention scores
        # [B, h, N, d_k] @ [B, h, d_k, N] -> [B, h, N, N]
        attn_scores = (Q @ K.transpose(-2, -1)) * self.scale
        
        # Apply masks if provided
        if attn_mask is not None:
            attn_scores = attn_scores + attn_mask
        
        if key_padding_mask is not None:
            # [B, N] -> [B, 1, 1, N]
            key_padding_mask = key_padding_mask.unsqueeze(1).unsqueeze(2)
            attn_scores = attn_scores.masked_fill(key_padding_mask, float('-inf'))
        
        # Softmax and dropout
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention to values
        # [B, h, N, N] @ [B, h, N, d_k] -> [B, h, N, d_k]
        out = attn_weights @ V
        
        # Reshape back
        # [B, h, N, d_k] -> [B, N, h, d_k] -> [B, N, D]
        out = out.transpose(1, 2).contiguous().view(B, N, D)
        
        # Output projection
        out = self.out_proj(out)
        
        return out


class FeedForward(nn.Module):
    """
    Feed-Forward Network.
    
    FFN(x) = W2 * ReLU(W1 * x + b1) + b2
    Expands from D to 4D then back to D.
    """
    
    def __init__(
        self, 
        hidden_dim: int, 
        feedforward_dim: int, 
        dropout: float = 0.1
    ):
        super().__init__()
        self.linear1 = nn.Linear(hidden_dim, feedforward_dim)
        self.linear2 = nn.Linear(feedforward_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.ReLU()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through FFN."""
        x = self.linear1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.linear2(x)
        x = self.dropout(x)
        return x


class TransformerEncoderLayer(nn.Module):
    """
    Single Transformer Encoder Layer.
    
    Pre-norm architecture:
    1. LayerNorm -> Self-Attention -> Residual
    2. LayerNorm -> FFN -> Residual
    """
    
    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        feedforward_dim: int,
        dropout: float = 0.1
    ):
        super().__init__()
        
        # Self-attention
        self.self_attn = MultiHeadSelfAttention(hidden_dim, num_heads, dropout)
        
        # Feed-forward
        self.ffn = FeedForward(hidden_dim, feedforward_dim, dropout)
        
        # Layer norms (pre-norm)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        
        # Dropout for residual
        self.dropout = nn.Dropout(dropout)
    
    def forward(
        self, 
        x: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass of encoder layer.
        
        Args:
            x: Input tensor [B, N, D]
            key_padding_mask: Padding mask [B, N]
            
        Returns:
            Output tensor [B, N, D]
        """
        # Self-attention with residual
        x_norm = self.norm1(x)
        attn_out = self.self_attn(x_norm, key_padding_mask)
        x = x + self.dropout(attn_out)
        
        # FFN with residual
        x_norm = self.norm2(x)
        ffn_out = self.ffn(x_norm)
        x = x + self.dropout(ffn_out)
        
        return x


class TransformerEncoder(nn.Module):
    """
    Full Transformer Encoder (6 layers).
    
    Output: Memory M ∈ R^{N x D} used by all decoders.
    """
    
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        
        # Build encoder layers
        self.layers = nn.ModuleList([
            TransformerEncoderLayer(
                hidden_dim=config.hidden_dim,
                num_heads=config.num_heads,
                feedforward_dim=config.dim_feedforward,
                dropout=config.dropout
            )
            for _ in range(config.num_encoder_layers)
        ])
        
        # Final layer norm
        self.final_norm = nn.LayerNorm(config.hidden_dim)
    
    def forward(
        self, 
        x: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass through all encoder layers.
        
        Args:
            x: Input features with positional encoding [B, N, D]
            key_padding_mask: Padding mask [B, N]
            
        Returns:
            Memory tensor [B, N, D]
        """
        for layer in self.layers:
            x = layer(x, key_padding_mask)
        
        # Final normalization
        x = self.final_norm(x)
        
        return x


if __name__ == "__main__":
    # Test encoder
    config = ModelConfig()
    encoder = TransformerEncoder(config)
    
    # Create dummy input (multi-scale concatenated features)
    # For 512x512 image: N = 64*64 + 32*32 + 16*16 = 5376
    N = 64*64 + 32*32 + 16*16
    x = torch.randn(2, N, config.hidden_dim)
    
    # Forward pass
    memory = encoder(x)
    print(f"Encoder Input: {x.shape}")
    print(f"Encoder Output (Memory): {memory.shape}")
    print(f"Number of parameters: {sum(p.numel() for p in encoder.parameters()):,}")
