"""
Base Decoder Layer with Self-Attention, Cross-Attention, and FFN.

This module provides the building blocks for all task-specific decoders.
Supports optional inter-decoder cross-attention for hierarchical information flow.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class MultiHeadCrossAttention(nn.Module):
    """
    Multi-Head Cross-Attention Layer.
    
    Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) V
    Where Q comes from queries and K, V come from memory/context.
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
        
        # Separate projections for Q, K, V
        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)
        
        # Output projection
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(
        self, 
        query: torch.Tensor,
        key_value: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass of cross-attention.
        
        Args:
            query: Query tensor [B, N_q, D]
            key_value: Key and Value tensor [B, N_kv, D]
            key_padding_mask: Mask for padding [B, N_kv]
            
        Returns:
            Output tensor [B, N_q, D]
        """
        B, N_q, D = query.shape
        N_kv = key_value.shape[1]
        
        # Project Q from queries, K and V from key_value
        Q = self.q_proj(query)      # [B, N_q, D]
        K = self.k_proj(key_value)  # [B, N_kv, D]
        V = self.v_proj(key_value)  # [B, N_kv, D]
        
        # Reshape for multi-head attention
        Q = Q.view(B, N_q, self.num_heads, self.head_dim).transpose(1, 2)   # [B, h, N_q, d_k]
        K = K.view(B, N_kv, self.num_heads, self.head_dim).transpose(1, 2)  # [B, h, N_kv, d_k]
        V = V.view(B, N_kv, self.num_heads, self.head_dim).transpose(1, 2)  # [B, h, N_kv, d_k]
        
        # Compute attention scores
        # [B, h, N_q, d_k] @ [B, h, d_k, N_kv] -> [B, h, N_q, N_kv]
        attn_scores = (Q @ K.transpose(-2, -1)) * self.scale
        
        # Apply mask if provided
        if key_padding_mask is not None:
            key_padding_mask = key_padding_mask.unsqueeze(1).unsqueeze(2)
            attn_scores = attn_scores.masked_fill(key_padding_mask, float('-inf'))
        
        # Softmax and dropout
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention to values
        out = attn_weights @ V  # [B, h, N_q, d_k]
        
        # Reshape back
        out = out.transpose(1, 2).contiguous().view(B, N_q, D)
        
        # Output projection
        out = self.out_proj(out)
        
        return out


class DecoderLayer(nn.Module):
    """
    Single Decoder Layer with optional inter-decoder cross-attention.
    
    Standard flow:
    1. Self-Attention (queries attend to each other)
    2. Cross-Attention with encoder memory
    3. (Optional) Cross-Attention with another decoder's features
    4. Feed-Forward Network
    
    All with pre-norm and residual connections.
    """
    
    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        feedforward_dim: int,
        dropout: float = 0.1,
        has_inter_decoder_attn: bool = False
    ):
        super().__init__()
        self.has_inter_decoder_attn = has_inter_decoder_attn
        
        # Self-attention
        self.self_attn = MultiHeadCrossAttention(hidden_dim, num_heads, dropout)
        self.norm1 = nn.LayerNorm(hidden_dim)
        
        # Cross-attention with encoder memory
        self.cross_attn_memory = MultiHeadCrossAttention(hidden_dim, num_heads, dropout)
        self.norm2 = nn.LayerNorm(hidden_dim)
        
        # Optional inter-decoder cross-attention
        if has_inter_decoder_attn:
            self.cross_attn_decoder = MultiHeadCrossAttention(hidden_dim, num_heads, dropout)
            self.norm3 = nn.LayerNorm(hidden_dim)
        
        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, feedforward_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(feedforward_dim, hidden_dim),
            nn.Dropout(dropout)
        )
        self.norm_ffn = nn.LayerNorm(hidden_dim)
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(
        self,
        queries: torch.Tensor,
        memory: torch.Tensor,
        decoder_context: Optional[torch.Tensor] = None,
        query_mask: Optional[torch.Tensor] = None,
        memory_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass of decoder layer.
        
        Args:
            queries: Query tensor [B, N_q, D]
            memory: Encoder memory [B, N_m, D]
            decoder_context: Optional features from another decoder [B, N_c, D]
            query_mask: Mask for queries [B, N_q]
            memory_mask: Mask for memory [B, N_m]
            
        Returns:
            Updated queries [B, N_q, D]
        """
        # 1. Self-attention
        q_norm = self.norm1(queries)
        self_attn_out = self.self_attn(q_norm, q_norm, query_mask)
        queries = queries + self.dropout(self_attn_out)
        
        # 2. Cross-attention with encoder memory
        q_norm = self.norm2(queries)
        cross_attn_out = self.cross_attn_memory(q_norm, memory, memory_mask)
        queries = queries + self.dropout(cross_attn_out)
        
        # 3. Optional inter-decoder cross-attention
        if self.has_inter_decoder_attn and decoder_context is not None:
            q_norm = self.norm3(queries)
            inter_attn_out = self.cross_attn_decoder(q_norm, decoder_context)
            queries = queries + self.dropout(inter_attn_out)
        
        # 4. Feed-forward network
        q_norm = self.norm_ffn(queries)
        ffn_out = self.ffn(q_norm)
        queries = queries + ffn_out
        
        return queries


class BaseDecoder(nn.Module):
    """
    Base Decoder class with 6 layers.
    
    Used as foundation for all task-specific decoders.
    """
    
    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        feedforward_dim: int,
        num_layers: int = 6,
        dropout: float = 0.1,
        has_inter_decoder_attn: bool = False,
        num_queries: int = 100
    ):
        super().__init__()
        
        # Learnable query embeddings
        self.query_embed = nn.Embedding(num_queries, hidden_dim)
        
        # Decoder layers
        self.layers = nn.ModuleList([
            DecoderLayer(
                hidden_dim=hidden_dim,
                num_heads=num_heads,
                feedforward_dim=feedforward_dim,
                dropout=dropout,
                has_inter_decoder_attn=has_inter_decoder_attn
            )
            for _ in range(num_layers)
        ])
        
        # Final norm
        self.final_norm = nn.LayerNorm(hidden_dim)
        
        self.hidden_dim = hidden_dim
        self.num_queries = num_queries
    
    def forward(
        self,
        memory: torch.Tensor,
        decoder_context: Optional[torch.Tensor] = None,
        memory_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass through all decoder layers.
        
        Args:
            memory: Encoder memory [B, N_m, D]
            decoder_context: Optional features from another decoder [B, N_c, D]
            memory_mask: Mask for memory [B, N_m]
            
        Returns:
            Decoder output features [B, num_queries, D]
        """
        B = memory.shape[0]
        device = memory.device
        
        # Initialize queries from learnable embeddings
        query_pos = torch.arange(self.num_queries, device=device)
        queries = self.query_embed(query_pos).unsqueeze(0).expand(B, -1, -1)
        
        # Pass through all layers
        for layer in self.layers:
            queries = layer(
                queries=queries,
                memory=memory,
                decoder_context=decoder_context,
                memory_mask=memory_mask
            )
        
        # Final normalization
        queries = self.final_norm(queries)
        
        return queries


if __name__ == "__main__":
    # Test decoder
    hidden_dim = 256
    num_heads = 8
    feedforward_dim = 1024
    
    decoder = BaseDecoder(
        hidden_dim=hidden_dim,
        num_heads=num_heads,
        feedforward_dim=feedforward_dim,
        num_layers=6,
        has_inter_decoder_attn=True,
        num_queries=100
    )
    
    # Create dummy inputs
    memory = torch.randn(2, 5376, hidden_dim)  # Encoder memory
    context = torch.randn(2, 50, hidden_dim)   # Another decoder's output
    
    # Forward pass
    output = decoder(memory, context)
    print(f"Decoder Input Memory: {memory.shape}")
    print(f"Decoder Context: {context.shape}")
    print(f"Decoder Output: {output.shape}")
    print(f"Number of parameters: {sum(p.numel() for p in decoder.parameters()):,}")
