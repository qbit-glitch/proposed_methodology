"""
Tracking Decoder - Algorithm 7.

Implements multi-object tracking with:
- Temporal query augmentation from previous frame
- Dual cross-attention to detection and segmentation features
- Track association with Hungarian matching
- Stopped time computation for alert generation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional, List
from scipy.optimize import linear_sum_assignment

from .base_decoder import BaseDecoder, DecoderLayer, MultiHeadCrossAttention

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import ModelConfig


class TemporalEmbedding(nn.Module):
    """
    Augment tracking queries with temporal information from previous frame.
    """
    
    def __init__(self, hidden_dim: int):
        super().__init__()
        
        # MLP to combine appearance, position, and velocity
        self.embed_proj = nn.Sequential(
            nn.Linear(hidden_dim + 4, hidden_dim),  # +4 for position/velocity
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Learnable queries for new tracks
        self.new_track_query = nn.Parameter(torch.randn(1, hidden_dim))
    
    def forward(
        self,
        prev_features: Optional[torch.Tensor],  # [B, M, D] previous track features
        prev_positions: Optional[torch.Tensor], # [B, M, 4] previous positions (x, y, w, h)
        velocities: Optional[torch.Tensor],     # [B, M, 2] velocity (dx, dy)
        num_new_tracks: int = 20,
        batch_size: int = 1,  # Explicit batch size for first frame
        device: str = 'cpu',  # Device for first frame
    ) -> torch.Tensor:
        """
        Generate tracking queries from previous state.
        
        Returns:
            queries: [B, M + num_new_tracks, D]
        """
        if prev_features is not None:
            B = prev_features.shape[0]
            device = prev_features.device
        else:
            B = batch_size
            device = device
        
        queries = []
        
        # Embed existing tracks
        if prev_features is not None and prev_features.shape[1] > 0:
            # Combine features with position and velocity
            if velocities is None:
                velocities = torch.zeros(B, prev_features.shape[1], 2, device=device)
            
            # Truncate positions to just x, y for velocity computation
            pos_vel = torch.cat([
                prev_positions[:, :, :2],  # x, y
                velocities
            ], dim=-1)  # [B, M, 4]
            
            combined = torch.cat([prev_features, pos_vel], dim=-1)
            existing_queries = self.embed_proj(combined)  # [B, M, D]
            queries.append(existing_queries)
        
        # Add new track queries
        new_queries = self.new_track_query.expand(B, num_new_tracks, -1)
        queries.append(new_queries)
        
        return torch.cat(queries, dim=1) if len(queries) > 1 else queries[0]


class DualCrossAttentionLayer(nn.Module):
    """
    Decoder layer with dual cross-attention to both detection and segmentation.
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
        self.self_attn = MultiHeadCrossAttention(hidden_dim, num_heads, dropout)
        self.norm1 = nn.LayerNorm(hidden_dim)
        
        # Cross-attention with encoder memory
        self.cross_attn_memory = MultiHeadCrossAttention(hidden_dim, num_heads, dropout)
        self.norm2 = nn.LayerNorm(hidden_dim)
        
        # Cross-attention with detection features
        self.cross_attn_det = MultiHeadCrossAttention(hidden_dim, num_heads, dropout)
        self.norm3 = nn.LayerNorm(hidden_dim)
        
        # Cross-attention with segmentation features
        self.cross_attn_seg = MultiHeadCrossAttention(hidden_dim, num_heads, dropout)
        self.norm4 = nn.LayerNorm(hidden_dim)
        
        # Feed-forward
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, feedforward_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(feedforward_dim, hidden_dim),
            nn.Dropout(dropout)
        )
        self.norm5 = nn.LayerNorm(hidden_dim)
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(
        self,
        queries: torch.Tensor,
        memory: torch.Tensor,
        detection_features: torch.Tensor,
        segmentation_features: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass with dual cross-attention.
        
        Args:
            queries: [B, N_q, D]
            memory: [B, N_m, D]
            detection_features: [B, 100, D]
            segmentation_features: [B, 50, D]
            
        Returns:
            Updated queries [B, N_q, D]
        """
        # Self-attention
        q_norm = self.norm1(queries)
        self_out = self.self_attn(q_norm, q_norm)
        queries = queries + self.dropout(self_out)
        
        # Cross-attention with memory
        q_norm = self.norm2(queries)
        mem_out = self.cross_attn_memory(q_norm, memory)
        queries = queries + self.dropout(mem_out)
        
        # Cross-attention with detection features
        q_norm = self.norm3(queries)
        det_out = self.cross_attn_det(q_norm, detection_features)
        queries = queries + self.dropout(det_out)
        
        # Cross-attention with segmentation features
        q_norm = self.norm4(queries)
        seg_out = self.cross_attn_seg(q_norm, segmentation_features)
        queries = queries + self.dropout(seg_out)
        
        # Feed-forward
        q_norm = self.norm5(queries)
        ffn_out = self.ffn(q_norm)
        queries = queries + ffn_out
        
        return queries


class TrackAssociation(nn.Module):
    """
    Learnable track association with Hungarian matching.
    """
    
    def __init__(self, hidden_dim: int):
        super().__init__()
        
        # Project features for similarity computation
        self.feat_proj = nn.Linear(hidden_dim, hidden_dim)
        
        # MLP for association score
        self.assoc_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2 + 5, 128),  # +5 for IoU and time diff
            nn.ReLU(),
            nn.Linear(128, 1)
        )
    
    def compute_similarity(
        self,
        track_features: torch.Tensor,  # [B, M, D]
        detection_features: torch.Tensor,  # [B, N, D]
    ) -> torch.Tensor:
        """Compute cosine similarity matrix."""
        track_proj = F.normalize(self.feat_proj(track_features), dim=-1)
        det_proj = F.normalize(self.feat_proj(detection_features), dim=-1)
        
        # [B, M, D] @ [B, D, N] -> [B, M, N]
        similarity = torch.bmm(track_proj, det_proj.transpose(1, 2))
        return similarity
    
    def forward(
        self,
        track_features: torch.Tensor,
        detection_features: torch.Tensor,
        track_boxes: Optional[torch.Tensor] = None,
        detection_boxes: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, List[Tuple]]:
        """
        Compute associations and match using Hungarian algorithm.
        
        Returns:
            - association_scores: [B, M, N]
            - matches: List of (track_idx, det_idx) tuples per batch
        """
        similarity = self.compute_similarity(track_features, detection_features)
        
        # Hungarian matching per batch
        matches = []
        for b in range(similarity.shape[0]):
            sim = similarity[b].detach().cpu().numpy()
            cost_matrix = -sim  # Hungarian minimizes cost
            row_ind, col_ind = linear_sum_assignment(cost_matrix)
            matches.append(list(zip(row_ind.tolist(), col_ind.tolist())))
        
        return similarity, matches


class TrackingHead(nn.Module):
    """Output heads for tracking."""
    
    def __init__(self, hidden_dim: int):
        super().__init__()
        
        # Trajectory prediction (delta x, y, w, h)
        self.trajectory_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 4)  # dx, dy, dw, dh
        )
        
        # Velocity estimation
        self.velocity_head = nn.Linear(hidden_dim, 2)  # vx, vy
        
        # Track confidence
        self.conf_head = nn.Linear(hidden_dim, 1)
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        return {
            'trajectory_delta': self.trajectory_head(x),
            'velocity': self.velocity_head(x),
            'track_confidence': self.conf_head(x).sigmoid(),
        }


class TrackingDecoder(nn.Module):
    """
    Tracking Decoder for multi-object tracking.
    
    Uses dual cross-attention to detection and segmentation features.
    """
    
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        
        # Temporal embedding
        self.temporal_embed = TemporalEmbedding(config.hidden_dim)
        
        # Dual cross-attention layers
        self.layers = nn.ModuleList([
            DualCrossAttentionLayer(
                hidden_dim=config.hidden_dim,
                num_heads=config.num_heads,
                feedforward_dim=config.dim_feedforward,
                dropout=config.dropout
            )
            for _ in range(config.num_decoder_layers)
        ])
        
        self.final_norm = nn.LayerNorm(config.hidden_dim)
        
        # Track association
        self.association = TrackAssociation(config.hidden_dim)
        
        # Output heads
        self.head = TrackingHead(config.hidden_dim)
        
        # Track state
        self.track_id_counter = 0
    
    def forward(
        self,
        memory: torch.Tensor,
        detection_features: torch.Tensor,
        segmentation_features: torch.Tensor,
        detection_boxes: torch.Tensor,
        prev_state: Optional[Dict] = None,
        memory_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[Dict[str, torch.Tensor], Dict]:
        """
        Forward pass of tracking decoder.
        
        Args:
            memory: Encoder memory [B, N, D]
            detection_features: [B, 100, D]
            segmentation_features: [B, 50, D]
            detection_boxes: [B, 100, 4]
            prev_state: Previous tracking state (features, boxes, velocities, stopped_times)
            
        Returns:
            - outputs: Dictionary with tracking predictions
            - new_state: Updated tracking state for next frame
        """
        B = memory.shape[0]
        device = memory.device
        
        # Initialize tracking queries
        if prev_state is not None and 'features' in prev_state:
            queries = self.temporal_embed(
                prev_features=prev_state['features'],
                prev_positions=prev_state['boxes'],
                velocities=prev_state.get('velocities'),
                num_new_tracks=self.config.num_tracking_queries // 2,
                batch_size=B,
                device=device,
            )
        else:
            # First frame: use learnable queries
            queries = self.temporal_embed(
                prev_features=None,
                prev_positions=None,
                velocities=None,
                num_new_tracks=self.config.num_tracking_queries,
                batch_size=B,
                device=device,
            )
        
        # Pass through dual cross-attention layers
        for layer in self.layers:
            queries = layer(
                queries=queries,
                memory=memory,
                detection_features=detection_features,
                segmentation_features=segmentation_features
            )
        
        queries = self.final_norm(queries)
        
        # Compute associations
        assoc_scores, matches = self.association(
            track_features=queries,
            detection_features=detection_features,
            detection_boxes=detection_boxes
        )
        
        # Apply output heads
        outputs = self.head(queries)
        outputs['association_scores'] = assoc_scores
        outputs['matches'] = matches
        outputs['features'] = queries
        
        # Update tracking state
        new_state = self._update_state(
            queries, detection_boxes, matches, outputs, prev_state, device
        )
        
        return outputs, new_state
    
    def _update_state(
        self,
        features: torch.Tensor,
        detection_boxes: torch.Tensor,
        matches: List[List[Tuple]],
        outputs: Dict,
        prev_state: Optional[Dict],
        device
    ) -> Dict:
        """Update tracking state for next frame."""
        B = features.shape[0]
        
        # For simplicity, use matched detection boxes as new positions
        # In full implementation, would use Kalman filter
        
        new_state = {
            'features': features,
            'boxes': detection_boxes[:, :features.shape[1], :] if detection_boxes.shape[1] >= features.shape[1] else 
                     torch.zeros(B, features.shape[1], 4, device=device),
            'velocities': outputs['velocity'],
            'matches': matches,
        }
        
        # Compute stopped times
        if prev_state is not None and 'stopped_times' in prev_state:
            prev_stopped = prev_state['stopped_times']
            velocity = outputs['velocity']
            speed = torch.sqrt((velocity ** 2).sum(dim=-1))  # [B, N]
            is_stopped = speed < 0.5  # Threshold: 0.5 pixels/frame
            
            # Update: add dt if stopped, else reset
            dt = 1.0 / 30.0  # Assuming 30 FPS
            new_stopped = torch.where(
                is_stopped,
                prev_stopped + dt,
                torch.zeros_like(prev_stopped)
            )
            new_state['stopped_times'] = new_stopped
        else:
            new_state['stopped_times'] = torch.zeros(B, features.shape[1], device=device)
        
        return new_state


if __name__ == "__main__":
    from config import ModelConfig
    
    config = ModelConfig()
    decoder = TrackingDecoder(config)
    
    # Dummy inputs
    memory = torch.randn(2, 5376, config.hidden_dim)
    det_features = torch.randn(2, 100, config.hidden_dim)
    seg_features = torch.randn(2, 50, config.hidden_dim)
    det_boxes = torch.rand(2, 100, 4)
    
    # Forward pass (first frame)
    outputs, state = decoder(memory, det_features, seg_features, det_boxes)
    
    print(f"📊 Tracking Decoder Output (Frame 1):")
    print(f"   Features: {outputs['features'].shape}")
    print(f"   Trajectory delta: {outputs['trajectory_delta'].shape}")
    print(f"   Velocity: {outputs['velocity'].shape}")
    print(f"   Track confidence: {outputs['track_confidence'].shape}")
    print(f"   Association scores: {outputs['association_scores'].shape}")
    print(f"   Matches: {len(outputs['matches'][0])} tracks matched")
    
    # Forward pass (second frame with previous state)
    outputs2, state2 = decoder(memory, det_features, seg_features, det_boxes, prev_state=state)
    print(f"\n📊 Tracking Decoder Output (Frame 2):")
    print(f"   Stopped times: {state2['stopped_times'].shape}")
    print(f"   Parameters: {sum(p.numel() for p in decoder.parameters()):,}")
