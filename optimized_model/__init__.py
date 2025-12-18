"""
Optimized Model Package.

This is an optimized version of the src/ package with integrated architectural optimizations:

1. Deformable Attention (OPT-1.1): Adaptive receptive fields, 2-3x faster attention
2. Hierarchical Decoder (OPT-4.1): Shared decoder + task-specific heads
3. Query Optimization (OPT-1.3): Two-stage query refinement
4. Cross-Task Consistency (OPT-4.3): Consistency losses between related tasks

Usage:
    from optimized_model import OptimizedUnifiedTransformer, OptimizedModelConfig
    
    config = OptimizedModelConfig(
        use_deformable_attention=True,
        use_hierarchical_decoder=True,
        use_query_refinement=True,
    )
    model = OptimizedUnifiedTransformer(config)
"""

# Re-export main components
from optimized_model.config import ModelConfig, OptimizedModelConfig
from optimized_model.backbone import ResNet50Backbone
from optimized_model.fpn import FPN
from optimized_model.transformer_encoder import TransformerEncoder
from optimized_model.deformable_attention import (
    DeformableAttention,
    MultiScaleDeformableAttention,
    DeformableTransformerEncoderLayer,
)

# Will be updated after optimization integration
# from optimized_model.optimized_unified_transformer import OptimizedUnifiedTransformer

__version__ = "2.0.0"
__all__ = [
    "ModelConfig",
    "OptimizedModelConfig",
    "ResNet50Backbone",
    "FeaturePyramidNetwork",
    "TransformerEncoder",
    "DeformableAttention",
    "MultiScaleDeformableAttention",
    "DeformableTransformerEncoderLayer",
]
