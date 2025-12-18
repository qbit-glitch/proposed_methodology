"""
Configuration module for Unified Multi-Task Vision Transformer.
Optimized for MacBook M4 Pro with MPS support.
"""

from dataclasses import dataclass, field
from typing import List, Tuple, Optional
import torch


def get_device() -> torch.device:
    """Get the best available device (MPS for M4 Pro, CUDA, or CPU)."""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


@dataclass
class ModelConfig:
    """Configuration for the Unified Multi-Task Transformer model."""
    
    # ===== Backbone =====
    backbone: str = "resnet50"
    pretrained: bool = True
    freeze_backbone: bool = False
    
    # ===== ViT Encoder (for global/temporal features) =====
    use_pretrained_vit: bool = True
    vit_model: str = "vit_b_16"
    freeze_vit: bool = False
    
    # ===== Image =====
    image_size: Tuple[int, int] = (512, 512)  # (H, W)
    
    # ===== Transformer Dimensions =====
    hidden_dim: int = 256  # D in the paper
    num_heads: int = 8  # h in the paper
    dim_feedforward: int = 1024  # D_ff = 4 * D
    dropout: float = 0.1
    
    # ===== Encoder =====
    num_encoder_layers: int = 6
    
    # ===== Decoder =====
    num_decoder_layers: int = 6
    
    # ===== Query Counts =====
    num_detection_queries: int = 100
    num_segmentation_queries: int = 50
    num_plate_queries: int = 50
    num_ocr_queries: int = 20
    num_tracking_queries: int = 100  # Dynamic, based on previous tracks
    
    # ===== Detection Classes =====
    num_vehicle_classes: int = 6  # car, scooty, bike, bus, truck, auto
    vehicle_classes: List[str] = field(default_factory=lambda: [
        "car", "scooty", "bike", "bus", "truck", "auto"
    ])
    num_colors: int = 10  # White, Black, Silver, Red, Blue, etc.
    num_types: int = 5  # Sedan, SUV, Hatchback, etc.
    
    # ===== Segmentation Classes =====
    num_seg_classes: int = 3  # Background, Driveway, Footpath
    seg_classes: List[str] = field(default_factory=lambda: [
        "background", "driveway", "footpath"
    ])
    
    # ===== OCR =====
    ocr_alphabet: str = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
    max_plate_length: int = 10
    
    # ===== Alert Thresholds =====
    stopped_time_threshold: float = 120.0  # seconds
    overlap_threshold: float = 0.3  # 30% IoU


@dataclass
class TrainingConfig:
    """Configuration for training."""
    
    # ===== Device =====
    device: torch.device = field(default_factory=get_device)
    
    # ===== Batch Size (small for M4 Pro memory) =====
    batch_size: int = 2
    gradient_accumulation_steps: int = 4  # Effective batch size = 8
    
    # ===== Learning Rate =====
    learning_rate: float = 1e-4
    backbone_lr: float = 1e-5  # Lower LR for pretrained backbone
    weight_decay: float = 1e-4
    
    # ===== Scheduler =====
    warmup_steps: int = 1000
    max_epochs: int = 100
    lr_scheduler: str = "cosine"
    
    # ===== Loss Weights (lambdas) =====
    lambda_det: float = 1.0
    lambda_seg: float = 1.0
    lambda_plate: float = 2.0  # Higher weight for plates
    lambda_ocr: float = 1.5
    lambda_track: float = 1.0
    
    # ===== Detection Loss =====
    focal_alpha: float = 0.25
    focal_gamma: float = 2.0
    bbox_loss_coef: float = 5.0
    giou_loss_coef: float = 2.0
    
    # ===== Optimization =====
    use_mixed_precision: bool = True
    clip_grad_norm: float = 0.1
    
    # ===== Checkpointing =====
    checkpoint_dir: str = "outputs/checkpoints"
    save_every: int = 5  # epochs
    
    # ===== Visualization =====
    visualize_every: int = 100  # iterations
    output_dir: str = "outputs"


@dataclass
class Config:
    """Combined configuration."""
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    
    def __post_init__(self):
        """Validate configuration."""
        # Ensure hidden_dim is divisible by num_heads
        assert self.model.hidden_dim % self.model.num_heads == 0, \
            f"hidden_dim ({self.model.hidden_dim}) must be divisible by num_heads ({self.model.num_heads})"
        
        # Print device info
        print(f"🖥️  Using device: {self.training.device}")
        if self.training.device.type == "mps":
            print("   ✓ Apple Silicon MPS acceleration enabled")
        elif self.training.device.type == "cuda":
            print(f"   ✓ CUDA GPU: {torch.cuda.get_device_name(0)}")


def get_config() -> Config:
    """Factory function to create default configuration."""
    return Config()


if __name__ == "__main__":
    # Test configuration
    config = get_config()
    print(f"\n📊 Model Configuration:")
    print(f"   - Hidden dim: {config.model.hidden_dim}")
    print(f"   - Num heads: {config.model.num_heads}")
    print(f"   - Encoder layers: {config.model.num_encoder_layers}")
    print(f"   - Image size: {config.model.image_size}")
    print(f"\n🎯 Query Counts:")
    print(f"   - Detection: {config.model.num_detection_queries}")
    print(f"   - Segmentation: {config.model.num_segmentation_queries}")
    print(f"   - Plate: {config.model.num_plate_queries}")
    print(f"   - OCR: {config.model.num_ocr_queries}")
    print(f"\n⚙️  Training Configuration:")
    print(f"   - Batch size: {config.training.batch_size}")
    print(f"   - Gradient accumulation: {config.training.gradient_accumulation_steps}")
    print(f"   - Effective batch size: {config.training.batch_size * config.training.gradient_accumulation_steps}")
