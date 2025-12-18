# Unified Multi-Task Transformer for Parking Violation Detection

A unified end-to-end deep learning system for automated parking violation detection, combining:
- **Vehicle Detection** (6 classes + attributes)
- **Scene Segmentation** (driveway/footpath)
- **License Plate Detection**
- **OCR Text Recognition**
- **Multi-Object Tracking**
- **Parking Violation Alert Generation**

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                    Input Images [B, 3, H, W]                        │
└─────────────────────────────────────────────────────────────────────┘
                                │
                    ┌───────────┴───────────┐
                    ▼                       ▼
        ┌───────────────────┐   ┌───────────────────┐
        │  ResNet50 Backbone │   │   ViT-B/16 Encoder│
        │   (Pretrained)     │   │   (Pretrained)    │
        │ Multi-scale C3,C4,C5│   │  Global Features  │
        └───────────────────┘   └───────────────────┘
                    │                       │
                    └───────────┬───────────┘
                                ▼
                    ┌───────────────────────┐
                    │   Feature Fusion       │
                    │  (Cross-Attention)     │
                    └───────────────────────┘
                                │
                                ▼
                    ┌───────────────────────┐
                    │  Transformer Encoder   │
                    │    (6 layers)          │
                    └───────────────────────┘
                                │
        ┌───────────┬───────────┼───────────┬───────────┐
        ▼           ▼           ▼           ▼           ▼
   ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐
   │Detection│ │Segment- │ │  Plate  │ │   OCR   │ │Tracking │
   │ Decoder │ │ation    │ │ Decoder │ │ Decoder │ │ Decoder │
   └─────────┘ └─────────┘ └─────────┘ └─────────┘ └─────────┘
```

## ✨ Features

- **Pretrained Backbones**: ResNet50 + ViT-B/16 with ImageNet weights
- **Multi-Task Learning**: All tasks trained jointly with uncertainty weighting
- **Cross-Platform**: Supports macOS (MPS), Linux (CUDA), and CPU
- **Multi-GPU Support**: Automatic distributed training with DDP
- **Memory Efficient**: Gradient checkpointing for training on 8GB GPUs
- **Mixed Precision**: FP16 training on CUDA for 2x speedup
- **Staged Training**: 3-stage curriculum with gradual unfreezing
- **Checkpoint Resume**: Full training state saved for resumable training

## 🚀 Quick Start

### 1. Setup Environment

```bash
# Clone the repository
git clone https://github.com/yourusername/parking-violation-detector.git
cd parking-violation-detector

# Run setup script (handles everything automatically)
chmod +x setup.sh
./setup.sh
```

### 2. Prepare Datasets

Place your dataset zip files in the `datasets/` folder:
- `coco2017.zip` - COCO 2017 (Detection)
- `cityscapes.zip` - Cityscapes (Segmentation)
- `CCPD2019.zip` - CCPD (License Plate OCR)
- `mot17.zip` - MOT17 (Tracking)

The setup script will extract them automatically.

### 3. Train the Model

```bash
# Activate environment
source venv/bin/activate

# Quick demo (synthetic data, 2 epochs)
python train_complete.py --demo

# Full training (20% of dataset, 15 epochs)
python train_complete.py --epochs 15 --pretrain-epochs 15 \
    --data-dir datasets --subset-ratio 0.2

# Resume from checkpoint
python train_complete.py --resume outputs/checkpoints/best_model.pt \
    --epochs 15 --data-dir datasets
```

## 🔥 Multi-GPU Training

The code automatically detects available GPUs and uses distributed training:

```bash
# Auto-detect GPUs and train (uses all available GPUs)
python train.py --epochs 15 --data-dir datasets

# Force single GPU mode
python train.py --single-gpu --epochs 15 --data-dir datasets

# Specify number of GPUs
python train.py --num-gpus 2 --epochs 15 --data-dir datasets

# Manual launch with torchrun (advanced)
torchrun --nproc_per_node=4 train_complete.py --epochs 15 --data-dir datasets
```

## 📊 Platform Support

| Platform | Device | GPUs | Batch Size | Mixed Precision |
|----------|--------|------|------------|-----------------|
| macOS M1 | MPS | 1 | 1 | ❌ |
| macOS M4 Pro | MPS | 1 | 2 | ❌ |
| RTX 3070 (8GB) | CUDA | 1 | 1 | ✅ |
| RTX 3090 (24GB) | CUDA | 1 | 4 | ✅ |
| 2x RTX 3090 | CUDA | 2 | 8 | ✅ |
| 4x A100 | CUDA | 4 | 16 | ✅ |

## 📁 Project Structure

```
├── src/
│   ├── config.py              # Configuration classes
│   ├── unified_transformer.py # Main model architecture
│   ├── backbone.py            # ResNet50 backbone
│   ├── vit_encoder.py         # ViT temporal encoder
│   ├── transformer_encoder.py # Transformer encoder
│   ├── staged_trainer.py      # 3-stage training pipeline
│   ├── distributed.py         # Multi-GPU DDP utilities
│   ├── m4_optimizations.py    # GPU optimizations (MPS + CUDA)
│   ├── decoders/              # Task-specific decoders
│   └── losses/                # Loss functions
├── train.py                   # Smart launcher (auto-detects GPUs)
├── train_complete.py          # Main training script
├── setup.sh                   # Environment setup script
├── requirements.txt           # Python dependencies
└── datasets/                  # Dataset directory (not in repo)
```

## 🔧 Configuration

Key training parameters:

```python
# Model
hidden_dim = 256
num_heads = 8
num_encoder_layers = 6
num_decoder_layers = 6

# Training
batch_size = auto  # Based on GPU memory
gradient_accumulation = auto
learning_rate = 1e-4
weight_decay = 1e-4
```

## 📈 Training Stages

1. **Stage 1: Initialization**
   - Load pretrained ResNet50 and ViT-B/16 weights
   - Xavier initialize transformer components

2. **Stage 2: Decoder Pretraining**
   - Train each decoder separately on task-specific data
   - Detection → Segmentation → Plate → OCR → Tracking

3. **Stage 3: Joint Training**
   - Train all tasks together with uncertainty weighting
   - Gradual unfreezing: decoders → encoder → backbone

## 🛠️ Requirements

- Python 3.6+ (tested on 3.6.x, 3.10+)
- PyTorch 1.10+
- 8GB+ GPU memory recommended

## 📝 License

MIT License

## 🤝 Contributing

Contributions welcome! Please open an issue or PR.
