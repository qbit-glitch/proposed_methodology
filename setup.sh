#!/bin/bash
# =============================================================================
# Unified Multi-Task Transformer - Setup Script
# =============================================================================
# This script sets up the complete environment for training:
# 1. Creates virtual environment
# 2. Installs dependencies
# 3. Extracts datasets from zip files
# 4. Verifies installation
#
# Usage:
#   chmod +x setup.sh
#   ./setup.sh
#
# =============================================================================

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}"
echo "============================================================"
echo "  Unified Multi-Task Transformer - Setup"
echo "  Parking Violation Detection System"
echo "============================================================"
echo -e "${NC}"

# =============================================================================
# 1. Detect System and Python
# =============================================================================
echo -e "${YELLOW}[1/5] Detecting system...${NC}"

# Detect OS
if [[ "$OSTYPE" == "darwin"* ]]; then
    OS="macos"
    echo -e "  ${GREEN}✓${NC} macOS detected"
    if [[ $(uname -m) == 'arm64' ]]; then
        ARCH="arm64"
        echo -e "  ${GREEN}✓${NC} Apple Silicon (M1/M2/M3/M4) detected - MPS acceleration available"
    else
        ARCH="x86_64"
        echo -e "  ${YELLOW}!${NC} Intel Mac detected - CPU only"
    fi
elif [[ "$OSTYPE" == "linux-gnu"* ]]; then
    OS="linux"
    echo -e "  ${GREEN}✓${NC} Linux detected"
    # Check for NVIDIA GPU
    if command -v nvidia-smi &> /dev/null; then
        echo -e "  ${GREEN}✓${NC} NVIDIA GPU detected - CUDA acceleration available"
        nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
        CUDA_AVAILABLE=true
    else
        echo -e "  ${YELLOW}!${NC} No NVIDIA GPU detected - CPU only"
        CUDA_AVAILABLE=false
    fi
else
    OS="unknown"
    echo -e "  ${YELLOW}!${NC} Unknown OS: $OSTYPE"
fi

# Find Python
PYTHON_CMD=""
for cmd in python3.6 python3.7 python3.8 python3.9 python3.10 python3.11 python3.12 python3 python; do
    if command -v $cmd &> /dev/null; then
        version=$($cmd --version 2>&1 | grep -oP '\d+\.\d+')
        echo -e "  ${GREEN}✓${NC} Found $cmd (version $version)"
        PYTHON_CMD=$cmd
        break
    fi
done

if [ -z "$PYTHON_CMD" ]; then
    echo -e "  ${RED}✗${NC} Python not found! Please install Python 3.6+"
    exit 1
fi

# =============================================================================
# 2. Create Virtual Environment
# =============================================================================
echo -e "\n${YELLOW}[2/5] Setting up virtual environment...${NC}"

if [ -d "venv" ]; then
    echo -e "  ${YELLOW}!${NC} Virtual environment already exists"
    read -p "  Do you want to recreate it? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        rm -rf venv
        $PYTHON_CMD -m venv venv
        echo -e "  ${GREEN}✓${NC} Virtual environment recreated"
    fi
else
    $PYTHON_CMD -m venv venv
    echo -e "  ${GREEN}✓${NC} Virtual environment created"
fi

# Activate virtual environment
source venv/bin/activate
echo -e "  ${GREEN}✓${NC} Virtual environment activated"

# Upgrade pip
pip install --upgrade pip setuptools wheel -q
echo -e "  ${GREEN}✓${NC} pip upgraded"

# =============================================================================
# 3. Install Dependencies
# =============================================================================
echo -e "\n${YELLOW}[3/5] Installing dependencies...${NC}"

# Install PyTorch with appropriate backend
if [[ "$OS" == "macos" && "$ARCH" == "arm64" ]]; then
    echo -e "  Installing PyTorch for Apple Silicon (MPS)..."
    pip install torch torchvision -q
elif [[ "$CUDA_AVAILABLE" == true ]]; then
    echo -e "  Installing PyTorch with CUDA support..."
    # For CUDA 11.8 (common for RTX 3070)
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118 -q
else
    echo -e "  Installing PyTorch (CPU only)..."
    pip install torch torchvision -q
fi
echo -e "  ${GREEN}✓${NC} PyTorch installed"

# Install other requirements
pip install -r requirements.txt -q
echo -e "  ${GREEN}✓${NC} All dependencies installed"

# =============================================================================
# 4. Extract Datasets
# =============================================================================
echo -e "\n${YELLOW}[4/5] Extracting datasets...${NC}"

mkdir -p datasets

# Dataset extraction order and expected structure
declare -A DATASETS=(
    ["coco2017"]="COCO 2017 (Detection)"
    ["cityscapes"]="Cityscapes (Segmentation)"
    ["CCPD2019"]="CCPD 2019 (License Plate OCR)"
    ["mot17"]="MOT17 (Tracking)"
    ["license_plates"]="License Plates (Plate Detection)"
)

for dataset in "${!DATASETS[@]}"; do
    zip_file="datasets/${dataset}.zip"
    tar_file="datasets/${dataset}.tar.gz"
    
    if [ -d "datasets/${dataset}" ]; then
        echo -e "  ${GREEN}✓${NC} ${DATASETS[$dataset]} already extracted"
    elif [ -f "$zip_file" ]; then
        echo -e "  Extracting ${DATASETS[$dataset]}..."
        unzip -q "$zip_file" -d datasets/
        echo -e "  ${GREEN}✓${NC} ${DATASETS[$dataset]} extracted"
    elif [ -f "$tar_file" ]; then
        echo -e "  Extracting ${DATASETS[$dataset]}..."
        tar -xzf "$tar_file" -C datasets/
        echo -e "  ${GREEN}✓${NC} ${DATASETS[$dataset]} extracted"
    else
        echo -e "  ${YELLOW}!${NC} ${DATASETS[$dataset]} not found (optional: ${dataset}.zip)"
    fi
done

# =============================================================================
# 5. Verify Installation
# =============================================================================
echo -e "\n${YELLOW}[5/5] Verifying installation...${NC}"

python -c "
import torch
import sys

print(f'  Python: {sys.version}')
print(f'  PyTorch: {torch.__version__}')

# Check available backends
if torch.cuda.is_available():
    print(f'  CUDA: {torch.version.cuda}')
    print(f'  GPU: {torch.cuda.get_device_name(0)}')
    print(f'  GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB')
elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
    print(f'  MPS (Apple Silicon): Available')
else:
    print(f'  Acceleration: CPU only')

# Test model import
try:
    from src.unified_transformer import build_model
    from src.config import ModelConfig
    print('  Model import: OK')
except Exception as e:
    print(f'  Model import: FAILED - {e}')
"

# =============================================================================
# Done!
# =============================================================================
echo -e "\n${GREEN}"
echo "============================================================"
echo "  Setup Complete!"
echo "============================================================"
echo -e "${NC}"
echo "To start training:"
echo "  source venv/bin/activate"
echo "  python train_complete.py --epochs 15 --pretrain-epochs 15 --data-dir datasets --subset-ratio 0.2"
echo ""
echo "For quick demo:"
echo "  python train_complete.py --demo"
echo ""
