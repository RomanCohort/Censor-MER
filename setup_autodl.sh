#!/bin/bash
# =============================================================================
# Censor: AutoDL One-Click Setup Script
# =============================================================================
# Run this script on AutoDL instance to set up the training environment.
#
# Usage:
#   wget https://raw.githubusercontent.com/RomanCohort/Censor-MER/master/setup_autodl.sh
#   chmod +x setup_autodl.sh
#   ./setup_autodl.sh
# =============================================================================

set -e

echo "============================================================"
echo "Censor: AutoDL Environment Setup"
echo "============================================================"

# 1. Check GPU
echo "[Step 1] Checking GPU..."
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi --query-gpu=name,memory.total --format=csv
    GPU_MEM=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1)
    echo "GPU Memory: ${GPU_MEM} MB"
else
    echo "[WARNING] nvidia-smi not found, may be CPU-only"
fi

# 2. Check Python/PyTorch
echo "[Step 2] Checking Python/PyTorch..."
PYTHON_VER=$(python --version 2>&1)
echo "Python: $PYTHON_VER"

TORCH_VER=$(python -c "import torch; print(torch.__version__)" 2>/dev/null || echo "not installed")
CUDA_AVAIL=$(python -c "import torch; print(torch.cuda.is_available())" 2>/dev/null || echo "false")
echo "PyTorch: $TORCH_VER"
echo "CUDA available: $CUDA_AVAIL"

# 3. Clone repository
echo "[Step 3] Cloning Censor repository..."
cd /root/autodl-tmp

if [ -d "Censor-MER" ]; then
    echo "Censor-MER already exists, updating..."
    cd Censor-MER
    git pull
else
    git clone https://github.com/RomanCohort/Censor-MER.git
    cd Censor-MER
fi

# 4. Install dependencies
echo "[Step 4] Installing dependencies..."
pip install -r requirements.txt -q

# Install OpenCV with optical flow support
pip uninstall opencv-python -y -q 2>/dev/null || true
pip install opencv-contrib-python -q

# Additional utilities
pip install tensorboard tqdm -q

echo "Dependencies installed."

# 5. Verify installation
echo "[Step 5] Verifying installation..."
python -c "
import torch
import cv2
import numpy as np

print('PyTorch:', torch.__version__)
print('CUDA:', torch.cuda.is_available())
if torch.cuda.is_available():
    print('GPU:', torch.cuda.get_device_name(0))
print('OpenCV:', cv2.__version__)
print('NumPy:', np.__version__)
"

# 6. Test model forward pass
echo "[Step 6] Testing model forward pass..."
python main.py 2>&1 | head -20

# 7. Create directories
echo "[Step 7] Creating directories..."
mkdir -p /root/autodl-tmp/data
mkdir -p /root/autodl-tmp/checkpoints
mkdir -p /root/autodl-tmp/logs

# 8. Determine batch_size based on GPU memory
echo "[Step 8] Configuring training parameters..."
if [ -n "$GPU_MEM" ]; then
    if [ "$GPU_MEM" -ge 40000 ]; then
        BATCH_SIZE=16
    elif [ "$GPU_MEM" -ge 24000 ]; then
        BATCH_SIZE=8
    elif [ "$GPU_MEM" -ge 16000 ]; then
        BATCH_SIZE=4
    else
        BATCH_SIZE=2
    fi
else
    BATCH_SIZE=2
fi

echo "Recommended batch_size: $BATCH_SIZE"

# 9. Create quick training script
echo "[Step 9] Creating training script..."
cat > /root/autodl-tmp/Censor-MER/quick_train.sh << EOF
#!/bin/bash
# Quick training script for AutoDL

cd /root/autodl-tmp/Censor-MER

# Check if real data exists
if [ -d "/root/autodl-tmp/data/CASME_II/videos" ]; then
    echo "Training on CASME II..."
    python train.py \
        --dataset casme2 \
        --data_root /root/autodl-tmp/data/CASME_II \
        --epochs 50 \
        --batch_size ${BATCH_SIZE} \
        --lr 1e-4 \
        --output_dir /root/autodl-tmp/checkpoints
elif [ -d "/root/autodl-tmp/data/SAMM/videos" ]; then
    echo "Training on SAMM..."
    python train.py \
        --dataset samm \
        --data_root /root/autodl-tmp/data/SAMM \
        --epochs 50 \
        --batch_size ${BATCH_SIZE} \
        --lr 1e-4
else
    echo "No real dataset found, using synthetic data for testing..."
    python train.py --synthetic_data --epochs 10 --batch_size ${BATCH_SIZE}
fi
EOF

chmod +x /root/autodl-tmp/Censor-MER/quick_train.sh

# 10. Summary
echo ""
echo "============================================================"
echo "Setup Complete!"
echo "============================================================"
echo ""
echo "Directories:"
echo "  Project:    /root/autodl-tmp/Censor-MER"
echo "  Data:       /root/autodl-tmp/data"
echo "  Checkpoints: /root/autodl-tmp/checkpoints"
echo ""
echo "Next steps:"
echo "  1. Download dataset to /root/autodl-tmp/data/"
echo "     - CASME II: http://casme.psych.ac.cn/casme/c2"
echo "     - SAMM:     https://www.mmu.ac.uk"
echo ""
echo "  2. Run training:"
echo "     cd /root/autodl-tmp/Censor-MER"
echo "     ./quick_train.sh"
echo ""
echo "  OR use synthetic data for testing:"
echo "     python train.py --synthetic_data --epochs 10 --batch_size ${BATCH_SIZE}"
echo ""
echo "============================================================"