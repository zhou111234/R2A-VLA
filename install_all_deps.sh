#!/bin/bash
# Complete dependency installation script for R2A → LIBERO
# Uses Python 3.11 with system PyTorch
# Updated to avoid Rust compilation issues

set -e

echo "=========================================="
echo "R2A to LIBERO - Complete Installation"
echo "(Python 3.11 + System PyTorch)"
echo "=========================================="

cd /mnt/nas/R2A-VLA

# Add conda to PATH
export PATH="/opt/conda/bin:$PATH"

# Step 0: Create virtual environment with Python 3.11
echo ""
echo "Step 0: Creating virtual environment..."
if [ -d ".venv_lerobot" ]; then
    echo "Removing old virtual environment..."
    rm -rf .venv_lerobot
fi

uv venv --python 3.11 .venv_lerobot
echo "✓ Virtual environment created (Python 3.11)"

# Step 1: Activate virtual environment
echo ""
echo "Step 1: Activating virtual environment..."
source .venv_lerobot/bin/activate
echo "✓ Activated .venv_lerobot"

# Step 2: Initialize git submodules
echo ""
echo "Step 2: Initializing git submodules..."
git submodule update --init third_party/libero
echo "✓ Git submodules initialized"

# Step 3: Install Python dependencies
# Using newer transformers to avoid Rust compilation
echo ""
echo "Step 3: Installing Python dependencies..."
uv pip install \
    "hydra-core==1.2.0" \
    "numpy>=1.23.0" \
    "wandb==0.13.1" \
    "easydict==1.9" \
    "transformers>=4.30.0" \
    "opencv-python==4.6.0.66" \
    "einops==0.4.1" \
    "thop==0.1.1-2209072238" \
    "robosuite" \
    "bddl==1.0.1" \
    "future==0.18.2" \
    "matplotlib==3.5.3" \
    "cloudpickle==2.1.0" \
    "gym==0.25.2" \
    "robomimic" \
    "imageio[ffmpeg]" \
    "tqdm" \
    "tyro" \
    "PyYaml"
echo "✓ Python dependencies installed"

# Step 4: Install projects
echo ""
echo "Step 4: Installing projects..."
uv pip install -e packages/openpi-client -e third_party/libero --no-deps -e . -e .venv_lerobot/lerobot --no-deps
echo "✓ Projects installed"

# Step 5: Set PYTHONPATH
echo ""
echo "Step 5: Setting up environment..."
export PYTHONPATH=$PYTHONPATH:$PWD/third_party/libero
echo "✓ PYTHONPATH configured"

# Step 6: Verify installation
echo ""
echo "Step 6: Verifying installation..."
echo "This will take 60-90 seconds for first import..."

python -c "
import sys
sys.path.insert(0, '/mnt/nas/R2A-VLA/third_party/libero')

print('Testing torch...')
import torch
print(f'  PyTorch: {torch.__version__}')
print(f'  CUDA: {torch.version.cuda}')

print('Testing openpi...')
from openpi.training import config
print('  ✓ openpi OK')

print('Testing libero...')
from libero.libero import benchmark
print('  ✓ libero OK')

print('Testing libero_r2a_policy...')
from openpi.policies.libero_r2a_policy import LiberoR2AInputs
print('  ✓ libero_r2a_policy OK')

print('')
print('✓ ALL DEPENDENCIES VERIFIED!')
"

echo ""
echo "=========================================="
echo "Installation Complete! ✓"
echo "=========================================="
echo ""
echo "Next steps:"
echo ""
echo "Terminal 1 (Policy Server):"
echo "  cd /mnt/nas/R2A-VLA"
echo "  source .venv_lerobot/bin/activate"
echo "  export PYTHONPATH=\$PYTHONPATH:\$PWD/third_party/libero"
echo "  uv run scripts/serve_policy.py \\"
echo "      --env LIBERO policy:checkpoint \\"
echo "      --policy.config acot_r2a_libero_eval_v2 \\"
echo "      --policy.dir ./checkpoints/acot_r2a_mymodal_temporal_noise/r2a_mymodal_v1/4000"
echo ""
echo "Terminal 2 (LIBERO Client):"
echo "  cd /mnt/nas/R2A-VLA"
echo "  source .venv_lerobot/bin/activate"
echo "  export PYTHONPATH=\$PYTHONPATH:\$PWD/third_party/libero"
echo "  python examples/libero/main.py --task-suite-name libero_spatial"
echo ""
