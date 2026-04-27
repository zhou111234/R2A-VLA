#!/bin/bash
set -euo pipefail

##############################################################################
# Training script for ACoT-VLA + MyModal (Temporal + Noise Expert)
# on the Reasoning2Action competition dataset.
#
# Usage:
#   bash scripts/train_r2a_mymodal.sh [--resume]
#
# Environment:
#   Single A100 80GB GPU on Alibaba Cloud DSW
##############################################################################

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ACOT_DIR="${ACOT_DIR:-$(dirname "$SCRIPT_DIR")}"
CONFIG_NAME="acot_r2a_mymodal_temporal_noise"
EXP_NAME="${EXP_NAME:-r2a_mymodal_v1}"

# Parse arguments
RESUME_FLAG=""
for arg in "$@"; do
    case $arg in
        --resume) RESUME_FLAG="--resume True" ;;
    esac
done

cd "$ACOT_DIR"

NORM_STATS_DIR="./assets/${CONFIG_NAME}/norm_stats"

echo "============================================"
echo "  Training: $CONFIG_NAME"
echo "  Experiment: $EXP_NAME"
echo "  Directory: $ACOT_DIR"
echo "============================================"

# JAX settings for single GPU training with max memory efficiency
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.90
export JAX_PLATFORMS=cuda
export TF_CPP_MIN_LOG_LEVEL=2

# Disable wandb if not configured
if [ -z "${WANDB_API_KEY:-}" ]; then
    echo "WANDB_API_KEY not set, disabling wandb logging."
    export WANDB_MODE=offline
fi

# ---- Step 1: Compute normalization statistics if not yet computed ----
if [ ! -f "${NORM_STATS_DIR}/norm_stats.json" ]; then
    echo ""
    echo "[Step 1/2] Computing normalization statistics..."
    echo "  This samples ~10% of the data and may take a while on the first run."
    echo ""
    python scripts/compute_norm_stats.py \
        --config-name "$CONFIG_NAME"

    mkdir -p "$NORM_STATS_DIR"
    mv ./norm_stats.json "$NORM_STATS_DIR/norm_stats.json"
    echo "  Norm stats saved to: $NORM_STATS_DIR/norm_stats.json"
else
    echo ""
    echo "[Step 1/2] Norm stats already exist at $NORM_STATS_DIR, skipping."
fi

# ---- Step 2: Train ----
echo ""
echo "[Step 2/2] Starting training..."
echo ""

python scripts/train.py \
    "$CONFIG_NAME" \
    --exp-name="$EXP_NAME" \
    $RESUME_FLAG
