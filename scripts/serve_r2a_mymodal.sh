#!/bin/bash
set -euo pipefail

##############################################################################
# Policy server for Genie Sim evaluation / competition submission
#
# Launches the OpenPI policy server with our trained model checkpoint.
# Genie Sim connects via WebSocket to get action predictions.
#
# Usage:
#   bash scripts/serve_r2a_mymodal.sh [CHECKPOINT_STEP]
#
# The server maintains a frame buffer for temporal encoding at inference.
##############################################################################

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ACOT_DIR="${ACOT_DIR:-$(dirname "$SCRIPT_DIR")}"
CONFIG_NAME="acot_r2a_mymodal_temporal_noise"
EXP_NAME="${EXP_NAME:-r2a_mymodal_v1}"
STEP="${1:-4000}"
PORT="${PORT:-8999}"

CKPT_DIR="$ACOT_DIR/checkpoints/$CONFIG_NAME/$EXP_NAME"

cd "$ACOT_DIR"

echo "============================================"
echo "  Policy Server: $CONFIG_NAME"
echo "  Checkpoint: step $STEP"
echo "  Port: $PORT"
echo "============================================"

export XLA_PYTHON_CLIENT_MEM_FRACTION=0.85
export JAX_PLATFORMS=cuda

CKPT_STEP_DIR="$CKPT_DIR/$STEP"

echo "Checkpoint dir: $CKPT_STEP_DIR"
echo ""
echo "Starting policy server on port $PORT..."
echo ""

python scripts/serve_policy.py \
    --port="$PORT" \
    policy:checkpoint \
    --policy.config="$CONFIG_NAME" \
    --policy.dir="$CKPT_STEP_DIR"
