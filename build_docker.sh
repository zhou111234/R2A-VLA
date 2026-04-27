#!/bin/bash
# Build script for ACoT-VLA R2A Docker image
# Copies the pre-built virtual environment from /mnt/nas/.venv_lerobot

set -e

VENV_SOURCE="/mnt/nas/.venv_lerobot"
VENV_DEST="./.venv_lerobot"
IMAGE_NAME="acot-r2a-mymodal"
DOCKERFILE="Dockerfile.r2a"

echo "=== ACoT-VLA R2A Docker Build Script ==="
echo ""

# Check if source venv exists
if [ ! -d "$VENV_SOURCE" ]; then
    echo "Error: Virtual environment not found at $VENV_SOURCE"
    exit 1
fi

echo "1. Copying virtual environment from $VENV_SOURCE..."
rm -rf "$VENV_DEST"
# Use cp -a to preserve all attributes and symlinks
cp -a "$VENV_SOURCE" "$VENV_DEST"
echo "   Done!"

echo ""
echo "2. Building Docker image: $IMAGE_NAME..."
echo "   Using: $DOCKERFILE"
echo ""

# Build docker image
docker build -t "$IMAGE_NAME" -f "$DOCKERFILE" .

echo ""
echo "=== Build Complete ==="
echo "Image: $IMAGE_NAME"
echo ""
echo "Next steps:"
echo "  docker tag $IMAGE_NAME sim-icra-registry.cn-beijing.cr.aliyuncs.com/aol/$IMAGE_NAME"
echo "  docker push sim-icra-registry.cn-beijing.cr.aliyuncs.com/aol/$IMAGE_NAME"
echo ""
echo "Or run locally:"
echo "  docker run --gpus all --rm -p 8999:8999 $IMAGE_NAME"
