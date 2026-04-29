# Use public NVIDIA CUDA base image
FROM nvidia/cuda:12.2.2-cudnn8-runtime-ubuntu22.04@sha256:2d913b09e6be8387e1a10976933642c73c840c0b735f0bf3c28d97fc9bc422e0

# Install uv for Python package management
COPY --from=ghcr.io/astral-sh/uv:0.5.1 /uv /uvx /bin/

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3.11 \
    python3.11-venv \
    python3-pip \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy project files
COPY . .

# Create virtual environment
ENV UV_PROJECT_ENVIRONMENT=/.venv
RUN uv venv --python 3.11 $UV_PROJECT_ENVIRONMENT

# Install Python dependencies
RUN uv pip sync requirements.docker.txt 2>/dev/null || \
    uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install project in editable mode
RUN uv pip install -e packages/openpi-client -e .

# Run the policy server
CMD ["/bin/bash", "-c", "source /.venv/bin/activate && uv run scripts/serve_policy.py $SERVER_ARGS"]
