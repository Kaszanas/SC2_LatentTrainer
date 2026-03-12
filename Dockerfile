# Use NVIDIA CUDA base image with Python
FROM nvidia/cuda:12.8.0-cudnn-runtime-ubuntu22.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3.11 \
    python3.11-dev \
    python3-pip \
    git \
    curl \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Set Python 3.11 as default
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1 && \
    update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1

# Install uv package manager and verify installation
RUN curl -LsSf https://astral.sh/uv/install.sh | sh && \
    /root/.local/bin/uv --version

# Add uv to PATH
ENV PATH="/root/.local/bin:${PATH}"

# Set working directory
WORKDIR /workspace

# Copy project files
COPY pyproject.toml ./
COPY README.md ./
COPY setup.cfg* ./
COPY uv.lock* ./

# Copy source code
COPY src/ ./src/

# Copy data and models directories if they exist
COPY data* ./data/
COPY models* ./models/

# Create a virtual environment with uv
RUN uv venv /opt/venv
ENV PATH="/opt/venv/bin:${PATH}"
ENV VIRTUAL_ENV="/opt/venv"

# Configure PyTorch CUDA index for uv
ENV UV_EXTRA_INDEX_URL="https://download.pytorch.org/whl/cu128"

# Sync dependencies from pyproject.toml using uv (includes PyTorch with CUDA)
RUN uv sync --no-dev || uv pip install -e .

# Create necessary directories
RUN mkdir -p /workspace/output/tensorboard_logs \
    /workspace/output/checkpoints \
    /workspace/output/predictions \
    /workspace/data/download \
    /workspace/data/unpack

# Expose ports for TensorBoard, Optuna Dashboard, and MLFlow
EXPOSE 5000 6006 8080

# Set the entrypoint
CMD ["/bin/bash"]