# Production-ready CharaConsist Dockerfile
# Base: RunPod PyTorch with CUDA 12.8.1 support
FROM runpod/pytorch:2.8.0-py3.11-cuda12.8.1-cudnn-devel-ubuntu22.04

# Platform specification for AMD64/x86_64 only
LABEL architecture="amd64"
LABEL description="CharaConsist: Fine-Grained Consistent Character Generation Container"
LABEL version="1.0.0"

# Set environment variables for optimization
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1
ENV PIP_DISABLE_PIP_VERSION_CHECK=1
ENV CUDA_VISIBLE_DEVICES=0

# System dependencies for CharaConsist
RUN apt-get update && apt-get install -y --no-install-recommends \
    # Core system utilities
    wget \
    curl \
    git \
    unzip \
    # Image processing libraries
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    # Development tools
    build-essential \
    pkg-config \
    # Jupyter notebook support
    nodejs \
    npm \
    # Clean up to reduce image size
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* \
    && rm -rf /tmp/* \
    && rm -rf /var/tmp/*

# Upgrade pip and install essential Python packages
RUN python -m pip install --upgrade pip setuptools wheel

# Create application directory with proper permissions
RUN mkdir -p /workspace/characonsist \
    && chmod 755 /workspace/characonsist

# Set working directory
WORKDIR /workspace/characonsist

# Create directories for application structure
RUN mkdir -p \
    models \
    examples \
    results \
    point_and_mask \
    notebooks \
    logs \
    && chmod -R 755 .

# Copy project files (excluding models and large files)
COPY --chown=root:root requirements.txt ./
COPY --chown=root:root inference.py ./
COPY --chown=root:root gen-*.ipynb ./notebooks/
COPY --chown=root:root models/ ./models/
COPY --chown=root:root examples/ ./examples/
COPY --chown=root:root point_and_mask/ ./point_and_mask/

# Create bootstrap script for runtime setup
RUN cat > /workspace/bootstrap.sh << 'EOF'
#!/bin/bash
set -e

echo "=== CharaConsist Container Bootstrap ==="
echo "Starting runtime initialization..."

# Function to log with timestamp
log() {
    echo "[$(date +'%Y-%m-%d %H:%M:%S')] $1"
}

# Check CUDA availability
log "Checking CUDA availability..."
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA devices: {torch.cuda.device_count()}')" || {
    log "ERROR: CUDA check failed"
    exit 1
}

# Install CharaConsist dependencies
log "Installing CharaConsist requirements..."
cd /workspace/characonsist
pip install -r requirements.txt --no-cache-dir

# Additional dependencies for notebook support
log "Installing Jupyter and visualization dependencies..."
pip install --no-cache-dir \
    jupyter \
    ipywidgets \
    matplotlib \
    seaborn \
    Pillow

# Set up model directory permissions
log "Setting up model directory..."
mkdir -p /workspace/models
chmod 755 /workspace/models

# Create results directory with proper permissions
log "Setting up results directory..."
mkdir -p /workspace/characonsist/results
chmod 755 /workspace/characonsist/results

# Verify installation
log "Verifying installation..."
python -c "
import torch
import diffusers
import transformers
import accelerate
print('✓ All dependencies installed successfully')
print(f'✓ PyTorch version: {torch.__version__}')
print(f'✓ Diffusers version: {diffusers.__version__}')
print(f'✓ Transformers version: {transformers.__version__}')
print(f'✓ CUDA available: {torch.cuda.is_available()}')
" || {
    log "ERROR: Dependency verification failed"
    exit 1
}

log "Bootstrap completed successfully!"
log "Ready to run CharaConsist inference"
echo ""
echo "Usage examples:"
echo "  python inference.py --init_mode 1 --prompts_file examples/prompts-bg_fg.txt --model_path /workspace/models/FLUX.1-dev --out_dir results/test"
echo "  jupyter notebook --ip=0.0.0.0 --port=8888 --no-browser --allow-root"
EOF

# Make bootstrap script executable
RUN chmod +x /workspace/bootstrap.sh

# Create entrypoint script
RUN cat > /workspace/entrypoint.sh << 'EOF'
#!/bin/bash
set -e

# Run bootstrap if not already done
if [ ! -f "/workspace/.bootstrap_done" ]; then
    echo "Running first-time setup..."
    /workspace/bootstrap.sh
    touch /workspace/.bootstrap_done
fi

# Execute the command passed to docker run
exec "$@"
EOF

RUN chmod +x /workspace/entrypoint.sh

# Set up environment for efficient inference
ENV TORCH_CUDA_ARCH_LIST="7.0 7.5 8.0 8.6 8.9 9.0"
ENV FORCE_CUDA="1"
ENV MAX_JOBS="4"

# Expose port for Jupyter notebooks
EXPOSE 8888

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=60s --retries=3 \
    CMD python -c "import torch; assert torch.cuda.is_available()" || exit 1

# Set entrypoint
ENTRYPOINT ["/workspace/entrypoint.sh"]

# Default command
CMD ["bash"]