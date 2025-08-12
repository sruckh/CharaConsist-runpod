#!/bin/bash
set -e

echo "=== CharaConsist Container Bootstrap ==="
log "Starting runtime initialization..."

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
    Pillow \
    gradio

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
import gradio
print('✓ All dependencies installed successfully')
print(f'✓ PyTorch version: {torch.__version__}')
print(f'✓ Diffusers version: {diffusers.__version__}')
print(f'✓ Transformers version: {transformers.__version__}')
print(f'✓ Gradio version: {gradio.__version__}')
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
echo "  python /workspace/characonsist/src/gradio_interface.py"
