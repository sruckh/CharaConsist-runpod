# CharaConsist Docker Deployment Guide

## Overview

This deployment setup provides a production-ready Docker container for CharaConsist, optimized for NVIDIA GPU environments with CUDA 12.8.1 support.

## Prerequisites

- Docker Engine 20.10+ with NVIDIA Container Toolkit
- NVIDIA GPU with CUDA compute capability 7.0+
- Minimum 26GB GPU memory (with CPU offload) or 37GB (single GPU)
- AMD64/x86_64 architecture

## Quick Start

### 1. Build the Container

```bash
# Build with default settings
./scripts/docker-build.sh

# Build with custom tag and no cache
./scripts/docker-build.sh --tag v1.0 --no-cache
```

### 2. Prepare Model Directory

Download the FLUX.1-dev model from HuggingFace:

```bash
# Create models directory
mkdir -p ./models/FLUX.1-dev

# Download model (requires HuggingFace CLI)
huggingface-cli download black-forest-labs/FLUX.1-dev --local-dir ./models/FLUX.1-dev
```

### 3. Run the Container

```bash
# Interactive mode with model mounting
./scripts/run-container.sh --model-path ./models/FLUX.1-dev

# Run specific inference command
./scripts/run-container.sh --model-path ./models/FLUX.1-dev \
    python inference.py --init_mode 1 --prompts_file examples/prompts-bg_fg.txt \
    --model_path /workspace/models/FLUX.1-dev --out_dir results/test

# Start Jupyter notebook in detached mode
./scripts/run-container.sh --detached --model-path ./models/FLUX.1-dev \
    jupyter notebook --ip=0.0.0.0 --port=8888 --no-browser --allow-root
```

## Container Architecture

### Directory Structure

```
/workspace/characonsist/
├── models/                 # CharaConsist model files
├── examples/              # Prompt examples
├── results/               # Output directory (mounted)
├── notebooks/             # Jupyter notebooks
├── point_and_mask/        # Point matching utilities
└── inference.py           # Main inference script
```

### Memory Optimization Modes

The container supports CharaConsist's memory optimization modes:

| Mode | Description | GPU Memory | Configuration |
|------|-------------|------------|---------------|
| 0    | Single GPU | 37GB | Standard mode |
| 1    | CPU Offload | 26GB | Model CPU offload |
| 2    | Multi-GPU | ≤20GB | Memory distributed |
| 3    | Sequential | 3GB | Sequential CPU offload |

### Runtime Bootstrap

The container includes an intelligent bootstrap system:

1. **First Run**: Installs Python dependencies and sets up environment
2. **Subsequent Runs**: Skips setup for faster container starts
3. **Health Checks**: Validates CUDA and dependency availability
4. **Cache Management**: Optimizes HuggingFace and PyTorch cache

## Usage Examples

### Batch Inference

```bash
# Fixed background character generation
docker run --gpus all --rm \
    -v $(pwd)/models/FLUX.1-dev:/workspace/models/FLUX.1-dev:ro \
    -v $(pwd)/results:/workspace/characonsist/results \
    characonsist:latest \
    python inference.py --init_mode 1 --prompts_file examples/prompts-bg_fg.txt \
    --model_path /workspace/models/FLUX.1-dev --out_dir results/bg_fg \
    --use_interpolate --save_mask --share_bg

# Variable background character generation
docker run --gpus all --rm \
    -v $(pwd)/models/FLUX.1-dev:/workspace/models/FLUX.1-dev:ro \
    -v $(pwd)/results:/workspace/characonsist/results \
    characonsist:latest \
    python inference.py --init_mode 1 --prompts_file examples/prompts-fg_only.txt \
    --model_path /workspace/models/FLUX.1-dev --out_dir results/fg_only \
    --use_interpolate --save_mask
```

### Jupyter Notebook Development

```bash
# Start notebook server
docker run --gpus all -d \
    -p 8888:8888 \
    -v $(pwd)/models/FLUX.1-dev:/workspace/models/FLUX.1-dev:ro \
    -v $(pwd)/results:/workspace/characonsist/results \
    --name characonsist-notebook \
    characonsist:latest \
    jupyter notebook --ip=0.0.0.0 --port=8888 --no-browser --allow-root

# Access at http://localhost:8888
```

## Docker Compose Deployment

For production environments, use the provided Docker Compose configuration:

```bash
# Start services
docker-compose up -d

# View logs
docker-compose logs -f

# Stop services
docker-compose down
```

### Configuration Options

Edit `docker-compose.yml` for your environment:

```yaml
environment:
  - CUDA_VISIBLE_DEVICES=0,1  # Specify GPU devices
  
volumes:
  - /path/to/your/models:/workspace/models:ro
  - /path/to/your/results:/workspace/characonsist/results:rw
```

## Performance Optimization

### GPU Memory Management

1. **Single GPU (≥37GB)**: Use `init_mode 0` for best performance
2. **Single GPU (≥26GB)**: Use `init_mode 1` with CPU offload
3. **Multiple GPUs**: Use `init_mode 2` for distributed memory
4. **Low Memory (≥3GB)**: Use `init_mode 3` with sequential offload

### Container Optimization

- **SHM Size**: Configured to 8GB for efficient data loading
- **Cache Volumes**: Persistent HuggingFace and PyTorch caches
- **Build Cache**: Multi-stage build with layer caching
- **Base Image**: Optimized RunPod PyTorch image

## Troubleshooting

### Common Issues

1. **CUDA Not Available**
   ```bash
   # Verify NVIDIA Docker setup
   docker run --rm --gpus all nvidia/cuda:11.0.3-base-ubuntu20.04 nvidia-smi
   ```

2. **Out of Memory Errors**
   - Switch to higher `init_mode` (1, 2, or 3)
   - Reduce batch size or image resolution
   - Monitor GPU memory: `nvidia-smi -l 1`

3. **Model Loading Issues**
   - Verify model path mounting: `docker run --rm -it characonsist:latest ls -la /workspace/models/`
   - Check model integrity: Ensure all FLUX.1-dev files are present

4. **Permission Errors**
   - Ensure results directory is writable: `chmod 755 ./results`
   - Check volume mount permissions

### Health Monitoring

```bash
# Check container health
docker inspect characonsist-inference --format='{{.State.Health.Status}}'

# View container logs
docker logs characonsist-inference

# Monitor resource usage
docker stats characonsist-inference
```

## Security Considerations

- Container runs with root privileges (required for GPU access)
- Model files mounted read-only for security
- Results directory isolated from host system
- No network access required for inference (offline capable)

## Maintenance

### Updates

```bash
# Update base image
docker pull runpod/pytorch:2.8.0-py3.11-cuda12.8.1-cudnn-devel-ubuntu22.04

# Rebuild container
./scripts/docker-build.sh --no-cache
```

### Cleanup

```bash
# Remove containers
docker container prune

# Remove unused images
docker image prune

# Remove volumes (careful!)
docker volume prune
```