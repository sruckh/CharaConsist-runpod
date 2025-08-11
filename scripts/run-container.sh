#!/bin/bash
# Container execution script for CharaConsist
set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Default values
IMAGE_NAME="characonsist:latest"
CONTAINER_NAME="characonsist-inference"
MODEL_PATH=""
RESULTS_DIR="./results"
JUPYTER_PORT=8888
INTERACTIVE=true
DETACHED=false

# Function to print colored output
print_status() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_header() {
    echo -e "${BLUE}================================${NC}"
    echo -e "${BLUE}  CharaConsist Container Runner${NC}"
    echo -e "${BLUE}================================${NC}"
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -m|--model-path)
            MODEL_PATH="$2"
            shift 2
            ;;
        -r|--results-dir)
            RESULTS_DIR="$2"
            shift 2
            ;;
        -p|--jupyter-port)
            JUPYTER_PORT="$2"
            shift 2
            ;;
        -d|--detached)
            DETACHED=true
            INTERACTIVE=false
            shift
            ;;
        --no-interactive)
            INTERACTIVE=false
            shift
            ;;
        -h|--help)
            echo "Usage: $0 [OPTIONS] [COMMAND]"
            echo ""
            echo "Options:"
            echo "  -m, --model-path PATH    Path to FLUX.1-dev model directory"
            echo "  -r, --results-dir PATH   Results output directory (default: ./results)"
            echo "  -p, --jupyter-port PORT  Jupyter notebook port (default: 8888)"
            echo "  -d, --detached           Run in detached mode"
            echo "      --no-interactive     Run without interactive mode"
            echo "  -h, --help               Show this help message"
            echo ""
            echo "Examples:"
            echo "  $0 --model-path /path/to/FLUX.1-dev"
            echo "  $0 --detached --model-path /models/FLUX.1-dev jupyter notebook"
            echo "  $0 python inference.py --init_mode 1 --prompts_file examples/prompts-bg_fg.txt"
            exit 0
            ;;
        *)
            break
            ;;
    esac
done

print_header

# Check if Docker is running
if ! docker info > /dev/null 2>&1; then
    print_error "Docker is not running or not accessible"
    exit 1
fi

# Check if NVIDIA Docker is available
if ! docker run --rm --gpus all nvidia/cuda:11.0.3-base-ubuntu20.04 nvidia-smi > /dev/null 2>&1; then
    print_warning "NVIDIA Docker support may not be available"
    print_warning "GPU acceleration might not work properly"
fi

# Create results directory if it doesn't exist
mkdir -p "$RESULTS_DIR"
print_status "Results will be saved to: $RESULTS_DIR"

# Prepare Docker run command
DOCKER_CMD="docker run --rm"

# GPU support
DOCKER_CMD="$DOCKER_CMD --gpus all"

# Container name
DOCKER_CMD="$DOCKER_CMD --name $CONTAINER_NAME"

# Interactive or detached mode
if [[ "$INTERACTIVE" == "true" ]]; then
    DOCKER_CMD="$DOCKER_CMD -it"
elif [[ "$DETACHED" == "true" ]]; then
    DOCKER_CMD="$DOCKER_CMD -d"
fi

# Port mapping
DOCKER_CMD="$DOCKER_CMD -p $JUPYTER_PORT:8888"

# Volume mounts
DOCKER_CMD="$DOCKER_CMD -v $(pwd)/$RESULTS_DIR:/workspace/characonsist/results"

# Model path mounting
if [[ -n "$MODEL_PATH" ]]; then
    if [[ ! -d "$MODEL_PATH" ]]; then
        print_error "Model path does not exist: $MODEL_PATH"
        exit 1
    fi
    DOCKER_CMD="$DOCKER_CMD -v $MODEL_PATH:/workspace/models/FLUX.1-dev:ro"
    print_status "Mounting model from: $MODEL_PATH"
else
    print_warning "No model path specified. You'll need to provide model path at runtime."
fi

# Environment variables
DOCKER_CMD="$DOCKER_CMD -e CUDA_VISIBLE_DEVICES=0"
DOCKER_CMD="$DOCKER_CMD -e PYTHONPATH=/workspace/characonsist"

# Resource limits
DOCKER_CMD="$DOCKER_CMD --shm-size=8g"

# Image
DOCKER_CMD="$DOCKER_CMD $IMAGE_NAME"

# Add command if provided
if [[ $# -gt 0 ]]; then
    DOCKER_CMD="$DOCKER_CMD $@"
fi

print_status "Starting CharaConsist container..."
print_status "Command: $DOCKER_CMD"
echo ""

# Execute the command
eval $DOCKER_CMD

if [[ $? -eq 0 ]]; then
    if [[ "$DETACHED" == "true" ]]; then
        print_status "Container started in detached mode"
        print_status "View logs: docker logs -f $CONTAINER_NAME"
        print_status "Stop container: docker stop $CONTAINER_NAME"
        if [[ -n "$MODEL_PATH" ]]; then
            echo ""
            print_status "Jupyter notebook should be available at: http://localhost:$JUPYTER_PORT"
        fi
    else
        print_status "Container execution completed"
    fi
else
    print_error "Container execution failed"
    exit 1
fi