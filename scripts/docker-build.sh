#!/bin/bash
# Docker build script for CharaConsist
set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Default values
IMAGE_NAME="characonsist"
TAG="latest"
PLATFORM="linux/amd64"
NO_CACHE=false
PUSH=false

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

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -t|--tag)
            TAG="$2"
            shift 2
            ;;
        --no-cache)
            NO_CACHE=true
            shift
            ;;
        --push)
            PUSH=true
            shift
            ;;
        -h|--help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  -t, --tag TAG     Docker image tag (default: latest)"
            echo "      --no-cache    Build without cache"
            echo "      --push        Push image after build"
            echo "  -h, --help        Show this help message"
            exit 0
            ;;
        *)
            print_error "Unknown option: $1"
            exit 1
            ;;
    esac
done

print_status "Building CharaConsist Docker image..."
print_status "Image: ${IMAGE_NAME}:${TAG}"
print_status "Platform: ${PLATFORM}"

# Check if Dockerfile exists
if [[ ! -f "Dockerfile" ]]; then
    print_error "Dockerfile not found in current directory"
    exit 1
fi

# Build Docker image
BUILD_ARGS="--platform ${PLATFORM} -t ${IMAGE_NAME}:${TAG}"

if [[ "$NO_CACHE" == "true" ]]; then
    BUILD_ARGS="$BUILD_ARGS --no-cache"
    print_warning "Building without cache"
fi

print_status "Starting Docker build..."
docker build $BUILD_ARGS .

if [[ $? -eq 0 ]]; then
    print_status "Docker build completed successfully!"
    
    # Show image info
    IMAGE_SIZE=$(docker images ${IMAGE_NAME}:${TAG} --format "table {{.Size}}" | tail -n 1)
    print_status "Image size: $IMAGE_SIZE"
    
    # Push if requested
    if [[ "$PUSH" == "true" ]]; then
        print_status "Pushing image to registry..."
        docker push ${IMAGE_NAME}:${TAG}
        if [[ $? -eq 0 ]]; then
            print_status "Image pushed successfully!"
        else
            print_error "Failed to push image"
            exit 1
        fi
    fi
    
    print_status "Build process completed!"
    echo ""
    echo "To run the container:"
    echo "  docker run --gpus all -it --rm ${IMAGE_NAME}:${TAG}"
    echo ""
    echo "To run with docker-compose:"
    echo "  docker-compose up -d"
    
else
    print_error "Docker build failed!"
    exit 1
fi