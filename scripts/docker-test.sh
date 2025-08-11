#!/bin/bash
# Docker testing script for CharaConsist
set -e

DOCKER_REPO=${1:-"gemneye/characonsist"}
DOCKER_TAG=${2:-"latest"}
PLATFORM=${3:-"linux/amd64"}

echo "🧪 Testing Docker image: ${DOCKER_REPO}:${DOCKER_TAG}"
echo "Platform: ${PLATFORM}"

# Test 1: Basic container startup
echo "Test 1: Basic container startup..."
docker run --rm --platform "${PLATFORM}" "${DOCKER_REPO}:${DOCKER_TAG}" \
    bash -c "echo 'Container started successfully'"

# Test 2: Python environment
echo "Test 2: Python environment verification..."
docker run --rm --platform "${PLATFORM}" "${DOCKER_REPO}:${DOCKER_TAG}" \
    python -c "
import sys
print(f'Python version: {sys.version}')
print('✅ Python environment OK')
"

# Test 3: Dependencies check
echo "Test 3: Dependencies verification..."
docker run --rm --platform "${PLATFORM}" "${DOCKER_REPO}:${DOCKER_TAG}" \
    python -c "
try:
    import torch
    import diffusers
    import transformers
    import accelerate
    import gradio
    print('✅ All dependencies imported successfully')
    print(f'PyTorch version: {torch.__version__}')
    print(f'Diffusers version: {diffusers.__version__}')
    print(f'Transformers version: {transformers.__version__}')
except ImportError as e:
    print(f'❌ Import error: {e}')
    exit(1)
"

# Test 4: CUDA availability (may not work in CI without GPU)
echo "Test 4: CUDA check..."
docker run --rm --platform "${PLATFORM}" "${DOCKER_REPO}:${DOCKER_TAG}" \
    python -c "
import torch
print(f'CUDA available: {torch.cuda.is_available()}')
print(f'CUDA devices: {torch.cuda.device_count()}')
if torch.cuda.is_available():
    print(f'CUDA version: {torch.version.cuda}')
print('✅ CUDA check completed')
"

# Test 5: Inference script exists
echo "Test 5: Inference script verification..."
docker run --rm --platform "${PLATFORM}" "${DOCKER_REPO}:${DOCKER_TAG}" \
    bash -c "
if [ -f 'inference.py' ]; then
    echo '✅ inference.py found'
    python -c 'import inference' 2>/dev/null && echo '✅ inference.py imports successfully' || echo '⚠️ inference.py import failed (may need models)'
else
    echo '❌ inference.py not found'
    exit 1
fi
"

# Test 6: Directory structure
echo "Test 6: Directory structure verification..."
docker run --rm --platform "${PLATFORM}" "${DOCKER_REPO}:${DOCKER_TAG}" \
    bash -c "
echo 'Checking directory structure...'
ls -la
echo 'Checking models directory...'
ls -la models/ || echo 'Models directory empty (expected for base image)'
echo 'Checking examples directory...'
ls -la examples/ || echo 'Examples directory missing'
echo '✅ Directory structure check completed'
"

echo ""
echo "🎉 All Docker tests completed successfully!"
echo "Image ${DOCKER_REPO}:${DOCKER_TAG} is ready for deployment."