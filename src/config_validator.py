#!/usr/bin/env python3
"""
Configuration validator for CharaConsist Gradio Interface
Helps users validate their setup before running the interface
"""

import os
import sys
import torch
import subprocess
import pkg_resources
from pathlib import Path

def check_cuda_availability():
    """Check CUDA availability and GPU information"""
    print("🔍 Checking CUDA availability...")
    
    if not torch.cuda.is_available():
        print("❌ CUDA is not available")
        return False
    
    print(f"✅ CUDA is available (version: {torch.version.cuda})")
    print(f"🎮 Available GPUs: {torch.cuda.device_count()}")
    
    for i in range(torch.cuda.device_count()):
        gpu_name = torch.cuda.get_device_name(i)
        gpu_memory = torch.cuda.get_device_properties(i).total_memory / 1024**3
        print(f"   GPU {i}: {gpu_name} ({gpu_memory:.1f} GB)")
    
    return True

def check_dependencies():
    """Check if all required dependencies are installed"""
    print("📦 Checking dependencies...")
    
    required_packages = [
        'torch',
        'torchvision', 
        'transformers',
        'diffusers',
        'accelerate',
        'gradio',
        'pillow',
        'numpy'
    ]
    
    missing_packages = []
    installed_packages = []
    
    for package in required_packages:
        try:
            pkg_resources.get_distribution(package)
            installed_packages.append(package)
            print(f"✅ {package}")
        except pkg_resources.DistributionNotFound:
            missing_packages.append(package)
            print(f"❌ {package} - NOT INSTALLED")
    
    if missing_packages:
        print(f"\n📋 Missing packages: {', '.join(missing_packages)}")
        print("Install with: pip install -r requirements.txt")
        return False
    
    print(f"✅ All {len(installed_packages)} required packages are installed")
    return True

def check_model_path(model_path):
    """Validate model path and structure"""
    print(f"🔍 Checking model path: {model_path}")
    
    if not os.path.exists(model_path):
        print(f"❌ Model path does not exist: {model_path}")
        return False
    
    # Check for essential model components
    required_files = [
        "transformer",
        "text_encoder_2", 
        "scheduler",
        "vae"
    ]
    
    missing_components = []
    for component in required_files:
        component_path = os.path.join(model_path, component)
        if os.path.exists(component_path):
            print(f"✅ {component}")
        else:
            missing_components.append(component)
            print(f"❌ {component} - NOT FOUND")
    
    if missing_components:
        print(f"\n📋 Missing model components: {', '.join(missing_components)}")
        print("Please ensure you have downloaded the complete FLUX.1-dev model")
        return False
    
    print("✅ Model structure appears valid")
    return True

def check_example_files():
    """Check if example prompt files exist"""
    print("📄 Checking example files...")
    
    example_files = [
        "examples/prompts-bg_fg.txt",
        "examples/prompts-fg_only.txt"
    ]
    
    existing_files = []
    missing_files = []
    
    for file_path in example_files:
        if os.path.exists(file_path):
            existing_files.append(file_path)
            print(f"✅ {file_path}")
        else:
            missing_files.append(file_path)
            print(f"⚠️ {file_path} - NOT FOUND (optional)")
    
    if existing_files:
        print(f"✅ {len(existing_files)} example files found")
    
    return len(existing_files) > 0

def estimate_memory_requirements(height, width, init_mode):
    """Estimate memory requirements for given configuration"""
    print("💾 Estimating memory requirements...")
    
    # Base memory estimates in GB (rough estimates)
    base_model_memory = 12.0  # FLUX.1-dev base memory
    
    # Image generation memory (depends on resolution)
    pixels = height * width
    image_memory = pixels * 4 * 1e-6  # Rough estimate in GB
    
    mode_multipliers = {
        0: 1.0,    # Single GPU - all memory on one GPU
        1: 0.7,    # CPU offload - reduced GPU memory
        2: 0.6,    # Multi-GPU - distributed memory  
        3: 0.5     # Sequential offload - most memory efficient
    }
    
    total_memory = (base_model_memory + image_memory) * mode_multipliers[init_mode]
    
    print(f"📊 Memory estimate for {height}x{width} with mode {init_mode}:")
    print(f"   Base model: ~{base_model_memory:.1f} GB")
    print(f"   Image generation: ~{image_memory:.1f} GB") 
    print(f"   Total estimate: ~{total_memory:.1f} GB")
    
    # Check against available GPU memory
    if torch.cuda.is_available():
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"   Available GPU 0 memory: {gpu_memory:.1f} GB")
        
        if total_memory > gpu_memory * 0.9:  # Leave 10% buffer
            print("⚠️ WARNING: Estimated memory usage exceeds available GPU memory")
            print("   Consider:")
            print("   - Using CPU offloading modes (1 or 3)")
            print("   - Reducing image dimensions")
            print("   - Using multi-GPU mode if available")
            return False
        else:
            print("✅ Memory requirements appear feasible")
            return True
    
    return True

def recommend_configuration():
    """Recommend optimal configuration based on hardware"""
    print("💡 Configuration recommendations:")
    
    if not torch.cuda.is_available():
        print("❌ No CUDA GPUs detected - CharaConsist requires GPU acceleration")
        return
    
    gpu_count = torch.cuda.device_count()
    gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
    
    print(f"🎮 Hardware: {gpu_count} GPU(s), {gpu_memory:.1f} GB memory")
    
    if gpu_memory >= 24:
        print("✅ Recommended: Mode 0 (Single GPU) - Best performance")
        print("✅ Image size: 1024x1024 or higher")
    elif gpu_memory >= 16:
        print("✅ Recommended: Mode 1 (Single GPU + CPU Offload)")
        print("✅ Image size: 1024x1024")
    elif gpu_memory >= 12:
        print("⚠️ Recommended: Mode 3 (Sequential CPU Offload)")
        print("⚠️ Image size: 768x768 or 1024x1024")
    else:
        print("❌ WARNING: Limited GPU memory detected")
        print("⚠️ Try: Mode 3 with 512x512 images")
        print("💡 Consider upgrading GPU for better performance")
    
    if gpu_count >= 2:
        print(f"💡 Alternative: Mode 2 (Multi-GPU) - Distribute across {gpu_count} GPUs")

def validate_configuration(model_path=None, init_mode=0, height=1024, width=1024):
    """Run comprehensive validation"""
    print("🔧 CharaConsist Configuration Validator")
    print("=" * 50)
    
    all_checks_passed = True
    
    # Check CUDA
    if not check_cuda_availability():
        all_checks_passed = False
    
    print()
    
    # Check dependencies
    if not check_dependencies():
        all_checks_passed = False
    
    print()
    
    # Check model if path provided
    if model_path:
        if not check_model_path(model_path):
            all_checks_passed = False
        print()
    
    # Check example files
    check_example_files()
    print()
    
    # Estimate memory requirements
    if torch.cuda.is_available():
        estimate_memory_requirements(height, width, init_mode)
        print()
    
    # Provide recommendations
    recommend_configuration()
    print()
    
    # Final summary
    if all_checks_passed:
        print("🎉 Configuration validation passed!")
        print("✅ You should be able to run CharaConsist successfully")
    else:
        print("⚠️ Some issues were detected")
        print("📋 Please address the issues above before running CharaConsist")
    
    return all_checks_passed

def main():
    """Main function for command line usage"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Validate CharaConsist configuration")
    parser.add_argument("--model-path", type=str, help="Path to FLUX.1-dev model")
    parser.add_argument("--init-mode", type=int, choices=[0,1,2,3], default=0, help="Initialization mode")
    parser.add_argument("--height", type=int, default=1024, help="Image height")
    parser.add_argument("--width", type=int, default=1024, help="Image width")
    
    args = parser.parse_args()
    
    success = validate_configuration(
        model_path=args.model_path,
        init_mode=args.init_mode, 
        height=args.height,
        width=args.width
    )
    
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()