# CharaConsist Gradio Interface

A comprehensive web-based interface for CharaConsist that supports all functionality from `inference.py` and the three generation modes demonstrated in the Jupyter notebooks.

## 🚀 Quick Start

### Prerequisites

1. **Hardware Requirements**:
   - NVIDIA GPU with CUDA support (recommended: 24GB+ VRAM)
   - For multi-GPU setup: 2+ NVIDIA GPUs
   - Sufficient system RAM (16GB+ recommended)

2. **Software Requirements**:
   - Python 3.8+
   - CUDA 11.8+ or 12.0+
   - FLUX.1-dev model files

### Installation

1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Launch the Interface**:
   ```bash
   python src/launch_gradio.py
   ```

3. **Access the Web Interface**:
   - Open your browser and go to `http://localhost:7860`
   - The interface will be available at the specified address

## 📋 Interface Overview

### Model Configuration Section

**Model Path**: Enter the path to your FLUX.1-dev model directory.

**Initialization Modes**:
- **Mode 0 - Single GPU**: Loads entire model on one GPU (fastest, requires most VRAM)
- **Mode 1 - Single GPU + CPU Offload**: Offloads some components to CPU to save VRAM
- **Mode 2 - Multi-GPU Balanced**: Distributes model across multiple GPUs
- **Mode 3 - Single GPU + Sequential CPU Offload**: Sequential offloading for memory efficiency

**GPU Configuration**: For multi-GPU mode, specify GPU IDs (e.g., "0,1,2").

**Image Dimensions**: Set output image dimensions (512-2048 pixels, must be multiples of 64).

### Generation Modes

#### 1. 📤 File Upload Mode
Upload custom prompt files with the format:
```
background_description#character_description#action_description
background_description#character_description#action_description

(empty line separates different character/scenario groups)
```

#### 2. 🏠 BG+FG Mode (Fixed Background + Fixed Character)
- Character maintains consistency in a fixed background
- Only actions change between images
- Best for: Action sequences, poses, expressions in same location
- Uses: `examples/prompts-bg_fg.txt`

#### 3. 🚶 FG Only Mode (Variable Background + Fixed Character)
- Character consistency across different backgrounds
- Background changes while character remains consistent
- Best for: Character in different environments, storytelling
- Uses: `examples/prompts-fg_only.txt`

#### 4. 🌀 Mixed Mode (Complex Scenarios)
- Character in partly fixed and partly varying backgrounds
- Most flexible mode for complex scene generation
- Best for: Complex narratives, varied scenarios

### Generation Parameters

- **Seed**: Random seed for reproducible results (integer)
- **Use Interpolation**: Enable for smoother character transitions
- **Share Background**: Share background features across images in a sequence
- **Save Character Masks**: Generate visual masks showing detected character regions

## 🎛️ Usage Instructions

### Step 1: Model Setup
1. Enter your FLUX.1-dev model path
2. Select appropriate initialization mode based on your hardware
3. Configure GPU IDs if using multi-GPU setup
4. Set desired image dimensions
5. Click "🚀 Initialize Model"
6. Wait for "✅ Model initialized successfully!" message

### Step 2: Prepare Prompts
Choose one of the following options:

**Option A - Use Example Files**:
- Select the "BG+FG Mode" or "FG Only Mode" tab
- Example files will be automatically loaded if available

**Option B - Upload Custom File**:
- Select "File Upload" tab
- Upload a `.txt` file with prompts in the specified format

**Option C - Mixed Mode**:
- Select "Mixed Mode" tab
- Upload a custom prompt file for complex scenarios

### Step 3: Configure Generation
1. Set **Seed** for reproducible results
2. Enable **Use Interpolation** for smoother results (recommended)
3. Set **Share Background** based on your generation mode:
   - `True` for BG+FG mode (fixed background)
   - `False` for FG Only mode (variable background)
4. Enable **Save Character Masks** to visualize detected character regions

### Step 4: Generate Images
1. Click "✨ Generate Images"
2. Monitor progress in the status area
3. View results in the image galleries
4. Download images by clicking on them

## 📁 File Format Specification

### Prompt File Format

Each line in the prompt file should follow this structure:
```
background_description#character_description#action_description
```

**Example**:
```
in a modern gym with equipment in the background,#a muscular man in his 30s, wearing a black tank top and shorts,#lifting polygon-shaped dumbbells with focused expression
in a modern gym with equipment in the background,#a muscular man in his 30s, wearing a black tank top and shorts,#squating down, polygon-shaped dumbbells on the ground

in a busy city street,#a young woman with pink hair, wearing a leather jacket,#walking confidently, side view
in a park setting,#a young woman with pink hair, wearing a leather jacket,#sitting on a bench, relaxed expression
```

### Grouping
- Empty lines separate different character/scenario groups
- Each group generates one identity image followed by frame images
- First prompt in each group becomes the identity (reference) image

## ⚙️ Advanced Configuration

### Command Line Options

```bash
python src/launch_gradio.py --help
```

Available options:
- `--host`: Host address (default: 127.0.0.1)
- `--port`: Port number (default: 7860)
- `--share`: Create public Gradio link
- `--debug`: Enable debug mode

### Hardware Optimization

**For 24GB+ VRAM**:
- Use Mode 0 (Single GPU) for best performance
- Set dimensions to 1024x1024 or higher

**For 12-16GB VRAM**:
- Use Mode 1 (Single GPU + CPU Offload)
- Consider reducing dimensions to 768x768

**For Multiple GPUs**:
- Use Mode 2 (Multi-GPU Balanced)
- Specify all available GPU IDs

**For Limited VRAM (<12GB)**:
- Use Mode 3 (Sequential CPU Offload)
- Reduce dimensions to 512x512
- Enable CPU offloading

## 🎨 Output Features

### Generated Images
- High-resolution images (up to 2048x2048)
- Consistent character appearance across images
- Professional quality suitable for various applications

### Character Masks (Optional)
- Visual overlay showing detected character regions
- Red highlighting on character areas
- Side-by-side comparison with original image
- Useful for understanding model behavior and debugging

## 🚨 Troubleshooting

### Common Issues

**"Model initialization failed"**:
- Verify model path is correct and accessible
- Check GPU memory availability
- Try a different initialization mode

**"CUDA out of memory"**:
- Reduce image dimensions
- Use CPU offloading modes (1 or 3)
- Close other GPU-intensive applications

**"No prompts found"**:
- Check prompt file format
- Ensure proper `#` separators
- Verify file encoding (UTF-8)

**Generation is slow**:
- Mode 0 is fastest but requires most VRAM
- Multi-GPU mode may be slower due to communication overhead
- CPU offloading modes trade speed for memory efficiency

### Performance Tips

1. **Batch Processing**: Group related prompts together for efficiency
2. **Consistent Seeds**: Use the same seed for reproducible results
3. **Appropriate Dimensions**: Use dimensions that match your use case
4. **Memory Management**: Monitor GPU memory usage and adjust modes accordingly

## 🔧 Technical Details

### Architecture
- Built on Gradio for web interface
- Integrates CharaConsist pipeline seamlessly
- Supports all inference.py functionality
- Real-time progress tracking
- Comprehensive error handling

### Model Integration
- Direct integration with CharaConsist attention processors
- Automatic mask generation and visualization
- Support for all FLUX.1-dev model configurations
- Efficient memory management across different hardware setups

### Security Considerations
- Interface runs locally by default
- No data is sent to external servers
- Model files remain on your local system
- Generated images are stored locally

## 📞 Support

For issues, questions, or contributions:
- Check the main project documentation
- Review troubleshooting section above
- Ensure all dependencies are correctly installed
- Verify CUDA and GPU drivers are up to date