#!/usr/bin/env python3
"""
Example script demonstrating CharaConsist Gradio Interface usage
"""

import sys
import os

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

def create_sample_prompt_file():
    """Create a sample prompt file for testing"""
    sample_prompts = """in a modern coffee shop with wooden tables and soft lighting in the background,#a young professional woman with shoulder-length brown hair, wearing a casual blazer and jeans,#sitting at a table with a laptop, typing with focused expression, front view
in a modern coffee shop with wooden tables and soft lighting in the background,#a young professional woman with shoulder-length brown hair, wearing a casual blazer and jeans,#holding a coffee cup with both hands, relaxed expression, side view
in a modern coffee shop with wooden tables and soft lighting in the background,#a young professional woman with shoulder-length brown hair, wearing a casual blazer and jeans,#standing by the window, looking outside thoughtfully, side view

in a public park with trees and benches in the background,#a young professional woman with shoulder-length brown hair, wearing a casual blazer and jeans,#walking on a path, confident stride, front view
in a busy office environment with computers and coworkers in the background,#a young professional woman with shoulder-length brown hair, wearing a casual blazer and jeans,#presenting at a meeting, enthusiastic expression, front view
in a home office with bookshelves and plants in the background,#a young professional woman with shoulder-length brown hair, wearing a casual blazer and jeans,#video calling on laptop, professional smile, front view"""

    # Create examples directory if it doesn't exist
    examples_dir = os.path.join(project_root, "examples")
    os.makedirs(examples_dir, exist_ok=True)
    
    # Write sample prompts
    sample_file = os.path.join(examples_dir, "sample_prompts.txt")
    with open(sample_file, "w") as f:
        f.write(sample_prompts)
    
    print(f"✅ Created sample prompt file: {sample_file}")
    return sample_file

def run_example():
    """Run the Gradio interface example"""
    try:
        # Create sample prompt file
        sample_file = create_sample_prompt_file()
        
        # Import and run the interface
        from src.gradio_interface import create_gradio_interface
        
        print("🚀 Starting CharaConsist Gradio Interface Example...")
        print(f"📄 Sample prompt file created: {sample_file}")
        print("🌐 Interface will open at: http://localhost:7860")
        print("\n📋 To test the interface:")
        print("1. Set your model path to your FLUX.1-dev directory")
        print("2. Choose initialization mode based on your GPU setup")
        print("3. Initialize the model")
        print(f"4. Upload the sample prompt file: {sample_file}")
        print("5. Configure generation parameters")
        print("6. Click 'Generate Images'")
        print("\n🛑 Press Ctrl+C to stop the interface")
        
        demo = create_gradio_interface()
        demo.launch(
            server_name="127.0.0.1",
            server_port=7860,
            share=False,
            debug=True
        )
        
    except ImportError as e:
        print(f"❌ Import Error: {e}")
        print("📋 Please install required dependencies:")
        print("   pip install -r requirements.txt")
    except KeyboardInterrupt:
        print("\n🛑 Interface stopped by user")
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    run_example()