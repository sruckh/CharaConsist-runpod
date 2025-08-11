#!/usr/bin/env python3
"""
Launcher script for CharaConsist Gradio Interface
"""

import sys
import os
import argparse

# Add project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

def main():
    parser = argparse.ArgumentParser(description="Launch CharaConsist Gradio Interface")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Host to bind to")
    parser.add_argument("--port", type=int, default=7860, help="Port to bind to")
    parser.add_argument("--share", action="store_true", help="Create public link")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    
    args = parser.parse_args()
    
    try:
        from src.gradio_interface import create_gradio_interface
        
        print("🚀 Starting CharaConsist Gradio Interface...")
        print(f"🌐 Host: {args.host}")
        print(f"🔌 Port: {args.port}")
        print(f"🔗 Share: {args.share}")
        print(f"🐛 Debug: {args.debug}")
        
        demo = create_gradio_interface()
        
        demo.launch(
            server_name=args.host,
            server_port=args.port,
            share=args.share,
            debug=args.debug
        )
        
    except ImportError as e:
        print(f"❌ Import Error: {e}")
        print("📋 Please install required dependencies:")
        print("   pip install -r requirements.txt")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Error launching interface: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()