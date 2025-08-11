#!/usr/bin/env python3
"""
Comprehensive Gradio Interface for CharaConsist
Supports all functionality from inference.py and the three generation modes
"""

import os
import sys
import torch
import numpy as np
import gradio as gr
from PIL import Image
import tempfile
import shutil
from pathlib import Path
import traceback
import logging

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.attention_processor_characonsist import (
    reset_attn_processor,
    set_text_len,
    reset_size,
    reset_id_bank,
)
from models.pipeline_characonsist import CharaConsistPipeline

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CharaConsistInterface:
    def __init__(self):
        self.pipe = None
        self.current_model_path = None
        self.current_init_mode = None
        
    def get_text_tokens_length(self, pipe, p):
        """Calculate text token length for prompt processing"""
        text_mask = pipe.tokenizer_2(
            p,
            padding="max_length",
            max_length=512,
            truncation=True,
            return_length=False,
            return_overflowing_tokens=False,
            return_tensors="pt",
        ).attention_mask
        return text_mask.sum().item() - 1

    def modify_prompt_and_get_length(self, bg, fg, act, pipe):
        """Process prompt components and calculate lengths"""
        bg += " "
        fg += " "
        prompt = bg + fg + act
        return prompt, self.get_text_tokens_length(pipe, bg), self.get_text_tokens_length(pipe, prompt)
        
    def init_model_mode_0(self, model_path):
        """Single GPU"""
        pipe = CharaConsistPipeline.from_pretrained(model_path, torch_dtype=torch.bfloat16)
        pipe.to("cuda:0")
        return pipe

    def init_model_mode_1(self, model_path):
        """Single GPU with model CPU offload"""
        pipe = CharaConsistPipeline.from_pretrained(model_path, torch_dtype=torch.bfloat16)
        pipe.enable_model_cpu_offload()
        return pipe

    def init_model_mode_2(self, model_path, gpu_ids=[0, 1]):
        """Multiple GPUs, memory distributed evenly"""
        from diffusers import FluxTransformer2DModel
        from transformers import T5EncoderModel
        
        # Set CUDA visible devices
        os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(map(str, gpu_ids))
        
        transformer = FluxTransformer2DModel.from_pretrained(
            model_path, subfolder="transformer", torch_dtype=torch.bfloat16, device_map="balanced")
        text_encoder_2 = T5EncoderModel.from_pretrained(
            model_path, subfolder="text_encoder_2", torch_dtype=torch.bfloat16, device_map="balanced")
        pipe = CharaConsistPipeline.from_pretrained(
            model_path, 
            transformer=transformer,
            text_encoder_2=text_encoder_2,
            torch_dtype=torch.bfloat16, 
            device_map="balanced")
        return pipe

    def init_model_mode_3(self, model_path):
        """Single GPU with sequential CPU offload"""
        pipe = CharaConsistPipeline.from_pretrained(model_path, torch_dtype=torch.bfloat16)
        pipe.enable_sequential_cpu_offload()
        return pipe

    def initialize_model(self, model_path, init_mode, gpu_ids, height, width, progress=gr.Progress()):
        """Initialize the model with specified configuration"""
        try:
            progress(0.1, desc="Loading model...")
            
            # Map init modes to functions
            MODEL_INIT_FUNCS = {
                0: self.init_model_mode_0,
                1: self.init_model_mode_1,
                2: self.init_model_mode_2,
                3: self.init_model_mode_3
            }
            
            if init_mode == 2:
                self.pipe = MODEL_INIT_FUNCS[init_mode](model_path, gpu_ids)
            else:
                self.pipe = MODEL_INIT_FUNCS[init_mode](model_path)
            
            progress(0.5, desc="Resetting attention processors...")
            reset_attn_processor(self.pipe, size=(height//16, width//16))
            
            self.current_model_path = model_path
            self.current_init_mode = init_mode
            
            progress(1.0, desc="Model initialized successfully!")
            return "✅ Model initialized successfully!"
            
        except Exception as e:
            error_msg = f"❌ Error initializing model: {str(e)}"
            logger.error(error_msg)
            logger.error(traceback.format_exc())
            return error_msg

    def parse_prompt_file(self, file_path):
        """Parse prompt file and return structured data"""
        try:
            with open(file_path, "r") as f:
                lines = f.readlines()
            
            all_prompt_groups = []
            current_group = []
            
            for line in lines:
                line = line.strip()
                if len(line) > 0:
                    parts = line.split("#")
                    if len(parts) != 3:
                        continue
                    bg, fg, act = parts
                    current_group.append({"bg": bg, "fg": fg, "act": act})
                else:
                    if current_group:
                        all_prompt_groups.append(current_group)
                        current_group = []
            
            if current_group:
                all_prompt_groups.append(current_group)
            
            return all_prompt_groups
            
        except Exception as e:
            logger.error(f"Error parsing prompt file: {e}")
            return []

    def load_prompt_file_for_processing(self, file_path):
        """Load and process prompt file for inference"""
        try:
            with open(file_path, "r") as f:
                all_lines = f.readlines()
            
            all_prompt_info = []
            curr_prompts, curr_bg_len, curr_real_len = [], [], []
            
            for line in all_lines:
                prompt = line.strip()
                if len(prompt) > 0:
                    bg, fg, act = prompt.split("#")
                    prompt, bg_len, real_len = self.modify_prompt_and_get_length(bg, fg, act, self.pipe)
                    curr_prompts.append(prompt)
                    curr_bg_len.append(bg_len)
                    curr_real_len.append(real_len)
                else:
                    all_prompt_info.append((curr_prompts, curr_bg_len, curr_real_len))
                    curr_prompts, curr_bg_len, curr_real_len = [], [], []
            
            if len(curr_prompts) > 0:
                all_prompt_info.append((curr_prompts, curr_bg_len, curr_real_len))
            
            return all_prompt_info
            
        except Exception as e:
            logger.error(f"Error loading prompt file: {e}")
            return []

    def overlay_mask_on_image(self, image, mask, color):
        """Overlay mask on image for visualization"""
        img_array = np.array(image).astype(np.float32) * 0.5
        mask_zero = np.zeros_like(img_array)

        mask_resized = Image.fromarray(mask.astype(np.uint8))
        mask_resized = mask_resized.resize(image.size, Image.NEAREST)
        mask_resized = np.array(mask_resized)
        mask_resized = mask_resized[:, :, None]
        color = np.array(color, dtype=np.float32).reshape(1, 1, -1)
        mask_resized_color = mask_resized * color
        img_array = img_array + mask_resized_color * 0.5
        mask_zero = mask_zero + mask_resized_color
        out_img = np.concatenate([img_array, mask_zero], axis=1)
        out_img[out_img > 255] = 255
        out_img = out_img.astype(np.uint8)
        return Image.fromarray(out_img)

    def generate_images(self, prompts_file, height, width, seed, use_interpolate, 
                       share_bg, save_mask, progress=gr.Progress()):
        """Generate images using the inference pipeline"""
        if self.pipe is None:
            return [], "❌ Please initialize the model first!", []
        
        try:
            progress(0.1, desc="Loading prompts...")
            all_prompt_info = self.load_prompt_file_for_processing(prompts_file.name)
            
            if not all_prompt_info:
                return [], "❌ No valid prompts found in file!", []
            
            pipe_kwargs = dict(
                height=height,
                width=width,
                use_interpolate=use_interpolate,
                share_bg=share_bg
            )
            
            all_results = []
            mask_results = []
            
            total_groups = len(all_prompt_info)
            
            for group_idx, (prompts, bg_lens, real_lens) in enumerate(all_prompt_info):
                progress(0.2 + 0.7 * group_idx / total_groups, 
                        desc=f"Processing group {group_idx + 1}/{total_groups}...")
                
                id_prompt = prompts[0]
                frm_prompts = prompts[1:]
                
                # Generate ID image
                set_text_len(self.pipe, bg_lens[0], real_lens[0])
                id_images, id_spatial_kwargs = self.pipe(
                    id_prompt, is_id=True, 
                    generator=torch.Generator("cpu").manual_seed(seed), 
                    **pipe_kwargs
                )
                
                id_fg_mask = id_spatial_kwargs["curr_fg_mask"]
                group_results = [id_images[0]]
                group_masks = []
                
                # Generate mask visualization for ID image if requested
                if save_mask:
                    mask_viz = self.overlay_mask_on_image(
                        id_images[0], 
                        id_fg_mask[0].cpu().numpy(), 
                        (255, 0, 0)
                    )
                    group_masks.append(mask_viz)
                
                # Generate frame images
                spatial_kwargs = dict(id_fg_mask=id_fg_mask, id_bg_mask=~id_fg_mask)
                
                for ind, prompt in enumerate(frm_prompts):
                    set_text_len(self.pipe, bg_lens[1:][ind], real_lens[1:][ind])
                    
                    # Pre-run
                    _, spatial_kwargs = self.pipe(
                        prompt, is_pre_run=True, 
                        generator=torch.Generator("cpu").manual_seed(seed),
                        spatial_kwargs=spatial_kwargs, 
                        **pipe_kwargs
                    )
                    
                    # Actual generation
                    images, spatial_kwargs = self.pipe(
                        prompt, 
                        generator=torch.Generator("cpu").manual_seed(seed),
                        spatial_kwargs=spatial_kwargs, 
                        **pipe_kwargs
                    )
                    
                    group_results.append(images[0])
                    
                    # Generate mask visualization if requested
                    if save_mask:
                        mask_viz = self.overlay_mask_on_image(
                            images[0], 
                            spatial_kwargs["curr_fg_mask"][0].cpu().numpy(), 
                            (255, 0, 0)
                        )
                        group_masks.append(mask_viz)
                
                all_results.extend(group_results)
                mask_results.extend(group_masks)
                
                # Reset ID bank for next group
                reset_id_bank(self.pipe)
            
            progress(1.0, desc="Generation complete!")
            success_msg = f"✅ Successfully generated {len(all_results)} images!"
            
            return all_results, success_msg, mask_results
            
        except Exception as e:
            error_msg = f"❌ Error during generation: {str(e)}"
            logger.error(error_msg)
            logger.error(traceback.format_exc())
            return [], error_msg, []

# Global interface instance
interface = CharaConsistInterface()

def create_gradio_interface():
    """Create the main Gradio interface"""
    
    with gr.Blocks(title="CharaConsist - Character Consistent Image Generation", theme=gr.themes.Soft()) as demo:
        gr.Markdown("""
        # 🎭 CharaConsist - Character Consistent Image Generation
        
        Generate consistent character images across different scenarios using FLUX.1-dev with CharaConsist.
        
        ## 📋 Instructions:
        1. **Configure Model**: Set model path and GPU configuration
        2. **Initialize Model**: Load the model with your chosen settings
        3. **Upload Prompts**: Choose a prompt file or use one of the generation modes
        4. **Set Parameters**: Configure generation settings
        5. **Generate**: Create your character-consistent images!
        """)
        
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### ⚙️ Model Configuration")
                
                model_path = gr.Textbox(
                    label="Model Path",
                    placeholder="/path/to/FLUX.1-dev",
                    value="/path/to/FLUX.1-dev",
                    info="Path to your FLUX.1-dev model directory"
                )
                
                init_mode = gr.Radio(
                    choices=[
                        (0, "Single GPU"),
                        (1, "Single GPU + CPU Offload"),
                        (2, "Multi-GPU Balanced"),
                        (3, "Single GPU + Sequential CPU Offload")
                    ],
                    label="Initialization Mode",
                    value=0,
                    info="Choose GPU configuration based on your hardware"
                )
                
                gpu_ids = gr.Textbox(
                    label="GPU IDs (for Multi-GPU mode)",
                    placeholder="0,1",
                    value="0,1",
                    info="Comma-separated GPU IDs for multi-GPU mode"
                )
                
                with gr.Row():
                    height = gr.Slider(
                        minimum=512, maximum=2048, step=64, value=1024,
                        label="Height"
                    )
                    width = gr.Slider(
                        minimum=512, maximum=2048, step=64, value=1024,
                        label="Width"
                    )
                
                init_btn = gr.Button("🚀 Initialize Model", variant="primary", size="lg")
                init_status = gr.Textbox(label="Initialization Status", interactive=False)
                
            with gr.Column(scale=2):
                gr.Markdown("### 📁 Generation Modes")
                
                with gr.Tabs():
                    with gr.TabItem("📤 File Upload"):
                        gr.Markdown("""
                        Upload a custom prompt file. Format should be:
                        ```
                        background#character#action
                        background#character#action
                        (empty line separates groups)
                        ```
                        """)
                        
                        prompts_file = gr.File(
                            label="Upload Prompt File",
                            file_types=[".txt"],
                            info="Upload a .txt file with prompts in the specified format"
                        )
                        
                    with gr.TabItem("🏠 BG+FG Mode"):
                        gr.Markdown("""
                        **Fixed Background + Fixed Character**: Character maintains consistency 
                        in a fixed background with different actions.
                        """)
                        
                        bg_fg_file = gr.File(
                            label="BG+FG Prompt File",
                            value="examples/prompts-bg_fg.txt" if os.path.exists("examples/prompts-bg_fg.txt") else None,
                            file_types=[".txt"]
                        )
                        
                    with gr.TabItem("🚶 FG Only Mode"):
                        gr.Markdown("""
                        **Variable Background + Fixed Character**: Character consistency 
                        across different backgrounds and scenarios.
                        """)
                        
                        fg_only_file = gr.File(
                            label="FG Only Prompt File",
                            value="examples/prompts-fg_only.txt" if os.path.exists("examples/prompts-fg_only.txt") else None,
                            file_types=[".txt"]
                        )
                        
                    with gr.TabItem("🌀 Mixed Mode"):
                        gr.Markdown("""
                        **Mixed Scenarios**: Character in partly fixed and partly varying 
                        backgrounds for complex scene generation.
                        """)
                        
                        mixed_file = gr.File(
                            label="Mixed Mode Prompt File",
                            file_types=[".txt"]
                        )
        
        gr.Markdown("### 🎛️ Generation Parameters")
        
        with gr.Row():
            seed = gr.Number(
                label="Seed",
                value=2025,
                precision=0,
                info="Random seed for reproducible results"
            )
            
            use_interpolate = gr.Checkbox(
                label="Use Interpolation",
                value=True,
                info="Enable interpolation for smoother results"
            )
            
            share_bg = gr.Checkbox(
                label="Share Background",
                value=True,
                info="Share background features across images"
            )
            
            save_mask = gr.Checkbox(
                label="Save Character Masks",
                value=False,
                info="Generate and display character mask visualizations"
            )
        
        generate_btn = gr.Button("✨ Generate Images", variant="primary", size="lg")
        
        with gr.Row():
            status_text = gr.Textbox(label="Generation Status", interactive=False)
        
        with gr.Row():
            with gr.Column():
                gr.Markdown("### 🖼️ Generated Images")
                result_gallery = gr.Gallery(
                    label="Results",
                    show_label=True,
                    elem_id="gallery",
                    columns=3,
                    rows=2,
                    object_fit="contain",
                    height="auto"
                )
            
            with gr.Column():
                gr.Markdown("### 🎭 Character Masks (Optional)")
                mask_gallery = gr.Gallery(
                    label="Character Masks",
                    show_label=True,
                    elem_id="mask_gallery",
                    columns=2,
                    rows=2,
                    object_fit="contain",
                    height="auto",
                    visible=False
                )
        
        # Event handlers
        def update_mask_visibility(save_masks):
            return gr.update(visible=save_masks)
        
        save_mask.change(
            update_mask_visibility,
            inputs=[save_mask],
            outputs=[mask_gallery]
        )
        
        def init_model_wrapper(model_path_val, init_mode_val, gpu_ids_val, height_val, width_val):
            try:
                gpu_list = [int(x.strip()) for x in gpu_ids_val.split(',') if x.strip().isdigit()]
                return interface.initialize_model(model_path_val, init_mode_val, gpu_list, height_val, width_val)
            except Exception as e:
                return f"❌ Error: {str(e)}"
        
        init_btn.click(
            init_model_wrapper,
            inputs=[model_path, init_mode, gpu_ids, height, width],
            outputs=[init_status]
        )
        
        def generate_wrapper(prompts_file_val, bg_fg_file_val, fg_only_file_val, mixed_file_val, 
                           height_val, width_val, seed_val, use_interpolate_val, share_bg_val, save_mask_val):
            # Determine which file to use
            active_file = None
            if prompts_file_val is not None:
                active_file = prompts_file_val
            elif bg_fg_file_val is not None:
                active_file = bg_fg_file_val
            elif fg_only_file_val is not None:
                active_file = fg_only_file_val
            elif mixed_file_val is not None:
                active_file = mixed_file_val
            
            if active_file is None:
                return [], "❌ Please upload a prompt file!", []
            
            return interface.generate_images(
                active_file, height_val, width_val, seed_val, 
                use_interpolate_val, share_bg_val, save_mask_val
            )
        
        generate_btn.click(
            generate_wrapper,
            inputs=[prompts_file, bg_fg_file, fg_only_file, mixed_file, 
                   height, width, seed, use_interpolate, share_bg, save_mask],
            outputs=[result_gallery, status_text, mask_gallery]
        )
    
    return demo

if __name__ == "__main__":
    demo = create_gradio_interface()
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        debug=True
    )