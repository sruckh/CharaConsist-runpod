import gradio as gr
import torch
import numpy as np
import os
import sys
from PIL import Image
import tempfile

# Add CharaConsist to path
sys.path.append("/workspace/CharaConsist")

from models.attention_processor_characonsist import (
    reset_attn_processor,
    set_text_len,
    reset_size,
    reset_id_bank,
)
from models.pipeline_characonsist import CharaConsistPipeline

# --- Model Loading ---
MODEL_PATH = "/workspace/CharaConsist/models/FLUX.1-dev"
pipe = None

def get_text_tokens_length(p):
    text_mask = pipe.tokenizer_2(
        p, padding="max_length", max_length=512, truncation=True,
        return_length=False, return_overflowing_tokens=False, return_tensors="pt",
    ).attention_mask
    return text_mask.sum().item() - 1

def modify_prompt_and_get_length(bg, fg, act):
    bg += " "
    fg += " "
    prompt = bg + fg + act
    return prompt, get_text_tokens_length(prompt), get_text_tokens_length(bg)

def load_model(init_mode):
    global pipe
    if pipe is not None:
        del pipe
        torch.cuda.empty_cache()

    if init_mode == 0:
        pipe = CharaConsistPipeline.from_pretrained(MODEL_PATH, torch_dtype=torch.bfloat16)
        pipe.to("cuda:0")
    elif init_mode == 1:
        pipe = CharaConsistPipeline.from_pretrained(MODEL_PATH, torch_dtype=torch.bfloat16)
        pipe.enable_model_cpu_offload()
    elif init_mode == 2:
        # This mode requires multiple GPUs and is complex to set up in a standard Gradio app
        # For simplicity, we'll treat it like mode 0.
        gr.Warning("Multi-GPU mode (2) is not fully supported in this interface. Using single GPU mode (0).")
        pipe = CharaConsistPipeline.from_pretrained(MODEL_PATH, torch_dtype=torch.bfloat16)
        pipe.to("cuda:0")
    elif init_mode == 3:
        pipe = CharaConsistPipeline.from_pretrained(MODEL_PATH, torch_dtype=torch.bfloat16)
        pipe.enable_sequential_cpu_offload()
    
    return f"Model loaded with init_mode {init_mode}"

# --- Main Generation Logic ---
def generate_images(mode, prompts_text, init_mode, use_interpolate, save_mask, height, width, seed):
    if pipe is None:
        load_model(init_mode)

    prompts_lines = [line.strip() for line in prompts_text.split('\n') if line.strip()]
    if not prompts_lines:
        raise gr.Error("Please provide at least one prompt.")

    all_prompts = []
    bg_lens = []
    real_lens = []
    
    for line in prompts_lines:
        parts = line.split('#')
        if len(parts) != 3:
            raise gr.Error(f"Invalid prompt format: '{line}'. Use 'bg#fg#act'.")
        bg, fg, act = parts
        prompt, real_len, bg_len = modify_prompt_and_get_length(bg, fg, act)
        all_prompts.append(prompt)
        bg_lens.append(bg_len)
        real_lens.append(real_len)

    share_bg = (mode == "Fixed Background")
    
    pipe_kwargs = {
        "height": height, "width": width,
        "use_interpolate": use_interpolate, "share_bg": share_bg
    }

    reset_attn_processor(pipe, size=(height // 16, width // 16))
    
    # ID Generation
    id_prompt = all_prompts[0]
    set_text_len(pipe, bg_lens[0], real_lens[0])
    id_images, id_spatial_kwargs = pipe(
        id_prompt, is_id=True, generator=torch.Generator("cpu").manual_seed(seed), **pipe_kwargs
    )
    id_fg_mask = id_spatial_kwargs["curr_fg_mask"]
    
    output_images = [id_images[0]]
    output_masks = []
    if save_mask:
        output_masks.append(id_fg_mask[0].cpu().numpy())

    # Frame Generation
    spatial_kwargs = {"id_fg_mask": id_fg_mask, "id_bg_mask": ~id_fg_mask}
    
    for i, prompt in enumerate(all_prompts[1:]):
        set_text_len(pipe, bg_lens[i+1], real_lens[i+1])
        
        # For mixed mode, update background if it's a new scene (simplified for Gradio)
        if mode == "Mixed/Story Mode" and bg_lens[i+1] != bg_lens[i]:
             spatial_kwargs["id_bg_mask"] = None # Force re-evaluation of background
        
        _, spatial_kwargs = pipe(
            prompt, is_pre_run=True, generator=torch.Generator("cpu").manual_seed(seed), 
            spatial_kwargs=spatial_kwargs, **pipe_kwargs
        )
        images, spatial_kwargs = pipe(
            prompt, generator=torch.Generator("cpu").manual_seed(seed), 
            spatial_kwargs=spatial_kwargs, **pipe_kwargs
        )
        output_images.append(images[0])
        if save_mask:
            output_masks.append(spatial_kwargs["curr_fg_mask"][0].cpu().numpy())

    reset_id_bank(pipe)
    
    if save_mask:
        return output_images, output_masks
    else:
        return output_images, None


# --- Gradio Interface ---
with gr.Blocks() as demo:
    gr.Markdown("# CharaConsist: Fine-Grained Consistent Character Generation")
    
    with gr.Row():
        with gr.Column(scale=1):
            mode = gr.Radio(
                ["Fixed Background", "Varying Background", "Mixed/Story Mode"],
                label="Generation Mode",
                value="Fixed Background"
            )
            prompts_text = gr.Textbox(
                lines=10,
                label="Prompts",
                placeholder="Enter prompts, one per line, in the format: background#character#action"
            )
            init_mode = gr.Dropdown([0, 1, 2, 3], label="Init Mode", value=0)
            
            with gr.Accordion("Advanced Options", open=False):
                use_interpolate = gr.Checkbox(label="Use Interpolate", value=True)
                save_mask = gr.Checkbox(label="Save Mask", value=False)
                height = gr.Slider(512, 1024, value=1024, step=64, label="Height")
                width = gr.Slider(512, 1024, value=1024, step=64, label="Width")
                seed = gr.Number(label="Seed", value=2025)
            
            run_button = gr.Button("Generate")
            status = gr.Textbox(label="Model Status", interactive=False)

        with gr.Column(scale=2):
            gallery = gr.Gallery(label="Generated Images", show_label=True, elem_id="gallery")
            mask_gallery = gr.Gallery(label="Masks", show_label=True, elem_id="mask_gallery", visible=False)

    run_button.click(
        fn=generate_images,
        inputs=[mode, prompts_text, init_mode, use_interpolate, save_mask, height, width, seed],
        outputs=[gallery, mask_gallery]
    )
    
    save_mask.change(
        fn=lambda x: gr.update(visible=x),
        inputs=save_mask,
        outputs=mask_gallery
    )
    
    init_mode.change(
        fn=load_model,
        inputs=init_mode,
        outputs=status
    )

if __name__ == "__main__":
    demo.launch(share=True)
