The goal of this project is to containerize the project https://github.com/Murray-Wang/CharaConsist.git so that it can run on a runpod server.  This is an exercise of repackaging, and not rebuilding source code.  This should only build on AMD64/x86_64 archeticture.
Use context7 to get documentation on best practices for Runpod contanerization. 
Change the github origin to be https://github.com/sruckh/CharaConsist-runpod.
Use SSH to to communicate with github.
Create github action to build and deploy container image to Dockerhub.  Use the gemneye/ repository.  Use the github secrets DOCKER_USERNAME and DOCKER_PASSWORD for authentication to Dockerhub.

This is the base image for the main container.
Use base image:runpod/pytorch:2.8.0-py3.11-cuda12.8.1-cudnn-devel-ubuntu22.04
 
Below are the RUNTIME steps to configure CharaConsist after container has loaded.  Very important that these are not part of the container image and are part of the bootstrap process after container is running.
**DO NOT BLINDY INSTALL, ONLY INSTALL THE BELOW ENVIRONMENT, IF IT HAS NOT ALREADY BEEN CONFIGURED**
**Instructions for setting up CharaConsist -- First Run**

  1) Create directory /workspace; if it does not already exist
  2) In the /workspace directory; git clone https://github.com/Murray-Wang/CharaConsist.git
 2a) cd to /workspace/CharaConsist
2a1) create a python virtual environment in the sub-directory for installing all the python modules and dependencies
2a2) upgrade pip:  pip install -U pip
 2b) pip install -r requirements.txt
  3) create a gradio interface that will support all the functionality of inference.py
  
      Generating consistent character in a fixed background
      python inference.py \
        --init_mode 0 \
        --prompts_file examples/prompts-bg_fg.txt \
        --model_path path/to/FLUX.1-dev \
        --out_dir results/bg_fg \
        --use_interpolate --save_mask --share_bg
        
      Generating consistent character across different backgrounds:
      python inference.py \
        --init_mode 0 \
        --prompts_file examples/prompts-fg_only.txt \
        --model_path path/to/FLUX.1-dev \
        --out_dir results/fg_only \
        --use_interpolate --save_mask
        
      gradio should support the 4 differt ini_mode (0-3_
        0 single GPU
        1 single GPU, with model cpu offload
        2 multiple GPUs, memory distribute evenly
        3 single GPU, with sequential cpu offload
        
      Also note the jupyter notebooks gen-bg_fg.ipynb, gen-fg_only.ipynb, and gen-mix.ipynb (the gradio interface should have a selector for each of the modes (maybe a radio button selector)
        gen-bg_fg.ipynb -- generating consistent character in a fixed background
        gen-fg_only.ipynb -- generating consistent character across different backgrounds
        gen-mix.ipynb -- generating the same character in partly fixed and partly varying backgrounds
        
  5) HF_HOME will be set in Runpod's environmental variables
  6) HF_TOKEN will be set in Runpd's environment variables; it will be needed to download flux models.
