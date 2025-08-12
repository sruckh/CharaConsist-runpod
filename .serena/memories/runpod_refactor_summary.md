Project has been refactored to run as a Runpod serverless endpoint.

Key changes:
1.  **Dockerfile:** The Dockerfile has been simplified to be a minimal base image (`runpod/pytorch:2.8.0-py3.11-cuda12.8.1-cudnn-devel-ubuntu22.04`) that only installs `git` and copies over the necessary handler and UI scripts. All application-specific setup is now done at runtime.

2.  **Runtime Handler (`runpod_handler.py`):** A new script has been created to manage the environment setup when the Runpod instance starts. Its responsibilities include:
    - Checking if the setup has already been performed.
    - Cloning the `CharaConsist` repository from GitHub.
    - Creating a Python virtual environment.
    - Installing all required dependencies from `requirements.txt` plus `gradio`.
    - Downloading the `FLUX.1-dev` model from Hugging Face using the `HF_TOKEN` environment variable.
    - Launching the Gradio web interface.

3.  **Gradio Interface (`src/gradio_interface.py`):** A comprehensive Gradio application has been created to provide a user-friendly UI for the model. It consolidates the functionality from the original `inference.py` script and the three Jupyter notebooks (`gen-bg_fg.ipynb`, `gen-fg_only.ipynb`, `gen-mix.ipynb`). Users can now select generation modes, input multiple prompts, and adjust parameters through the web interface.

4.  **GitHub Actions (`.github/workflows/docker-build-deploy.yml`):** The workflow has been updated to build the new, simplified Docker image and push it to Docker Hub under the `gemneye/characonsist-runpod` repository.

This new structure makes the Docker image much smaller and more portable, with the heavy setup tasks deferred to the runtime environment, as per the project goals.