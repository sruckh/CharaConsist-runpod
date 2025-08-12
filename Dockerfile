FROM runpod/pytorch:2.8.0-py3.11-cuda12.8.1-cudnn-devel-ubuntu22.04

# Install git
RUN apt-get update && apt-get install -y git

# Copy the handler and interface scripts
COPY src/gradio_interface.py /app/gradio_interface.py
COPY runpod_handler.py /app/runpod_handler.py

# Set the working directory
WORKDIR /app

# Set the entrypoint
ENTRYPOINT ["python", "/app/runpod_handler.py"]