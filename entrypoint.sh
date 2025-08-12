#!/bin/bash
set -e

# Start services
service ssh start
service nginx start

# Run bootstrap if not already done
if [ ! -f "/workspace/.bootstrap_done" ]; then
    echo "Running first-time setup..."
    /workspace/bootstrap.sh
    touch /workspace/.bootstrap_done
fi

# Start jupyter lab
echo "Starting Jupyter Lab..."
mkdir -p "/workspace/characonsist" && \
cd / && \
nohup jupyter lab --allow-root --no-browser --port=8888 --ip=* --NotebookApp.token='' --NotebookApp.password='' --FileContentsManager.delete_to_trash=False --ServerApp.terminado_settings='{"shell_command":["/bin/bash"]}' --ServerApp.allow_origin=* --ServerApp.preferred_dir="/workspace/characonsist" &> /jupyter.log &
echo "Jupyter Lab started without a password"

# Launch the gradio interface
echo "Launching Gradio Interface..."
python /workspace/characonsist/src/gradio_interface.py --share
