import os
import subprocess
import sys
from pathlib import Path

# Define the CharaConsist directory path as a global variable
characonsist_dir = Path("/workspace/CharaConsist")

def setup_environment():
    """
    Clones the repository, creates a venv, installs dependencies,
    and downloads the model if not already done.
    """
    setup_complete_marker = characonsist_dir / ".setup_complete"

    if setup_complete_marker.exists():
        print("Setup already complete. Skipping.")
        return

    print("Performing first-time setup...")

    # 1. Create workspace and clone repo
    if not characonsist_dir.exists():
        print("Cloning repository...")
        subprocess.run(
            ["git", "clone", "https://github.com/Murray-Wang/CharaConsist.git", str(characonsist_dir)],
            check=True
        )

    # 2. Add missing dependencies to the cloned repo's requirements file
    requirements_file_path = characonsist_dir / "requirements.txt"
    missing_deps = ['opencv-python', 'einops', 'sentencepiece']
    try:
        with open(requirements_file_path, 'r+') as f:
            content = f.read()
            # Move to the end of the file to append
            f.seek(0, 2)
            if not content.endswith('\n'):
                f.write('\n')

            for dep in missing_deps:
                if dep not in content:
                    f.write(f'{dep}\n')
                    print(f"Patched requirements.txt with {dep}.")
    except FileNotFoundError:
        print(f"Warning: {requirements_file_path} not found. Skipping dependency patch.")


    # 3. Create and activate virtual environment
    venv_dir = characonsist_dir / "venv"
    if not venv_dir.exists():
        print("Creating virtual environment...")
        subprocess.run([sys.executable, "-m", "venv", str(venv_dir)], check=True)

    pip_executable = str(venv_dir / "bin" / "pip")
    python_executable = str(venv_dir / "bin" / "python")

    # Upgrade pip
    print("Upgrading pip...")
    subprocess.run([pip_executable, "install", "-U", "pip"], check=True)

    # 4. Install Python dependencies
    requirements_file = characonsist_dir / "requirements.txt"
    print("Installing requirements...")
    subprocess.run([pip_executable, "install", "-r", str(requirements_file)], check=True)
    print("Installing Gradio...")
    subprocess.run([pip_executable, "install", "gradio"], check=True)

    # 5. Download the model using HF_TOKEN
    hf_token = os.getenv("HF_TOKEN")
    if not hf_token:
        print("WARNING: HF_TOKEN environment variable not set. Model download may fail.")
    else:
        print("Downloading model...")
        model_dir = str(characonsist_dir / "models" / "FLUX.1-dev")
        os.makedirs(model_dir, exist_ok=True)
        
        download_script = f"""
from huggingface_hub import snapshot_download
import os
snapshot_download(
    repo_id="black-forest-labs/FLUX.1-dev",
    local_dir="{model_dir}",
    token=os.getenv("HF_TOKEN"),
    resume_download=True
)
"""
        subprocess.run([python_executable, "-c", download_script], check=True)

    # 6. Create marker file
    print("Setup complete.")
    setup_complete_marker.touch()

def launch_gradio():
    """
    Launches the Gradio interface.
    """
    print("Launching Gradio interface...")
    venv_python = str(characonsist_dir / "venv" / "bin" / "python")
    gradio_script_path = "/app/gradio_interface.py"
    
    # Add the CharaConsist directory to the python path to resolve imports
    env = os.environ.copy()
    env["PYTHONPATH"] = f"{str(characonsist_dir)}:{env.get('PYTHONPATH', '')}"
    # Set PYTHONUNBUFFERED to 1 to prevent output buffering
    env["PYTHONUNBUFFERED"] = "1"

    subprocess.run([venv_python, gradio_script_path], env=env)


if __name__ == "__main__":
    # Force re-installation by removing the setup complete marker.
    # This is to ensure that any changes to dependencies are applied.
    setup_complete_marker = characonsist_dir / ".setup_complete"
    if setup_complete_marker.exists():
        setup_complete_marker.unlink()
        print("Removed setup marker to force re-installation.")

    setup_environment()
    launch_gradio()
