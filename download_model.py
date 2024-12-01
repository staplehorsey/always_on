#!/usr/bin/env python3
import os
import subprocess
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def download_models():
    print("Creating models directory...")
    os.makedirs("models", exist_ok=True)
    
    print("\nDownloading Silero VAD...")
    model_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
    model_path = os.path.join(model_dir, "silero_vad.jit")
    model_url = "https://github.com/snakers4/silero-vad/raw/master/models/silero_vad.jit"
    
    if os.path.exists(model_path):
        logger.info(f"Model already exists at {model_path}")
    else:
        logger.info(f"Downloading model to {model_path}")
        try:
            # Use curl command for downloading
            subprocess.run([
                "curl",
                "-L",  # Follow redirects
                "-o", model_path,  # Output file
                "--create-dirs",  # Create directories if needed
                "-C", "-",  # Resume download if possible
                model_url
            ], check=True)
            
            logger.info("Model downloaded successfully")
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to download model: {e}")
            raise
    
    print("\nDownloading Whisper base model...")
    model_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
    model_path = os.path.join(model_dir, "whisper_base.pt")
    model_url = "https://github.com/openai/whisper/blob/main/whisper/base.pt?raw=true"
    
    if os.path.exists(model_path):
        logger.info(f"Model already exists at {model_path}")
    else:
        logger.info(f"Downloading model to {model_path}")
        try:
            # Use curl command for downloading
            subprocess.run([
                "curl",
                "-L",  # Follow redirects
                "-o", model_path,  # Output file
                "--create-dirs",  # Create directories if needed
                "-C", "-",  # Resume download if possible
                model_url
            ], check=True)
            
            logger.info("Model downloaded successfully")
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to download model: {e}")
            raise
    
    print("\nDownloading Mistral-7B-Instruct-v0.3...")
    model_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
    model_path = os.path.join(model_dir, "mistral-7b-instruct-v0.3")
    model_url = "https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.3/resolve/main/pytorch_model.bin"
    
    if os.path.exists(model_path):
        logger.info(f"Model already exists at {model_path}")
    else:
        logger.info(f"Downloading model to {model_path}")
        try:
            # Use curl command for downloading
            subprocess.run([
                "curl",
                "-L",  # Follow redirects
                "-o", model_path,  # Output file
                "--create-dirs",  # Create directories if needed
                "-C", "-",  # Resume download if possible
                model_url
            ], check=True)
            
            logger.info("Model downloaded successfully")
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to download model: {e}")
            raise
    
    print("\nAll models downloaded successfully!")

if __name__ == "__main__":
    download_models()
