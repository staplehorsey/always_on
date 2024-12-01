import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import whisper

def download_models():
    print("Creating models directory...")
    os.makedirs("models", exist_ok=True)
    
    print("\nDownloading Silero VAD...")
    model, utils = torch.hub.load(repo_or_dir="snakers4/silero-vad",
                                model="silero_vad",
                                force_reload=True,
                                onnx=False)
    
    print("Saving Silero VAD model...")
    model = model.to('cpu')  # Ensure model is on CPU
    model.eval()
    torch.jit.save(model, "models/silero_vad.jit")
    
    print("\nDownloading Whisper base model...")
    whisper.load_model("base")
    
    print("\nDownloading Mistral-7B-Instruct-v0.3...")
    model_name = "mistralai/Mistral-7B-Instruct-v0.3"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    
    print("\nSaving models...")
    tokenizer.save_pretrained("models/mistral")
    model.save_pretrained("models/mistral")
    
    print("\nAll models downloaded successfully!")

if __name__ == "__main__":
    download_models()
