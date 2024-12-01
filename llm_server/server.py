import os
import logging
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
from llama_cpp import Llama, LlamaGrammar
import json
import time
import requests
from tqdm import tqdm
import gc
import psutil

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('llm-server.log')
    ]
)
logger = logging.getLogger('llm-server')

def log_system_info():
    """Log system memory information."""
    mem = psutil.virtual_memory()
    logger.info(f"System memory: {mem.total / (1024**3):.1f}GB total, {mem.available / (1024**3):.1f}GB available")
    
    if hasattr(psutil, 'cuda'):
        try:
            gpu_info = psutil.cuda.get_device_info(0)
            logger.info(f"GPU memory: {gpu_info['total_memory'] / (1024**3):.1f}GB total")
        except:
            logger.info("Could not get GPU information")

def download_file(url: str, dest_path: str):
    """Download a file with progress bar."""
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    
    with open(dest_path, 'wb') as f, tqdm(
        desc="Downloading model",
        total=total_size,
        unit='iB',
        unit_scale=True,
        unit_divisor=1024,
    ) as pbar:
        for data in response.iter_content(chunk_size=1024):
            size = f.write(data)
            pbar.update(size)

# Create models directory if it doesn't exist
models_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
os.makedirs(models_dir, exist_ok=True)

# Set environment variables for GPU memory management
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

app = FastAPI()

class GenerateRequest(BaseModel):
    prompt: str
    max_tokens: Optional[int] = 512
    temperature: Optional[float] = 0.7
    top_p: Optional[float] = 0.9
    presence_penalty: Optional[float] = 0.0
    frequency_penalty: Optional[float] = 0.0
    stop: Optional[List[str]] = None

# Initialize LLM Engine
llm = None

@app.on_event("startup")
async def startup_event():
    global llm
    logger.info("Starting LLM server...")
    try:
        # Log system information
        log_system_info()
        
        model_path = os.path.join(models_dir, "mistral-7b-instruct-v0.2.Q4_K_M.gguf")
        if not os.path.exists(model_path):
            logger.info("Downloading model...")
            model_url = "https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.2-GGUF/resolve/main/mistral-7b-instruct-v0.2.Q4_K_M.gguf"
            download_file(model_url, model_path)
            logger.info("Model downloaded successfully")
        
        # Force garbage collection before loading model
        gc.collect()
        
        logger.info("Initializing model with conservative memory settings...")
        llm = Llama(
            model_path=model_path,
            n_gpu_layers=20,        # Reduced from 35 for stability
            n_ctx=2048,            # Reduced context window
            n_batch=256,           # Smaller batch size
            verbose=True,          # Keep verbose for debugging
            f16_kv=True,          # Use float16 for key/value cache
            use_mmap=True,        # Use memory mapping
            use_mlock=False,      # Don't lock memory
            embedding=False,       # Disable embedding to save memory
            logits_all=False,     # Don't compute all logits
            vocab_only=False      # Load full model
        )
        
        # Log model information
        logger.info(f"Model loaded successfully:")
        logger.info(f"- Model path: {llm.model_path}")
        logger.info(f"- Context window: {llm.n_ctx()}")
        logger.info(f"- Vocabulary size: {llm.n_vocab()}")
        logger.info(f"- System memory: {psutil.Process().memory_info().rss / (1024 * 1024):.2f} MB")
        
        # Force another GC after model load
        gc.collect()
        
    except Exception as e:
        logger.error(f"Failed to initialize LLM engine: {str(e)}", exc_info=True)
        raise

@app.post("/v1/chat/completions")
async def create_chat_completion(request: GenerateRequest):
    try:
        logger.info("Starting request processing")
        logger.info(f"Received prompt: {request.prompt[:100]}...")
        
        # Format prompt for chat
        prompt = f"### Human: {request.prompt}\n### Assistant: "
        logger.info("Formatted prompt")
        
        # Generate response with error handling
        try:
            logger.info("Starting generation...")
            response = llm(
                prompt,
                max_tokens=min(request.max_tokens, 1024),  # Limit max tokens
                temperature=request.temperature,
                top_p=request.top_p,
                stop=request.stop or ["### Human:", "\n### Human:"],
                echo=False
            )
            logger.info("Generation completed")
            
            result = {
                "id": "chatcmpl-" + os.urandom(12).hex(),
                "object": "chat.completion",
                "created": int(time.time()),
                "model": "mistral-7b-instruct",
                "choices": [{
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": response["choices"][0]["text"].strip()
                    },
                    "finish_reason": "stop"
                }],
                "usage": {
                    "prompt_tokens": response["usage"]["prompt_tokens"],
                    "completion_tokens": response["usage"]["completion_tokens"],
                    "total_tokens": response["usage"]["total_tokens"]
                }
            }
            logger.info("Response formatted successfully")
            return result
            
        except Exception as e:
            logger.error(f"Error during generation: {str(e)}", exc_info=True)
            raise HTTPException(status_code=500, detail="Generation failed")

    except Exception as e:
        logger.error(f"Error handling request: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
