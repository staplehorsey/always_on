from typing import List, Literal, Optional, Union
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from llama_cpp import Llama
import asyncio
import uvicorn
import json
import logging
import logging.handlers
import os
from datetime import datetime
import time
import requests
from tqdm import tqdm

# Set PyTorch memory management settings
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.handlers.RotatingFileHandler(
            'logs/server.log',
            maxBytes=10485760,  # 10MB
            backupCount=5
        ),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('llm-server')

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

# Initialize FastAPI app
app = FastAPI(title="Local LLM Server")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize LLM Engine
llm = None

@app.on_event("startup")
async def startup_event():
    global llm
    logger.info("Starting LLM server...")
    try:
        model_path = os.path.join(models_dir, "mistral-7b-instruct-v0.2.Q4_K_M.gguf")
        if not os.path.exists(model_path):
            logger.info("Downloading model...")
            model_url = "https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.2-GGUF/resolve/main/mistral-7b-instruct-v0.2.Q4_K_M.gguf"
            download_file(model_url, model_path)
            logger.info("Model downloaded successfully")
            
        llm = Llama(
            model_path=model_path,
            n_gpu_layers=35,  # Adjust based on your GPU memory
            n_ctx=4096,       # Context window
            n_batch=512,      # Reduce for less memory usage
            verbose=True      # For debugging
        )
        logger.info("LLM engine initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize LLM engine: {str(e)}", exc_info=True)
        raise

# Pydantic models for API
class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant"] = Field(default="user")
    content: str

class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[ChatMessage]
    temperature: Optional[float] = 0.7
    top_p: Optional[float] = 0.9
    max_tokens: Optional[int] = 200
    stream: Optional[bool] = False

class ChatChoice(BaseModel):
    index: int
    message: ChatMessage
    finish_reason: Optional[str] = None

class ChatCompletionResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: List[ChatChoice]

@app.post("/v1/chat/completions", response_model=ChatCompletionResponse)
async def create_chat_completion(request: ChatCompletionRequest):
    try:
        logger.info(f"Received chat completion request for model: {request.model}")
        
        # Format prompt for chat
        prompt = ""
        for msg in request.messages:
            if msg.role == "system":
                prompt += f"<s>[INST] {msg.content} [/INST]"
            elif msg.role == "user":
                prompt += f"<s>[INST] {msg.content} [/INST]"
            elif msg.role == "assistant":
                prompt += f"{msg.content}</s>"
        prompt = f"### Human: {prompt}\n### Assistant: "
        
        # Generate response
        response = llm(
            prompt,
            max_tokens=request.max_tokens,
            temperature=request.temperature,
            top_p=request.top_p,
            stop=["### Human:", "\n### Human:"],
            echo=False
        )

        # Extract generated text
        generated_text = response["choices"][0]["text"].strip()
        logger.info(f"Generated response")
        logger.debug(f"Generated text: {generated_text[:100]}...")
        
        # Create response
        response = ChatCompletionResponse(
            id=f"chat-{int(time.time())}",
            created=int(time.time()),
            model=request.model,
            choices=[
                ChatChoice(
                    index=0,
                    message=ChatMessage(role="assistant", content=generated_text),
                    finish_reason="stop"
                )
            ]
        )
        
        return response
    except Exception as e:
        logger.error("Error in chat completion", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    logger.info("Starting server on http://0.0.0.0:8000")
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=False)
