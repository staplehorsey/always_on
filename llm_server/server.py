from typing import List, Literal, Optional, Union
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from vllm import AsyncLLMEngine, AsyncEngineArgs, SamplingParams
import asyncio
import uvicorn
import json
import logging
import logging.handlers
import os
from datetime import datetime

# Set PyTorch memory management settings
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

# Set up logging
os.makedirs('logs', exist_ok=True)
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

# Create models directory if it doesn't exist
models_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
os.makedirs(models_dir, exist_ok=True)

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
engine_args = AsyncEngineArgs(
    model="TheBloke/Mistral-7B-v0.1-GGUF",  # Switch to GGUF model
    download_dir=models_dir,  # Use absolute path to models directory
    gpu_memory_utilization=0.7,  # Reduced to avoid OOM
    tensor_parallel_size=1,
    dtype="float16",
    trust_remote_code=True,
    max_model_len=4096,  # Reduced context window
    enforce_eager=True,
    max_num_batched_tokens=4096,  # Limit batch size
    max_num_seqs=1,  # Process one sequence at a time
    quantization=None  # Remove GPTQ quantization
)
engine = None

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

@app.on_event("startup")
async def startup_event():
    global engine
    logger.info("Starting LLM server...")
    try:
        # Initialize engine with pre-quantized model
        engine = AsyncLLMEngine.from_engine_args(engine_args)
        logger.info("LLM engine initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize LLM engine: {str(e)}", exc_info=True)
        raise

def create_prompt(messages: List[ChatMessage]) -> str:
    """Convert chat messages to Mistral prompt format."""
    prompt = ""
    for msg in messages:
        if msg.role == "system":
            prompt += f"<s>[INST] {msg.content} [/INST]"
        elif msg.role == "user":
            prompt += f"<s>[INST] {msg.content} [/INST]"
        elif msg.role == "assistant":
            prompt += f"{msg.content}</s>"
    return prompt

@app.post("/v1/chat/completions", response_model=ChatCompletionResponse)
async def create_chat_completion(request: ChatCompletionRequest):
    try:
        logger.info(f"Received chat completion request for model: {request.model}")
        
        # Create sampling params
        sampling_params = SamplingParams(
            temperature=request.temperature,
            top_p=request.top_p,
            max_tokens=request.max_tokens,
        )
        
        # Create prompt
        prompt = create_prompt(request.messages)
        logger.debug(f"Generated prompt: {prompt[:100]}...")
        
        # Generate response
        start_time = datetime.now()
        result_generator = engine.generate(prompt, sampling_params)
        result = await result_generator.__anext__()
        generation_time = (datetime.now() - start_time).total_seconds()
        
        # Extract generated text
        generated_text = result.outputs[0].text.strip()
        logger.info(f"Generated response in {generation_time:.2f} seconds")
        logger.debug(f"Generated text: {generated_text[:100]}...")
        
        # Create response
        response = ChatCompletionResponse(
            id=f"chat-{result.request_id}",
            created=int(asyncio.get_event_loop().time()),
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
