# Local LLM Server

An OpenAI-compatible API server for running TinyLlama locally.

## Requirements

- NVIDIA GPU (tested with 1080Ti)
- CUDA 11.8 or higher
- Python 3.8 or higher

## GPU Configuration

The server is configured for optimal performance on the GTX 1080 Ti with memory constraints:
- Uses float16 precision (bfloat16 not supported on compute capability < 8.0)
- 2048 token context window (reduced for memory efficiency)
- TinyLlama 1.1B model (optimized for consumer GPUs)
- 80% GPU memory utilization
- Single sequence processing
- Memory expansion enabled via PyTorch settings

You may need to adjust these settings in `server.py` based on your specific GPU:
```python
# Environment settings
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

# Engine settings
engine_args = AsyncEngineArgs(
    model="TinyLlama/TinyLlama-1.1B-Chat-v1.0",  # Smaller model for better compatibility
    download_dir="models",
    gpu_memory_utilization=0.8,      # Can use more memory with smaller model
    tensor_parallel_size=1,          # Single GPU setup
    dtype="float16",                # Required for GTX 1080 Ti
    trust_remote_code=True,
    max_model_len=2048,             # Reduced context window
    enforce_eager=True,             # More stable on older GPUs
    max_num_batched_tokens=2048,    # Smaller batches for stability
    max_num_seqs=1                  # Process one sequence at a time
)

### Memory Management Tips
- If you're still experiencing OOM errors:
  1. Reduce `gpu_memory_utilization` (try 0.7 or 0.6)
  2. Further reduce `max_model_len` (try 1024)
  3. Reduce `max_num_batched_tokens` (try 1024)
  4. Clear CUDA cache between requests if needed

## Installation

1. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # Linux
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Start the server:
```bash
python server.py
```

The server will run on `http://localhost:8000` and provide an OpenAI-compatible API endpoint at `/v1/chat/completions`.

## API Usage

The server implements the OpenAI chat completions API. Example request:

```python
import openai

openai.api_key = "dummy"  # Not used but required
openai.api_base = "http://localhost:8000/v1"

response = openai.ChatCompletion.create(
    model="tinyllama",  # Model name doesn't matter, server uses TinyLlama
    messages=[
        {"role": "user", "content": "Hello, how are you?"}
    ],
    temperature=0.7,
    max_tokens=200
)

print(response.choices[0].message.content)

## Performance

- Uses vLLM for optimized inference
- Supports tensor parallelism
- Efficient memory usage
- Batching support

## Configuration

Adjust these parameters in `server.py`:

- `gpu_memory_utilization`: GPU memory usage (0.0 to 1.0)
- `tensor_parallel_size`: Number of GPUs to use
- `max_tokens`: Maximum tokens to generate
- `temperature`: Sampling temperature
