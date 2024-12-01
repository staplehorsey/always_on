# Local LLM Server

An OpenAI-compatible API server for running Mistral-7B locally.

## Requirements

- NVIDIA GPU (tested with 1080Ti)
- CUDA 11.8 or higher
- Python 3.8 or higher

## GPU Configuration

The server is configured for optimal performance on the GTX 1080 Ti:
- Uses float16 precision (bfloat16 not supported on compute capability < 8.0)
- 8192 token context window (adjustable based on GPU memory)
- Eager execution mode for better stability
- 90% GPU memory utilization

You may need to adjust these settings in `server.py` based on your specific GPU:
```python
engine_args = AsyncEngineArgs(
    model="mistralai/Mistral-7B-Instruct-v0.3",
    download_dir="models",
    gpu_memory_utilization=0.9,  # Adjust if running out of memory
    tensor_parallel_size=1,      # Increase for multi-GPU setup
    dtype="float16",            # Required for GTX 1080 Ti
    trust_remote_code=True,
    max_model_len=8192,         # Adjust based on available memory
    enforce_eager=True          # More stable on older GPUs
)
```

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
    model="mistral",  # Model name doesn't matter, server uses Mistral
    messages=[
        {"role": "user", "content": "Hello, how are you?"}
    ],
    temperature=0.7,
    max_tokens=200
)

print(response.choices[0].message.content)
```

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
