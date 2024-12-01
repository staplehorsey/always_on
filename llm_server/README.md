# Local LLM Server

An OpenAI-compatible API server for running Mistral 7B locally using llama.cpp.

## Requirements

- Python 3.8+
- CUDA-compatible GPU (GTX 1080 Ti or better)
- 8GB+ GPU VRAM

## GPU Configuration

The server is configured for optimal performance on the GTX 1080 Ti:
- Uses GGUF quantized model (4-bit)
- 4096 token context window
- Mistral 7B model (optimized for consumer GPUs)
- 35 layers offloaded to GPU
- 512 batch size for stability
- CUDA acceleration via llama.cpp

The server uses llama.cpp which is highly optimized for older GPUs like the GTX 1080 Ti. Key features:
- 4-bit quantization for reduced memory usage
- Efficient CUDA kernels
- Layer-wise GPU offloading
- Optimized for Pascal architecture

## Installation

1. Create and activate virtual environment:
```bash
python3 -m venv venv
source venv/bin/activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the server:
```bash
python server.py
```

The model will be automatically downloaded on first run.

## Usage

The server provides an OpenAI-compatible API endpoint at `http://localhost:8000/v1/chat/completions`.

Example Python usage:
```python
import openai

openai.api_key = "not-needed"
openai.api_base = "http://localhost:8000/v1"

response = openai.ChatCompletion.create(
    model="mistral-7b",  # Model name doesn't matter, server uses Mistral
    messages=[
        {"role": "user", "content": "Hello, how are you?"}
    ],
    temperature=0.7,
    max_tokens=512
)

print(response.choices[0].message.content)
```

## Memory Management Tips
- If you're experiencing OOM errors:
  1. Reduce `n_gpu_layers` (try 25 or 20)
  2. Reduce `n_batch` (try 256)
  3. Use Q4_0 model variant for less memory usage
  4. Reduce context window with `n_ctx`
