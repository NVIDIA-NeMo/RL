# LLaDA OpenAI API Server

This repository provides an OpenAI-compatible API server for LLaDA (Large Language Diffusion Models) with Fast-dLLM acceleration and support for DCP (Distributed Checkpoint) format checkpoints.

## What is LLaDA?

LLaDA (Large Language Diffusion Models) is a state-of-the-art approach that applies diffusion processes to language generation. Unlike traditional autoregressive models that generate tokens sequentially, LLaDA uses an iterative diffusion process to refine masked text over multiple steps.

### Key Features of LLaDA:
- **Diffusion-based generation**: Iteratively refines masked tokens
- **Parallel generation**: Can generate multiple tokens simultaneously within blocks
- **Quality control**: More diffusion steps generally lead to higher quality output
- **Flexible parameters**: Control creativity, guidance, and generation strategy
- **Fast-dLLM acceleration**: Up to 11x speedup with KV caching and parallel decoding

## Quick Start

### Prerequisites

For local execution:
```bash
# Recommended: Use uv to sync from project dependencies (for DCP support)
uv sync --locked --extra vllm --no-install-project
uv pip install fastapi uvicorn aiohttp requests

# Minimal for HuggingFace model testing only
pip install fastapi uvicorn torch transformers aiohttp requests

# Or using conda
conda install fastapi uvicorn pytorch transformers aiohttp requests -c conda-forge
```

**Note**: DCP checkpoint functionality requires NeMo-RL dependencies. For quick local testing, use HuggingFace model names instead.

For SLURM execution:
- Access to a SLURM cluster with container support
- Set `ACCOUNT` environment variable: `export ACCOUNT=your_slurm_account`

### Option 1: Local Execution

Run the server locally on your machine:

```bash
# Easiest: HuggingFace model (from Hub) - no setup required
./start_llada_server.sh --local --model-path GSAI-ML/LLaDA-8B-Instruct

# Local HuggingFace model (if downloaded)
./start_llada_server.sh --local --model-path /path/to/llada-model

# DCP checkpoint (requires NeMo-RL dependencies)
uv sync --locked --extra vllm --no-install-project  # First install dependencies
./start_llada_server.sh --local --dcp-path /path/to/checkpoint.dcp --base-model GSAI-ML/LLaDA-8B-Instruct
```

**💡 Tip**: For quick local testing, use HuggingFace model names - they work immediately without additional setup!

### Option 2: SLURM Job (Default)

Run as a SLURM job with GPU resources:

```bash
# Set up environment
export ACCOUNT=your_slurm_account

# Submit SLURM job with HuggingFace model (local path)
./start_llada_server.sh --model-path /path/to/llada-model

# Submit SLURM job with HuggingFace model (from Hub)  
./start_llada_server.sh --model-path GSAI-ML/LLaDA-8B-Instruct

# Submit SLURM job with DCP checkpoint (recommended for NeMo-RL users)
./start_llada_server.sh --dcp-path /path/to/checkpoint.dcp --base-model GSAI-ML/LLaDA-8B-Instruct

# Custom SLURM resources
./start_llada_server.sh --dcp-path /path/to/checkpoint.dcp --gpus 2 --mem 128G --time 8:00:00
```

### Connecting to SLURM Server

When running on SLURM, the server runs on a compute node and automatically displays connection instructions:

```bash
# Submit your job
export ACCOUNT=your_account
./start_llada_server.sh --model-path GSAI-ML/LLaDA-8B-Instruct

# Connection instructions appear automatically in the terminal output when server starts
# All logs are shown in real-time - no need to monitor separate log files

# Or use the helper script for manual connection setup
./connect_to_llada_server.sh --job-id $JOB_ID
```

**Connection Instructions Appear Automatically**: The server output will show SSH tunnel commands and local URLs when the server starts up, directly in your terminal in real-time.

### Basic Usage

Once the server is running, you can use it like any OpenAI API:

```python
import requests

response = requests.post("http://localhost:8000/v1/chat/completions", json={
    "model": "llada-8b-instruct",
    "messages": [{"role": "user", "content": "What is artificial intelligence?"}],
    "temperature": 0.0,
    "max_tokens": 128,
    "steps": 128,  # LLaDA: diffusion steps
    "use_cache": True,  # Fast-dLLM: enable acceleration
    "use_dual_cache": True
})

print(response.json()["choices"][0]["message"]["content"])
```

## API Reference

The server provides OpenAI-compatible endpoints with additional LLaDA-specific parameters:

### POST `/v1/chat/completions`

Standard OpenAI parameters:
- `model`: Model name (default: "llada-8b-instruct")
- `messages`: Array of chat messages
- `temperature`: Sampling temperature (0.0-2.0, default: 0.0)
- `max_tokens`: Maximum tokens to generate (default: 128)
- `stream`: Enable streaming responses (default: false)

**LLaDA-specific parameters:**
- `steps`: Number of diffusion steps (1-512, default: 128)
  - More steps = higher quality but slower generation
  - Recommended: 64-256 for most use cases
- `block_length`: Semi-autoregressive block size (default: 32)
  - Smaller blocks = more parallel generation
  - Must divide evenly into `max_tokens`
- `cfg_scale`: Classifier-free guidance scale (≥0.0, default: 0.0)
  - Higher values = more guided generation
  - Recommended: 0.0-3.0
- `remasking`: Token selection strategy (default: "low_confidence")
  - `"low_confidence"`: Select tokens with lowest confidence
  - `"random"`: Random token selection

**Fast-dLLM acceleration parameters:**
- `use_cache`: Enable KV caching (default: true)
  - Provides 2-3x speedup with minimal quality impact
- `use_dual_cache`: Enable dual cache for prefix and suffix (default: true)
  - Maximum acceleration when combined with KV cache
- `threshold`: Confidence threshold for parallel decoding (optional)
  - Range: 0.0-1.0, recommended: 0.7-0.9
  - Higher values = more conservative parallel decoding
- `factor`: Dynamic parallel decoding factor (optional)
  - Range: 1.0-4.0, recommended: 1.5-2.5
  - Controls aggressiveness of parallel token generation

### GET `/v1/models`

List available models.

### GET `/health`

Server health check with model loading status.

## Usage Examples

### 1. Basic Chat Completion

```python
import requests

def basic_chat():
    response = requests.post("http://localhost:8000/v1/chat/completions", json={
        "messages": [{"role": "user", "content": "Explain quantum computing"}],
        "temperature": 0.0,
        "max_tokens": 100,
        "steps": 64,  # Fast generation
        "use_cache": True,  # Enable acceleration
        "use_dual_cache": True
    })
    
    return response.json()["choices"][0]["message"]["content"]
```

### 2. Streaming Response

```python
import requests

def streaming_chat():
    response = requests.post("http://localhost:8000/v1/chat/completions", 
        json={
            "messages": [{"role": "user", "content": "Write a short story"}],
            "stream": True,
            "temperature": 0.5,
            "max_tokens": 200,
            "steps": 128,
            "use_cache": True,
            "use_dual_cache": True
        },
        stream=True
    )
    
    for line in response.iter_lines():
        if line.startswith(b'data: '):
            data = line[6:].decode('utf-8')
            if data != '[DONE]':
                chunk = json.loads(data)
                if 'content' in chunk['choices'][0]['delta']:
                    print(chunk['choices'][0]['delta']['content'], end='')
```

### 3. Quality vs Speed Trade-offs

```python
# Fast generation with maximum acceleration
fast_params = {
    "steps": 32,
    "temperature": 0.0,
    "max_tokens": 80,
    "use_cache": True,
    "use_dual_cache": True
}

# High quality generation
quality_params = {
    "steps": 256,
    "temperature": 0.0,
    "max_tokens": 80,
    "use_cache": True,
    "use_dual_cache": True
}

# Creative generation with parallel decoding
creative_params = {
    "steps": 64,
    "temperature": 0.8,
    "cfg_scale": 0.0,
    "max_tokens": 80,
    "use_cache": True,
    "use_dual_cache": True,
    "threshold": 0.7
}

# Conservative parallel generation
conservative_params = {
    "steps": 128,
    "temperature": 0.0,
    "max_tokens": 80,
    "use_cache": True,
    "use_dual_cache": True,
    "threshold": 0.9
}
```

## Advanced Configuration

### Local Server Configuration

```bash
# Custom host and port (local mode) with local model
./start_llada_server.sh --local --host 127.0.0.1 --port 8080 --model-path /path/to/model

# Custom host and port (local mode) with HuggingFace Hub model
./start_llada_server.sh --local --host 127.0.0.1 --port 8080 --model-path GSAI-ML/LLaDA-8B-Instruct

# Using DCP with custom temp directory (local mode)
./start_llada_server.sh --local --dcp-path /path/to/checkpoint.dcp \
                       --base-model GSAI-ML/LLaDA-8B-Instruct \
                       --temp-dir /custom/temp/dir
```

### SLURM Configuration

```bash
# Custom SLURM job resources
./start_llada_server.sh --dcp-path /path/to/checkpoint.dcp \
                       --job-name my-llada-server \
                       --time 12:00:00 \
                       --gpus 2 \
                       --cpus 32 \
                       --mem 256G \
                       --partition gpu

# Custom container image with local model
./start_llada_server.sh --model-path /path/to/model \
                       --container /path/to/custom/container.sqsh

# Custom container image with HuggingFace Hub model
./start_llada_server.sh --model-path GSAI-ML/LLaDA-8B-Instruct \
                       --container /path/to/custom/container.sqsh

# Custom port (useful for running multiple servers)
./start_llada_server.sh --dcp-path /path/to/checkpoint.dcp --port 8080
```

### Environment Variables

For SLURM jobs, you can set these environment variables:

```bash
export ACCOUNT=your_slurm_account          # Required
export LOG=/path/to/your/log/directory     # Optional, defaults to ./logs
```

### Performance Tuning

#### GPU Memory Optimization
- Use `torch.bfloat16` for reduced memory usage (automatically enabled)
- Monitor GPU memory with `nvidia-smi`
- Adjust `block_length` to balance memory and parallelism

#### Generation Speed vs Quality
- **Maximum speed**: `steps=32-64`, `use_cache=true`, `use_dual_cache=true`
- **Balanced quality**: `steps=128`, `use_cache=true`, `threshold=0.8`
- **High quality**: `steps=256+`, `use_cache=true`, conservative settings
- **Disable acceleration**: `use_cache=false` (for quality comparison)

#### Batch Processing
- Server handles one request at a time (suitable for single-user scenarios)
- For production, consider load balancing multiple server instances

## DCP Checkpoint Conversion

The server automatically converts DCP checkpoints to HuggingFace format when needed:

1. **Automatic conversion**: Happens on first startup with DCP checkpoint
2. **Caching**: Converted models are cached in temp directory
3. **Base model**: Used for tokenizer and configuration
4. **Memory efficient**: Conversion preserves model weights structure

### Manual Conversion (Optional)

You can also manually convert DCP to HF format:

```python
from nemo_rl.utils.native_checkpoint import convert_dcp_to_hf

convert_dcp_to_hf(
    dcp_ckpt_path="/path/to/checkpoint.dcp",
    hf_ckpt_path="/path/to/output/hf/model",
    model_name_or_path="GSAI-ML/LLaDA-8B-Instruct",
    tokenizer_name_or_path="GSAI-ML/LLaDA-8B-Instruct",
    overwrite=True
)
```

## Testing the Server

Use the provided client example:

```bash
# Run all examples (from project root)
python xp/llada_api/examples/llada_api_client.py

# Or from llada_api directory
cd xp/llada_api
python examples/llada_api_client.py

# Test server health
curl http://localhost:8000/health

# Test basic completion with Fast-dLLM acceleration
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [{"role": "user", "content": "Hello!"}],
    "temperature": 0.0,
    "max_tokens": 64,
    "steps": 64,
    "use_cache": true,
    "use_dual_cache": true
  }'
```

## Troubleshooting

### Common Issues

#### Local Mode Issues

1. **Model not loading**
   - Check file paths and permissions
   - Verify CUDA availability for GPU models
   - Check server logs for detailed error messages

2. **Out of memory errors**
   - Reduce `max_tokens` and `block_length`
   - Use CPU if GPU memory is insufficient
   - Close other GPU-intensive applications

3. **Slow generation**
   - Reduce `steps` parameter
   - Use smaller `block_length`
   - Enable GPU acceleration

4. **DCP conversion fails**
   - Verify DCP checkpoint path is correct
   - Ensure base model name is accessible
   - Check disk space for temporary files

#### SLURM Mode Issues

1. **Job fails to submit**
   - Check `ACCOUNT` environment variable is set
   - Verify partition and resource limits
   - Check container image path exists

2. **Server not accessible**
   - Verify SSH tunnel is set up correctly
   - Check job is still running: `squeue -u $USER`
   - Check job logs for errors

3. **Container issues**
   - Ensure container image supports GPU (if using GPUs)
   - Check container has required dependencies
   - Verify mount paths are correct

4. **Job gets killed**
   - Increase time limit: `--time 24:00:00`
   - Increase memory: `--mem 128G`
   - Check cluster job limits

### Server Logs

#### Local Mode Logs
The server provides detailed logging for debugging:

```bash
# Enable debug logging (local mode)
# Make sure dependencies are installed with uv first
uv sync --locked --extra vllm --no-install-project
uv pip install fastapi uvicorn

export PYTHONPATH=/path/to/NeMo-RL:$PYTHONPATH
python llada_openai_server.py --model-path /path/to/model --log-level DEBUG
```

#### SLURM Mode Logs
For SLURM jobs, all output appears in real-time in your terminal:

```bash
# All server output is shown directly - no log files needed!
# Connection instructions appear automatically when the server starts

# Check job status
squeue -j JOB_ID

# Check job details
scontrol show job JOB_ID

# View resource usage
sacct -j JOB_ID --format=JobID,JobName,ReqMem,MaxRSS,Elapsed,State
```

### Performance Monitoring

#### Local Mode
```bash
# Monitor GPU usage
nvidia-smi -l 1

# Monitor server resources  
htop

# Test server load
ab -n 100 -c 10 -p test_request.json -T application/json http://localhost:8000/v1/chat/completions
```

#### SLURM Mode
```bash
# Monitor job in real-time
watch squeue -u $USER

# Monitor GPU usage on compute node (if accessible)
ssh COMPUTE_NODE nvidia-smi -l 1

# Check job efficiency after completion
seff JOB_ID
```

## Integration Examples

### With OpenAI Python Library

```python
import openai

# Configure client for local server
client = openai.OpenAI(base_url="http://localhost:8000/v1", api_key="dummy")

response = client.chat.completions.create(
    model="llada-8b-instruct",
    messages=[{"role": "user", "content": "Hello!"}],
    temperature=0.0,
    max_tokens=100,
    extra_body={  # LLaDA + Fast-dLLM parameters
        "steps": 128,
        "block_length": 32,
        "cfg_scale": 0.0,
        "remasking": "low_confidence",
        "use_cache": True,
        "use_dual_cache": True,
        "threshold": 0.8
    }
)

print(response.choices[0].message.content)
```

### With LangChain

```python
from langchain.llms.base import LLM
import requests

class LLaDALLM(LLM):
    base_url: str = "http://localhost:8000"
    temperature: float = 0.0
    steps: int = 128
    use_cache: bool = True
    use_dual_cache: bool = True
    
    def _call(self, prompt: str, stop=None) -> str:
        response = requests.post(f"{self.base_url}/v1/chat/completions", json={
            "messages": [{"role": "user", "content": prompt}],
            "temperature": self.temperature,
            "max_tokens": 200,
            "steps": self.steps,
            "use_cache": self.use_cache,
            "use_dual_cache": self.use_dual_cache
        })
        return response.json()["choices"][0]["message"]["content"]
    
    @property
    def _llm_type(self) -> str:
        return "llada"

# Usage
llm = LLaDALLM(temperature=0.0, steps=64)
response = llm("What is machine learning?")
```

## Contributing

When contributing to the LLaDA API server:

1. Follow the existing code style and patterns
2. Add appropriate error handling for new features
3. Update documentation for any new parameters
4. Test with both HuggingFace and DCP checkpoint formats
5. Ensure backward compatibility with OpenAI API standards

## License

This implementation follows the same license as the NeMo-RL project.
