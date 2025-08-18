# LLaDA API Troubleshooting Guide

Common issues and solutions for the LLaDA OpenAI API server.

## Import Errors

### `ModuleNotFoundError: No module named 'nemo_rl'`

**Problem**: When running locally, you get an import error for `nemo_rl`.

**Solutions**:

1. **Easiest**: Use HuggingFace models instead of DCP checkpoints:
   ```bash
   ./start_llada_server.sh --local --model-path GSAI-ML/LLaDA-8B-Instruct
   ```

2. **For DCP support**: Install NeMo-RL dependencies:
   ```bash
   # Make sure you're in the NeMo-RL project root
   uv sync --locked --extra vllm --no-install-project
   uv pip install fastapi uvicorn
   
   # Now DCP checkpoints will work
   ./start_llada_server.sh --local --dcp-path /path/to/checkpoint.dcp
   ```

3. **Manual PYTHONPATH**: The script automatically sets PYTHONPATH, but you can also set it manually:
   ```bash
   export PYTHONPATH=/path/to/NeMo-RL:$PYTHONPATH
   ./start_llada_server.sh --local --dcp-path /path/to/checkpoint.dcp
   ```

## Model Loading Issues

### `Failed to load model from HuggingFace model name`

**Problem**: HuggingFace model fails to download or load.

**Solutions**:
- Check your internet connection
- Verify the model name is correct: `GSAI-ML/LLaDA-8B-Instruct`
- Try with `trust_remote_code=True` (automatically enabled)
- Check HuggingFace Hub status

### `Model path does not exist`

**Problem**: Local model path not found.

**Solutions**:
- Verify the path is correct and accessible
- Use absolute paths instead of relative paths
- Try a HuggingFace model name instead: `--model-path GSAI-ML/LLaDA-8B-Instruct`

## Dependency Issues

### Missing PyTorch, FastAPI, or other packages

**Problem**: Required packages not installed.

**Solutions**:
```bash
# Using uv (recommended)
uv sync --locked --extra vllm --no-install-project
uv pip install fastapi uvicorn

# Using pip
pip install torch transformers fastapi uvicorn

# Using conda
conda install pytorch transformers fastapi uvicorn -c conda-forge -c pytorch
```

## SLURM Issues

### `ACCOUNT environment variable must be set`

**Problem**: SLURM account not configured.

**Solution**:
```bash
export ACCOUNT=your_slurm_account
./start_llada_server.sh --model-path GSAI-ML/LLaDA-8B-Instruct
```

### Container or job submission issues

**Problem**: SLURM job fails to submit or start.

**Solutions**:
- Check your SLURM account and partition settings
- Verify container image path exists
- Check resource limits (GPUs, memory, time)
- Look at job logs: `tail -f logs/llada_server/llada_server_JOBID.log`

## Performance Issues

### Slow model loading

**Problem**: Model takes a long time to load.

**Solutions**:
- Models download from HuggingFace on first use (normal)
- Use local models if you have them downloaded
- Check your internet speed for HuggingFace downloads
- Consider using smaller models for testing

### Out of memory errors

**Problem**: GPU or CPU memory exhausted.

**Solutions**:
- Use smaller models
- Reduce `max_tokens` in requests
- Reduce `block_length` parameter
- Close other GPU-intensive applications
- Use CPU mode if necessary (slower)

## API Usage Issues

### Connection refused

**Problem**: Can't connect to the API server.

**Solutions**:
- Check if the server is actually running: `ps aux | grep llada_openai_server`
- Verify the correct host and port
- For SLURM jobs, set up SSH tunnel properly
- Check firewall settings

### API returns errors

**Problem**: API requests fail with HTTP errors.

**Solutions**:
- Check the request format matches OpenAI API spec
- Verify required fields are present
- Check server logs for detailed error messages
- Use the provided examples as reference

## Quick Tests

### Test HuggingFace model loading
```bash
python3 -c "
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained('GSAI-ML/LLaDA-8B-Instruct', trust_remote_code=True)
print('✅ HuggingFace model loading works')
"
```

### Test server dependencies
```bash
python3 -c "
import fastapi, uvicorn, torch, transformers
print('✅ All required packages available')
"
```

### Test NeMo-RL availability
```bash
python3 -c "
try:
    import nemo_rl.utils.native_checkpoint
    print('✅ NeMo-RL available - DCP support enabled')
except ImportError:
    print('⚠️ NeMo-RL not available - use HF models only')
"
```

## Getting Help

1. **Check server logs**: Always check the server output for detailed error messages
2. **Use examples**: Start with the provided examples before customizing
3. **Try simpler configurations**: Use HuggingFace models before DCP checkpoints
4. **Check resources**: Ensure adequate GPU memory and disk space
5. **Verify environment**: Make sure all dependencies are properly installed

## Common Workflow

For troubleshooting, follow this workflow:

1. **Start simple**: Test with HuggingFace model first
   ```bash
   ./start_llada_server.sh --local --model-path GSAI-ML/LLaDA-8B-Instruct
   ```

2. **Check basic connectivity**: Test the health endpoint
   ```bash
   curl http://localhost:8000/health
   ```

3. **Try a simple request**: Use the provided client examples
   ```bash
   python xp/llada_api/examples/llada_api_client.py
   ```

4. **Add complexity gradually**: Move to DCP checkpoints or SLURM only after basic functionality works

This approach helps isolate issues and identify the root cause more quickly.
