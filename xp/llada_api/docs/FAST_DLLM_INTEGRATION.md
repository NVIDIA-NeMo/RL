# Fast-dLLM Integration

The LLaDA OpenAI server has been enhanced with [Fast-dLLM](https://github.com/NVlabs/Fast-dLLM) acceleration, providing significant speedup improvements for LLaDA diffusion model inference through KV caching and parallel decoding optimizations.

## Overview

Fast-dLLM provides three key optimizations:
1. **KV Cache for Block-Wise Decoding** - Reuses attention Key-Value activations across multiple steps
2. **DualCache Extension** - Caches both prefix and suffix tokens for greater speedup  
3. **Confidence-Aware Parallel Decoding** - Decodes tokens in parallel based on confidence thresholds

## Performance Benefits

According to the Fast-dLLM paper, you can expect:
- **Significant speedup** with KV Cache mechanism alone
- **Enhanced performance** when combined with parallel decoding strategies
- **Maximum acceleration** on longer sequences with dual cache + parallel decoding

## New API Parameters

The server now supports additional Fast-dLLM specific parameters:

### Cache Optimization Parameters
- `use_cache` (bool, default: `true`): Enable KV caching
- `use_dual_cache` (bool, default: `true`): Enable dual cache (both prefix and suffix)

### Parallel Decoding Parameters  
- `threshold` (float, optional): Confidence threshold for parallel decoding
- `factor` (float, optional): Factor for dynamic parallel decoding strategy

### Updated Defaults
- `temperature`: `0.0` (optimized for Fast-dLLM)
- `max_tokens`: `128` (Fast-dLLM default)  
- `steps`: `128` (increased for better quality)
- `block_length`: `32` (optimized block size)

## Usage Examples

### Basic Request (Dual Cache Enabled)
```json
{
  "model": "llada-8b-instruct",
  "messages": [{"role": "user", "content": "What is 2+2?"}],
  "max_tokens": 128,
  "use_cache": true,
  "use_dual_cache": true
}
```

### Confidence-Aware Parallel Decoding
```json
{
  "model": "llada-8b-instruct", 
  "messages": [{"role": "user", "content": "Solve this math problem step by step."}],
  "max_tokens": 256,
  "use_cache": true,
  "use_dual_cache": true,
  "threshold": 0.8,
  "steps": 128,
  "block_length": 32
}
```

### Dynamic Parallel Decoding  
```json
{
  "model": "llada-8b-instruct",
  "messages": [{"role": "user", "content": "Write a short story."}], 
  "max_tokens": 512,
  "use_cache": true,
  "use_dual_cache": true,
  "factor": 2.0,
  "steps": 256,
  "block_length": 64
}
```

## Cache Strategies

The server automatically selects the optimal generation strategy based on your parameters:

1. **No Cache** (`use_cache: false`): Uses basic generation without any caching
2. **Prefix Cache** (`use_cache: true, use_dual_cache: false`): Caches prefix activations only  
3. **Dual Cache** (`use_cache: true, use_dual_cache: true`): Caches both prefix and suffix (fastest)

## Testing

### API Testing
Use the provided test script to verify the integration:

```bash
# Start the server  
python llada_openai_server.py --model-path GSAI-ML/LLaDA-8B-Instruct

# Run tests in another terminal
python test_fast_dllm_integration.py
```

The test script will benchmark different cache configurations and show performance improvements.

### Evaluation with NeMo-Skills

The `eval_llada.py` script now supports Fast-dLLM parameters for comprehensive benchmarking:

```bash
# Default evaluation with Fast-dLLM acceleration
python xp/nemo-skills/eval_llada.py

# Compare different acceleration strategies
python xp/nemo-skills/eval_llada.py --use-cache --use-dual-cache --threshold 0.8
python xp/nemo-skills/eval_llada.py --no-cache  # Baseline comparison
python xp/nemo-skills/eval_llada.py --factor 2.0 --steps 256  # Dynamic parallel decoding

# Quick test mode for development
python xp/nemo-skills/eval_llada.py --quick-test --use-dual-cache
```

### Automated Benchmarking

Use the comprehensive benchmark script to compare all acceleration strategies:

```bash
python xp/nemo-skills/benchmark_fast_dllm.py
```

This script will automatically run multiple configurations and provide performance comparisons.

## Requirements

- Fast-dLLM submodule must be present in `../../3rdparty/Fast-dLLM/`
- PyTorch with CUDA support recommended for best performance
- LLaDA model compatible with Fast-dLLM optimizations

## Troubleshooting

### Fast-dLLM Import Errors
If you see "Fast-dLLM not available" warnings:
1. Ensure the git submodule was added correctly: `git submodule update --init --recursive`
2. Verify the Fast-dLLM path exists: `ls 3rdparty/Fast-dLLM/llada/`
3. Check Python can import the modules from that path

### Block Length Compatibility
The server automatically adjusts `gen_length` to be divisible by `block_length` for compatibility with Fast-dLLM's block-wise generation requirements.

### Memory Requirements
Fast-dLLM caching may require additional GPU memory. If you encounter OOM errors:
- Reduce `max_tokens` or `block_length` 
- Set `use_cache: false` to disable caching
- Use CPU inference for smaller models

## Integration Details

The integration replaces the custom `generate_llada()` function with Fast-dLLM's optimized generation functions:
- `generate()` - Basic generation
- `generate_with_prefix_cache()` - Prefix caching  
- `generate_with_dual_cache()` - Dual caching (recommended)

The server loads models using `LLaDAModelLM` from Fast-dLLM when available, falling back to standard HuggingFace `AutoModel` otherwise.
