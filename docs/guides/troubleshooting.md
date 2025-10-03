---
description: "Comprehensive troubleshooting guide for NeMo RL covering common errors, configuration issues, and solutions for different deployment scenarios"
categories: ["deployment-operations"]
tags: ["troubleshooting", "debugging", "errors", "configuration", "support"]
personas: ["mle-focused", "admin-focused", "devops-focused"]
difficulty: "intermediate"
content_type: "reference"
modality: "universal"
---

# Troubleshoot NeMo RL

This guide covers common issues, error messages, and solutions for NeMo RL. If you encounter a problem not covered here, please check the [GitHub Issues](https://github.com/NVIDIA/NeMo-RL/issues) or create a new one.

## Installation Issues

### Missing Submodules (ModuleNotFoundError)

**Error Message:**
```
ModuleNotFoundError: No module named 'megatron'
```

**Why This Happens:**
The NeMo RL repository uses git submodules for optional third-party dependencies like Megatron-LM and Megatron-Bridge. If you cloned the repository without the `--recursive` flag, these submodules won't be initialized, causing import errors when trying to use Megatron backend features.

**Solutions:**

1. **Initialize submodules** (if you forgot during initial clone):
   ```bash
   git submodule update --init --recursive
   ```

2. **Force rebuild virtual environments** (required after adding submodules):
   
   After initializing submodules, your existing virtual environments won't have access to the new dependencies. Force a rebuild:
   
   ```bash
   NRL_FORCE_REBUILD_VENVS=true uv run examples/run_grpo_math.py
   ```
   
   This environment variable tells NeMo RL to rebuild all isolated virtual environments with the newly available submodules.

3. **Verify submodules are properly initialized:**
   ```bash
   # Check if Megatron-LM is present
   ls 3rdparty/Megatron-LM-workspace/Megatron-LM
   
   # Check if Megatron-Bridge is present
   ls 3rdparty/Megatron-Bridge-workspace/Megatron-Bridge
   ```

**Prevention:**
Always clone with the `--recursive` flag to automatically initialize all submodules:
```bash
git clone --recursive git@github.com:NVIDIA-NeMo/RL.git nemo-rl
```

**When Switching Branches:**
Different branches may have different submodule versions. After switching branches or pulling updates:
```bash
git submodule update --init --recursive
NRL_FORCE_REBUILD_VENVS=true uv run <your_command>
```

**Note:** Most users can use the default HuggingFace/DTensor backend without initializing submodules. Only initialize submodules if you specifically need Megatron backend support.

## Common Errors

### CUDA Out of Memory

**Error Message:**
```
RuntimeError: CUDA out of memory. Tried to allocate X MiB
```

**Solutions:**

1. **Reduce batch size:**
   ```yaml
   algorithm:
     batch_size: 2  # Reduce from 4 to 2
   ```

2. **Enable gradient accumulation:**
   ```yaml
   algorithm:
     gradient_accumulation_steps: 4
     batch_size: 1
   ```

3. **Reduce model size or use smaller model:**
   ```yaml
   model:
     name: "llama2-7b"  # Use smaller model
   ```

4. **Enable memory optimization:**
   ```yaml
   model:
     use_cache: false
     torch_dtype: "float16"
   ```

5. **Use actual memory management patterns from codebase:**
   ```python
   # Force memory cleanup
   import torch
   import gc
   
   torch.cuda.empty_cache()
   gc.collect()
   
   # Move model to CPU if needed
   model = model.cpu()
   torch.cuda.empty_cache()
   ```

6. **Fix memory fragmentation** (for models without FlashAttention2 support):
   
   **Why This Happens:**
   Large amounts of memory fragmentation can occur when running models without FlashAttention2 support. If OOM occurs after a few iterations of training, the PyTorch CUDA memory allocator may benefit from different settings.
   
   **Solutions:**
   
   **Option A: Set environment variable globally** (applies to all Ray actors):
   ```bash
   PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:64 uv run python examples/run_dpo.py ...
   ```
   
   **Option B: Configure in YAML** (more permanent, policy-specific):
   ```yaml
   policy:
     dtensor_cfg:
       env_vars:
         PYTORCH_CUDA_ALLOC_CONF: "max_split_size_mb:64"
   ```
   
   Or for Megatron backend:
   ```yaml
   policy:
     megatron_cfg:
       env_vars:
         PYTORCH_CUDA_ALLOC_CONF: "max_split_size_mb:64"
   ```
   
   **What this does:**
   The `max_split_size_mb` parameter controls how PyTorch's caching allocator splits large memory blocks. Setting it to 64MB can reduce fragmentation at the cost of slightly more allocation overhead.
   
   **Related PyTorch documentation:**
   See [PyTorch CUDA memory management](https://docs.pytorch.org/docs/stable/notes/cuda.html#optimizing-memory-usage-with-pytorch-cuda-alloc-conf) for more allocator configuration options.

### Model Loading Errors

**Error Message:**
```
OSError: We couldn't connect to 'https://huggingface.co/...'
```

**Solutions:**

1. **Check internet connection**
2. **Use local model path:**
   ```yaml
   model:
     name: "/path/to/local/model"
   ```
3. **Set Hugging Face token:**
   ```bash
   export HF_TOKEN="your_token"
   ```

**Real Model Loading Patterns from Codebase:**

```python
# Actual model loading pattern from codebase
from transformers import AutoModelForCausalLM, AutoTokenizer

try:
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="cpu",  # Load on CPU first for large models
        torch_dtype=torch.float32,  # Use float32 for stability
        trust_remote_code=True,  # Required for custom models
        **sliding_window_overwrite(model_name),  # For specific models
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
except Exception as e:
    print(f"Model loading failed: {e}")
    # Try with different parameters
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True
    )
```

### Configuration Validation Errors

**Error Message:**
```
ValidationError: Invalid configuration parameter
```

**Solutions:**

1. **Use actual configuration loading:**
   ```python
   from nemo_rl.utils.config import load_config, parse_hydra_overrides
   
   try:
       config = load_config("training.yaml")
       # Apply overrides
       config = parse_hydra_overrides(config, ["algorithm.learning_rate=1e-5"])
   except Exception as e:
       print(f"Configuration error: {e}")
   ```

2. **Check parameter names and types**
3. **Use configuration inheritance:**
   ```yaml
   # child.yaml
   defaults: parent.yaml
   algorithm:
     learning_rate: 1e-5
   ```

### DevOps Automation for Configuration Management

For DevOps professionals managing multiple environments:

```yaml
# config-management.yaml
environments:
  development:
    algorithm:
      batch_size: 2
      learning_rate: 1e-5
    model:
      use_cache: true
      torch_dtype: "float16"
  
  staging:
    algorithm:
      batch_size: 4
      learning_rate: 1e-5
    model:
      use_cache: false
      torch_dtype: "float16"
  
  production:
    algorithm:
      batch_size: 8
      learning_rate: 5e-6
    model:
      use_cache: false
      torch_dtype: "bfloat16"
```

```bash
# Automated configuration deployment
#!/bin/bash
ENVIRONMENT=$1
CONFIG_FILE="configs/${ENVIRONMENT}.yaml"

if [ ! -f "$CONFIG_FILE" ]; then
    echo "Configuration file $CONFIG_FILE not found"
    exit 1
fi

# Validate configuration using actual codebase patterns
uv run python -c "
from nemo_rl.utils.config import load_config
try:
    config = load_config('$CONFIG_FILE')
    print('Configuration valid')
except Exception as e:
    print(f'Configuration error: {e}')
    exit(1)
"

# Deploy with environment-specific settings
uv run python examples/run_grpo_math.py --config "$CONFIG_FILE"
```

## Configuration Issues

### Missing Required Parameters

**Error Message:**
```
Missing required parameter: algorithm.name
```

**Solutions:**

1. **Add missing parameter to configuration:**
   ```yaml
   algorithm:
     name: "dpo"  # Add this line
   ```

2. **Use configuration inheritance:**
   ```yaml
   # base.yaml
   algorithm:
     name: "dpo"
     learning_rate: 1e-5
   
   # training.yaml
   defaults: base.yaml
   algorithm:
     batch_size: 4
   ```

### Invalid Parameter Values

**Error Message:**
```
Invalid value for learning_rate: must be positive
```

**Solutions:**

1. **Check parameter ranges:**
   - Learning rates: positive values
   - Batch sizes: positive integers
   - Model names: valid paths or names

2. **Use reasonable defaults:**
   ```yaml
   algorithm:
     learning_rate: 1e-5  # Use small positive value
     batch_size: 4        # Use reasonable batch size
   ```

### Environment Variable Issues

**Problem:** Environment variables not being recognized

**Solutions:**

1. **Check variable naming:**
   ```bash
   # Correct format
   export NEMO_RL_ALGORITHM_LEARNING_RATE=2e-5
   export NEMO_RL_MODEL_NAME="llama2-7b"
   ```

2. **Verify variable is set:**
   ```bash
   echo $NEMO_RL_ALGORITHM_LEARNING_RATE
   ```

3. **Restart terminal after setting variables**

## Distributed Training Issues

### Ray Connection Errors

**Error Message:**
```
RayConnectionError: Failed to connect to Ray cluster
```

**Solutions:**

1. **Start Ray cluster:**
   ```bash
   ray start --head
   ```

2. **Check Ray status:**
   ```bash
   ray status
   ray list nodes
   ```

3. **Use local mode for testing:**
   ```yaml
   distributed:
     backend: "ray"
     local_mode: true
   ```

### Worker Allocation Errors

**Error Message:**
```
Resource allocation failed: insufficient resources
```

**Solutions:**

1. **Reduce resource requirements:**
   ```yaml
   distributed:
     num_workers: 2        # Reduce from 4
     num_gpus_per_worker: 1
     memory_per_worker: "8GB"  # Reduce from 16GB
   ```

2. **Check available resources:**
   ```bash
   nvidia-smi  # Check GPU availability
   ray status  # Check Ray cluster resources
   ```

3. **Use single GPU mode:**
   ```yaml
   distributed:
     backend: "torch"  # Use PyTorch DDP instead of Ray
   ```

### Communication Errors

**Error Message:**
```
NCCL error: unhandled cuda error
```

**Solutions:**

1. **Check GPU connectivity:**
   ```bash
   nvidia-smi topo -m
   ```

2. **Use different communication backend:**
   ```yaml
   distributed:
     communication:
       backend: "gloo"  # Use gloo instead of nccl
   ```

3. **Reduce batch size and workers**

## Model Issues

### Model Loading Failures

**Error Message:**
```
RuntimeError: Error(s) in loading state_dict
```

**Solutions:**

1. **Check model compatibility:**
   ```yaml
   model:
     backend: "huggingface"  # Ensure correct backend
     trust_remote_code: true  # For custom models
   ```

2. **Use actual model loading patterns:**
   ```python
   # From codebase - actual model loading pattern
   model = AutoModelForCausalLM.from_pretrained(
       model_name,
       device_map="cpu",  # Load on CPU first
       torch_dtype=torch.float32,  # Use float32 for stability
       trust_remote_code=True,
       **sliding_window_overwrite(model_name),
   )
   ```

3. **Verify model path and format**
4. **Use compatible model version**

### Generation Errors

**Error Message:**
```
RuntimeError: Expected tensor to have size X but got size Y
```

**Solutions:**

1. **Check input format:**
   ```python
   # Ensure inputs are properly tokenized
   inputs = tokenizer(prompts, return_tensors="pt", padding=True)
   ```

2. **Verify model configuration:**
   ```yaml
   generation:
     max_new_tokens: 512
     pad_token_id: tokenizer.pad_token_id
     eos_token_id: tokenizer.eos_token_id
   ```

### Memory Issues During Training

**Error Message:**
```
RuntimeError: CUDA out of memory during training
```

**Solutions:**

1. **Enable memory optimization:**
   ```yaml
   model:
     use_cache: false
     torch_dtype: "float16"
     device_map: "auto"
   ```

2. **Use gradient checkpointing:**
   ```yaml
   algorithm:
     gradient_checkpointing: true
   ```

3. **Use actual memory management from codebase:**
   ```python
   # From codebase - actual memory management
   def offload_before_refit(self):
       torch.cuda.empty_cache()
   
   def offload_after_refit(self):
       self.model = self.move_to_cpu(self.model)
       self.model.eval()
       torch.randn(1).cuda()  # wake up torch allocator
       self.offload_before_refit()
       
       # Clean up held tensors
       if self._held_sharded_state_dict_reference is not None:
           del self._held_sharded_state_dict_reference
           self._held_sharded_state_dict_reference = None
       
       gc.collect()
       torch.cuda.empty_cache()
   ```

4. **Reduce model precision:**
   ```yaml
   model:
     torch_dtype: "bfloat16"  # Use bfloat16 for better memory efficiency
   ```

## Data Issues

### Dataset Loading Errors

**Error Message:**
```
DatasetNotFoundError: Dataset not found
```

**Solutions:**

1. **Check dataset path:**
   ```yaml
   data:
     dataset: "/correct/path/to/dataset"  # Use absolute path
   ```

2. **Verify dataset format:**
   - Ensure dataset is in expected format
   - Check required columns are present

3. **Use HuggingFace dataset:**
   ```yaml
   data:
     dataset: "huggingface-dataset-name"
   ```

### Data Preprocessing Errors

**Error Message:**
```
TokenizationError: Input too long
```

**Solutions:**

1. **Reduce sequence length:**
   ```yaml
   data:
     preprocessing:
       max_length: 1024  # Reduce from 2048
   ```

2. **Enable truncation:**
   ```yaml
   data:
     preprocessing:
       truncation: true
       padding: "max_length"
   ```

## Performance Issues

### Slow Training

**Problem:** Training is slower than expected

**Solutions:**

1. **Optimize data loading:**
   ```yaml
   data:
     num_workers: 8        # Increase from 4
     pin_memory: true
     prefetch_factor: 2
   ```

2. **Use mixed precision:**
   ```yaml
   algorithm:
     mixed_precision: true
     fp16: true
   ```

3. **Enable optimizations:**
   ```yaml
   model:
     use_flash_attention: true
     use_cache: false
   ```

### High Memory Usage

**Problem:** Excessive memory consumption

**Solutions:**

1. **Reduce batch size and use gradient accumulation:**
   ```yaml
   algorithm:
     batch_size: 1
     gradient_accumulation_steps: 4
   ```

2. **Enable memory optimizations:**
   ```yaml
   model:
     use_cache: false
     torch_dtype: "float16"
   ```

3. **Use smaller model or model parallelism**

## Debugging Tips

### Enable Debug Logging

```bash
uv run python examples/run_grpo_math.py --config training.yaml --log-level DEBUG
```

### Validate Configuration

```python
# Use actual configuration validation
from nemo_rl.utils.config import load_config

try:
    config = load_config("training.yaml")
    print("Configuration loaded successfully")
except Exception as e:
    print(f"Configuration error: {e}")
```

### Check System Resources

```bash
# Check GPU usage
nvidia-smi

# Check memory usage
free -h

# Check Ray cluster
ray status
ray list nodes
ray memory --stats-only
```

### Dry Run Mode

```bash
uv run python examples/run_grpo_math.py --config training.yaml --dry-run
```

## Get Help

### Before Asking for Help

1. **Check this troubleshooting guide**
2. **Search existing issues on GitHub**
3. **Validate your configuration**
4. **Check system requirements**
5. **Try with minimal configuration**

### Providing Information

When reporting issues, include:

1. **Error message and stack trace**
2. **Configuration file (sanitized)**
3. **System information:**
   ```bash
   uv run python --version
   nvidia-smi
   ray status
   ```
4. **Steps to reproduce**
5. **Expected vs actual behavior**

### Useful Commands

```bash
# Check NeMo RL version
uv run python -c "import nemo_rl; print(nemo_rl.__version__)"

# Validate configuration using actual codebase
uv run python -c "
from nemo_rl.utils.config import load_config
try:
    config = load_config('training.yaml')
    print('Configuration valid')
except Exception as e:
    print(f'Configuration error: {e}')
"

# Test model loading
uv run python -c "
from transformers import AutoModelForCausalLM
model = AutoModelForCausalLM.from_pretrained('llama2-7b', trust_remote_code=True)
print('Model loading successful')
"

# Check Ray cluster
ray status
ray list nodes
ray memory --stats-only
```

## Common Workarounds

### Temporary Solutions

1. **Use smaller model for testing**
2. **Reduce batch size and sequence length**
3. **Use single GPU mode**
4. **Disable optimizations temporarily**

### Performance vs Memory Trade-offs

- **Higher batch size** → Better performance, more memory
- **Longer sequences** → More context, more memory
- **Mixed precision** → Less memory, potential precision loss
- **Gradient accumulation** → Same effective batch size, less memory

### Configuration Templates

Use provided templates as starting points:

```yaml
# Example configuration structure from codebase
model:
  name: "llama2-7b"
  backend: "huggingface"
  torch_dtype: "bfloat16"
  trust_remote_code: true

algorithm:
  name: "dpo"
  learning_rate: 1e-5
  batch_size: 4

distributed:
  backend: "ray"
  num_workers: 2
  num_gpus_per_worker: 1
```
 