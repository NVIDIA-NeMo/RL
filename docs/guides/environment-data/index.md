# Environment and Data

This section covers environment setup, data management, debugging, and performance profiling for NeMo RL training workflows.

## Overview

NeMo RL training requires careful attention to environment configuration, data preparation, and debugging tools. This guide covers all aspects of setting up and maintaining a robust training environment.

## Quick Navigation

For GPU profiling and performance optimization, see the [Development](../../development/index) section.

::::{grid} 1 2 2 2
:gutter: 2 2 2 2

:::{grid-item-card} {octicon}`gear;1.5em;sd-mr-1` Environment Development
:link: environment-development
:link-type: doc

Set up and configure your development environment for NeMo RL.

+++
{bdg-primary}`Setup`
:::

:::{grid-item-card} {octicon}`container;1.5em;sd-mr-1` Environments
:link: environments
:link-type: doc

Create and manage custom RL environments.

+++
{bdg-info}`Intermediate`
:::

::::

## Environment Setup

### System Requirements

NeMo RL has specific requirements for optimal performance:

**Hardware Requirements:**
- **GPU**: NVIDIA GPU with 8GB+ VRAM (recommended: A100, H100)
- **CPU**: 8+ cores recommended
- **RAM**: 32GB+ system memory
- **Storage**: Fast SSD for data loading

**Software Requirements:**
- **CUDA**: 11.8+ or 12.1+
- **Python**: 3.9-3.11
- **PyTorch**: 2.0+ with CUDA support
- **Ray**: Latest stable version

### Installation

```bash
# Install NeMo RL
pip install nemo-rl

# Optional extras
pip install nemo-rl[vllm]
pip install nemo-rl[mcore]
```

### Environment Variables

Configure your environment with these variables:

```bash
# Logging
export NEMO_RL_LOG_LEVEL=INFO
export NEMO_RL_LOG_FILE=/path/to/logs/nemo_rl.log

# Cache and data
export NEMO_RL_CACHE_DIR=/path/to/cache
export HF_HOME=/path/to/huggingface/cache

# Ray configuration
export RAY_ADDRESS=auto
export RAY_DISABLE_IMPORT_WARNING=1

# CUDA configuration
export CUDA_VISIBLE_DEVICES=0,1,2,3
export NCCL_DEBUG=INFO
```

## Data Management

### Dataset Preparation

NeMo RL supports various data formats and sources:

**HuggingFace Datasets:**
```python
# See examples/run_dpo.py for dataset setup via nemo_rl.data.hf_datasets and AllTaskProcessedDataset
```

**Custom Data Formats:**
```python
# DPO format
dpo_data = [
    {
        "prompt": "What is 2+2?",
        "chosen": "The answer is 4.",
        "rejected": "I don't know."
    }
]

# GRPO format
grpo_data = [
    {
        "prompt": "Solve: 3x + 5 = 20",
        "response": "x = 5",
        "reward": 1.0
    }
]
```

### Data Preprocessing

```python
# Preprocessing is implemented in examples (see run_dpo.py dpo_preprocessor and AllTaskProcessedDataset)
```

### Data Validation

```python
# Validate via lightweight checks in your pipeline; no standalone validate_dataset API in nemo_rl.data
```

## Performance Optimization

### Memory Management

**Gradient Checkpointing:**
```yaml
# In configuration file
training:
  gradient_checkpointing: true
  gradient_accumulation_steps: 4
```

**Mixed Precision:**
```yaml
training:
  precision: "bf16"  # or "fp16"
  autocast: true
```

### Data Loading Optimization

```python
# Optimize data loading
dataset = dataset.with_format("torch")
dataset = dataset.map(
    preprocess_function,
    batched=True,
    num_proc=4,
    remove_columns=dataset.column_names
)
```

## Monitoring and Logging

### Training Metrics

NeMo RL automatically logs key metrics:

- **Loss**: Training and validation loss
- **Reward**: Average reward per episode (GRPO)
- **Accuracy**: Preference accuracy (DPO)
- **Memory**: GPU and system memory usage
- **Throughput**: Samples per second

### Custom Logging

```python
from nemo_rl.utils import get_logger

logger = get_logger(__name__)

# Log custom metrics
logger.info(f"Custom metric: {value}")

# Log to external systems
logger.add_handler(custom_handler)
```

## Troubleshooting

### Common Issues

1. **Out of Memory (OOM)**
   ```bash
   # Reduce batch size
   nemo-rl train --config config.yaml training.batch_size=2
   
   # Enable gradient checkpointing
   nemo-rl train --config config.yaml training.gradient_checkpointing=true
   ```

2. **Data Loading Issues**
   ```bash
   # Validate dataset
   nemo-rl validate --config config.yaml --check-data
   
   # Debug data loading
   nemo-rl debug --config config.yaml --check-data
   ```

3. **Ray Connection Issues**
   ```bash
   # Check Ray status
   ray status
   
   # Restart Ray cluster
   ray stop
   ray start --head
   ```

### Debug Mode

Enable debug mode for detailed logging:

```bash
nemo-rl train --config config.yaml --debug
```

## Best Practices

### Environment Setup

1. **Use virtual environments** for isolation
2. **Pin dependency versions** for reproducibility
3. **Monitor system resources** during training
4. **Use SSD storage** for data loading
5. **Configure proper logging** for debugging

### Data Management

1. **Validate datasets** before training
2. **Use efficient data formats** (parquet, arrow)
3. **Implement proper preprocessing** pipelines
4. **Monitor data quality** and distribution
5. **Cache processed data** when possible

### Performance

1. **Profile training** with NSYS
2. **Optimize data loading** with multiple workers
3. **Use mixed precision** when possible
4. **Monitor memory usage** and adjust batch sizes
5. **Scale horizontally** with Ray clusters

## Next Steps

- [Debugging Guide](../../development/debugging) - Learn debugging techniques
- [GPU Profiling](../../development/nsys-profiling) - Profile training performance with NSYS

For additional learning resources, visit the main [Guides](../index) page.

---

::::{toctree}
:hidden:
:caption: Environment and Data
:maxdepth: 2
environment-development
environments
:::: 
 