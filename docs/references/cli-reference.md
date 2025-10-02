# CLI Reference

This document provides comprehensive reference documentation for all NeMo RL command-line interface tools and utilities.

## Overview

NeMo RL provides Python scripts for training, evaluation, and model management. These scripts follow a consistent pattern and provide detailed help information.

## Command Structure

```bash
python examples/run_<algorithm>.py --config path/to/config.yaml [options]
```

## Main Commands

### Training Commands

#### `python examples/run_dpo.py`

Start DPO (Direct Preference Optimization) training with the specified configuration.

```bash
python examples/run_dpo.py --config path/to/config.yaml [options]
```

**Options:**
- `--config`: Path to configuration file (required)
- Additional Hydra overrides can be passed as command-line arguments

**Examples:**
```bash
# Basic DPO training
python examples/run_dpo.py --config examples/configs/dpo.yaml

# With Hydra overrides
python examples/run_dpo.py --config examples/configs/dpo.yaml data.train_data_path=/path/to/data
```

#### `python examples/run_grpo_math.py`

Start GRPO (Group Relative Policy Optimization) training for mathematical reasoning tasks.

```bash
python examples/run_grpo_math.py --config path/to/config.yaml [options]
```

**Options:**
- `--config`: Path to configuration file (required)
- Additional Hydra overrides can be passed as command-line arguments

**Examples:**
```bash
# Basic GRPO math training
python examples/run_grpo_math.py --config examples/configs/grpo_deepscaler-1.5b-24K.yaml

# With custom parameters
python examples/run_grpo_math.py --config examples/configs/grpo_deepscaler-1.5b-24K.yaml policy.model_name=deepscaler-1.5b
```

#### `python examples/run_grpo_sliding_puzzle.py`

Start GRPO training for sliding puzzle tasks.

```bash
python examples/run_grpo_sliding_puzzle.py --config path/to/config.yaml [options]
```

**Options:**
- `--config`: Path to configuration file (required)
- Additional Hydra overrides can be passed as command-line arguments

#### `python examples/run_sft.py`

Start SFT (Supervised Fine-Tuning) training.

```bash
python examples/run_sft.py --config path/to/config.yaml [options]
```

**Options:**
- `--config`: Path to configuration file (required)
- Additional Hydra overrides can be passed as command-line arguments

### Evaluation Commands

#### `python examples/run_eval.py`

Evaluate a trained model on mathematical reasoning tasks.

```bash
python examples/run_eval.py --config path/to/eval_config.yaml [options]
```

**Options:**
- `--config`: Path to evaluation configuration file (required)
- Additional Hydra overrides can be passed as command-line arguments

**Examples:**
```bash
# Basic evaluation
python examples/run_eval.py --config examples/configs/eval.yaml

# With custom model path
python examples/run_eval.py --config examples/configs/eval.yaml policy.model_name=deepscaler-1.5b
```

### Model Management

#### `python examples/converters/convert_dcp_to_hf.py`

Convert models from DCP (Distributed Checkpoint) format to HuggingFace format.

```bash
python examples/converters/convert_dcp_to_hf.py --input path/to/model --output path/to/output
```

**Options:**
- `--input`: Input model path (required)
- `--output`: Output path (required)
- `--config`: Conversion configuration file

#### `python examples/converters/convert_megatron_to_hf.py`

Convert models from Megatron format to HuggingFace format.

```bash
python examples/converters/convert_megatron_to_hf.py --input path/to/model --output path/to/output
```

**Options:**
- `--input`: Input model path (required)
- `--output`: Output path (required)
- `--config`: Conversion configuration file

## Configuration Options

### Global Options

All scripts support these global options:

- `--config`: Path to YAML configuration file (required)
- Additional Hydra overrides can be passed as command-line arguments

### Environment Variables

NeMo RL respects these environment variables:

- `RAY_ADDRESS`: Ray cluster address
- `NRL_NSYS_WORKER_PATTERNS`: Enable nsys profiling
- `HF_HOME`: HuggingFace cache directory
- `HF_TOKEN`: HuggingFace authentication token
- `WANDB_API_KEY`: Weights & Biases API key for logging
- `HF_DATASETS_CACHE`: HuggingFace datasets cache directory
- `RAY_ENABLE_MULTI_PROCESSES_GPU_COUNT`: Enable multi-process GPU support

## Examples

### Complete Training Workflow

```bash
# 1. Start DPO training
python examples/run_dpo.py --config examples/configs/dpo.yaml

# 2. Start GRPO training for math
python examples/run_grpo_math.py --config examples/configs/grpo_deepscaler-1.5b-24K.yaml

# 3. Evaluate model
python examples/run_eval.py --config examples/configs/eval.yaml

# 4. Convert model to HuggingFace format
python examples/converters/convert_dcp_to_hf.py --input checkpoints/model.pt --output hf_model
```

### Development Workflow

```bash
# Local development with custom parameters
python examples/run_dpo.py --config examples/configs/dpo.yaml data.train_data_path=/path/to/data

# Quick evaluation
python examples/run_eval.py --config examples/configs/eval.yaml

# Model conversion
python examples/converters/convert_megatron_to_hf.py --input megatron_model --output hf_model
```

## Troubleshooting

### Common Issues

1. **Configuration Errors**
   ```bash
   # Check configuration file syntax
   python -c "from omegaconf import OmegaConf; OmegaConf.load('your_config.yaml')"
   ```

2. **Memory Issues**
   ```bash
   # Use smaller batch size
   python examples/run_dpo.py --config examples/configs/dpo.yaml data.batch_size=1
   ```

3. **Ray Connection Issues**
   ```bash
   # Check Ray status
   ray status
   
   # Start local Ray cluster
   ray start --head
   ```

## Getting Help

- [Configuration Reference](configuration-reference) - Complete configuration options
- [Troubleshoot NeMo RL](../guides/troubleshooting) - Common issues and solutions
- [API Documentation](../apidocs/index) - Complete API documentation
- [Community Support](https://github.com/NVIDIA/NeMo-RL/issues) - GitHub issues and discussions

## Advanced Usage

### Custom Scripts

You can integrate NeMo RL scripts into custom workflows:

```bash
#!/bin/bash
# Training script example

CONFIG="examples/configs/dpo.yaml"
CHECKPOINT_DIR="checkpoints"

# Start training
python examples/run_dpo.py --config $CONFIG

# Wait for completion and evaluate
if [ -f "$CHECKPOINT_DIR/best.pt" ]; then
    python examples/run_eval.py --config examples/configs/eval.yaml
fi
```

### Batch Processing

```bash
# Process multiple configurations
for config in configs/*.yaml; do
    echo "Processing $config"
    python examples/run_dpo.py --config "$config"
done
```

## Integration with Other Tools

### Ray Dashboard

When using Ray clusters, access the dashboard at `http://localhost:8265` to monitor training progress.

### TensorBoard

Training logs are automatically saved and can be viewed with TensorBoard:

```bash
tensorboard --logdir logs/
```

### MLflow

NeMo RL integrates with MLflow for experiment tracking:

```bash
# Enable MLflow logging in configuration
python examples/run_dpo.py --config config_with_mlflow.yaml
```

## Best Practices

1. **Always validate configurations** before training
2. **Use descriptive configuration names** for organization
3. **Monitor resources** when using clusters
4. **Save checkpoints regularly** for recovery
5. **Log all experiments** for reproducibility
6. **Use Hydra overrides** for quick parameter changes

## Reference

- [Configuration Reference](configuration-reference)
- [Troubleshoot NeMo RL](../guides/troubleshooting)
- [API Documentation](../apidocs/index) 