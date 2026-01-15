# Configuration Reference

NeMo RL configuration can be done programmatically through function parameters or via configuration dictionaries.

## Configuration Methods

### Method 1: Function Parameters (Recommended)

The simplest approach - pass parameters directly:

```python
import nemo_rl

result = nemo_rl.train(
    model="Qwen/Qwen2.5-1.5B",
    dataset="nvidia/HelpSteer2",
    reward_fn=my_reward,
    
    # Training parameters
    algorithm="grpo",
    max_steps=1000,
    learning_rate=1e-6,
    batch_size=32,
)
```

### Method 2: Trainer Constructor (More Control)

Pass configuration dictionary to trainer:

```python
from nemo_rl import GRPOTrainer

config = {
    "policy": {
        "model_name": "Qwen/Qwen2.5-1.5B",
        "learning_rate": 1e-6,
    },
    "grpo": {
        "num_prompts_per_step": 32,
        "num_generations_per_prompt": 16,
    },
}

trainer = GRPOTrainer(config)
```

### Method 3: YAML Files (Legacy)

Load configuration from YAML:

```python
from omegaconf import OmegaConf

cfg = OmegaConf.load("config.yaml")
trainer = GRPOTrainer.from_config(cfg)
```

## Configuration Sections

### Policy Configuration

Model and training settings:

```python
{
    "policy": {
        "model_name": "Qwen/Qwen2.5-1.5B",      # Model identifier
        "train_global_batch_size": 512,          # Total batch size
        "train_micro_batch_size": 4,             # Per-GPU batch size
        "max_total_sequence_length": 512,        # Max token length
        "precision": "bfloat16",                 # Model precision
        "max_grad_norm": 1.0,                    # Gradient clipping
        
        # Optimizer
        "optimizer": {
            "name": "torch.optim.AdamW",
            "kwargs": {
                "lr": 1e-6,
                "weight_decay": 0.01,
                "betas": [0.9, 0.999],
            },
        },
        
        # Scheduler
        "scheduler": [
            {
                "name": "torch.optim.lr_scheduler.LinearLR",
                "kwargs": {"start_factor": 0.1, "total_iters": 50},
            },
        ],
    }
}
```

### GRPO Configuration

GRPO-specific settings:

```python
{
    "grpo": {
        "num_prompts_per_step": 32,              # Prompts per step
        "num_generations_per_prompt": 16,        # Samples per prompt
        "max_num_steps": 1000,                   # Max training steps
        "max_num_epochs": 1,                     # Max epochs
        "normalize_rewards": True,               # Normalize rewards
        "use_leave_one_out_baseline": True,      # LOO baseline
        "val_period": 10,                        # Validation frequency
        "val_at_start": False,                   # Validate before training
        "max_val_samples": 256,                  # Max validation samples
        "seed": 42,                              # Random seed
    }
}
```

### SFT Configuration

SFT-specific settings:

```python
{
    "sft": {
        "batch_size": 32,                        # Training batch size
        "max_num_steps": 5000,                   # Max steps
        "max_num_epochs": 3,                     # Max epochs
    }
}
```

### DPO Configuration

DPO-specific settings:

```python
{
    "dpo": {
        "batch_size": 32,                        # Training batch size
        "max_num_steps": 1000,                   # Max steps
        "beta": 0.1,                             # KL penalty
    }
}
```

### Checkpointing Configuration

```python
{
    "checkpointing": {
        "enabled": True,                         # Enable checkpointing
        "checkpoint_dir": "results/grpo",        # Save directory
        "metric_name": "val:accuracy",           # Metric to track
        "higher_is_better": True,                # Metric direction
        "keep_top_k": 3,                         # Keep best N
        "save_period": 10,                       # Save frequency
    }
}
```

### Logger Configuration

```python
{
    "logger": {
        "log_dir": "logs",                       # Log directory
        "wandb_enabled": False,                  # W&B logging
        "tensorboard_enabled": True,             # TensorBoard
        "mlflow_enabled": False,                 # MLflow
        "monitor_gpus": True,                    # GPU monitoring
        
        "wandb": {
            "project": "my-project",
            "name": "run-001",
        },
        
        "tensorboard": {},
    }
}
```

### Cluster Configuration

```python
{
    "cluster": {
        "gpus_per_node": 8,                      # GPUs per node
        "num_nodes": 1,                          # Number of nodes
    }
}
```

## Common Parameters Reference

### Model Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model` | `str` | *required* | Model name or path |
| `learning_rate` | `float` | `1e-6` | Learning rate |
| `max_sequence_length` | `int` | `512` | Max tokens |
| `precision` | `str` | `"bfloat16"` | Model precision |

### Training Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `batch_size` | `int` | `32` | Prompts per step |
| `max_steps` | `int` | `1000` | Max training steps |
| `max_epochs` | `int` | `1` | Max epochs |

### GRPO Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `num_generations_per_prompt` | `int` | `16` | Samples per prompt |
| `normalize_rewards` | `bool` | `True` | Normalize rewards |
| `use_leave_one_out_baseline` | `bool` | `True` | LOO baseline |

### DPO Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `beta` | `float` | `0.1` | KL penalty coefficient |

## Advanced Configuration

### DTensor Configuration

For distributed training with PyTorch DTensor:

```python
{
    "policy": {
        "dtensor_cfg": {
            "enabled": True,
            "cpu_offload": False,
            "tensor_parallel_size": 1,
            "context_parallel_size": 1,
            "activation_checkpointing": False,
            "sequence_parallel": False,
        }
    }
}
```

### Megatron Configuration

For Megatron-LM backend:

```python
{
    "policy": {
        "megatron_cfg": {
            "enabled": True,
            "tensor_model_parallel_size": 4,
            "pipeline_model_parallel_size": 2,
            "activation_checkpointing": True,
        }
    }
}
```

### Generation Configuration

```python
{
    "policy": {
        "generation": {
            "backend": "vllm",                   # vllm or megatron
            "max_new_tokens": 512,
            "temperature": 1.0,
            "top_p": 1.0,
            "top_k": None,
            
            "vllm_cfg": {
                "tensor_parallel_size": 1,
                "gpu_memory_utilization": 0.6,
                "enforce_eager": False,
            },
        }
    }
}
```

### Sequence Packing

```python
{
    "policy": {
        "sequence_packing": {
            "enabled": True,
            "algorithm": "modified_first_fit_decreasing",
            "sequence_length_round": 64,
        }
    }
}
```

## Environment Variables

Some settings can be overridden via environment variables:

| Variable | Description |
|----------|-------------|
| `NRL_CONTAINER` | Set when running in container |
| `NRL_FORCE_REBUILD_VENVS` | Force rebuild virtual environments |
| `NRL_IGNORE_VERSION_MISMATCH` | Skip version checking |
| `RAY_USAGE_STATS_ENABLED` | Ray telemetry (default: 0) |

## Validation

Configuration is validated at trainer initialization:

```python
from nemo_rl import GRPOTrainer

# Invalid config will raise ValueError
try:
    trainer = GRPOTrainer({
        "policy": {"model_name": ""},  # Empty model name
    })
except ValueError as e:
    print(f"Invalid config: {e}")
```

## Related APIs

- [train()](train.md) - Simple parameter passing
- [Trainers](trainers.md) - Trainer configuration
- [Backends](backends.md) - Backend configuration
