# Backend Selection and Configuration

NeMo RL supports multiple training and generation backends for different use cases and hardware configurations.

## Overview

| Backend Type | Options | Description |
|-------------|---------|-------------|
| Training | DTensor, Megatron | Model training backend |
| Generation | vLLM, Megatron | Text generation backend |

## Training Backends

### DTensor Backend (Default)

Native PyTorch distributed training using DTensor:

```python
from nemo_rl import GRPOTrainer

trainer = GRPOTrainer.from_pretrained(
    "Qwen/Qwen2.5-1.5B",
    # DTensor is default
    tensor_parallel_size=1,
)
```

#### DTensor Configuration

```python
{
    "policy": {
        "dtensor_cfg": {
            "enabled": True,                     # Use DTensor
            "cpu_offload": False,                # CPU offloading
            "tensor_parallel_size": 1,           # TP degree
            "context_parallel_size": 1,          # CP degree
            "activation_checkpointing": False,   # Gradient checkpointing
            "sequence_parallel": False,          # Sequence parallelism
            "custom_parallel_plan": None,        # Custom parallelization
        }
    }
}
```

#### When to Use DTensor

- Standard GPU setups (1-8 GPUs)
- Native PyTorch experience
- Simpler debugging
- Models that fit in memory

### Megatron Backend

NVIDIA Megatron-LM for large-scale distributed training:

```python
from nemo_rl import GRPOTrainer

config = {
    "policy": {
        "model_name": "Qwen/Qwen2.5-72B",
        "dtensor_cfg": {"enabled": False},
        "megatron_cfg": {
            "enabled": True,
            "tensor_model_parallel_size": 8,
            "pipeline_model_parallel_size": 2,
        },
    },
}

trainer = GRPOTrainer(config)
```

#### Megatron Configuration

```python
{
    "policy": {
        "megatron_cfg": {
            "enabled": True,
            "converter_type": "Qwen2ForCausalLM",  # Model architecture
            
            # Parallelism
            "tensor_model_parallel_size": 8,       # Tensor parallelism
            "pipeline_model_parallel_size": 2,     # Pipeline parallelism
            "context_parallel_size": 1,            # Context parallelism
            
            # MoE settings (for Mixture-of-Experts models)
            "expert_tensor_parallel_size": 1,
            "expert_model_parallel_size": 1,
            "freeze_moe_router": True,
            
            # Performance
            "activation_checkpointing": True,
            "sequence_parallel": True,
            "apply_rope_fusion": True,
            "bias_activation_fusion": True,
            "empty_unused_memory_level": 1,
            
            # Optimizer
            "optimizer": {
                "optimizer": "adam",
                "lr": 5e-6,
                "weight_decay": 0.01,
                "use_distributed_optimizer": True,
            },
        }
    }
}
```

#### When to Use Megatron

- Very large models (>70B parameters)
- Multi-node training
- Maximum training efficiency
- Production deployments

## Generation Backends

### vLLM Backend (Default)

High-throughput inference using vLLM:

```python
{
    "policy": {
        "generation": {
            "backend": "vllm",
            "max_new_tokens": 512,
            "temperature": 1.0,
            
            "vllm_cfg": {
                "tensor_parallel_size": 1,
                "pipeline_parallel_size": 1,
                "gpu_memory_utilization": 0.6,
                "max_model_len": 512,
                "enforce_eager": False,
            },
        }
    }
}
```

#### vLLM Features

- Paged attention for efficient memory
- Continuous batching
- CUDA graph optimization
- High throughput generation

### Megatron Inference Backend

For models already loaded with Megatron:

```python
{
    "policy": {
        "generation": {
            "backend": "megatron",
            "max_new_tokens": 512,
            
            "mcore_generation_config": {
                "buffer_size_gb": 20,
                "num_cuda_graphs": 16,
                "block_size_tokens": 256,
                "enable_chunked_prefill": True,
            },
        }
    }
}
```

## Colocated vs Dedicated Generation

### Colocated Mode (Default)

Generation shares GPUs with training:

```python
{
    "policy": {
        "generation": {
            "colocated": {
                "enabled": True,  # Share GPUs
            }
        }
    }
}
```

**Pros:**
- Simpler setup
- Lower resource usage
- Good for small-medium models

**Cons:**
- Context switching overhead
- Memory pressure

### Dedicated Mode

Separate GPUs for generation:

```python
{
    "policy": {
        "generation": {
            "colocated": {
                "enabled": False,
                "resources": {
                    "gpus_per_node": 4,   # Gen GPUs per node
                    "num_nodes": 1,       # Gen nodes
                },
            }
        }
    }
}
```

**Pros:**
- Better throughput
- No context switching
- Recommended for large models

**Cons:**
- More resources needed
- Complex orchestration

## Backend Selection Guide

### Decision Tree

```
Model Size
  │
  ├─ <10B parameters ──────► DTensor + vLLM
  │
  ├─ 10B-70B parameters
  │   │
  │   ├─ Single node ──────► DTensor + vLLM (TP=8)
  │   └─ Multi-node ───────► Megatron + vLLM
  │
  └─ >70B parameters ──────► Megatron + Megatron Gen
```

### Recommended Configurations

#### Small Models (<10B)

```python
import nemo_rl

result = nemo_rl.train(
    model="Qwen/Qwen2.5-1.5B",
    dataset="...",
    reward_fn=my_reward,
    # Uses DTensor + vLLM by default
)
```

#### Medium Models (10B-70B)

```python
from nemo_rl import GRPOTrainer

trainer = GRPOTrainer.from_pretrained(
    "Qwen/Qwen2.5-32B",
    tensor_parallel_size=8,  # Spread across 8 GPUs
)
```

#### Large Models (>70B)

```python
from nemo_rl import GRPOTrainer

config = {
    "policy": {
        "model_name": "Qwen/Qwen2.5-72B",
        "dtensor_cfg": {"enabled": False},
        "megatron_cfg": {
            "enabled": True,
            "tensor_model_parallel_size": 8,
            "pipeline_model_parallel_size": 4,
        },
    },
    "cluster": {
        "gpus_per_node": 8,
        "num_nodes": 4,
    },
}

trainer = GRPOTrainer(config)
```

## Performance Tuning

### Memory Optimization

```python
{
    "policy": {
        # Reduce memory usage
        "precision": "bfloat16",              # Use BF16
        "train_micro_batch_size": 2,          # Smaller batches
        
        "dtensor_cfg": {
            "activation_checkpointing": True,  # Trade compute for memory
            "cpu_offload": True,               # Offload to CPU
        },
        
        "generation": {
            "vllm_cfg": {
                "gpu_memory_utilization": 0.5,  # Less VRAM for gen
            },
        },
    }
}
```

### Throughput Optimization

```python
{
    "policy": {
        # Maximize throughput
        "train_micro_batch_size": 8,
        
        "sequence_packing": {
            "enabled": True,                   # Pack sequences
        },
        
        "megatron_cfg": {
            "apply_rope_fusion": True,        # Fused ops
            "bias_activation_fusion": True,
        },
        
        "generation": {
            "vllm_cfg": {
                "enforce_eager": False,        # CUDA graphs
            },
        },
    }
}
```

## FP8 Training

For NVIDIA Hopper GPUs with FP8 support:

```python
{
    "policy": {
        "megatron_cfg": {
            "enabled": True,
            "fp8_cfg": {
                "enabled": True,
                "fp8_format": "hybrid",
            },
        },
    }
}
```

## Related APIs

- [Config](config.md) - Full configuration reference
- [Trainers](trainers.md) - Trainer classes
- [train()](train.md) - Simple training API
