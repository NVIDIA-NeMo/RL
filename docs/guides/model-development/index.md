---
description: "Guides for extending NeMo RL with custom models and handling model-specific quirks and special cases"
categories: ["model-development"]
tags: ["model-development", "custom-models", "model-integration", "model-quirks", "architecture"]
personas: ["mle-focused", "researcher-focused"]
difficulty: "advanced"
content_type: "concept"
modality: "universal"
---

# Model Development

Welcome to the NeMo RL Model Development guide! This section covers everything you need to know about extending NeMo RL with custom models, handling model-specific behaviors, and integrating new architectures into the reinforcement learning framework.

## Overview

NeMo RL provides model-agnostic support for a wide range of language model architectures and backends. Whether you're working with Hugging Face models, custom architectures, or specialized implementations, this guide will help you integrate your models seamlessly into the RL training pipeline.

## Core Concepts

::::{grid} 1 2 2 2
:gutter: 2 2 2 2

:::{grid-item-card} {octicon}`plus;1.5em;sd-mr-1` Add New Models
:link: add-new-models
:link-type: doc

Learn how to integrate custom models and architectures into NeMo RL training pipelines.

+++
{bdg-warning}`Advanced`
:::

:::{grid-item-card} {octicon}`alert;1.5em;sd-mr-1` Model Quirks and Special Cases
:link: model-quirks
:link-type: doc

Handle model-specific behaviors, special cases, and edge cases in NeMo RL.

+++
{bdg-warning}`Advanced`
:::

:::{grid-item-card} {octicon}`rocket;1.5em;sd-mr-1` Use DeepSeek Models
:link: deepseek
:link-type: doc

Convert and configure DeepSeek models for training in NeMo RL.

+++
{bdg-primary}`Practical`
:::

::::

## Supported Model Types

### Hugging Face Models
NeMo RL provides first-class support for Hugging Face models:

- **AutoModelForCausalLM**: GPT-style models (Llama, Mistral, Qwen)
- **AutoModelForSeq2SeqLM**: Encoder-decoder models (T5, BART)
- **Custom Architectures**: Any model with compatible interfaces

### Backend Support
Multiple backends for different use cases:

- **Hugging Face**: Standard HF model loading and inference
- **Megatron-LM**: NVIDIA's optimized training framework
- **Custom Backends**: Extensible backend system

## Model Integration Workflow

### 1. Model Selection
Choose the right model for your use case:

```python
# Hugging Face model
model_config = {
    "name": "llama2-7b",
    "backend": "huggingface",
    "max_length": 2048
}

# Custom model
model_config = {
    "name": "my-custom-model",
    "backend": "custom",
    "model_class": "MyCustomModel"
}
```

### 2. Backend Configuration
Configure the appropriate backend:

```yaml
# config.yaml
model:
  name: "llama2-7b"
  backend: "huggingface"
  max_length: 2048
  trust_remote_code: true
  
  # Model-specific parameters
  use_cache: false  # For training
  gradient_checkpointing: true
```

### 3. Training Integration
Integrate with RL training algorithms:

```python
# Training with Policy and DPO (see examples/run_dpo.py for a complete pipeline)
from nemo_rl.models.policy.lm_policy import Policy
from nemo_rl.algorithms.dpo import setup as dpo_setup, dpo_train
```

## Architecture Considerations

### Model Size and Memory
- **Small Models (1B-7B)**: Good for experimentation and prototyping
- **Medium Models (7B-30B)**: Balance of performance and resource usage
- **Large Models (30B+)**: Require careful memory management

### Memory Optimization Techniques
- **Gradient Checkpointing**: Trade compute for memory
- **Model Parallelism**: Distribute across many GPUs
- **Activation Checkpointing**: Reduce memory footprint
- **Mixed Precision**: Use FP16/BF16 for efficiency

### Performance Optimization
- **Kernel Fusion**: Optimize CUDA kernels
- **Attention Optimization**: Flash attention, grouped query attention
- **Quantization**: INT8/INT4 for inference
- **Distributed Training**: Scale across many nodes

## Custom Model Development

### Create Custom Models
Extend the base model interface:

```python
# Extend or customize behavior by wrapping interfaces and workers used by Policy
# See nemo_rl.models.policy.*_policy_worker for extension points
```

### Backend Integration
Implement custom backends for specialized hardware:

```python
# Policy selects backends via configuration (FSDP/DTensor/Megatron)
# Example fields: policy.dtensor_cfg.enabled, policy.megatron_cfg.enabled
```

## Model-Specific Considerations

### Tokenization
Handle different tokenizer requirements:

```python
# Custom tokenizer integration
tokenizer_config = {
    "name": "my-tokenizer",
    "special_tokens": ["<|start|>", "<|end|>"],
    "padding_side": "left"
}
```

### Position Embeddings
Manage position embedding requirements:

```python
# Position embedding configuration
model_config = {
    "position_embedding_type": "rope",  # or "learned", "sinusoidal"
    "max_position_embeddings": 4096,
    "rope_scaling": {"type": "linear", "factor": 2.0}
}
```

### Attention Mechanisms
Configure attention patterns:

```python
# Attention configuration
attention_config = {
    "attention_type": "flash_attention_2",
    "use_sliding_window": True,
    "sliding_window_size": 4096
}
```

## Common Integration Patterns

### Hugging Face Integration
Standard pattern for HF models:

```yaml
# Standard HF integration
model:
  name: "meta-llama/Llama-2-7b-hf"
  backend: "huggingface"
  trust_remote_code: true
  use_auth_token: true  # For gated models
```

### Custom Architecture Integration
For proprietary or research models:

```yaml
# Custom model integration
model:
  name: "my-research-model"
  backend: "custom"
  model_class: "MyResearchModel"
  custom_config:
    architecture: "transformer"
    hidden_size: 4096
    num_layers: 32
```

### Multi-Model Training
Training with many models:

```yaml
# Multi-model configuration
models:
  policy:
    name: "llama2-7b"
    backend: "huggingface"
  reference:
    name: "llama2-7b-base"
    backend: "huggingface"
  reward:
    name: "reward-model"
    backend: "custom"
```

## Testing and Validation

### Model Compatibility
Test your model integration:

```python
# Compatibility testing
def test_model_compatibility(model_config):
    model = ModelFactory.create_model(model_config)
    
    # Test forward pass
    input_ids = torch.randint(0, 1000, (1, 10))
    outputs = model(input_ids)
    
    # Test generation
    generated = model.generate(input_ids, max_length=20)
    
    return outputs is not None and generated is not None
```

### Performance Benchmarking
Measure model performance:

```python
# Performance benchmarking
def benchmark_model(model, test_data):
    start_time = time.time()
    
    for batch in test_data:
        outputs = model(batch)
        
    end_time = time.time()
    return end_time - start_time
```

## Troubleshooting

### Common Issues

#### Memory Errors
- **Problem**: CUDA out of memory
- **Solution**: Reduce batch size, enable gradient checkpointing
- **Prevention**: Monitor memory usage, use memory profiling tools

#### Tokenization Mismatches
- **Problem**: Tokenizer/model vocabulary mismatch
- **Solution**: Ensure tokenizer matches model architecture
- **Prevention**: Validate tokenizer configuration

#### Generation Issues
- **Problem**: Incorrect or unexpected outputs
- **Solution**: Check generation parameters, check model outputs
- **Prevention**: Test generation with known inputs

### Debugging Tools

#### Model Inspection
```python
# Inspect model architecture
def inspect_model(model):
    print(f"Model type: {type(model)}")
    print(f"Parameters: {sum(p.numel() for p in model.parameters())}")
    print(f"Device: {next(model.parameters()).device}")
```

#### Memory Profiling
```python
# Memory profiling
import torch.profiler

with torch.profiler.profile(
    activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]
) as prof:
    outputs = model(input_ids)
    
print(prof.key_averages().table(sort_by="cuda_time_total"))
```

## Best Practices

### 1. Start Simple
- Begin with well-tested models (Llama, Mistral)
- Use standard configurations before customization
- Validate basic functionality before optimization

### 2. Gradual Complexity
- Add custom features incrementally
- Test each modification thoroughly
- Maintain backward compatibility

### 3. Performance Monitoring
- Track memory usage and training speed
- Monitor for regressions
- Profile bottlenecks regularly

### 4. Documentation
- Document custom model requirements
- Maintain configuration examples
- Update integration guides

## Next Steps

- [Add New Models](add-new-models) - Step-by-step integration guide
- [Model Quirks and Special Cases](model-quirks) - Handle edge cases and special behaviors
- [Training Algorithms](../training-algorithms/index) - Apply your models to RL training
- [Performance Optimization](../training-optimization/index) - Optimize model performance
- [API Reference](../../apidocs/index) - Complete model API reference

## Get Help

- [Troubleshooting](../troubleshooting) - Common model integration issues
- Model configuration parameters
- [Community Support](https://github.com/NVIDIA/NeMo-RL/issues) - GitHub discussions
- [Model Examples](https://github.com/NVIDIA/NeMo-RL/tree/main/examples) - Working model integration examples

---

::::{toctree}
:hidden:
:caption: Model Development
:maxdepth: 2
add-new-models
model-quirks
deepseek
::::::
 