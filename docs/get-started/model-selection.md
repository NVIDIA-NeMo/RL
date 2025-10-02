---
description: "Choose the right model size and type for your NeMo RL use case, with hardware requirements and performance expectations"
categories: ["getting-started"]
tags: ["model-selection", "hardware-requirements", "performance", "scaling", "gpu-memory"]
personas: ["mle-focused", "researcher-focused", "admin-focused"]
difficulty: "beginner"
content_type: "guide"
modality: "universal"
---

# Choose Your Model

This guide helps you choose the right model size and type for your NeMo RL training needs, with clear hardware requirements and performance expectations.

## Overview

NeMo RL supports models ranging from 0.5B to 70B+ parameters. The choice of model size significantly impacts training time, memory requirements, and final performance. This guide helps you make informed decisions based on your specific use case and available hardware.

## Model Size Categories

### Small Models (0.5B - 1B Parameters)

**Best for:**
- Learning and experimentation
- Debugging training pipelines
- Prototyping new algorithms
- Educational purposes
- Quick iteration cycles

**Examples:**
- `Qwen/Qwen2.5-0.5B`
- `microsoft/DialoGPT-small`
- `distilbert-base-uncased`

**Hardware Requirements:**
- **GPU Memory**: 8-16 GB VRAM
- **Training Time**: 1-4 hours per epoch
- **Batch Size**: 4-8 (depending on sequence length)

**Performance Expectations:**
- Fast training (minutes to hours)
- Good for understanding RL concepts
- Limited reasoning capabilities
- Suitable for simple tasks

### Medium Models (1B - 7B Parameters)

**Best for:**
- Most research and development
- Production applications
- Domain-specific fine-tuning
- Balanced performance and speed

**Examples:**
- `Qwen/Qwen2.5-1.5B`
- `meta-llama/Llama-2-7b-hf`
- `microsoft/DialoGPT-medium`

**Hardware Requirements:**
- **GPU Memory**: 16-32 GB VRAM
- **Training Time**: 4-24 hours per epoch
- **Batch Size**: 2-4 (depending on sequence length)

**Performance Expectations:**
- Good reasoning capabilities
- Suitable for most practical applications
- Reasonable training time
- Good balance of performance and efficiency

### Large Models (7B - 70B+ Parameters)

**Best for:**
- Advanced research
- State-of-the-art performance
- Complex reasoning tasks
- Production systems with high accuracy requirements

**Examples:**
- `meta-llama/Llama-2-13b-hf`
- `meta-llama/Llama-2-70b-hf`
- `Qwen/Qwen2.5-7B`

**Hardware Requirements:**
- **GPU Memory**: 32+ GB VRAM (may require multiple GPUs)
- **Training Time**: 24+ hours per epoch
- **Batch Size**: 1-2 (may require gradient accumulation)

**Performance Expectations:**
- Excellent reasoning capabilities
- State-of-the-art performance
- Longer training times
- Higher computational costs

## Hardware Requirements by Model Size

### Single GPU Training

| Model Size | GPU Memory | Recommended GPU | Training Time |
|------------|------------|-----------------|---------------|
| 0.5B | 8-12 GB | RTX 3080, RTX 4070 | 1-4 hours |
| 1B | 12-16 GB | RTX 3080 Ti, RTX 4070 Ti | 2-6 hours |
| 1.5B | 16-20 GB | RTX 3090, RTX 4080 | 4-8 hours |
| 3B | 20-24 GB | RTX 3090, RTX 4080 | 8-16 hours |
| 7B | 24-32 GB | RTX 4090, A100 | 16-32 hours |
| 13B+ | 32+ GB | A100, H100 | 24+ hours |

### Multi-GPU Training

For models larger than 7B parameters, consider distributed training:

- **2-4 GPUs**: Suitable for 7B-13B models
- **4-8 GPUs**: Suitable for 13B-30B models  
- **8+ GPUs**: Required for 30B+ models

## Use Case Recommendations

### For Learning and Experimentation

**Recommended Model Size:** 0.5B - 1B
**Reasoning:**
- Fast iteration cycles
- Lower hardware requirements
- Good for understanding concepts
- Easy to debug and experiment

**Example Configuration:**
```yaml
policy:
  model_name: "Qwen/Qwen2.5-0.5B"
  max_seq_length: 512
algorithm:
  batch_size: 4
  gradient_accumulation_steps: 1
```

### For Research and Development

**Recommended Model Size:** 1B - 7B
**Reasoning:**
- Good performance for most tasks
- Reasonable training time
- Suitable for most research questions
- Balanced cost and performance

**Example Configuration:**
```yaml
policy:
  model_name: "Qwen/Qwen2.5-1.5B"
  max_seq_length: 1024
algorithm:
  batch_size: 2
  gradient_accumulation_steps: 2
```

### For Production Applications

**Recommended Model Size:** 7B - 13B
**Reasoning:**
- High accuracy requirements
- Complex reasoning tasks
- Production-grade performance
- Justified computational investment

**Example Configuration:**
```yaml
policy:
  model_name: "meta-llama/Llama-2-7b-hf"
  max_seq_length: 2048
algorithm:
  batch_size: 1
  gradient_accumulation_steps: 4
```

## Model Architecture Considerations

### Base Models vs. Fine-tuned Models

**Base Models:**
- Start from scratch or minimal fine-tuning
- More flexible for custom tasks
- Longer training time required
- Better for research and experimentation

**Pre-fine-tuned Models:**
- Already fine-tuned for specific domains
- Faster convergence
- Less flexible for custom tasks
- Better for production applications

### Model Family Selection

**Llama Family:**
- Strong general capabilities
- Good for most applications
- Requires Hugging Face access
- Well-documented and supported

**Qwen Family:**
- Good performance on Chinese and English
- Open access (no special permissions)
- Smaller model sizes available
- Good for experimentation

**Custom Models:**
- Domain-specific performance
- May require custom tokenization
- Limited documentation
- Higher maintenance overhead

## Performance Optimization Tips

### Memory Optimization

For models that barely fit in GPU memory:

1. **Enable Gradient Checkpointing:**
   ```yaml
   model:
     gradient_checkpointing: true
   ```

2. **Use Mixed Precision:**
   ```yaml
   model:
     torch_dtype: "float16"
   ```

3. **Reduce Batch Size and Use Gradient Accumulation:**
   ```yaml
   algorithm:
     batch_size: 1
     gradient_accumulation_steps: 4
   ```

### Training Speed Optimization

For faster training:

1. **Increase Batch Size** (if memory allows)
2. **Use Multiple GPUs** for distributed training
3. **Enable Flash Attention** (if supported)
4. **Use Optimized Data Loading**

## Decision Matrix

| Use Case | Model Size | Training Time | Hardware | Cost |
|----------|------------|---------------|----------|------|
| Learning | 0.5B | 1-4 hours | Single GPU | Low |
| Research | 1-7B | 4-24 hours | Single GPU | Medium |
| Production | 7B+ | 24+ hours | Multi-GPU | High |

## Next Steps

After selecting your model:

1. **Check Hardware Compatibility**: Ensure your GPU meets the memory requirements
2. **Start with Quickstart**: Use the [Quickstart Guide](quickstart.md) with your chosen model
3. **Monitor Performance**: Track training metrics during your runs
4. **Scale Up**: Consider distributed training for larger models

## Common Pitfalls

### Memory Issues
- **Problem**: CUDA out of memory errors
- **Solution**: Reduce batch size, enable gradient checkpointing, or use smaller model

### Training Time
- **Problem**: Training takes too long
- **Solution**: Use smaller model for experimentation, then scale up

### Performance Expectations
- **Problem**: Model doesn't perform well enough
- **Solution**: Try larger model or different architecture

## Get Help

- [Hardware Requirements](installation.md#hardware-requirements) - Detailed system requirements
- [Troubleshooting](../guides/troubleshooting) - Common issues and solutions 