# FusedLinearCrossEntropy Configuration Guide

This document explains how to configure the `FusedLinearCrossEntropy` loss function in various nemo-rl training scenarios.

## 📋 **Configuration Overview**

The FusedLinearCrossEntropy can be configured in three main ways:

1. **YAML Configuration Files** (most common)
2. **Python Code** (programmatic setup)
3. **Config Dictionaries** (hybrid approach)

## 🔧 **YAML Configuration**

### Pattern 1: Dedicated Loss Function Section (GRPO Style)

This is the recommended pattern for algorithms that support configurable loss functions:

```yaml
# grpo_with_fused_loss.yaml
loss_fn:
  type: "FusedLinearCrossEntropyLoss"
  ignore_index: -100          # Default: -100
  logit_softcapping: 0.0       # Default: 0.0 (no capping)
  reduction: "sum"             # Default: "sum"

policy:
  model_name: "meta-llama/Llama-3.2-1B-Instruct"
  # CRITICAL: Enable hidden states output
  model_config:
    output_hidden_states: true
  
  # Memory optimization: Use logits_to_keep=1 pattern
  train_global_batch_size: 32
  train_micro_batch_size: 2
  max_total_sequence_length: 1024
  precision: "bfloat16"
  
  dtensor_cfg:
    enabled: true
    tensor_parallel_size: 2
    context_parallel_size: 1
    
# ... rest of GRPO config
grpo:
  num_prompts_per_step: 16
  # ... other GRPO settings
```

### Pattern 2: Algorithm-Specific Section (SFT Style)

For algorithms like SFT that have their own loss handling:

```yaml
# sft_with_fused_loss.yaml
sft:
  max_num_steps: 1000
  loss_fn:
    type: "FusedLinearCrossEntropyLoss"
    ignore_index: -100
    logit_softcapping: 30.0      # Enable softcapping
    reduction: "sum"

policy:
  model_name: "Qwen/Qwen2.5-7B"
  model_config:
    output_hidden_states: true   # REQUIRED
  train_global_batch_size: 64
  train_micro_batch_size: 4
  max_total_sequence_length: 2048
```

### Pattern 3: Custom Loss Configuration

For advanced use cases with custom loss parameters:

```yaml
# advanced_fused_loss.yaml
loss_fn:
  type: "FusedLinearCrossEntropyLoss"
  config:
    ignore_index: -1           # Custom ignore index
    logit_softcapping: 50.0    # Higher softcapping for large vocabs
    reduction: "mean"          # Use mean reduction instead of sum

policy:
  model_name: "microsoft/DialoGPT-large"
  model_config:
    output_hidden_states: true
    # Additional model-specific config
    use_cache: false
  
  # Optimize for memory efficiency
  train_global_batch_size: 128
  train_micro_batch_size: 1    # Smaller micro-batches for memory
  
  dtensor_cfg:
    enabled: true
    cpu_offload: true          # Additional memory savings
    tensor_parallel_size: 4
```

## 🐍 **Python Configuration**

### Direct Instantiation

```python
from nemo_rl.algorithms.loss_functions import (
    FusedLinearCrossEntropyLoss,
    FusedLinearCrossEntropyLossConfig
)

# Basic configuration
config: FusedLinearCrossEntropyLossConfig = {
    "ignore_index": -100,
    "logit_softcapping": 0.0,
    "reduction": "sum"
}

loss_fn = FusedLinearCrossEntropyLoss(config)
```

### Integration with Training Algorithms

```python
# GRPO Training Example
from nemo_rl.algorithms.grpo import grpo_train

# Create loss function
loss_config = {
    "ignore_index": -100,
    "logit_softcapping": 0.0,
    "reduction": "sum"
}
loss_fn = FusedLinearCrossEntropyLoss(loss_config)

# Configure model to output hidden states
policy_config = {
    "model_name": "meta-llama/Llama-3.1-8B",
    "model_config": {
        "output_hidden_states": True
    },
    # ... other policy config
}

# Run training
grpo_train(
    policy=policy,
    dataloader=dataloader,
    loss_fn=loss_fn,  # Pass the FusedLinearCrossEntropy
    # ... other training args
)
```

### With Model Integration

```python
# Ensure model is properly configured
import torch
from transformers import AutoModelForCausalLM

# Load model with hidden states enabled
model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2.5-1.5B",
    output_hidden_states=True,  # CRITICAL
    torch_dtype=torch.bfloat16,
    device_map="auto"
)

# Create loss function
loss_fn = FusedLinearCrossEntropyLoss({
    "ignore_index": -100,
    "reduction": "sum"
})

# In training loop
outputs = model(**batch)
loss = calculate_loss(
    loss_fn=loss_fn,
    model=model,
    hidden_states=outputs.hidden_states[-1],
    data=batch_data,
    global_valid_seqs=valid_seqs,
    global_valid_toks=valid_toks
)
```

## 📊 **Configuration Parameters**

### Core Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `ignore_index` | int | -100 | Token IDs to ignore in loss computation |
| `logit_softcapping` | float | 0.0 | Softcapping value (0 = disabled) |
| `reduction` | str | "sum" | Loss reduction: "sum", "mean", or "none" |

### Model Requirements

| Setting | Required | Purpose |
|---------|----------|---------|
| `output_hidden_states: true` | ✅ Yes | Enables hidden states output |
| `use_cache: false` | 🔄 Recommended | Reduces memory usage |
| Accessible `lm_head.weight` | ✅ Yes | Required for fused computation |

### Memory Optimization Settings

```yaml
# Memory-efficient configuration
policy:
  # Use smaller micro-batches
  train_micro_batch_size: 1
  
  # Enable memory optimizations
  dtensor_cfg:
    cpu_offload: true
    activation_checkpointing: true
  
  # Sequence packing for efficiency
  sequence_packing:
    enabled: true
    train_mb_tokens: 2048
```

## 🔄 **Migration from Standard Loss Functions**

### From NLLLoss

```yaml
# BEFORE (Standard NLL Loss)
sft:
  # No loss_fn specified - uses NLLLoss by default
  max_num_steps: 1000

# AFTER (FusedLinearCrossEntropy)
sft:
  loss_fn:
    type: "FusedLinearCrossEntropyLoss"
    ignore_index: -100
    reduction: "sum"
  max_num_steps: 1000

policy:
  model_config:
    output_hidden_states: true  # ADD THIS
```

### From ClippedPGLoss (for SFT components)

```yaml
# BEFORE (Standard GRPO)
loss_fn:
  reference_policy_kl_penalty: 0.01
  ratio_clip_min: 0.2
  ratio_clip_max: 0.2
  # ... other PG settings

# AFTER (GRPO with custom SFT loss)
loss_fn:
  reference_policy_kl_penalty: 0.01
  ratio_clip_min: 0.2
  ratio_clip_max: 0.2
  # Keep PG settings, add SFT loss config
  sft_loss_fn:
    type: "FusedLinearCrossEntropyLoss"
    ignore_index: -100
    reduction: "sum"
```

## 🎯 **Algorithm-Specific Integration**

### GRPO (Group Relative Policy Optimization)

```yaml
# grpo_fused_loss.yaml
loss_fn:
  # Standard GRPO parameters
  reference_policy_kl_penalty: 0.01
  ratio_clip_min: 0.2
  ratio_clip_max: 0.2
  ratio_clip_c: null
  use_on_policy_kl_approximation: false
  use_importance_sampling_correction: false
  sequence_level_importance_ratios: false
  token_level_loss: true
  
  # Optional: Override SFT component with fused loss
  sft_loss_override:
    type: "FusedLinearCrossEntropyLoss"
    ignore_index: -100
    logit_softcapping: 0.0
    reduction: "sum"
```

### SFT (Supervised Fine-Tuning)

```yaml
# sft_fused_loss.yaml
sft:
  max_num_steps: 1000
  max_num_epochs: 3
  val_period: 100
  
  loss_fn:
    type: "FusedLinearCrossEntropyLoss"
    ignore_index: -100
    logit_softcapping: 0.0
    reduction: "sum"
```

### DPO (Direct Preference Optimization)

```yaml
# dpo_fused_loss.yaml
dpo:
  max_num_steps: 500
  reference_policy_kl_penalty: 0.05
  preference_loss_weight: 1.0
  sft_loss_weight: 0.5
  
  # Override SFT component with fused loss
  sft_loss_fn:
    type: "FusedLinearCrossEntropyLoss"
    ignore_index: -100
    reduction: "sum"
```

## ⚡ **Performance Optimization**

### Memory-Efficient Configuration

```yaml
# High-memory-efficiency setup
policy:
  train_global_batch_size: 32
  train_micro_batch_size: 1      # Smallest possible micro-batch
  
  dtensor_cfg:
    enabled: true
    cpu_offload: true             # Offload to CPU when possible
    activation_checkpointing: true # Trade compute for memory
    tensor_parallel_size: 4       # Distribute across GPUs
    
  sequence_packing:
    enabled: true                 # Pack sequences efficiently
    algorithm: "modified_first_fit_decreasing"

loss_fn:
  type: "FusedLinearCrossEntropyLoss"
  reduction: "sum"                # More memory-efficient than mean
```

### High-Performance Configuration

```yaml
# Speed-optimized setup
policy:
  train_global_batch_size: 128
  train_micro_batch_size: 8       # Larger micro-batches for throughput
  
  dtensor_cfg:
    enabled: true
    tensor_parallel_size: 8       # Maximum parallelization
    activation_checkpointing: false # Trade memory for speed
    
loss_fn:
  type: "FusedLinearCrossEntropyLoss"
  logit_softcapping: 50.0         # May improve stability at high throughput
```

## 🔍 **Validation and Testing**

### Configuration Validation

```python
# Validate your configuration
def validate_fused_loss_config(config, model):
    """Validate FusedLinearCrossEntropy configuration."""
    
    # Check model has required attributes
    assert hasattr(model, 'lm_head'), "Model must have lm_head"
    assert hasattr(model.lm_head, 'weight'), "lm_head must have weight"
    
    # Check hidden states are enabled
    if hasattr(model, 'config'):
        assert getattr(model.config, 'output_hidden_states', False), \
            "Model must have output_hidden_states=True"
    
    # Validate loss config
    loss_config = config.get('loss_fn', {})
    if loss_config.get('type') == 'FusedLinearCrossEntropyLoss':
        print("✅ FusedLinearCrossEntropy configuration is valid")
        return True
    
    return False
```

### Runtime Verification

```yaml
# Add verification in your config
logger:
  # Enable detailed logging to verify fused loss is working
  log_dir: "logs/fused_loss_verification"
  tensorboard_enabled: true
  
  # Monitor memory usage
  gpu_monitoring:
    collection_interval: 5
    flush_interval: 10
```

## 🚨 **Common Configuration Issues**

### Issue 1: Missing Hidden States

```yaml
# ❌ WRONG - Missing hidden states
policy:
  model_name: "meta-llama/Llama-3.1-8B"
  # Missing: output_hidden_states: true

# ✅ CORRECT
policy:
  model_name: "meta-llama/Llama-3.1-8B"
  model_config:
    output_hidden_states: true
```

### Issue 2: Incompatible Model Architecture

```yaml
# ❌ WRONG - Model without lm_head
policy:
  model_name: "bert-base-uncased"  # BERT doesn't have lm_head

# ✅ CORRECT - Causal LM with lm_head
policy:
  model_name: "meta-llama/Llama-3.1-8B"
```

### Issue 3: Memory Configuration

```yaml
# ❌ WRONG - Too aggressive memory settings
policy:
  train_global_batch_size: 1024
  train_micro_batch_size: 128     # Too large for fused loss

# ✅ CORRECT - Balanced memory usage
policy:
  train_global_batch_size: 128
  train_micro_batch_size: 2       # Start small and increase
```

## 📚 **Configuration Templates**

### Small Model (1-7B parameters)

```yaml
# template_small_model.yaml
loss_fn:
  type: "FusedLinearCrossEntropyLoss"
  ignore_index: -100
  reduction: "sum"

policy:
  train_global_batch_size: 64
  train_micro_batch_size: 4
  max_total_sequence_length: 2048
  
  dtensor_cfg:
    enabled: true
    tensor_parallel_size: 1
```

### Large Model (70B+ parameters)

```yaml
# template_large_model.yaml
loss_fn:
  type: "FusedLinearCrossEntropyLoss"
  ignore_index: -100
  logit_softcapping: 30.0
  reduction: "sum"

policy:
  train_global_batch_size: 32
  train_micro_batch_size: 1
  max_total_sequence_length: 4096
  
  dtensor_cfg:
    enabled: true
    cpu_offload: true
    tensor_parallel_size: 8
    context_parallel_size: 2
```

This comprehensive configuration guide should help you integrate FusedLinearCrossEntropy into any nemo-rl training setup! 🚀
