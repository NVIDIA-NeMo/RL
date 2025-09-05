# FusedLinearCrossEntropy Integration Guide

This document explains how to use the `FusedLinearCrossEntropy` loss function from nemo-automodel within the nemo-rl framework.

## Overview

The `FusedLinearCrossEntropy` performs fused linear transformation and cross-entropy computation, which can be more memory efficient than computing logits and then applying cross-entropy separately. This is especially beneficial for large models where the vocabulary size is very large.

📚 **See Also**: For detailed configuration examples and patterns, refer to the [FusedLinearCrossEntropy Configuration Guide](fused_linear_cross_entropy_configuration.md).

## Setup

### Prerequisites

1. Ensure that the `cut_cross_entropy` library is installed (required by nemo-automodel's FusedLinearCrossEntropy)
2. Make sure your model configuration has `output_hidden_states=True` to enable hidden state outputs

### Configuration

```python
from nemo_rl.algorithms.loss_functions import (
    FusedLinearCrossEntropyLoss,
    FusedLinearCrossEntropyLossConfig
)

# Configure the loss function
loss_config: FusedLinearCrossEntropyLossConfig = {
    "ignore_index": -100,  # Default: -100
    "logit_softcapping": 0.0,  # Default: 0.0 (no capping)
    "reduction": "sum",  # Default: "sum"
}

# Create the loss function instance
loss_fn = FusedLinearCrossEntropyLoss(loss_config)
```

### Model Configuration

Ensure your model is configured to output hidden states:

```python
# In your model config
model_config = {
    # ... other config
    "output_hidden_states": True,
}
```

## Usage

The integration automatically handles the differences between the nemo-automodel and nemo-rl interfaces:

### Direct Usage with calculate_loss

```python
from nemo_rl.algorithms.loss_functions import calculate_loss

# In your training loop
output = model(**batch)  # Model automatically outputs hidden states when needed
loss = calculate_loss(
    loss_fn=loss_fn,
    logits=output.logits,  # Not used by FusedLinearCrossEntropy but required for interface
    labels=labels,
    model=model,  # Required for accessing lm_head.weight
    hidden_states=output.hidden_states[-1],  # Last layer hidden states
    data=data_dict,  # Your BatchedDataDict with training data
    global_valid_seqs=global_valid_seqs,
    global_valid_toks=global_valid_toks,
)
```

### Integration with Training Workers

The integration automatically works with DTensorPolicyWorker, DTensorPolicyWorkerV2, and MegatronPolicyWorker:

```python
# When using DTensorPolicyWorker, DTensorPolicyWorkerV2, or MegatronPolicyWorker
policy = Policy(
    # ... other config
)

# Train with FusedLinearCrossEntropy
results = policy.train(
    data=data_batch,
    loss_fn=loss_fn,  # The FusedLinearCrossEntropyLoss instance
    eval_mode=False,
)
```

## Key Features

### Automatic Hidden State Management

- The integration automatically detects when FusedLinearCrossEntropy is being used
- Model calls are modified to output hidden states (`output_hidden_states=True`)
- Hidden states are automatically extracted and passed to the loss function

### Memory Efficiency

- Uses `logits_to_keep=1` pattern similar to nemo-automodel for memory efficiency
- Avoids computing full logits matrix when using FusedLinearCrossEntropy
- Fused computation reduces memory overhead

### Compatibility

- Fully compatible with existing nemo-rl training pipelines
- Works with sequence packing (`SequencePackingLossWrapper`)
- Supports distributed training (tensor parallel, context parallel)
- Compatible with DTensorPolicyWorker, DTensorPolicyWorkerV2, and MegatronPolicyWorker

## Data Requirements

The loss function expects the following in your `BatchedDataDict`:

```python
data_dict = {
    "input_ids": torch.Tensor,     # Shape: [batch_size, seq_len]
    "token_mask": torch.Tensor,    # Shape: [batch_size, seq_len] 
    "sample_mask": torch.Tensor,   # Shape: [batch_size]
    "hidden_states": torch.Tensor, # Shape: [batch_size, seq_len, hidden_size] - automatically added
    # ... other fields as needed
}
```

## Performance Considerations

### Memory Usage

- FusedLinearCrossEntropy can significantly reduce memory usage for large vocabulary models
- The peak memory reduction is most pronounced when vocabulary size is very large (>100K tokens)

### Speed

- The fused computation can provide speed improvements by avoiding separate linear + cross-entropy operations
- Benefits are most significant on newer GPU architectures with good memory bandwidth

## Troubleshooting

### Common Issues

1. **Missing hidden states error**: Ensure `output_hidden_states=True` in your model configuration

2. **Import errors**: Make sure the `cut_cross_entropy` library is properly installed

3. **Model compatibility**: Ensure your model has an accessible `lm_head.weight` parameter

### Error Messages

- `"FusedLinearCrossEntropy requires hidden states but model output doesn't contain them"`: Set `model.config.output_hidden_states=True`
- `"Model must have accessible lm_head.weight parameter"`: Verify your model architecture has the expected language model head structure

## Example Configuration

Here's a complete example of setting up FusedLinearCrossEntropy in a training configuration:

```python
from nemo_rl.algorithms.loss_functions import FusedLinearCrossEntropyLoss

# Training configuration
config = {
    "model": {
        "output_hidden_states": True,  # Required for FusedLinearCrossEntropy
        # ... other model config
    },
    "loss": {
        "type": "FusedLinearCrossEntropyLoss",
        "config": {
            "ignore_index": -100,
            "logit_softcapping": 0.0,
            "reduction": "sum",
        }
    },
    # ... other training config
}

# Create loss function
loss_fn = FusedLinearCrossEntropyLoss(config["loss"]["config"])
```

## Testing

The integration includes comprehensive unit tests to ensure correctness and reliability.

### Running Tests

To run the FusedLinearCrossEntropy tests:

```bash
# Run all FusedLinearCrossEntropy tests
python -m pytest tests/unit/algorithms/test_loss_functions.py -k "fused" -v

# Or use the provided test runner
python test_fused_linear_cross_entropy.py
```

### Test Coverage

The unit tests cover:

1. **Initialization and Configuration**
   - Default and custom configurations
   - Error handling when dependencies are missing

2. **Loss Computation**
   - Basic loss computation with various input sizes
   - Proper tensor shapes and value validation
   - Numerical stability checks

3. **Error Handling**
   - Missing model parameter
   - Model without accessible lm_head.weight
   - Invalid data structures

4. **Masking and Normalization**
   - Token-level masking behavior
   - Sample-level masking behavior
   - Global normalization with distributed training

5. **Integration Tests**
   - calculate_loss utility function
   - SequencePackingLossWrapper compatibility
   - Comparison with standard loss functions

6. **Protocol Compliance**
   - LossFunction interface implementation
   - Correct loss_type attribute

### Test Dependencies

Tests automatically handle missing dependencies:
- When `cut_cross_entropy` is not available, tests are skipped with appropriate messages
- GPU availability is checked and tests are skipped when running on CPU-only environments

### Mock Infrastructure

Tests use realistic mock models and data:
- MockModel with proper lm_head structure
- Realistic tensor shapes and data distributions
- Deterministic initialization for reproducible testing
