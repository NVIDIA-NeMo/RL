# Padding in NeMo RL

This document explains padding in NeMo RL and why consistent padding is critical for the framework.

## Padding Approach

NeMo RL uses **right padding** for all tensor operations, where padding tokens are added to the right/end of sequences:

```
[101, 2054, 2003,    0,    0]  # Length 3
[101, 2054, 2003, 2001, 1996]  # Length 5 (no padding needed)
[101, 2054,    0,    0,    0]  # Length 2
```

This approach:
1. **Naturally aligns with LLM processing**: Tokens are processed from left to right.
2. **Keeps meaningful tokens contiguous**: All valid tokens appear at the beginning of tensors.
3. **Simplifies indexing and operations**: Valid token boundaries are easily defined with a single length value.

## Right-Padded Generation Example

Input (right-padded) → Generation → Final (right-padded):
```
[101, 2054, 2003,    0,    0]  # Original input (length 3)
                ↓
[101, 2054, 2003, 2001, 1996, 4568, 7899,    0]  # After generation
|-- input --|    |----- generation -----|  |pad|
```

Corresponding logprobs:
```
[   0,    0,    0, -1.2, -0.8, -1.5, -2.1,    0]
|-- zeros for input --|  |- gen logprobs -|  |pad|
```

## Verify Right Padding

NeMo RL provides utilities to verify correct padding. For example:

```{testcode}
import torch
from nemo_rl.distributed.batched_data_dict import BatchedDataDict
from nemo_rl.models.generation.interfaces import verify_right_padding

# For input data (BatchedDataDict containing input_ids and input_lengths)
input_data = BatchedDataDict({
    "input_ids": torch.tensor([
        [101, 2054, 2003, 0, 0],  # Example input sequence
        [101, 2054, 0, 0, 0]       # Another input sequence
    ]),
    "input_lengths": torch.tensor([3, 2])  # Length of each sequence
})

# Check if input data is properly right-padded
is_right_padded, error_msg = verify_right_padding(input_data, pad_value=0)

# For generation output data (BatchedDataDict containing output_ids and generation_lengths)
output_data = BatchedDataDict({
    "output_ids": torch.tensor([
        [101, 2054, 2003, 2001, 1996, 0, 0],  # Example output sequence
        [101, 2054, 2001, 4568, 0, 0, 0]       # Another output sequence
    ]),
    "generation_lengths": torch.tensor([2, 2]),  # Length of generated response
    "unpadded_sequence_lengths": torch.tensor([5, 4])  # Total valid tokens
})

# Check if output data is properly right-padded
is_right_padded, error_msg = verify_right_padding(output_data, pad_value=0)

if not is_right_padded:
    print(f"Padding error: {error_msg}")
```

<!-- This testoutput is intentionally empty-->
```{testoutput}
:hide:
```

The {py:class}`verify_right_padding() <nemo_rl.models.generation.interfaces.verify_right_padding>` function checks that:
1. All padding (zeros or padding token provided by the user) appears after valid tokens.
2. The padding starts at the position specified by the length tensor.

The function automatically detects whether you're passing input or output data:
- For input data: Requires `input_ids` and `input_lengths` fields.
- For output data: Requires `output_ids` and either `generation_lengths` or `unpadded_sequence_lengths`.


## Best Practices

1. **Always Use Right Padding**: All components expect this format.

2. **Track Length Tensors**: Include appropriate length tensors with your data.

3. **Verify Padding**: Use {py:class}`verify_right_padding() <nemo_rl.models.generation.interfaces.verify_right_padding>` when in doubt.

4. **Mask Padding in Operations**: Use lengths to exclude padding tokens from loss calculations.
