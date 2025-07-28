# Sequence Packing and Dynamic Batching

This document describes the sequence packing and dynamic batching features implemented in NeMo-RL to optimize training efficiency for variable-length sequences.

## Table of Contents

1. [Problem](#problem)
2. [Sequence Packing and Dynamic Batching](#sequence-packing-and-dynamic-batching)
3. [Sequence Packing](#sequence-packing)
4. [Dynamic Batching](#dynamic-batching)
5. [Configuration](#configuration)
6. [Integration with Training Pipeline](#integration-with-training-pipeline)
7. [Metrics and Monitoring](#metrics-and-monitoring)
8. [Usage](#usage)

## Problem

### Challenge: Variable Sequence Lengths in RL/SFT

RL and SFT exhibit highly variable sequence lengths due to many datasets having seqlens following Zipf's law:

- **Skewed Distribution**: Most sequences are short, with a few very long sequences
- **Padding Inefficiency**: Traditional fixed-length batching requires padding all sequences to the maximum length, resulting in:
  - Wasted computation on pad tokens
  - Underutilized GPU memory
  - Poor GPU compute efficiency
- **Memory Constraints**: Batch size is often limited by the longest sequences in the batch

Without optimization, 50-70% of computation can be wasted on padding tokens.

## Sequence Packing and Dynamic Batching
NeMo-RL implements two exclusive approaches to address variable sequence lengths:

1. **Sequence Packing**: Concatenates multiple sequences into a single "packed" sequence, eliminating most padding.
2. **Dynamic Batching**: Groups sequences of similar lengths and adjusts microbatch sizes based on total token count, reducing the excess padding.

### Important Notes

- Dynamic batching and sequence packing cannot be enabled simultaneously, **they are exclusive**.
- Compatible with Context Parallelism (CP)
- Requires FlashAttention-2 for packed sequences

## Sequence Packing

Sequence packing concatenates multiple variable-length sequences into a single sequence, eliminating the need for padding tokens. This approach maximizes GPU utilization by ensuring all computational resources are used for meaningful tokens.

```
Unpacked: (# == useful token, p == padding token)
0 0 0 p p p
1 1 1 1 1 1
2 2 p p p p
3 3 3 p p p
~40% padding

Packed:
0 0 0 1 1 1 1 1 1 2 2 3 3 3 p # some padding may still be required as discussed later, but it is significantly reduced
```

### Implementation Details

#### 1. Packing Process (`BatchedDataDict.shard_by_batch_size`)
```python
# Located in: nemo_rl/distributed/batched_data_dict.py
def shard_by_batch_size(
    self,
    shards: int,
    sequence_packing_args: Optional[SequencePackingArgs] = None
):
    # 1. Get bin packer for specified algorithm
    bin_packer = get_packer(
        algorithm=sequence_packing_args["algorithm"],
        bin_capacity=sequence_packing_args["max_tokens_per_microbatch"]
    )
    
    # 2. Pack sequences into bins per chunk
    for chunk_idx in range(num_chunks):
        chunk_bin_assignments = bin_packer.pack(
            sequence_lengths=chunk_padded_seqlens_list
        )
    
    # 3. Create sharded microbatches from packed bins
```
This method **does not** actually concatenate the sequences and create the packed tensor. Rather, it reorders the elements in the batch and creates metadata such that after you call your workers with `RayWorkerGroup.run_all_workers_sharded_data`, each worker can call `BatchedDataDict.make_microbatch_iterator_for_packable_sequences` locally to return an iterator over batches, where each batch contains elements that should be packed together. For an example of this, you can take a look at the `MegatronPolicyWorker`'s train function.

We have the policy backends perform the actual packing because implementations can vary widely on how exactly it should be done and what metadata needs to be collected.


#### 2. Packing Algorithms (`nemo_rl/data/packing/algorithms.py`)

Four packing algorithms are implemented, but we recommend you just use Modified First Fit Decreasing for the best packing efficiency:

##### Concatenative Packer 
- Sequential concatenation until bin capacity is reached
- O(n)
- Simple, deterministic packing for debugging

##### Modified First Fit Decreasing (MFFD)
- Johnson & Garey (1985) heuristic with 5-phase packing strategy
- O(n log n + n*m)
- Best bin utilization
- Phases:
  1. Classify items (large: >C/2, medium: >C/3, small: >C/6, tiny: ≤C/6)
  2. Create one bin per large item
  3. Add medium items to large bins (forward pass)
  4. Add pairs of small items (backward pass)
  5. Greedy fit remaining items
  6. Apply FFD to leftovers

##### First Fit Decreasing (FFD)
- Sort sequences by length (descending), place each in first fitting bin
- O(n log n + n*m) where m = number of bins
- Good general-purpose algorithm

##### First Fit Shuffle
- Randomly shuffle sequences, then apply first-fit
- O(n*m)
- When sequence order doesn't matter

### Usage with Context Parallelism

For long sequences with context parallelism (CP > 1):
- Individual sequences must be padded to a multiple of `cp_size * 2 * tp_size`, where the factor of 2 ensures load balancing for causal attention

#### Understanding CP Load balancing:
```
Given a sequence of length 6, CP 2:

0 1 2 3 4 5

The attention mask is:
  | 0 1 2 3 4 5
--+------------
0 | 1 0 0 0 0 0
1 | 1 1 0 0 0 0
2 | 1 1 1 0 0 0
3 | 1 1 1 1 0 0
4 | 1 1 1 1 1 0
5 | 1 1 1 1 1 1


If we were to naively chunk this sequence into CP chunks, we would have:

CP0:
  | 0 1 2
--+------
0 | 1 0 0
1 | 1 1 0   +   send KV 0 1 2
2 | 1 1 1

CP1:
  | 3 4 5                            | 0 1 2
--+------                          --+------
3 | 1 0 0                          3 | 1 1 1 
4 | 1 1 0   +   recv KV 0 1 2   +  4 | 1 1 1
5 | 1 1 1                          5 | 1 1 1

Here, CP1 ends up with more than double the work of CP0, stalling training on CP0.

To fix this, we can chunk the sequence into 2*CP chunks (and pad to accommodate):

| 0 1 | 2 3 | 4 5 | p p |
|--V--|--V--|--V--|--V--|
| CP0 | CP1 | CP1 | CP0 |

Now, the work looks like this:

CP0:
  | 0 1                                           | 2 3 4 5 p p
--+----                                         --+------------
0 | 1 0   +   send KV 0 1, recv KV 2 3 4 5   +  p | 1 1 1 1 1 0
1 | 1 1                                         p | 1 1 1 1 1 1


CP1:
  | 2 3 4 5                                           | 0 1
--+--------                                         --+----
2 | 1 0 0 0                                         2 | 1 1
3 | 1 1 0 0   +   send KV 2 3 4 5, recv KV 0 1   +  3 | 1 1
4 | 1 1 1 0                                         4 | 1 1
5 | 1 1 1 1                                         5 | 1 1

Much more even!
```

With Sequence packing + CP, we pack and CP-shard _per sequence_ to take full advantage of the load-balancing properties of CP-sharding.

```
Input batch:
0 0 0 0 0 p p p
1 1 1 1 1 1 1 1
2 p p p p p p p
3 3 3 p p p p p

CP = 2

First pack every sequence to 2 * CP * TP = 4:
[
0 0 0 0 0 p p p,
1 1 1 1 1 1 1 1,
2 p p p,
3 3 3 p
]

Now CP-shard each individual sequence and pack
CP0:
0 0 p p
1 1 1 1
2 p
3 p
packed:
0 0 p p 1 1 1 1 2 p 3 p

CP1:
0 0 0 p
1 1 1 1
p p
3 3
packed:
0 0 0 p 1 1 1 1 p p 3 3
```

Internally, DTensor and Megatron-Core are made aware of sequence packing with either `FlashAttentionArgs` or `PackedSeqParams`, which contain `cu_seqlens_q` and `cu_seqlens_kv`, which are the cumulative sequence lengths of the sequence in the packed batch without CP.

### Nuances
- With using Sequence Packing with Megatron + Pipeline Parallelism (PP), note that all packed sequences will be padded up to the maximum packed sequence length because PP requires maintaining a fixed-size batch x seqlen buffer for PP communications. In practice, however, we find that packing is _so efficient_ that this hardly makes a difference.

All together, we see **speedups in the ~2-3x range** when enabling sequence packing.

## Dynamic Batching

Dynamic batching optimizes microbatch formation by:
1. Sorting sequences by length within batches (and respects chunk boundaries, so there are no training order diffs).
2. Grouping sequences to achieve target token count per microbatch.
3. Padding sequences to configurable multiples for hardware alignment.

**Cannot be used with sequence packing**

### Architecture

#### Processing Pipeline

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Input Batch   │ ── │ Sort by Length   │ ── │ Group by Tokens │
│                 │    │ (within chunks)  │    │                 │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                                                        │
┌─────────────────┐    ┌──────────────────┐    ┌────────V────────┐
│ Dynamic Micros  │ <─ │ Pad to Multiple  │ <─ │ Calculate Sizes │
│                 │    │                  │    │                 │
└─────────────────┘    └──────────────────┘    └─────────────────┘
```

```
Input batch:
0 0 p p p p p
1 1 1 1 p p p
2 2 2 2 2 2 2
3 3 3 3 3 3 p
4 4 4 p p p p
5 5 5 5 p p p

MBS = 16 tokens

Dynamic Batching will re-order this batch to minimize padding

1. Sort:
2 2 2 2 2 2 2
3 3 3 3 3 3 p
1 1 1 1 p p p
5 5 5 5 p p p
4 4 4 p p p p
0 0 p p p p p

2. Chunk by MBS token count
MBS 0:
2 2 2 2 2 2 2
3 3 3 3 3 3 p

MBS 1:
1 1 1 1 
5 5 5 5 
4 4 4 p 
0 0 p p 

Note how we're able to remove a huge chunk of padding this way and do the full batch with fewer microbatches than we would otherwise need.
```

#### Implementation Details

**Sorting and Load Balancing** (`nemo_rl/distributed/batched_data_dict.py`)
```python
if dynamic_batching_args is not None:
    # Sort sequences by length within each chunk
    for chunk_idx in range(num_chunks):
        chunk_seqlens = self.data[input_lengths_key][chunk_start:chunk_end]
        chunk_idx_indices = sorted(range(batch_size), 
                                 key=lambda i: chunk_seqlens[i])
        # Stride sorted sequences across DP ranks for load balancing
        chunk_idx_indices = [chunk_idx_indices[i::shards] for i in range(shards)]
```

**Dynamic Shape Processing** (`nemo_rl/distributed/batched_data_dict.py`)
```python
# In the batched datadict, everything is padded up to the max seqlen. This truncates
# everything in one dynamic batch to just pad up to the max within this batch.
def make_microbatch_iterator_with_dynamic_shapes(self):
    for seqlen, (start_idx, end_idx) in zip(self.micro_batch_lengths[0], 
                                           self.micro_batch_indices[0]):
        mb = self.slice(start_idx, end_idx)
        mb.truncate_tensors(dim=sequence_dim, truncated_len=seqlen)
        yield mb
```

### Interface
```python
class BatchedDataDict(UserDict, Generic[DictT]):
    def shard_by_batch_size(
        self,
        shards: int,
        dynamic_batching_args: Optional[DynamicBatchingArgs] = None,
        sequence_packing_args: Optional[SequencePackingArgs] = None
    ) -> list[SlicedDataDict]:
        # Main entry point for both packing and dynamic batching
```

Similar to Sequence Packing, we do not actually create the dynamic batches upon the call to shard_by_batch_size, but just reorder sequences and create metadata internally. With a call to `RayWorkerGroup.run_all_workers_sharded_data`, the workers can run `make_microbatch_iterator_with_dynamic_shapes` to get the true dynamic batches.

### Nuances
- Dynamic batching **cannot** be used with Megatron + PP because PP requires a fixed [batch x seqlen] buffer for PP communication. Please use Sequence Packing.
- Dynamic batching is almost always slower than Sequence Packing, but does not require that your model (and in particular, your attention variant) have Sequence-packing implemented (which can be complicated). We'd recommend always using Sequence Packing where possible, and falling back to Dynamic batching as a last resort.

## Configuration

### Dynamic Batching Configuration
```python
class DynamicBatchingArgs(TypedDict):
    max_tokens_per_microbatch: int  # Target tokens per microbatch
    sequence_length_round: int      # Padding alignment multiple
    input_key: str                  # Input tensor key ("input_ids")
    input_lengths_key: str          # Sequence lengths key ("input_lengths")
```

### Sequence Packing Configuration
```python
class SequencePackingArgs(TypedDict):
    max_tokens_per_microbatch: int     # Bin capacity for packing
    input_key: str                     # Input tensor key
    input_lengths_key: str             # Sequence lengths key
    algorithm: str                     # Packing algorithm name
    sequence_length_pad_multiple: int  # CP/TP alignment factor
```

## Integration with Training Pipeline

### Loss Function Integration
A key design consideration was that we wanted to avoid the loss function writer needing to be aware of if there is sequence packing or not. To do this, we created a `SequencePackingLossWrapper` which takes the packed next_token_logits and the unpacked auxiliary loss function data and runs the loss function on each sequence individually. Since the loss function's computation time is typically trivial, we don't see a slowdown from this approach. With this, the loss function can be written as though it just deals with typical, unpacked batched data (as long as it is capable of processing one sequence at a time).

If your loss function cannot assume batch-independence, however, then both Dynamic Batching and Sequence Packing won't work. (I.e. DPO [issue #719](https://github.com/NVIDIA-NeMo/RL/issues/719)).

## Metrics and Monitoring

### Packing Efficiency Metrics (`nemo_rl/data/packing/metrics.py`)

- **Bin Utilization**: Percentage of bin capacity used
- **Waste Ratio**: Fraction of capacity unused due to packing constraints
- **Bin Balance**: Measure of load distribution evenness across bins
- **Packing Efficiency**: Ratio of theoretical minimum to actual bins used

## Usage
### Sequence Packing Configuration
```yaml
# examples/configs/grpo_math_1B.yaml
policy:
  sequence_packing:
    enabled: True
    train_mb_tokens: 2048  # Target tokens per microbatch
    logprob_mb_tokens: 2048
    algorithm: "modified_first_fit_decreasing"  # Best algorithm
    sequence_length_round: 64  # Hardware alignment
  
  dynamic_batching:
    enabled: False  # Mutually exclusive
```

### Dynamic Batching Configuration
```yaml
# examples/configs/grpo_math_8B.yaml
policy:
  dynamic_batching:
    enabled: True
    train_mb_tokens: 4096
    logprob_mb_tokens: 8192
    sequence_length_round: 64
  
  sequence_packing:
    enabled: False  # Mutually exclusive
```

### Framework Compatibility

**Sequence Packing Requirements:**
- Megatron or DTensor policy
- FlashAttention-2 for efficient packed attention
- If using CP with Megatron, you _must_ use sequence packing. If using CP with Dtensor, you _cannot_ yet use packing (WIP, [Issue #520](https://github.com/NVIDIA-NeMo/RL/issues/520))

**Dynamic Batching Requirements:**
- Any policy framework
- Pipeline parallelism size = 1
- Cannot be used with torch.compile since shapes change.

---

## References
[Johnson & Garey (1985) - Modified First Fit Decreasing](https://doi.org/10.1016/0885-064X(85)90022-6)
