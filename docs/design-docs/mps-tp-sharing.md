# MPS weight sharing with different tensor parallel sizes

The experimental TP sharing path supports training at TP1 and inference at
TP2 or higher. Both sides must use the same EP, with ETP1, CP1 and PP1. The GPU
count must support both layouts. GPU and full-model validation are required
before treating a new layout as a supported performance configuration.

For inference-optimized Megatron models, set both inference options:

```yaml
policy:
  generation:
    mcore_generation_config:
      tensor_model_parallel_size: 2
      sequence_parallel: true
```

The existing MPS mode and `engine_weights: shared` select the sharing path. Keep
the training TP at 1 and both expert layouts unchanged. No extra sharing switch
is required. Changing inference TP also changes its data-parallel degree and
aggregate request capacity; account for those changes in comparisons.

Each inference rank stays on the GPU with the matching training rank. Dense
training weights are complete at TP1, and unchanged EP/ETP preserves local
expert ownership. The native Megatron shard planner maps destination weights
to source slices using logical coordinates, including partition strides and
packed components. It rejects incomplete local coverage.

| Destination layout | Storage and update behavior |
|---|---|
| Contiguous source view, compatible parameter dtype | Shares training storage |
| Noncontiguous source view, such as an input-column slice | Private inference weight, copied at publication |
| Packed projection requiring several separated slices | Private inference weight, updated directly from those slices |
| Unchanged compatible expert stacks | Retains existing expert sharing |
| Persistent buffer | Private, updated at publication |

The existing optimizer reader pause, parameter publication and inference cache
refresh remain responsible for consistency. Fragment copying allocates no extra
full-weight staging buffer on each update. Shared parameters can still become
private if the inference backend replaces their storage; subsequent refits
continue to copy them correctly.

This path rejects GTP and quantized source storage. Unsupported attention or
other model-specific sharding metadata fails the native logical-shape or
coverage check. Training TP greater than 1, lowering inference TP, simultaneous
EP changes and mixed CP are not enabled by this extension. Those cases require
additional placement or access to shards on other GPUs.
