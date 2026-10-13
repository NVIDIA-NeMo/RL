# DFlash Draft Body

`DFlashBody` turns selected target-model hidden states into draft-block hidden
states. Use `build_dflash_batch_plan` to select the blocks before calling the
body. The caller supplies token embeddings for each block and computes the
draft loss from the returned hidden states.

## Use the Body

Let `B` be the batch size, `S` the target sequence length, `A` the number of
anchors per sample, `G` the number of draft tokens per block, `T` the number of
target hidden-state taps, and `H` the hidden size.

1. Build a boolean `token_valid_mask` with shape `[B, S]`. Give each sample a
   stable `int64` ID in `sample_ids` with shape `[B]`.
2. Build the plan. Use the same valid-token mask, sample IDs, optimizer step,
   and seed to select the same anchors again.
3. Collect `target_taps` with shape `[B, S, T, H]`. Build
   `block_embeddings` with shape `[B * A, G + 1, H]`. Slot 0 holds the anchor
   embedding; the other slots hold draft-token embeddings.
4. Pass these tensors and the plan to `DFlashBody`. The result has shape
   `[B * A, G + 1, H]`. Use `plan.loss_mask` to exclude the anchor and invalid
   slots from the draft loss.

The example assumes that the caller has prepared the input tensors, process
groups, and a `DFlashBodyConfig` that matches the model dimensions.

```python
from nemo_rl.models.megatron.draft.block_plan import build_dflash_batch_plan
from nemo_rl.models.megatron.draft.dflash import DFlashBody

plan = build_dflash_batch_plan(
    token_valid_mask,
    sample_ids,
    anchors_per_sample=anchors_per_sample,
    gamma=gamma,
    optimizer_step=optimizer_step,
    seed=seed,
)
body = DFlashBody(
    body_config,
    tp_group=tp_group,
    parallel_config=parallel_config,
)
draft_hidden = body(
    target_taps=target_taps,
    block_embeddings=block_embeddings,
    plan=plan,
)
loss_mask = plan.loss_mask
```

The plan and both input tensors must be on the same device. Both input tensors
must have the same floating-point dtype. The body does not create token
embeddings or vocabulary logits. Its checkpoint contains body weights, not
token embeddings or an output head.

## Block Selection

`build_dflash_batch_plan` selects `A` valid windows per sample. Each window
contains one anchor position and `G` following positions. The plan stores
positions, sample rows, and validity masks as tensors on the input device.
The anchor slot does not contribute to the loss. If a sample has no valid
window, its blocks are invalid and contribute no loss.

## Body Computation

`DFlashBody` projects the `T` target hidden states at each sequence position
into one hidden state. Each decoder layer forms keys and values from these
projected target states. It forms queries, keys, and values for the block
slots. `dflash_block_only_attention` computes attention for block slots only:

- A block can read valid target tokens before its anchor.
- A block can read all slots in that same block.
- A block cannot read slots in another block.

The GPU path uses FlexAttention. The CPU path uses a reference implementation.
The body sets the output of invalid slots to zero.

## Parallelism

The body supports tensor parallelism (TP) with context parallelism (CP) set to
1. Tests cover TP sizes 1, 2, and 4. The body rejects sequence parallelism and
CP greater than 1. Pipeline-parallel integration is outside this component.
