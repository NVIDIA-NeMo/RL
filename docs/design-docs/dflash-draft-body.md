# DFlash Draft Body

The target model provides hidden states for an input sequence. The DFlash draft
body uses these states as context for short blocks of draft-token positions.
It updates all positions in a block together and returns hidden states for a
separate draft-loss calculation. The body does not produce vocabulary logits.

## How It Works

1. **Choose anchors.** An anchor is a valid position in the target sequence.
   `build_dflash_batch_plan` chooses anchors and records which draft positions
   belong to each one. The same input mask, sample IDs, optimizer step, and seed
   produce the same plan. A sample with no valid window contributes no draft
   loss.
2. **Build draft blocks.** Each block has one anchor position followed by
   `gamma` draft positions. The caller supplies an embedding for each position.
   The target model supplies selected hidden states from its layers.
3. **Update the blocks.** `DFlashBody` projects the target hidden states and
   passes the block embeddings through draft decoder layers. Within a layer,
   each block can read the target prefix before its anchor and all positions
   in that same block. It cannot read another block. Attention computes outputs
   for block positions only; it does not update the target sequence.
4. **Compute the loss.** The body returns one hidden state per block position.
   The caller uses `plan.loss_mask` to exclude the anchor and invalid positions
   from the draft loss. The body returns zero for invalid positions.

This separation lets the caller reuse target-model context across draft blocks
while keeping each block's attention independent.

## Component Boundary

`DFlashBody` owns the draft projection, decoder layers, and normalization.
It does not own token embeddings or an output head. The caller provides block
embeddings and handles the output projection and loss. A standalone body
checkpoint therefore contains body weights only.

The entry points are `build_dflash_batch_plan` in `block_plan.py`,
`DFlashBody.forward` in `dflash.py`, and `dflash_block_only_attention` in
`block_attention.py`. The GPU attention path uses FlexAttention. The CPU path
is a reference implementation.

## Input Contract

The plan takes a valid-token mask with shape `[batch, sequence]` and one stable
`int64` sample ID per row. The body takes target hidden-state taps with shape
`[batch, sequence, taps, hidden]` and block embeddings with shape
`[batch * anchors_per_sample, gamma + 1, hidden]`. It returns block hidden
states with the same shape as the block embeddings. The plan and input tensors
must share a device, and both input tensors must share a floating-point dtype.

The body supports tensor parallelism (TP) with context parallelism (CP) set to
1. Tests cover TP sizes 1, 2, and 4. The body rejects sequence parallelism and
CP greater than 1. Pipeline-parallel integration is outside this component.
