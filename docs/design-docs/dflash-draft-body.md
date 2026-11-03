# DFlash Draft Body

The target model provides hidden states for an input sequence. The DFlash draft
body uses these states as context for short blocks of draft-token positions.
It updates all positions in a block together and returns hidden states for a
separate draft-loss calculation. The body does not produce vocabulary logits.

## How It Works

1. **Choose anchors.** An anchor is a valid position in the target sequence.
   `build_dflash_batch_plan` chooses anchors and records which draft positions
   belong to each one. The same input mask, sample IDs, optimizer step, and seed
   produce the same plan. Rows with the same sample ID get the same anchors, so
   give each rollout row a unique sample ID. A sample with no valid window
   contributes no draft loss.
2. **Build draft blocks.** Each block has one anchor position followed by
   `gamma` draft positions. The caller supplies an embedding for each position.
   The target model supplies selected hidden states from its layers.
3. **Update the blocks.** `DFlashBody` projects the target hidden states and
   passes the block embeddings through draft decoder layers. Within a layer,
   each block can read the target prefix before its anchor and all positions
   in that same block. It cannot read another block. Attention computes outputs
   for block positions only. It does not update the target sequence.
4. **Compute the loss.** The body returns one hidden state per block position.
   The caller uses `plan.loss_mask` to exclude the anchor and invalid positions
   from the draft loss. The body returns zero for invalid positions.

With this separation, the caller can use the same target-model context for
all draft blocks. The attention of each block stays independent.

## Scope

The body implements the draft-state update only. The caller supplies token
embeddings and uses the returned states to compute vocabulary logits and loss.

The body supports tensor parallelism. The tensor-parallel size must divide
`num_key_value_heads`, so the default config supports a maximum of 8. The body
does not support sequence parallelism or context parallelism greater than 1.
The caller controls pipeline-parallel placement.
