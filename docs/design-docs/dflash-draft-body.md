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

## Scope

The body implements the draft-state update only. The caller supplies token
embeddings and uses the returned states to compute vocabulary logits and loss.
