# DFlash Draft Body

The DFlash body updates draft blocks from target hidden-state taps. It shares
the target model's embeddings and output head. This change provides the draft
body, block planning, and block-only attention. PR #3715 supplies the projected
loss and body-only export contract. Generation and refit integration are in PR
#3744.

## Block Plan

`build_dflash_batch_plan` selects anchors from valid input windows. It uses
sample IDs, the optimizer step, and a seed to select anchors deterministically.
The plan contains query positions, label positions, and validity masks. A block
has one anchor slot and `gamma` draft slots. The anchor slot does not contribute
to the draft loss.

## Attention

`dflash_block_only_attention` computes queries for block slots only. Each block
can read its valid trunk prefix and its own block slots. It cannot read another
block. The function returns zero for invalid slots. CUDA uses FlexAttention;
the CPU path provides a reference implementation.

## Supported Parallelism

`DFlashBody` supports tensor parallelism (TP) with context parallelism (CP) set
to 1. Tests cover TP1, TP2, and TP4. The body rejects draft-body sequence
parallelism and CP greater than 1. Full pipeline-parallel integration is not
part of this change.

The body keeps target-owned parameters out of the standalone draft checkpoint.
Its sharded state dictionary stores the draft-body parameters required for
checkpoint loading.
