# DFlash Draft Body

The DFlash body updates draft blocks from selected target hidden states. This
change adds the body, block planning, and block-only attention. [PR #3715](https://github.com/NVIDIA-NeMo/RL/pull/3715)
adds the projected loss and body-only export contract. [PR #3744](https://github.com/NVIDIA-NeMo/RL/pull/3744)
adds generation and refit integration.

## Block Plan

`build_dflash_batch_plan` selects anchors from valid input windows. It uses
sample IDs, the optimizer step, and a seed. The same inputs produce the same
anchors. Each block has one anchor slot and `gamma` draft slots. Here, `gamma`
is the number of draft tokens per block. The plan records query positions,
label positions, and validity masks. The anchor slot does not contribute to
the draft loss.

## Attention

`dflash_block_only_attention` computes queries for block slots only. Each block
can read valid target-sequence tokens before its anchor and its own block slots.
It cannot read another block. Invalid slots produce zero output. The GPU path
uses FlexAttention. The CPU path is a reference implementation.

## Supported Parallelism

`DFlashBody` supports tensor parallelism (TP) when context parallelism (CP) is
1. Tests cover TP sizes 1, 2, and 4. The body rejects sequence parallelism and
CP greater than 1. This change does not add pipeline-parallel integration.

## Weight Ownership

The body does not own token embeddings or an output head (`lm_head`). Its caller
supplies block embeddings. The standalone draft checkpoint stores body weights
only. The body does not train or export embedding or output-head weights.
Ownership of these weights across draft backends is under discussion in
[issue #4691](https://github.com/NVIDIA-NeMo/RL/issues/4691).
