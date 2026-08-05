# Out-of-tree changes for Super Omni MTP at CP>1

`mtp-sp-scatter.patch` applies to the nested Megatron-LM submodule:

    git -C 3rdparty/Megatron-Bridge-workspace/Megatron-Bridge/3rdparty/Megatron-LM \
        apply ../../../../../patches/mtp-sp-scatter.patch

Required for MTP under context parallelism on Nemotron Omni. Without it,
`_concat_embeddings` fails at 16384 tokens with CP=2 / TP=8:

    RuntimeError: Sizes of tensors must match except in dimension 2.
    Expected size 8192 but got size 1024 for tensor number 1 in the list.

## Reapply after any submodule sync

`git submodule update --init --recursive` resets Megatron-LM to the commit
Megatron-Bridge pins and silently discards this patch, so the next MTP run
fails with the error above. Re-run the apply command afterwards. To check
whether it is currently applied:

    git -C 3rdparty/Megatron-Bridge-workspace/Megatron-Bridge/3rdparty/Megatron-LM \
        diff --quiet megatron/core/transformer/multi_token_prediction.py \
        && echo "NOT applied" || echo "applied"

## Why a patch rather than a submodule bump

The fix is committed as Megatron-LM `8590b5210` and proposed upstream, but
carrying it here as a pointer bump would require bumping Megatron-Bridge to a
commit that does not exist upstream, which breaks `git submodule update` for
everyone else. The Megatron-Bridge revision this branch pins (`2f0f8c80`)
already points at the Megatron-LM base the fix is built on (`cd4afffa6`), so
the patch applies cleanly.

Delete this directory once the Megatron-LM change lands and Megatron-Bridge
picks it up; the pin then moves forward on its own.
