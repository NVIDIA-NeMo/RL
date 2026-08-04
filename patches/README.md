# Out-of-tree changes for Super Omni MTP at CP>1

`mtp-sp-scatter.patch` applies to the nested Megatron-LM submodule:

    git -C 3rdparty/Megatron-Bridge-workspace/Megatron-Bridge/3rdparty/Megatron-LM \
        apply ../../../../../patches/mtp-sp-scatter.patch

Required for MTP under context parallelism on Nemotron Omni. Without it,
_concat_embeddings fails at 16384 tokens with CP=2 / TP=8:

    RuntimeError: Sizes of tensors must match except in dimension 2.
    Expected size 8192 but got size 1024 for tensor number 1 in the list.

Committed upstream-side as Megatron-LM 8590b5210; this patch exists so the
branch is reproducible before that lands.
