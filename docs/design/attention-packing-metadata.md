# Packed attention metadata

Megatron packing retains the CPU offsets used to construct `cu_seqlens` and registers them with Transformer Engine when its optional `register_cu_seqlens` API is available. Logical and padded boundaries keep their existing meanings. There is no model, head-count, or sequence-length restriction in this integration.

The ordinary and VLM packers register their actual constructed boundaries. Prepacked CPU boundaries are also registered. Prepacked GPU-only boundaries are left unchanged; no device-to-host read is introduced to recover metadata.

The policy worker wraps forward and backward in TE's optional `attention_backend_workspace()` scope. TE captures ownership during forward, reuses scratch during training, and releases it on normal or exceptional exit. This does not select an attention backend or alter evaluation, losses, masks, or rollout.

Older TE versions continue through the existing path. TE owns backend selection and eligibility. One intended consumer is its opt-in cuDNN compact GQA backward; other consumers can use the same metadata interface. GPU-only or transformed, unregistered prefixes use the regular attention backend.
