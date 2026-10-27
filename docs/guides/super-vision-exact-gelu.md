# Exact GELU for the Super vision encoder

Super's Megatron vision MLPs can use the tanh approximation (`fast_gelu`),
while the vLLM RADIO path uses exact GELU. Matching checkpoint weights and
tokens does not eliminate this activation mismatch.

Enable the opt-in override in a Super Megatron recipe:

```yaml
policy:
  megatron_cfg:
    super_vision_exact_gelu: true
```

The flag is disabled by default. After model construction, the policy worker
validates all 32 vision MLPs before replacing their activation with
`torch.nn.functional.gelu` (the default `approximate="none"`).
Language-model activations, parameters, state-dict keys, and gradient flow
are unchanged. No forward hooks are installed.

This override targets the tested Super layout: all 32 vision MLPs must be
present on the worker, under names containing `vision_model.` and ending in
`.mlp`. It rejects missing or different layer counts, non-GELU activations,
Transformer Engine activation modules, and bias-activation fusion. It is
not a general activation override for other architectures or pipeline
layouts that split/omit the vision encoder.

## Separate precision setting

FP32 LM-head support already exists and is independent of this override.
When used, its Megatron and vLLM settings must agree:

```yaml
policy:
  megatron_cfg:
    fp32_lm_head: true
  generation:
    vllm_cfg:
      fp32_lm_head: true
```

The exact-GELU flag does not enable FP32 logits or change these defaults.
Backend parity should be measured using the same checkpoint, token sequence,
and multimodal inputs, with matching routing replay settings. Stale-policy
TMPE during asynchronous training also includes real policy drift; low
same-checkpoint TMPE does not guarantee reward or entropy stability.
