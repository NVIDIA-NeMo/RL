# Qwen

This is the landing page for Qwen model guidance in NeMo RL. It links to
version-specific subpages and points to the recipes and known issues for each
supported variant.

For the full list of supported Qwen models, see
[Model Support](../../../about/model-support.md).

## Version Guides

- **[Qwen3.5](qwen3.5.md)** — LLM and VLM recipes for `Qwen3.5-9B-Base`,
  `Qwen3.5-35B-A3B-Base`, and `Qwen3.5-397B-A17B` on the Megatron and AutoModel
  backends. Covers generation-length defaults for thinking-mode runs, the
  `flash-linear-attention` performance requirement, and known issues.

Subpages for other Qwen versions are added as distinct, recipe-backed guidance
accumulates. Until then, the GRPO, evaluation, and recipe YAMLs remain the source of
truth for those models.

## Quick Tips for Qwen Thinking Models

> [!WARNING]
> Qwen3 and Qwen3.5 thinking models need a large generation budget. Default
> `max_new_tokens` values from non-thinking baselines (3K–4K) can truncate the
> reasoning trace before the final answer and make evaluation accuracy appear near
> zero while training metrics look normal. Set
> `policy.generation.max_new_tokens >= 8192` and size
> `policy.max_total_sequence_length` / `policy.generation.vllm_cfg.max_model_len`
> to match. See [Qwen3.5 → Key Defaults](qwen3.5.md#key-defaults) and
> [#2725](https://github.com/NVIDIA-NeMo/RL/issues/2725).
