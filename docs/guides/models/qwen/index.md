# Qwen

This is the landing page for Qwen model guidance in NeMo RL. It links to
version-specific subpages and points to the recipes and known issues for each
supported variant.

For the full list of supported Qwen models, see
[Model Support](../../../about/model-support.md).

## Version Guides

- **[Qwen3.5](qwen3-5.md)** — LLM and VLM recipes for `Qwen3.5-9B-Base`,
  `Qwen3.5-35B-A3B-Base`, and `Qwen3.5-397B-A17B` on the Megatron and AutoModel
  backends. Covers backend/parallelism support, example recipes, and the
  `flash-linear-attention` performance requirement.

Subpages for other Qwen versions are added as distinct, recipe-backed guidance
accumulates. Until then, the GRPO, evaluation, and recipe YAMLs remain the source of
truth for those models.

> [!NOTE]
> Qwen3 and Qwen3.5 thinking models need a large generation budget. See
> [Qwen3.5 → Example Recipes](qwen3-5.md#example-recipes) for the
> `max_new_tokens` guidance.
