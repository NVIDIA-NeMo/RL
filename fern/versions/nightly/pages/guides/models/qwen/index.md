---
title: "Qwen"
---
This is the landing page for Qwen model guidance in NeMo RL. It links to
version-specific subpages and points to the recipes and known issues for each
supported variant.

For the full list of supported Qwen models, see
[Model Support](/about/model-support).

## Version Guides

- **[Qwen3.8](/guides/model-guides/qwen/qwen3-8)** — initial dense `Qwen3.8-27B` text-only GRPO
  support on the Megatron backend, with a low-cost functional nightly recipe.
- **[Qwen3.5](/guides/model-guides/qwen/qwen3-5)** — LLM and VLM recipes for `Qwen3.5-9B-Base`,
  `Qwen3.5-35B-A3B-Base`, and `Qwen3.5-397B-A17B` on the Megatron and AutoModel
  backends. This guide covers backend and parallelism support, example recipes, and the
  `flash-linear-attention` performance notes.

Subpages for other Qwen versions are added as distinct, recipe-backed guidance
accumulates. Until then, the GRPO, evaluation, and recipe YAML files remain the source of
truth for those models.

> [!NOTE]
> Qwen3 and Qwen3.5 thinking models need a large generation budget. See
> [Qwen3.5 → Example Recipes](/guides/model-guides/qwen/qwen3-5#example-recipes) for the
> `max_new_tokens` guidance.
