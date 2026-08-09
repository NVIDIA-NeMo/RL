# Model Guides

Model-family guidance for post-training with NeMo RL. Each family hub links to
version-specific pages covering recipe selection, recommended generation settings,
and known issues. Recipe YAML files under `examples/configs/recipes/` remain the source of
truth; these pages explain *when and why* to choose a recipe.

For the full list of supported models, see
[Model Support](../../about/model-support.md).

## Families

- **[Muse Glimmer](muse-glimmer.md)** — 29B DAPO recipes on the AutoModel backend,
  with context-parallel variants at 4k, 8k and 16k and the memory budgeting that
  makes them fit.
- **[Qwen](qwen/index.md)** — Qwen3.5 LLM and VLM recipes (dense and MoE) on the
  Megatron and AutoModel backends, including thinking-mode generation-length
  guidance.

Other model-specific guides currently live directly under
[Guides](../../index.md) (for example, [DeepSeek](../deepseek.md) and the
Nemotron guides) and are migrated into this hub as their guidance grows.
