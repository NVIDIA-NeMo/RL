---
title: "Model Guides"
---
Model-family guidance for post-training with NeMo RL. Each family hub links to
version-specific pages covering recipe selection, recommended generation settings,
and known issues. Recipe YAML files under `examples/configs/recipes/` remain the source of
truth; these pages explain *when and why* to choose a recipe.

For the full list of supported models, see
[Model Support](/about/model-support).

## Families

- **[DeepSeek](/guides/model-guides/deepseek)** — DeepSeek V4 Flash GRPO with AutoModel
  training and block-FP8 vLLM generation.
- **[GLM](/guides/model-guides/glm)** — GLM-5.1 and GLM-5.2 GRPO recipes on the Megatron
  backend, colocated and non-colocated with vLLM.
- **[Gemma](/guides/model-guides/gemma)** — Gemma 4 text-only DAPO and VLM GRPO recipes on
  AutoModel, including Context Parallel layouts for the text-only variants.
- **[MiniMax](/guides/model-guides/minimax)** — MiniMax-M3 GRPO with AutoModel training
  and BF16 vLLM generation, including validated scope and known limitations.
- **[Nemotron](/guides/model-guides/nemotron)** — post-training recipes for Nemotron 3
  Nano, Nano Omni, Super, Ultra, and Nemotron 3.5 Lightning, spanning the
  Megatron and AutoModel backends.
- **[Qwen](/guides/model-guides/qwen)** — Qwen3.5 and Qwen3.8 LLM and VLM recipes (dense and
  MoE), with backend availability documented per version and thinking-mode
  generation-length guidance.

Other model-specific guides currently live directly under
[Guides](/about) and are migrated into this hub as their guidance grows.
