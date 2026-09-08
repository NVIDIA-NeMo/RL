# Model Guides

Model-family guidance for post-training with NeMo RL. Each family hub links to
version-specific pages covering recipe selection, recommended generation settings,
and known issues. Recipe YAML files under `examples/configs/recipes/` remain the source of
truth; these pages explain *when and why* to choose a recipe.

For the full list of supported models, see
[Model Support](../../about/model-support.md).

## Families

- **[GLM](glm/index.md)** — GLM-5.1 and GLM-5.2 GRPO recipes on the Megatron
  backend, colocated and non-colocated with vLLM.
- **[Nemotron](nemotron/index.md)** — post-training recipes for Nemotron 3
  Nano, Nano Omni, Super, Ultra, and Nemotron 3.5 Lightning, spanning the
  Megatron and AutoModel backends.
- **[Qwen](qwen/index.md)** — Qwen3.5 and Qwen3.8 LLM and VLM recipes (dense and
  MoE), with backend availability documented per version and thinking-mode
  generation-length guidance.

Other model-specific guides currently live directly under
[Guides](../../index.md) (for example, [DeepSeek](../deepseek.md)) and are
migrated into this hub as their guidance grows.

```{toctree}
:hidden:

glm/index.md
nemotron/index.md
qwen/index.md
```
