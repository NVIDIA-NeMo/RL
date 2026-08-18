# Qwen3.8

This page describes the initial NeMo RL support for the dense
`Qwen/Qwen3.8-27B` model.

## Support Status

`Qwen/Qwen3.8-27B` is **Functionally Ready** for text-only GRPO on the Megatron
(MBridge) training backend, with vLLM inference. The shipped recipe is a short
functional smoke test. It validates model loading, rollout generation, weight
refit, log-probability computation, and an optimizer step; it is not a long-run
convergence recipe.

## What's Supported

| Model | Modality | Training backend | Parallelism | Inference |
| --- | --- | --- | --- | --- |
| `Qwen/Qwen3.8-27B` | LLM (dense) | Megatron | TP + PP + CP | vLLM |

The model follows the same dense hybrid-attention integration path as the
supported Qwen3.5 dense models. Context parallelism requires sequence packing,
which is enabled in the example recipe below.

> [!NOTE]
> AutoModel backend support is tracked in
> [issue #3675](https://github.com/NVIDIA-NeMo/RL/issues/3675) and deferred until
> [NeMo RL PR #3498](https://github.com/NVIDIA-NeMo/RL/pull/3498) upgrades the
> pinned AutoModel revision to one that provides the Qwen3.5-family dense
> state-dict adapter. A follow-up PR will add and validate the AutoModel example
> recipe.

## Example Recipes

The recipe below is an example starting point. Recipe YAML files under
`examples/configs/recipes/` are the source of truth; check the YAML file for the
authoritative settings.

| Model | Modality | Algorithm | Backend | Scale | Recipe |
|---|---|---|---|---|---|
| Qwen3.8-27B | LLM | GRPO | Megatron | 2n8g | [`grpo-qwen3.8-27b-2n8g-megatron-tp2pp2cp2.yaml`](../../../../examples/configs/recipes/llm/grpo-qwen3.8-27b-2n8g-megatron-tp2pp2cp2.yaml) |

## Choose a Recipe

### 27B GRPO (Megatron)

Use the Megatron recipe to validate the setup, launch mechanics, logging, and
checkpointing.

```sh
uv run examples/run_grpo.py \
  --config examples/configs/recipes/llm/grpo-qwen3.8-27b-2n8g-megatron-tp2pp2cp2.yaml
```

This is a functional validation recipe rather than a long-run convergence
recipe; validate longer training separately for the target workload.
