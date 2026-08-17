# Qwen3.8

This page describes the initial NeMo RL support for the dense
`Qwen/Qwen3.8-27B` model.

## Support Status

`Qwen/Qwen3.8-27B` is **Functionally Ready** for text-only GRPO on the Megatron
(MBridge) training backend, with vLLM inference. The shipped recipe is a short
functional smoke test. It validates model loading, rollout generation, weight
refit, log-probability computation, and an optimizer step; it is not a long-run
convergence recipe.

| Model | Modality | Training backend | Parallelism | Inference |
| --- | --- | --- | --- | --- |
| `Qwen/Qwen3.8-27B` | LLM (dense) | Megatron | TP4 + DP4 | vLLM TP8 |

The model follows the same dense hybrid-attention integration path as the
supported Qwen3.5 dense models. Context parallelism is not enabled in this
recipe.

> [!NOTE]
> AutoModel backend support is tracked in
> [issue #3675](https://github.com/NVIDIA-NeMo/RL/issues/3675) and deferred until
> [NeMo RL PR #3498](https://github.com/NVIDIA-NeMo/RL/pull/3498) upgrades the
> pinned AutoModel revision to one that provides the Qwen3.5-family dense
> state-dict adapter. A follow-up PR will add and validate the AutoModel recipe
> and nightly coverage.

## Nightly Smoke Recipe

The recipe uses two nodes with eight GPUs per node, a 512-token total sequence
length, and a deliberately small rollout batch. The nightly driver runs one
training step.

| Algorithm | Backend | Scale | Recipe |
| --- | --- | --- | --- |
| GRPO | Megatron | 2n8g | [`grpo-qwen3.8-27b-2n8g-megatron.yaml`](../../../../examples/configs/recipes/llm/grpo-qwen3.8-27b-2n8g-megatron.yaml) |

Run the recipe directly with:

```sh
uv run examples/run_grpo.py \
  --config examples/configs/recipes/llm/grpo-qwen3.8-27b-2n8g-megatron.yaml
```

For a real training run, increase the sequence length, rollout batch, and step
count for the target workload, then validate convergence separately.
