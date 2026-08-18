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
| `Qwen/Qwen3.8-27B` | LLM (dense) | Megatron | TP2 + PP2 + CP2 + DP2 | vLLM TP4 |

The model follows the same dense hybrid-attention integration path as the
supported Qwen3.5 dense models. Context parallelism requires sequence packing,
which is enabled in this recipe. With TP2, CP2, and sequence parallelism, packed
sequences are padded to a multiple of 8.

> [!NOTE]
> AutoModel backend support is tracked in
> [issue #3675](https://github.com/NVIDIA-NeMo/RL/issues/3675) and deferred until
> [NeMo RL PR #3498](https://github.com/NVIDIA-NeMo/RL/pull/3498) upgrades the
> pinned AutoModel revision to one that provides the Qwen3.5-family dense
> state-dict adapter. A follow-up PR will add and validate the AutoModel recipe
> and nightly coverage.

## Nightly Recipe

The recipe inherits the Qwen3.5-9B Megatron GRPO defaults, including a
4,096-token maximum sequence length. It uses two nodes with eight GPUs per node,
32 prompts per step, eight generations per prompt, and a global training batch
size of 256. MCore uses TP2, PP2, and CP2 across the 16 GPUs, leaving DP2;
vLLM uses TP4. The nightly driver runs 10 training steps with checkpointing
enabled, saving at step 10.

| Algorithm | Backend | Scale | Recipe |
| --- | --- | --- | --- |
| GRPO | Megatron | 2n8g, TP2/PP2/CP2 | [`grpo-qwen3.8-27b-2n8g-megatron-tp2pp2cp2.yaml`](../../../../examples/configs/recipes/llm/grpo-qwen3.8-27b-2n8g-megatron-tp2pp2cp2.yaml) |

Run the recipe directly with:

```sh
uv run examples/run_grpo.py \
  --config examples/configs/recipes/llm/grpo-qwen3.8-27b-2n8g-megatron-tp2pp2cp2.yaml
```

This remains a functional validation recipe rather than a convergence recipe;
validate longer training separately for the target workload.
