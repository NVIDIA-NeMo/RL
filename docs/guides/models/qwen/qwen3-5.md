# Qwen3.5

This page collects NeMo RL guidance for Qwen3.5 LLM and VLM post-training. Use it to
choose a starting recipe, install the dependencies needed for full performance, and
understand Qwen3.5-specific setup.

## When to Use This Page

Use this page when training or evaluating Qwen3.5 models, including:

- `Qwen/Qwen3.5-9B-Base`
- `Qwen/Qwen3.5-35B-A3B-Base` (MoE; LLM and VLM)
- `Qwen/Qwen3.5-397B-A17B` (MoE)

For family-wide Qwen guidance, see the [Qwen family hub](index.md). For the full list
of supported models, see [Model Support](../../../about/model-support.md).

## Support Status

We track model support in two stages:

| Stage | Meaning |
| --- | --- |
| **Functional Ready** | Runnable end-to-end and numerically validated with an initial training run. |
| **Long-run convergence validated** | Trains stably over a full-length run with a healthy, reproducible reward curve. |

The Qwen3.5 family is supported on both the Megatron (MCore) and AutoModel (DTensor)
backends. The specific configurations shipped as [example recipes](#example-recipes)
below are the ones that have been **long-run convergence validated**. Other variants
and configurations are runnable but have not all been validated for long-run
convergence.

## What's Supported

| Model | Modality | Training backend | Parallelism | Inference |
| --- | --- | --- | --- | --- |
| `Qwen/Qwen3.5-9B-Base` | LLM (dense) | Megatron | TP | vLLM |
| `Qwen/Qwen3.5-35B-A3B-Base` | LLM (MoE) | Megatron | EP + TP + CP | vLLM |
| `Qwen/Qwen3.5-35B-A3B-Base` | LLM (MoE) | AutoModel (DTensor) | EP | vLLM |
| `Qwen/Qwen3.5-35B-A3B-Base` | VLM (MoE) | Megatron / AutoModel | EP | vLLM |
| `Qwen/Qwen3.5-397B-A17B` | LLM (MoE) | Megatron | TP + PP + EP | vLLM |

Notes on backends and parallelism:

- **Megatron (MCore)** supports the widest parallelism for Qwen3.5 MoE, including
  Context Parallel (CP) for longer sequences (see [#2312](https://github.com/NVIDIA-NeMo/RL/pull/2312)).
- **AutoModel (DTensor)** supports Expert Parallel (EP). For Qwen3.5 MoE + CP on
  AutoModel, the TE backend plus `flash-linear-attention` is required; **dense
  Qwen3.5 does not support CP on AutoModel** (set `cp_size = 1`). See
  [Performance: flash-linear-attention](#performance-flash-linear-attention).

## Example Recipes

The recipes below are example starting points. Recipe YAMLs under
`examples/configs/recipes/` are the source of truth; check the YAML for the
authoritative settings.

| Model | Modality | Algorithm | Backend | Scale | Recipe |
|---|---|---|---|---|---|
| Qwen3.5-9B-Base | LLM | GRPO | Megatron | 1n8g | [`grpo-qwen3.5-9b-1n8g-megatron.yaml`](../../../../examples/configs/recipes/llm/grpo-qwen3.5-9b-1n8g-megatron.yaml) |
| Qwen3.5-35B-A3B-Base | LLM | GRPO | Megatron | 2n8g | [`grpo-qwen3.5-35ba3b-2n8g-megatron-ep16tp2cp2.yaml`](../../../../examples/configs/recipes/llm/grpo-qwen3.5-35ba3b-2n8g-megatron-ep16tp2cp2.yaml) |
| Qwen3.5-35B-A3B-Base | LLM | GRPO | AutoModel | 2n8g | [`grpo-qwen3.5-35ba3b-2n8g-automodel-ep16.yaml`](../../../../examples/configs/recipes/llm/grpo-qwen3.5-35ba3b-2n8g-automodel-ep16.yaml) |
| Qwen3.5-35B-A3B-Base | LLM | DAPO-style GRPO | AutoModel | 4n8g | [`grpo-qwen3.5-35ba3b-dapo-4n8g-automodel.yaml`](../../../../examples/configs/recipes/llm/grpo-qwen3.5-35ba3b-dapo-4n8g-automodel.yaml) |
| Qwen3.5-397B-A17B | LLM | GRPO | Megatron | 32n8g | [`grpo-qwen3.5-397ba17b-32n8g-megatron.v2.yaml`](../../../../examples/configs/recipes/llm/grpo-qwen3.5-397ba17b-32n8g-megatron.v2.yaml) |
| Qwen3.5-35B-A3B-Base | VLM | GRPO | Megatron | 2n8g | [`vlm_grpo-qwen3.5-35ba3b-geo3k-2n8g-megatron-ep16.yaml`](../../../../examples/configs/recipes/vlm/vlm_grpo-qwen3.5-35ba3b-geo3k-2n8g-megatron-ep16.yaml) |
| Qwen3.5-35B-A3B-Base | VLM | GRPO | AutoModel | 2n8g | [`vlm_grpo-qwen3.5-35ba3b-geo3k-2n8g-automodel-ep16.yaml`](../../../../examples/configs/recipes/vlm/vlm_grpo-qwen3.5-35ba3b-geo3k-2n8g-automodel-ep16.yaml) |

> [!NOTE]
> Qwen3.5 thinking-mode and long-reasoning runs need a large generation budget. If
> `policy.generation.max_new_tokens` (and the matching `policy.max_total_sequence_length`
> / `policy.generation.vllm_cfg.max_model_len`) is too small, the reasoning trace can
> be truncated before the final answer, and evaluation accuracy can appear near zero
> even when training metrics look normal. Use `max_new_tokens >= 8192` for reasoning
> tasks. See [#2725](https://github.com/NVIDIA-NeMo/RL/issues/2725).

## Choosing a Recipe

### Small LLM smoke run

Use the 9B Megatron recipe when validating setup, launch mechanics, logging, and
checkpointing.

```sh
uv run examples/run_grpo.py \
  --config examples/configs/recipes/llm/grpo-qwen3.5-9b-1n8g-megatron.yaml
```

### 35B-A3B GRPO (Megatron or AutoModel)

Pick the backend you want to validate. Megatron supports the widest parallelism
(including CP); AutoModel uses Expert Parallel on the DTensor backend.

```sh
# Megatron (EP16 TP2 CP2)
uv run examples/run_grpo.py \
  --config examples/configs/recipes/llm/grpo-qwen3.5-35ba3b-2n8g-megatron-ep16tp2cp2.yaml

# AutoModel (EP16)
uv run examples/run_grpo.py \
  --config examples/configs/recipes/llm/grpo-qwen3.5-35ba3b-2n8g-automodel-ep16.yaml
```

For long-reasoning tasks, override the generation length explicitly:

```sh
uv run examples/run_grpo.py \
  --config examples/configs/recipes/llm/grpo-qwen3.5-35ba3b-2n8g-megatron-ep16tp2cp2.yaml \
  policy.max_total_sequence_length=9216 \
  policy.generation.max_new_tokens=8192 \
  policy.generation.vllm_cfg.max_model_len=9216
```

### DAPO-style GRPO (long reasoning, ready out of the box)

The 4n8g DAPO recipe already sets `max_new_tokens: 8192` and
`max_total_sequence_length: 9216`, so it is a good starting point for long-reasoning
runs. See the [DAPO guide](../../dapo.md) for algorithm details.

```sh
uv run examples/run_grpo.py \
  --config examples/configs/recipes/llm/grpo-qwen3.5-35ba3b-dapo-4n8g-automodel.yaml
```

### Large MoE (397B-A17B)

The 397B-A17B Megatron recipe targets 32 nodes (256 GPUs) with TP8 PP8 EP32 and
`max_new_tokens: 8192`.

```sh
uv run examples/run_grpo.py \
  --config examples/configs/recipes/llm/grpo-qwen3.5-397ba17b-32n8g-megatron.v2.yaml
```

### VLM (Geo3K)

The VLM recipes target the Geo3K task. They train the vision tower per their
`freeze_config` (see [Model Quirks](../../../model-quirks.md)).

```sh
uv run examples/run_grpo.py \
  --config examples/configs/recipes/vlm/vlm_grpo-qwen3.5-35ba3b-geo3k-2n8g-megatron-ep16.yaml
```

## Performance: flash-linear-attention

Qwen3.5 relies on `flash-linear-attention` (FLA) and `causal-conv1d` kernels for
full speed. There are two distinct cases:

- **Performance fallback (AutoModel / DTensor).** Several `nemo-automodel` kernels
  dispatch to FLA when it is importable and fall back to slower PyTorch
  implementations otherwise. Without FLA, Qwen3.5 (dense or MoE) trains roughly **2x
  slower** on the AutoModel path, with no error. The `-megatron` recipes use MCore
  kernels directly and are not affected. See
  [#2722](https://github.com/NVIDIA-NeMo/RL/issues/2722) and
  [#2324](https://github.com/NVIDIA-NeMo/RL/issues/2324).
- **Hard requirement (AutoModel Qwen3.5 MoE + Context Parallel).** When
  `context_parallel_size > 1` for a Qwen3.5 MoE model on the AutoModel backend,
  NeMo RL requires FLA and raises `ImportError` if it is missing (see the `import
  fla` guard in `nemo_rl/models/automodel/setup.py`). CP for Qwen3.5 MoE on
  AutoModel also requires the TE backend; **dense Qwen3.5 does not support CP on
  AutoModel** — set `cp_size = 1`.

> [!NOTE]
> The container does not yet install FLA in the AutoModel worker venv by default.
> The root-cause fix is upstream [Automodel#1894](https://github.com/NVIDIA-NeMo/Automodel/pull/1894),
> which moves FLA out of the dev-only group so `nemo-automodel[moe]` pulls it in;
> tracked on the NeMo RL side by [#2324](https://github.com/NVIDIA-NeMo/RL/issues/2324).
> Until that bump lands, install it in the AutoModel venv:
>
> ```sh
> pip install flash-linear-attention
> ```
