# Qwen3.5

This page collects NeMo RL guidance for Qwen3.5 LLM and VLM post-training recipes.
Use it to choose a starting recipe, set generation length safely, install the
dependencies needed for full performance, and avoid Qwen3.5-specific training and
evaluation pitfalls.

## When to Use This Page

Use this page when training or evaluating:

- `Qwen/Qwen3.5-9B-Base`
- `Qwen/Qwen3.5-35B-A3B-Base` (MoE; LLM and VLM recipes)
- `Qwen/Qwen3.5-397B-A17B` (MoE)

For family-wide Qwen guidance, see the [Qwen family hub](index.md). For the full list
of supported models, see [Model Support](../../../about/model-support.md).

## Support Status

We track model support in two stages:

| Stage | Meaning |
| --- | --- |
| **Functional Ready** | Runnable end-to-end and numerically validated with an initial training run. |
| **Long-run convergence validated** | Trains stably over a full-length run with a healthy, reproducible reward curve. |

Qwen3.5 is **Functional Ready** across the recipes listed below, on both the Megatron
(MCore) and AutoModel (DTensor) backends. Treat the shipped recipes as validated
starting points and follow the [Key Defaults](#key-defaults) before launching
reasoning workloads.

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

## Key Defaults

> [!WARNING]
> For Qwen3.5 thinking-mode math or long-reasoning runs, use
> `policy.generation.max_new_tokens >= 8192` and keep `policy.max_total_sequence_length`
> and `policy.generation.vllm_cfg.max_model_len` large enough for the same response
> budget.
>
> Smaller values such as 2048, 3072, or 4096 can truncate the reasoning trace before
> the final `\boxed{}` answer and make evaluation accuracy appear near zero **even
> when training loss and reward metrics look normal**. See
> [NVIDIA-NeMo/RL#2725](https://github.com/NVIDIA-NeMo/RL/issues/2725).

| Setting | Recommendation | Notes |
|---|---|---|
| `policy.generation.max_new_tokens` | `8192` or higher | Especially for math, long CoT, and thinking mode |
| `policy.max_total_sequence_length` | input length plus response budget | Must fit prompt plus generated tokens |
| `policy.generation.vllm_cfg.max_model_len` | at least total sequence length | Required when overriding vLLM context |
| `data.max_input_seq_length` | task dependent | Leave room for long responses |
| `grpo.overlong_filtering` | task dependent | Consider for long-reasoning tasks |

## Recipe Matrix

Recipe YAMLs are the source of truth. The values below summarize each recipe at the
time of writing; check the YAML for the authoritative settings.

| Model | Modality | Algorithm | Backend | Scale | Recipe | Notes |
|---|---|---|---|---|---|---|
| Qwen3.5-9B-Base | LLM | GRPO | Megatron | 1n8g | [`grpo-qwen3.5-9b-1n8g-megatron.yaml`](../../../../examples/configs/recipes/llm/grpo-qwen3.5-9b-1n8g-megatron.yaml) | Dense; short-context baseline (`max_total_sequence_length: 4096`, TP4) |
| Qwen3.5-35B-A3B-Base | LLM | GRPO | Megatron | 2n8g | [`grpo-qwen3.5-35ba3b-2n8g-megatron-ep16tp2cp2.yaml`](../../../../examples/configs/recipes/llm/grpo-qwen3.5-35ba3b-2n8g-megatron-ep16tp2cp2.yaml) | EP16 TP2 CP2; TP=2 resolves the ep16 OOM ([#2619](https://github.com/NVIDIA-NeMo/RL/pull/2619)). Raise generation length for reasoning tasks |
| Qwen3.5-35B-A3B-Base | LLM | GRPO | AutoModel | 2n8g | [`grpo-qwen3.5-35ba3b-2n8g-automodel-ep16.yaml`](../../../../examples/configs/recipes/llm/grpo-qwen3.5-35ba3b-2n8g-automodel-ep16.yaml) | EP16; DTensor variant. See FLA note below |
| Qwen3.5-35B-A3B-Base | LLM | DAPO-style GRPO | AutoModel | 4n8g | [`grpo-qwen3.5-35ba3b-dapo-4n8g-automodel.yaml`](../../../../examples/configs/recipes/llm/grpo-qwen3.5-35ba3b-dapo-4n8g-automodel.yaml) | EP32; `max_new_tokens: 8192`, `max_total_sequence_length: 9216` |
| Qwen3.5-397B-A17B | LLM | GRPO | Megatron | 32n8g | [`grpo-qwen3.5-397ba17b-32n8g-megatron.v2.yaml`](../../../../examples/configs/recipes/llm/grpo-qwen3.5-397ba17b-32n8g-megatron.v2.yaml) | Large MoE; TP8 PP8 EP32, `max_new_tokens: 8192` |
| Qwen3.5-35B-A3B-Base | VLM | GRPO | Megatron | 2n8g | [`vlm_grpo-qwen3.5-35ba3b-geo3k-2n8g-megatron-ep16.yaml`](../../../../examples/configs/recipes/vlm/vlm_grpo-qwen3.5-35ba3b-geo3k-2n8g-megatron-ep16.yaml) | Geo3K VLM; response budget `2048` (raise for reasoning) |
| Qwen3.5-35B-A3B-Base | VLM | GRPO | AutoModel | 2n8g | [`vlm_grpo-qwen3.5-35ba3b-geo3k-2n8g-automodel-ep16.yaml`](../../../../examples/configs/recipes/vlm/vlm_grpo-qwen3.5-35ba3b-geo3k-2n8g-automodel-ep16.yaml) | Geo3K VLM; DTensor variant; response budget `2048` |

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

These recipes default to `max_total_sequence_length: 4096`. For long-reasoning
tasks, override the generation length explicitly:

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

The VLM recipes target the Geo3K task with a `2048` response budget. Raise the
budget for longer multimodal reasoning, and note that VLM recipes train the vision
tower per their `freeze_config` (see [Model Quirks](../../../model-quirks.md)).

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

## Evaluation

Use the general [evaluation guide](../../eval.md) for command structure. For
thinking-mode evaluation, match the generation settings to the training prompt
format and keep the generation budget large (the same `max_new_tokens` /
`max_model_len` concern from [Key Defaults](#key-defaults) applies to eval):

```sh
uv run python examples/run_eval.py \
    generation.model_name=/path/to/qwen3.5/checkpoint/hf \
    generation.temperature=0.6 \
    generation.top_p=0.95 \
    generation.vllm_cfg.max_model_len=9216 \
    generation.max_new_tokens=8192 \
    tokenizer.chat_template_kwargs.enable_thinking=true
```

## Known Issues

### Truncated thinking traces (near-zero eval accuracy)

Symptoms:

- Validation accuracy is near zero.
- Training reward or loss may look normal.
- Samples contain reasoning but no final answer.

Fix:

- Increase `policy.generation.max_new_tokens` (>= 8192 for reasoning tasks).
- Increase `policy.max_total_sequence_length`.
- Increase `policy.generation.vllm_cfg.max_model_len` if the vLLM context is lower.
- Re-run evaluation with matching generation length.

See [#2725](https://github.com/NVIDIA-NeMo/RL/issues/2725).

### Token id out of vocabulary on policy drift

For some Qwen-based models, a `ValueError: Token id ... is out of vocabulary` error
may occur when the policy drifts from its initial distribution. Setting
`policy.generation.top_p` to `0.9999` is a recommended workaround. See
[#237](https://github.com/NVIDIA-NeMo/RL/issues/237).

### Slow AutoModel training (missing FLA)

If AutoModel-path Qwen3.5 training is ~2x slower than expected, `flash-linear-attention`
is likely not installed. See [Performance: flash-linear-attention](#performance-flash-linear-attention).
