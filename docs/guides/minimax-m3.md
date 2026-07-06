# MiniMax-M3 Support

This guide summarizes the current MiniMax-M3 support in NeMo-RL, including the
validated scope, the reference GRPO recipes, and known limitations.

> [!IMPORTANT]
> **Status: Functional Ready.** MiniMax-M3 is runnable in NeMo-RL, and short
> GRPO training runs have been validated with a BF16 MiniMax-M3 checkpoint.
> Long-run convergence has not been validated yet, so treat this as an
> early-access integration.

## Support Status

| Model                | Training backend    | Training parallelism                              | Inference backend | Precision                                       | Status           |
| -------------------- | ------------------- | ------------------------------------------------- | ----------------- | ----------------------------------------------- | ---------------- |
| MiniMaxAI/MiniMax-M3 | AutoModel (DTensor) | EP, PP and CP  | vLLM              | BF16 training weights with BF16 vLLM generation | Functional Ready |

Validated scope:

- **Training backend**: [NeMo AutoModel](https://github.com/NVIDIA-NeMo/Automodel).
- **Training parallelism**: Expert Parallel (EP), Pipeline Parallel (PP) and Context Parallel (CP).
- **Inference backend**: [vLLM](https://github.com/vllm-project/vllm).
- **Precision**: BF16 training weights and BF16 vLLM generation.

## How to Run

### 1. Build the Environment

MiniMax-M3 support has landed on the main branches of both AutoModel and vLLM.
Clone those sources into the `3rdparty` paths used by NeMo-RL's editable
installs.

Sources:

- AutoModel: [main branch](https://github.com/NVIDIA-NeMo/Automodel/tree/main) — MiniMax-M3 support is merged.
- vLLM: [main branch](https://github.com/vllm-project/vllm) — MiniMax-M3 support ([PR #45381](https://github.com/vllm-project/vllm/pull/45381)) is merged.

From the NeMo-RL repository root, run:

```bash
mkdir -p 3rdparty/Automodel-workspace 3rdparty/vLLM-workspace

git clone https://github.com/NVIDIA-NeMo/Automodel.git \
  3rdparty/Automodel-workspace/Automodel

git clone https://github.com/vllm-project/vllm.git \
  3rdparty/vLLM-workspace/vllm
```

Published NeMo-RL containers do not yet include the full MiniMax-M3 runtime
environment. Force a rebuild of the per-worker `uv` virtual environments at
launch time so that Ray workers pick up the local AutoModel and vLLM sources:

```bash
export NRL_FORCE_REBUILD_VENVS=true
```

### 2. Choose a Reference Recipe

Two reference GRPO recipes are provided. Both run non-colocated vLLM generation
(`generation.colocated.enabled: false`) on the DAPO Math datasets
(`DAPOMath17K` for training, `DAPOMathAIME2024` for validation).

**EP + PP — `exp/grpo-minimax-m3-32n8g-non-colocated-fused-adam-pp.yaml`**

- AutoModel (DTensor) training with `expert_parallel_size: 32` and
  `pipeline_parallel_size: 4`.
- Fused Adam optimizer and a 4096-token maximum sequence length.

**EP + CP — `exp/grpo-m3-32n8g-non-colocated-fused-adam-cp8.yaml`**

- AutoModel (DTensor) training with `expert_parallel_size: 128` and
  `context_parallel_size: 8`.
- Fused Adam optimizer and an 8192-token maximum sequence length.

### 3. Launch

MiniMax-M3 uses the standard GRPO entrypoint. Pick one of the reference recipes
above:

```bash
export NRL_FORCE_REBUILD_VENVS=true

# EP + PP
uv run examples/run_grpo.py \
  --config exp/grpo-minimax-m3-32n8g-non-colocated-fused-adam-pp.yaml

# EP + CP
uv run examples/run_grpo.py \
  --config exp/grpo-m3-32n8g-non-colocated-fused-adam-cp8.yaml
```

### Reference Training Curves

The following curve was produced with the reference recipes above, covering
both the EP + PP and EP + CP configurations:

![MiniMax-M3 GRPO training curve (EP + PP and EP + CP)](../assets/minimax_m3_grpo_curve.png)

## Known Issues

- **Long-run validation**: Current validation covers short training runs only;
  long-run convergence has not been established.

## What's Next

- Validate long-run MiniMax-M3 training.
