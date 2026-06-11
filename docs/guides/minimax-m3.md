# MiniMax-M3 Support

This guide summarizes the current MiniMax-M3 support in NeMo-RL, including the
validated scope, a reference GRPO recipe, and known limitations.

:::{warning}
**Status: Functional Ready.** MiniMax-M3 is runnable in NeMo-RL, and short GRPO
training runs have been validated with a BF16 MiniMax-M3 checkpoint. Long-run
convergence has not been validated yet, so treat this as an early-access
integration.
:::

## Support Status


| Model                      | Training backend    | Training parallelism      | Inference backend | Precision                                       | Status           |
| -------------------------- | ------------------- | ------------------------- | ----------------- | ----------------------------------------------- | ---------------- |
| MiniMax-M3 BF16 checkpoint | AutoModel (DTensor) | Expert Parallel (EP) only | vLLM              | BF16 training weights with BF16 vLLM generation | Functional Ready |


Validated scope:

- **Training backend**: [NeMo AutoModel](https://github.com/NVIDIA-NeMo/Automodel).
- **Training parallelism**: Expert Parallel (EP) only.
- **Inference backend**: [vLLM](https://github.com/vllm-project/vllm).
- **Precision**: BF16 training weights and BF16 vLLM generation.

## How to Run

### 1. Build the Environment

MiniMax-M3 requires matching development branches of AutoModel and vLLM.
NeMo-RL resolves both packages from local editable sources, so place the
checkouts at the expected `3rdparty` paths:

```bash
mkdir -p 3rdparty/Automodel-workspace 3rdparty/vLLM-workspace

git clone --branch athitten/minimax_m3 --single-branch \
  https://github.com/athitten/Automodel-private.git \
  3rdparty/Automodel-workspace/Automodel

git clone --branch m3_release --single-branch \
  https://github.com/vllm-project/vllm.git \
  3rdparty/vLLM-workspace/vllm
```

Branches:

- AutoModel: [https://github.com/athitten/Automodel-private/tree/athitten/minimax_m3](https://github.com/athitten/Automodel-private/tree/athitten/minimax_m3)
- vLLM: [https://github.com/vllm-project/vllm/tree/m3_release](https://github.com/vllm-project/vllm/tree/m3_release)

Published NeMo-RL containers do not yet include the full MiniMax-M3 runtime
environment. Force a rebuild of the per-worker `uv` virtual environments at
launch time so Ray workers pick up the local AutoModel and vLLM sources:

```bash
export NRL_FORCE_REBUILD_VENVS=true
```

### 2. Use the Reference Recipe

The reference recipe is:

```text
exp/grpo-m3-32n8g-non-colocated-adamw.yaml
```

Before launching, update `policy.model_name` to the path of your BF16
MiniMax-M3 checkpoint.

Key settings:

- AutoModel (DTensor) training with `expert_parallel_size: 128`.
- Non-colocated vLLM generation
  (`generation.colocated.enabled: false`).
- DAPO Math datasets (`DAPOMath17K` train / `DAPOMathAIME2024` validation).

### 3. Launch

MiniMax-M3 uses the standard GRPO entrypoint:

```bash
export NRL_FORCE_REBUILD_VENVS=true

uv run examples/run_grpo.py \
  --config exp/grpo-m3-32n8g-non-colocated-adamw.yaml
```

### Reference Training Curve

The following curve was produced with the reference recipe above:

![MiniMax-M3 GRPO training curve](../assets/minimax_m3_grpo_curve.png)

## Known Issues

- **Sequence length**: The validated configuration uses EP=128 and a 2k maximum
  sequence length. Longer sequences may OOM and will likely require additional
  parallelism such as Context Parallel (CP) or Pipeline Parallel (PP).
- **Long-run validation**: Current validation covers short training runs only.
  Long-run convergence has not been established.
- **Additional parallelism**: CP, PP, TP, and sequence packing are not part of
  the validated MiniMax-M3 training scope yet.

## What's Next

- Validate long-run MiniMax-M3 training.
- Add and validate more training parallelism, especially CP and PP, to support
  longer contexts.
