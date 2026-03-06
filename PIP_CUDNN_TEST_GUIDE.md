# pip-only cuDNN 9.19 Test Guide

Branch: `sj/pip-cudnn-test`

## Overview

This branch replaces the tarball-based cuDNN installation with pip-installed `nvidia-cudnn-cu12==9.19.0.56`.
The goal is to enable **FusedAttention** (requires cuDNN ≥ 9.18) without downloading/extracting cuDNN tarballs at runtime.

### Core changes (3 files)

| File | Change |
|------|--------|
| `pyproject.toml` | `nvidia-cudnn-cu12==9.19.0.56` in `override-dependencies` (overrides torch's 9.10 pin) |
| `nemo_rl/models/policy/utils.py` | Auto-detect pip cuDNN lib path and prepend to `LD_LIBRARY_PATH` for Ray workers |
| `cluster_config.sh` | Fix H100 container path |

## Setup on a New Server

```bash
git fetch origin
git checkout sj/pip-cudnn-test
git submodule update --init --recursive
```

## Known Issue: System cuDNN Override

The container ships cuDNN 9.10.1 via apt (`/usr/lib/x86_64-linux-gnu/libcudnn.so.9`).
Even with pip cuDNN 9.19 installed, the system library loads first unless `LD_LIBRARY_PATH` is configured.

**Two things are needed before running:**

### 1. Set LD_LIBRARY_PATH (head process)

The `utils.py` change handles Ray workers automatically, but the **head process** (where `uv run` executes) needs manual setup:

```bash
cd /path/to/RL-pip-cudnn-test  # or wherever this branch is checked out

export PIP_CUDNN_LIB=$(uv run python3 -c "import nvidia.cudnn, pathlib; print(pathlib.Path(list(nvidia.cudnn.__path__)[0]) / 'lib')")
export LD_LIBRARY_PATH="${PIP_CUDNN_LIB}:${LD_LIBRARY_PATH}"
```

### 2. Create symlink for unversioned libcudnn.so

The pip package only provides `libcudnn.so.9` (versioned), not `libcudnn.so` (unversioned).
Some code paths use `dlopen("libcudnn.so")` which falls through to the system 9.10 library.

```bash
ln -sf libcudnn.so.9 ${PIP_CUDNN_LIB}/libcudnn.so
```

## Verification Commands

Run these **after** the LD_LIBRARY_PATH and symlink setup above:

```bash
# Check installed pip package version (should be 9.19.0.56)
uv run python3 -c "import importlib.metadata; print('nvidia-cudnn-cu12:', importlib.metadata.version('nvidia-cudnn-cu12'))"

# Check runtime cuDNN via versioned SONAME (should be 9.19.0)
uv run python3 -c "
import ctypes
cudnn = ctypes.CDLL('libcudnn.so.9')
cudnn.cudnnGetVersion.restype = ctypes.c_size_t
v = cudnn.cudnnGetVersion()
print(f'libcudnn.so.9: {v // 10000}.{(v % 10000) // 100}.{v % 100}')
"

# Check runtime cuDNN via unversioned name (should also be 9.19.0 after symlink)
uv run python3 -c "
import ctypes
cudnn = ctypes.CDLL('libcudnn.so')
cudnn.cudnnGetVersion.restype = ctypes.c_size_t
v = cudnn.cudnnGetVersion()
print(f'libcudnn.so: {v // 10000}.{(v % 10000) // 100}.{v % 100}')
"

# Check PyTorch's loaded cuDNN version
uv run python3 -c "import torch; print(f'torch cudnn: {torch.backends.cudnn.version()}')"
```

**All four should report 9.19.x.** If any shows 9.10.x, the system cuDNN is being loaded instead.

## Running Experiments

### GPT-OSS 20B (2 nodes, interactive)

```bash
# Submit interactive job
./exp_gptoss20b_experiments.sh interactive_2node

# After attaching (bash JOBID-attach.sh), run:
export PIP_CUDNN_LIB=$(uv run python3 -c "import nvidia.cudnn, pathlib; print(pathlib.Path(list(nvidia.cudnn.__path__)[0]) / 'lib')")
export LD_LIBRARY_PATH="${PIP_CUDNN_LIB}:${LD_LIBRARY_PATH}"
ln -sf libcudnn.so.9 ${PIP_CUDNN_LIB}/libcudnn.so

NVTE_DEBUG=1 CUDNN_INSTALL=0 NRL_FORCE_REBUILD_VENVS=true uv run ./examples/run_grpo.py \
  --config examples/configs/recipes/llm/grpo-gptoss-20b-8n8g-megatron.yaml \
  cluster.num_nodes=2 cluster.gpus_per_node=8 \
  policy.generation.vllm_cfg.tensor_parallel_size=4 \
  policy.megatron_cfg.tensor_model_parallel_size=2 \
  policy.megatron_cfg.expert_model_parallel_size=8 \
  policy.megatron_cfg.moe_permute_fusion=true \
  policy.sequence_packing.enabled=true \
  grpo.max_num_steps=3 checkpointing.enabled=false \
  logger.wandb_enabled=True \
  logger.wandb.project=sync-grpo-h100-gptoss-exp \
  logger.wandb.name=GPTOSS20B_pip_cudnn919 \
  2>&1 | tee ~/gpt-oss-20b_pip_cudnn919.log
```

### GPT-OSS 120B (8 nodes, batch)

```bash
# Submit batch interactive job
./exp_gptoss120b_experiments.sh interactive_8node

# After attaching (bash JOBID-attach.sh), run:
export PIP_CUDNN_LIB=$(uv run python3 -c "import nvidia.cudnn, pathlib; print(pathlib.Path(list(nvidia.cudnn.__path__)[0]) / 'lib')")
export LD_LIBRARY_PATH="${PIP_CUDNN_LIB}:${LD_LIBRARY_PATH}"
ln -sf libcudnn.so.9 ${PIP_CUDNN_LIB}/libcudnn.so

NVTE_DEBUG=1 CUDNN_INSTALL=0 NRL_FORCE_REBUILD_VENVS=true uv run ./examples/run_grpo.py \
  --config examples/configs/recipes/llm/grpo-gptoss-120b-8n8g-megatron.yaml \
  cluster.num_nodes=8 cluster.gpus_per_node=8 \
  policy.generation.vllm_cfg.tensor_parallel_size=8 \
  policy.megatron_cfg.tensor_model_parallel_size=4 \
  policy.megatron_cfg.expert_model_parallel_size=8 \
  policy.megatron_cfg.pipeline_model_parallel_size=2 \
  policy.megatron_cfg.moe_permute_fusion=true \
  policy.sequence_packing.enabled=true \
  grpo.max_num_steps=3 checkpointing.enabled=false \
  logger.wandb_enabled=True \
  logger.wandb.project=sync-grpo-h100-gptoss120b-exp \
  logger.wandb.name=GPTOSS120B_pip_cudnn919_8node \
  2>&1 | tee ~/gpt-oss-120b_pip_cudnn919_8node.log
```

### Key env vars

| Variable | Value | Purpose |
|----------|-------|---------|
| `CUDNN_INSTALL=0` | Disable tarball cuDNN download in `ray.sub` |
| `NRL_FORCE_REBUILD_VENVS=true` | Rebuild Ray worker venvs with new cuDNN |
| `NVTE_DEBUG=1` | Show TE attention backend selection in logs |

## What to Check in Logs

1. **FusedAttention enabled**: Look for `[TE] using FusedAttention` (not `UnfusedDotProductAttention`)
2. **cuDNN version**: `cudnn_version` should show `91900` or `9.19.0`
3. **No `cuDNN version < 9.18` warning**: This warning means system cuDNN 9.10 is being loaded

## Troubleshooting

- If cuDNN shows 9.10.x → `LD_LIBRARY_PATH` is not set or the symlink is missing
- If `uv sync` fails → run `git submodule update --init --recursive` first
- If Ray workers load 9.10.x → check that `utils.py` changes are present (the `_get_pip_cudnn_lib_path` function)

## Reference

- PR #1962: https://github.com/NVIDIA-NeMo/RL/pull/1962 (uses newer container without system cuDNN)
- This branch takes the alternative approach: keep existing container, override cuDNN via pip + LD_LIBRARY_PATH
