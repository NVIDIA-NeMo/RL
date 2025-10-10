# Quick Start

Use this quick start to get going with either the native PyTorch DTensor or Megatron Core training backends.

```{note}
Both training backends are independent — you can install and use either one on its own.
```

For more examples and setup details, continue to the [Prerequisites](installation.md) section.

## Quick Start Options

| Native PyTorch (DTensor) | Megatron Core |
|--------------------------|---------------|
| **Clone and create the environment** | |

```sh
git clone git@github.com:NVIDIA-NeMo/RL.git nemo-rl
cd nemo-rl
git submodule update --init --recursive
uv venv
```

```{note}
If you previously ran without checking out the submodules, you may need to rebuild virtual environments by setting `NRL_FORCE_REBUILD_VENVS=true`. See [Tips and Tricks](tips-and-tricks.md).
```

| Native PyTorch (DTensor) | Megatron Core |
|--------------------------|---------------|
| **Run GRPO (DTensor)** | **Run GRPO (Megatron)** |

```sh
# DTensor
uv run python examples/run_grpo_math.py
```

```sh
# Megatron
uv run examples/run_grpo_math.py \
  --config examples/configs/grpo_math_1B_megatron.yaml
```

