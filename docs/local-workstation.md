# Run on Your Local Workstation

When launching examples locally with `uv`, {py:class}`init_ray() <nemo_rl.distributed.virtual_cluster.init_ray>` will first attempt to connect to an existing cluster. If none is found, it will start a local one and connect to it using all available GPU and CPU resources on your node.

To launch a job outside of a container, simply run:

```sh
uv run examples/run_grpo.py
```

In the logs, you will see that Ray has started a local cluster instance, along with details on the resources made available to it:
```
2025-03-17 13:37:45,360 INFO worker.py:1841 -- Started a local Ray instance.
...
INFO:nemo_rl.distributed.virtual_cluster:Started local cluster with: {'node:__internal_head__': 1.0, 'CPU': 24.0, 'object_store_memory': 80448493977.0, 'accelerator_type:RTX': 1.0, 'memory': 177713152615.0, 'GPU': 1.0, 'node:10.0.0.1': 1.0}
```

To have more precise control over the GPUs Ray uses locally, please use `CUDA_VISIBLE_DEVICES`:

```sh
# Use the 0th and 3rd indexed GPU (for a total of 2 GPUs)
CUDA_VISIBLE_DEVICES=0,3 uv run examples/run_grpo.py
```

We also allow multiple colocated local clusters, which are uniquely identified by the values in `CUDA_VISIBLE_DEVICES`. Concretely:

```sh
# (1) Start a fresh cluster on GPU=0
CUDA_VISIBLE_DEVICES=0 uv run examples/run_grpo.py

# (2) While (1) is running, this will start a new cluster using GPUs 1 and 2 without interfering with (1)
# Ensure that the CUDA_VISIBLE_DEVICES do not overlap already running jobs.
CUDA_VISIBLE_DEVICES=1,2 uv run examples/run_grpo.py
```

## Set `TORCH_CUDA_ARCH_LIST` for Your GPU

PyTorch builds the CUDA extensions NeMo RL relies on (for example TransformerEngine, DeepEP, and flash-attention) for the GPU architectures listed in `TORCH_CUDA_ARCH_LIST`. If the variable is unset, or does not cover your GPU, a run can fail with errors such as `RuntimeError: No CUDA GPUs are available` even though the GPUs are visible and CUDA is in `PATH`. The official container already sets this variable; on bare metal or in a custom container, set it yourself to match the GPU on the node:

```sh
# A100 node
export TORCH_CUDA_ARCH_LIST="8.0"
uv run examples/run_grpo.py
```

Find your GPU's value with `nvidia-smi --query-gpu=compute_cap --format=csv,noheader`, or look it up here:

| GPU | Compute capability | `TORCH_CUDA_ARCH_LIST` |
| --- | --- | --- |
| V100 | 7.0 | `"7.0"` |
| T4 | 7.5 | `"7.5"` |
| A100 | 8.0 | `"8.0"` |
| A10 / RTX 30-series | 8.6 | `"8.6"` |
| L4 / L40S / RTX 40-series | 8.9 | `"8.9"` |
| H100 / H200 | 9.0 | `"9.0"` |
| B100 / B200 | 10.0 | `"10.0"` |
| RTX 50-series | 12.0 | `"12.0"` |

`examples/run_grpo.py` detects the local GPU and defaults the variable to its compute capability when it is unset; an explicitly set value is never overridden. On multi-node runs, set the variable on every node, or list every architecture (for example `TORCH_CUDA_ARCH_LIST="8.0 9.0"`), or pass it through the recipe with `policy.megatron_cfg.env_vars` / `policy.automodel_cfg.env_vars`, since the driver can only detect its own node's GPU. The Megatron backend additionally fails fast at startup with a clear error when the variable is unset; the Automodel backend only warns, so on that path a missing value surfaces later as a CUDA error instead.
