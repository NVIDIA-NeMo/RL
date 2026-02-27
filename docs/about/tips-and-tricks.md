# Tips and Tricks

## Missing Submodules Error

If you forget to initialize the NeMo and Megatron submodules when cloning the NeMo-RL repository, you may run into an error like this:

```sh
ModuleNotFoundError: No module named 'megatron'
```

If you see this error, there is likely an issue with your virtual environments. To fix this, first initialize the submodules:

```sh
git submodule update --init --recursive
```

and then force a rebuild of the virtual environments by setting `NRL_FORCE_REBUILD_VENVS=true` next time you launch a run:

```sh
NRL_FORCE_REBUILD_VENVS=true uv run examples/run_grpo.py ...
```

## Memory Fragmentation

Large amounts of memory fragmentation might occur when running models without support for FlashAttention2. If OOM occurs after a few iterations of training, it may help to tweak the allocator settings to reduce memory fragmentation. To do so, specify [`max_split_size_mb`](https://docs.pytorch.org/docs/stable/notes/cuda.html#optimizing-memory-usage-with-pytorch-alloc-conf) at **either** one of the following places:

1. Launch training with:

```sh
# This will globally apply to all Ray actors
PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:64 uv run python examples/run_dpo.py ...
```

2. Make the change more permanently by adding this flag in the training configuration:

```yaml
policy:
  # ...
  dtensor_cfg:
    env_vars:
      PYTORCH_CUDA_ALLOC_CONF: "max_split_size_mb:64"
```

