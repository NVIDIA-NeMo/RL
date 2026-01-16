# DAPO

[Dual-Clip Asymmetric Policy Optimization (DAPO)](https://arxiv.org/pdf/2503.14476) extends GRPO by allowing asymmetric clipping with distinct minimum and maximum clip parameters. This provides more fine-grained control over policy updates.

DAPO is implemented through the same `ClippedPGLossFn` as GRPO, but with the ability to set different values for `ratio_clip_min` and `ratio_clip_max`. For standard GRPO/PPO, these parameters are set to the same value.

## Key Differences from GRPO

- **Asymmetric Clipping**: DAPO allows `ratio_clip_min` ≠ `ratio_clip_max`, providing asymmetric bounds on the probability ratio
- **Same Infrastructure**: Uses the same training infrastructure and configurations as GRPO

## DAPO Single Node

To run DAPO on a single GPU, use the GRPO script with asymmetric clip parameters:

```sh
# Run DAPO with asymmetric clipping
uv run python examples/run_grpo_math.py \
  policy.model_name="Qwen/Qwen2.5-1.5B" \
  grpo.ratio_clip_min=0.15 \
  grpo.ratio_clip_max=0.25 \
  checkpointing.checkpoint_dir="results/dapo_math" \
  logger.wandb_enabled=True \
  logger.wandb.name="dapo-math"
```

For multi-GPU setups:

```sh
uv run python examples/run_grpo_math.py \
  cluster.gpus_per_node=8 \
  grpo.ratio_clip_min=0.15 \
  grpo.ratio_clip_max=0.25 \
  checkpointing.checkpoint_dir="results/dapo_8gpu" \
  logger.wandb_enabled=True \
  logger.wandb.name="dapo-8gpu"
```

## DAPO Multi-node

DAPO can be run on multiple nodes using the same approach as GRPO:

```sh
# Run from the root of NeMo RL repo
NUM_ACTOR_NODES=2

COMMAND="uv run ./examples/run_grpo_math.py \
  --config examples/configs/grpo_math_8B.yaml \
  cluster.num_nodes=2 \
  grpo.ratio_clip_min=0.15 \
  grpo.ratio_clip_max=0.25 \
  checkpointing.checkpoint_dir='results/dapo_2nodes' \
  logger.wandb_enabled=True \
  logger.wandb.name='dapo-multinode'" \
CONTAINER=YOUR_CONTAINER \
MOUNTS="$PWD:$PWD" \
sbatch \
    --nodes=${NUM_ACTOR_NODES} \
    --account=YOUR_ACCOUNT \
    --job-name=YOUR_JOBNAME \
    --partition=YOUR_PARTITION \
    --time=4:0:0 \
    --gres=gpu:8 \
    ray.sub
```

> [!NOTE]
> For GB200 systems with 4 GPUs per node, use `--gres=gpu:4` instead.

## Configuration

DAPO uses the same configuration structure as GRPO. The key parameters are:

```yaml
grpo:
  ratio_clip_min: 0.15  # Minimum clip value (can be different from max)
  ratio_clip_max: 0.25  # Maximum clip value (can be different from min)
  # ... other GRPO parameters ...
```

For more details on other configuration options, refer to the [GRPO documentation](grpo.md).

## Additional Resources

- [DAPO Paper](https://arxiv.org/pdf/2503.14476)
- [GRPO Documentation](grpo.md)
- [Training Backends](../../design-docs/training-backends.md)
