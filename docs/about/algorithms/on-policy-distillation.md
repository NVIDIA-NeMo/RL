# On-policy Distillation

We provide an example on-policy distillation experiment using the [DeepScaler dataset](https://huggingface.co/agentica-org/DeepScaleR-1.5B-Preview).

> [!NOTE]
> Distillation currently supports the DTensor and vLLM generation backend. Megatron generation/training paths are not supported yet.

## On-policy Distillation Single Node

To run on-policy distillation on a single GPU using `Qwen/Qwen3-1.7B-Base` as the student and `Qwen/Qwen3-4B` as the teacher:

```sh
uv run python examples/run_distillation_math.py
```

Customize parameters with command-line overrides. For example:

```sh
uv run python examples/run_distillation_math.py \
  policy.model_name="Qwen/Qwen3-1.7B-Base" \
  teacher.model_name="Qwen/Qwen3-4B" \
  cluster.gpus_per_node=8
```

## On-policy Distillation Multi-node

```sh
# Run from the root of NeMo RL repo
NUM_ACTOR_NODES=2

COMMAND="uv run ./examples/run_distillation_math.py --config examples/configs/distillation_math.yaml cluster.num_nodes=2 cluster.gpus_per_node=8 checkpointing.checkpoint_dir='results/distill_2nodes' logger.wandb_enabled=True logger.wandb.name='distill-2nodes'" \
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
