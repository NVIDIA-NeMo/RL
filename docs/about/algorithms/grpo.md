# GRPO

We provide a reference GRPO configuration for math benchmarks using the [OpenInstructMath2](https://huggingface.co/datasets/nvidia/OpenMathInstruct-2) dataset.

You can read about the details of the GRPO implementation [here](../../guides/grpo.md).

## GRPO Single Node

To run GRPO on a single GPU for `Qwen/Qwen2.5-1.5B`:

```sh
# Run the GRPO math example using a 1B parameter model
uv run python examples/run_grpo_math.py
```

By default, this uses the configuration in `examples/configs/grpo_math_1B.yaml`. You can customize parameters with command-line overrides. For example, to run on 8 GPUs:

```sh
# Run the GRPO math example using a 1B parameter model using 8 GPUs
uv run python examples/run_grpo_math.py \
  cluster.gpus_per_node=8
```

You can override any of the parameters listed in the YAML configuration file. For example:

```sh
uv run python examples/run_grpo_math.py \
  policy.model_name="meta-llama/Llama-3.2-1B-Instruct" \
  checkpointing.checkpoint_dir="results/llama1b_math" \
  logger.wandb_enabled=True \
  logger.wandb.name="grpo-llama1b_math" \
  logger.num_val_samples_to_print=10
```

The default configuration uses the DTensor training backend. We also provide a config `examples/configs/grpo_math_1B_megatron.yaml` which is set up to use the Megatron backend out of the box.

To train using this config on a single GPU:

```sh
# Run a GRPO math example on 1 GPU using the Megatron backend
uv run python examples/run_grpo_math.py \
  --config examples/configs/grpo_math_1B_megatron.yaml
```

For additional details on supported backends and how to configure the training backend to suit your setup, refer to the [Training Backends documentation](../../design-docs/training-backends.md).

## GRPO Multi-node

```sh
# Run from the root of NeMo RL repo
NUM_ACTOR_NODES=2

# grpo_math_8b uses Llama-3.1-8B-Instruct model
COMMAND="uv run ./examples/run_grpo_math.py --config examples/configs/grpo_math_8B.yaml cluster.num_nodes=2 checkpointing.checkpoint_dir='results/llama8b_2nodes' logger.wandb_enabled=True logger.wandb.name='grpo-llama8b_math'" \
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

The required `CONTAINER` can be built by following the instructions in the [Docker documentation](../../docker.md).

## GRPO Qwen2.5-32B

This section outlines how to run GRPO for Qwen2.5-32B with a 16k sequence length.

```sh
# Run from the root of NeMo RL repo
NUM_ACTOR_NODES=32

# Download Qwen before the job starts to avoid spending time downloading during the training loop
HF_HOME=/path/to/hf_home huggingface-cli download Qwen/Qwen2.5-32B

# Ensure HF_HOME is included in your MOUNTS
HF_HOME=/path/to/hf_home \
COMMAND="uv run ./examples/run_grpo_math.py --config examples/configs/grpo_math_8B.yaml policy.model_name='Qwen/Qwen2.5-32B' policy.generation.vllm_cfg.tensor_parallel_size=4 policy.max_total_sequence_length=16384 cluster.num_nodes=${NUM_ACTOR_NODES} policy.dtensor_cfg.enabled=True policy.dtensor_cfg.tensor_parallel_size=8 policy.dtensor_cfg.sequence_parallel=True policy.dtensor_cfg.activation_checkpointing=True checkpointing.checkpoint_dir='results/qwen2.5-32b' logger.wandb_enabled=True logger.wandb.name='qwen2.5-32b'" \
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

## GRPO Multi-Turn

We also support multi-turn generation and training (tool use, games, etc.). Reference example for training to play a Sliding Puzzle Game:

```sh
uv run python examples/run_grpo_sliding_puzzle.py
```

