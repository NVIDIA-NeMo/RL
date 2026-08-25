# OAPL

Offline Advantage-weighted Partition-function Loss (OAPL) is an offline RL-free
algorithm for training on a fixed dataset of `(prompt, trajectory, reward,
reference_logprob)` tuples, such as agent trajectories with tool calls and a
final task reward. Unlike DPO, OAPL does not require pairwise preference
labels: it regresses the policy's implicit reward directly onto the observed
reward of each trajectory.

## OAPL Single Node

The default OAPL experiment is configured to run on a single GPU. To launch the experiment, point `data.train.data_path` (and `data.validation.data_path`) in `examples/configs/oapl.yaml` at your own dataset, then run:

```sh
uv run python examples/run_oapl.py
```

This trains `Llama3.2-1B-Instruct` on 1 GPU.

Any of the OAPL parameters can be customized from the command line. For example:

```sh
uv run python examples/run_oapl.py \
  oapl.beta=0.05 \
  checkpointing.checkpoint_dir="results/llama_oapl" \
  logger.wandb_enabled=True \
  logger.wandb.name="llama-oapl"
```

Refer to `examples/configs/oapl.yaml` for a full list of parameters that can be overridden. For an in-depth explanation of the OAPL dataset format, refer to the [OAPL documentation](../../guides/oapl.md).

## OAPL Multi-node

For distributed OAPL training across multiple nodes, modify the following script for your use case:

```sh
# Run from the root of NeMo RL repo
## number of nodes to use for your job
NUM_ACTOR_NODES=2

COMMAND="uv run ./examples/run_oapl.py --config examples/configs/oapl.yaml cluster.num_nodes=2 cluster.gpus_per_node=8 oapl.val_global_batch_size=32 checkpointing.checkpoint_dir='results/oapl_llama1b_2nodes' logger.wandb_enabled=True logger.wandb.name='oapl-llama1b'" \
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
