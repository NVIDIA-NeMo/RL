# RM

We provide a sample RM experiment that uses the [HelpSteer3 dataset](https://huggingface.co/datasets/nvidia/HelpSteer3) for preference-based training.

## RM Single Node

The default RM experiment is configured to run on a single GPU. To launch the experiment:

```sh
uv run python examples/run_rm.py
```

This trains a RM based on `meta-llama/Llama-3.2-1B-Instruct` on 1 GPU.

If you have access to more GPUs, you can update the experiment accordingly. To run on 8 GPUs, we update the cluster configuration:

```sh
uv run python examples/run_rm.py cluster.gpus_per_node=8
```

Refer to the [RM documentation](../../guides/rm.md) for more information.

## RM Multi-node

For distributed RM training across multiple nodes, modify the following script for your use case:

```sh
# Run from the root of NeMo RL repo
## number of nodes to use for your job
NUM_ACTOR_NODES=2

COMMAND="uv run ./examples/run_rm.py --config examples/configs/rm.yaml cluster.num_nodes=2 cluster.gpus_per_node=8 checkpointing.checkpoint_dir='results/rm_llama1b_2nodes' logger.wandb_enabled=True logger.wandb.name='rm-llama1b-2nodes'" \
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
