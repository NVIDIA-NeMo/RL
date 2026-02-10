# DPO

We provide a sample DPO experiment that uses the [HelpSteer3 dataset](https://huggingface.co/datasets/nvidia/HelpSteer3) for preference-based training.

## DPO Single Node

The default DPO experiment is configured to run on a single GPU. To launch the experiment:

```sh
uv run python examples/run_dpo.py
```

This trains `Llama3.2-1B-Instruct` on 1 GPU.

If you have access to more GPUs, you can update the experiment accordingly. To run on 8 GPUs, we update the cluster configuration and switch to an 8B Llama3.1 Instruct model:

```sh
uv run python examples/run_dpo.py \
  policy.model_name="meta-llama/Llama-3.1-8B-Instruct" \
  policy.train_global_batch_size=256 \
  cluster.gpus_per_node=8
```

Any of the DPO parameters can be customized from the command line. For example:

```sh
uv run python examples/run_dpo.py \
  dpo.sft_loss_weight=0.1 \
  dpo.preference_average_log_probs=True \
  checkpointing.checkpoint_dir="results/llama_dpo_sft" \
  logger.wandb_enabled=True \
  logger.wandb.name="llama-dpo-sft"
```

Refer to `examples/configs/dpo.yaml` for a full list of parameters that can be overridden. For an in-depth explanation of how to add your own DPO dataset, refer to the [DPO documentation](../../guides/dpo.md).

## DPO Multi-node

For distributed DPO training across multiple nodes, modify the following script for your use case:

```sh
# Run from the root of NeMo RL repo
## number of nodes to use for your job
NUM_ACTOR_NODES=2

COMMAND="uv run ./examples/run_dpo.py --config examples/configs/dpo.yaml cluster.num_nodes=2 cluster.gpus_per_node=8 dpo.val_global_batch_size=32 checkpointing.checkpoint_dir='results/dpo_llama81_2nodes' logger.wandb_enabled=True logger.wandb.name='dpo-llama1b'" \
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
