# Supervised Fine-Tuning (SFT)

We provide example SFT experiments using various datasets including [SQuAD](https://rajpurkar.github.io/SQuAD-explorer/), OpenAI format datasets (with tool calling support), and custom JSONL datasets. For detailed documentation on supported datasets and configurations, see the [SFT documentation](../../guides/sft.md).

## SFT Single Node

The default SFT configuration is set to run on a single GPU. To start the experiment:

```sh
uv run python examples/run_sft.py
```

This fine-tunes the `Llama3.2-1B` model on the SQuAD dataset using a 1 GPU.

To use multiple GPUs on a single node, you can modify the cluster configuration. This adjustment will also let you potentially increase the model and batch size:

```sh
uv run python examples/run_sft.py \
  policy.model_name="meta-llama/Meta-Llama-3-8B" \
  policy.train_global_batch_size=128 \
  sft.val_global_batch_size=128 \
  cluster.gpus_per_node=8
```

Refer to `examples/configs/sft.yaml` for a full list of parameters that can be overridden.

## SFT Multi-node

```sh
# Run from the root of NeMo RL repo
NUM_ACTOR_NODES=2

COMMAND="uv run ./examples/run_sft.py --config examples/configs/sft.yaml cluster.num_nodes=2 cluster.gpus_per_node=8 checkpointing.checkpoint_dir='results/sft_llama8b_2nodes' logger.wandb_enabled=True logger.wandb.name='sft-llama8b'" \
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
