# SFT v2 with an Energon CLEVR dataset

This guide runs Qwen2.5-VL-3B-Instruct supervised fine-tuning with the Megatron backend on two GPUs. It starts with the public CLEVR-CoGenT data, prepares the indexed Energon dataset expected by the recipe, and launches `examples/run_sft_v2.py`. Run the commands from the repository root inside a NeMo RL container on a machine with at least two GPUs. For cluster allocation and container setup, see [Cluster Setup](../cluster.md) and [Installation](../about/installation.md).

## 1. Get the code and environment

Clone NeMo RL with its pinned submodules, then mount that checkout at `/opt/nemo-rl` in the container. Mount a persistent writable data directory at `/data`; both data preparation and training must see the same path. The Megatron dependency set is selected by `--extra mcore` during preparation and by the SFT v2 worker environments during training.

```bash
git clone --recursive git@github.com:NVIDIA-NeMo/RL.git nemo-rl
cd nemo-rl
# Inside the container, with this checkout at /opt/nemo-rl:
cd /opt/nemo-rl
git submodule update --init --recursive
```

On HSG, the `sft_v2_docs` rem project already has a checkout at `/home/rohitkumarj/code/RL/nemo-rl-sft_v2_docs` and uses `rl.nightly.sep30.2026.sqsh`. Its standard container mount maps the shared host data tree to `/data`. The project's `coreai_dlalgo_nemorl` account requires a four-GPU allocation; set `CUDA_VISIBLE_DEVICES=0,1` inside that allocation for this two-GPU recipe.

## 2. Prepare CLEVR for Energon

The [preparation script](../../nemo_rl/data/energon/scripts/prepare_energon_dataset.py) downloads `MMInstruction/Clevr_CoGenT_TrainA_70K_Complex`, converts each example into a conversation with its image, writes WebDataset shards, and builds Energon's `.nv-meta` index. The recipe reads the `train` split at `/data/nemorl-datasets/clevr`. Its validation dataset is disabled, so `train` is sufficient for this run.

```bash
mkdir -p /data/nemorl-datasets
uv run --locked --extra mcore \
  nemo_rl/data/energon/scripts/prepare_energon_dataset.py \
  --splits train \
  --output-dir /data/nemorl-datasets/clevr
```

The output directory must be absent or empty before preparation. If the dataset has already been prepared, check for `/data/nemorl-datasets/clevr/.nv-meta/dataset.yaml` and `/data/nemorl-datasets/clevr/.nv-meta/split.yaml` and reuse it. On HSG, the shared CLEVR dataset is already present at this path. For another location, pass `data.train.path=/your/mounted/path` to the training command.

For a small data and training smoke test, prepare a separate directory with `--max-samples 16 --max-samples-per-shard 16 --num-workers 1 --download-workers 1`. Do not point the full recipe at that small dataset without also changing `data.train.virtual_epoch_length`.

## 3. Run SFT v2

The [Qwen CLEVR Energon recipe](../../examples/configs/recipes/vlm/vlm_sft-qwen2.5-vl-3b-instruct-clevr-1n2g-megatrontp1-energon.v1.yaml) inherits `examples/configs/sft_vlm_3B.yaml` and `examples/configs/sft.yaml`. It selects Qwen2.5-VL-3B-Instruct, two GPUs, Megatron tensor parallel size 1, the Energon `generic_conversation` cooker and `generic_sft` task encoder, and the prepared `train` split. The model and tokenizer are fetched through Hugging Face on first use; make sure the compute environment can reach or already has the model cache.

```bash
uv run examples/run_sft_v2.py \
  --config examples/configs/recipes/vlm/vlm_sft-qwen2.5-vl-3b-instruct-clevr-1n2g-megatrontp1-energon.v1.yaml
```

The recipe runs 1,000 steps with global batch size 16 and no validation. It enables Weights & Biases; log in first or add `logger.wandb_enabled=false`. By default, logs go to `logs/sft-qwen2.5-vl-3b-clevr-energon-v2` and checkpoints to `results/sft_Qwen/Qwen2.5-VL-3B-Instruct_clevr_energon_v2` relative to the checkout. Put `checkpointing.checkpoint_dir` on persistent storage for a cluster run.

To check the full loading and training path before a longer run, use the prepared dataset and run three steps:

```bash
uv run examples/run_sft_v2.py \
  --config examples/configs/recipes/vlm/vlm_sft-qwen2.5-vl-3b-instruct-clevr-1n2g-megatrontp1-energon.v1.yaml \
  sft.max_num_steps=3 \
  logger.wandb_enabled=false \
  checkpointing.enabled=false
```

For a 16-sample smoke dataset, also set `data.train.path=/path/to/smoke/clevr`, `data.train.virtual_epoch_length=6`, `policy.train_global_batch_size=2`, `data.energon.num_workers=0`, and `data.energon.shuffle_buffer_size=16`, as in the [functional smoke script](../../tests/functional/sft_v2_energon.sh).

## 4. Adapt the data

The recipe's `data.backend: energon` loads a prepared Energon directory, not a raw JSONL file or Hugging Face dataset ID. Each shard record produced by the preparation script has a `json` conversation and an image member. The conversation contains `messages` with a user turn referencing `media_index: 0`, an assistant turn holding the answer, and a `media` entry identifying the image member and its dimensions. The `generic_conversation` cooker and `generic_sft` task encoder turn those records into Qwen multimodal training samples; assistant content supplies the supervised loss.

To use another dataset, produce the same Energon record shape and `.nv-meta` index, mount the directory into every training worker, and override `data.train.path`, `data.train.split`, and `data.train.virtual_epoch_length` to match it. See [SFT datasets](sft.md#datasets) for the other SFT data backends and customization options.
