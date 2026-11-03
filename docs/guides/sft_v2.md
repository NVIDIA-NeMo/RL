# SFT v2 with an Energon CLEVR dataset

This guide runs Qwen2.5-VL-3B-Instruct supervised fine-tuning with the Megatron backend on two GPUs. It downloads the public CLEVR-CoGenT data, prepares the indexed Energon dataset expected by the recipe, and launches `examples/run_sft_v2.py`. Run the commands from the repository root. For Slurm launch details, see [Cluster Setup](../cluster.md).

## 1. Launch an SFT v2 run

The [SFT v2 runner](../../examples/run_sft_v2.py) launches an experiment locally or through Slurm. As with [SFT](sft.md#launch-an-sft-run), launch it with `uv` and supply a YAML config plus any command line overrides:

```bash
uv run examples/run_sft_v2.py --config <PATH TO YAML CONFIG> <OVERRIDES>
```

The [CLEVR recipe](../../examples/configs/recipes/vlm/vlm_sft-qwen2.5-vl-3b-instruct-clevr-1n2g-megatrontp1-energon.v1.yaml) is the example config used below. Prepare the dataset first, then set `data.train.path` to its output directory. The recipe requests two GPUs.

## 2. Prepare CLEVR for Energon

The [preparation script](../../nemo_rl/data/energon/scripts/prepare_energon_dataset.py) downloads `MMInstruction/Clevr_CoGenT_TrainA_70K_Complex`, converts each example into a conversation with its image, writes WebDataset shards, and builds Energon's `.nv-meta` index. The recipe uses the `train` split and disables validation, so `train` is sufficient for this run.

```bash
uv run --locked --extra mcore \
  nemo_rl/data/energon/scripts/prepare_energon_dataset.py \
  --splits train \
  --output-dir /path/to/data/path
```

The output directory must be absent or empty before preparation. If the dataset has already been prepared, check for `/path/to/data/path/.nv-meta/dataset.yaml` and `/path/to/data/path/.nv-meta/split.yaml` and reuse it. Use this same directory as `data.train.path` when training.

For a small data and training smoke test, prepare a separate directory with `--max-samples 16 --max-samples-per-shard 16 --num-workers 1 --download-workers 1`. Do not point the full recipe at that small dataset without also changing `data.train.virtual_epoch_length`.

## 3. Create a config

The [Qwen CLEVR Energon recipe](../../examples/configs/recipes/vlm/vlm_sft-qwen2.5-vl-3b-instruct-clevr-1n2g-megatrontp1-energon.v1.yaml) inherits `examples/configs/sft_vlm_3B.yaml` and `examples/configs/sft.yaml`. It selects Qwen2.5-VL-3B-Instruct, two GPUs, Megatron tensor parallel size 1, the Energon `generic_conversation` cooker and `generic_sft` task encoder, and the prepared `train` split. The model and tokenizer are fetched through Hugging Face on first use; make sure the compute environment can reach or already has the model cache.

To make your own config, copy the recipe within the same directory so its relative `defaults` path still resolves:

```bash
cp examples/configs/recipes/vlm/vlm_sft-qwen2.5-vl-3b-instruct-clevr-1n2g-megatrontp1-energon.v1.yaml \
  examples/configs/recipes/vlm/my-clevr-sft-v2.yaml
```

Edit `my-clevr-sft-v2.yaml` to point at your prepared data and choose the run length and outputs. These are the key fields in the recipe:

```yaml
sft:
  max_num_steps: 1000
  val_period: 0
  val_at_start: false
data:
  backend: energon
  train:
    path: /path/to/data/path
    split: train
    virtual_epoch_length: 4375
  validation: null
logger:
  log_dir: logs/sft-qwen2.5-vl-3b-clevr-energon-v2
checkpointing:
  checkpoint_dir: results/sft_Qwen/Qwen2.5-VL-3B-Instruct_clevr_energon_v2
```

Keep the rest of the copied recipe, including its `data.energon`, `data_plane`, and policy settings. `virtual_epoch_length` is the number of batches in a virtual epoch; change it when changing dataset size or batch size. As in [SFT configuration](sft.md#example-configuration-file), you can edit YAML fields or pass `key=value` overrides on the command line.

## 4. Run SFT v2

```bash
uv run examples/run_sft_v2.py \
  --config examples/configs/recipes/vlm/vlm_sft-qwen2.5-vl-3b-instruct-clevr-1n2g-megatrontp1-energon.v1.yaml \
  data.train.path=/path/to/data/path
```

The recipe runs 1,000 steps with global batch size 16 and no validation. It enables Weights & Biases; log in first or add `logger.wandb_enabled=false`. By default, logs go to `logs/sft-qwen2.5-vl-3b-clevr-energon-v2` and checkpoints to `results/sft_Qwen/Qwen2.5-VL-3B-Instruct_clevr_energon_v2` relative to the repository root. Set `checkpointing.checkpoint_dir` to a persistent location if needed.

If you copied the config, replace the `--config` path in the command with `examples/configs/recipes/vlm/my-clevr-sft-v2.yaml`.

To check the full loading and training path before a longer run, use the prepared dataset and run three steps:

```bash
uv run examples/run_sft_v2.py \
  --config examples/configs/recipes/vlm/vlm_sft-qwen2.5-vl-3b-instruct-clevr-1n2g-megatrontp1-energon.v1.yaml \
  data.train.path=/path/to/data/path \
  sft.max_num_steps=3 \
  logger.wandb_enabled=false \
  checkpointing.enabled=false
```

For a 16-sample smoke dataset, also set `data.train.path=/path/to/smoke/clevr`, `data.train.virtual_epoch_length=6`, `policy.train_global_batch_size=2`, `data.energon.num_workers=0`, and `data.energon.shuffle_buffer_size=16`, as in the [functional smoke script](../../tests/functional/sft_v2_energon.sh).

## 5. Adapt the data

The recipe's `data.backend: energon` loads a prepared Energon directory, not a raw JSONL file or Hugging Face dataset ID. Each shard record produced by the preparation script has a `json` conversation and an image member. The conversation contains `messages` with a user turn referencing `media_index: 0`, an assistant turn holding the answer, and a `media` entry identifying the image member and its dimensions. The `generic_conversation` cooker and `generic_sft` task encoder turn those records into Qwen multimodal training samples; assistant content supplies the supervised loss.

To use another dataset, produce the same Energon record shape and `.nv-meta` index, then set `data.train.path`, `data.train.split`, and `data.train.virtual_epoch_length` to match it. See [SFT datasets](sft.md#datasets) for the other SFT data backends and customization options.
