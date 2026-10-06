# Nemotron 3.5 Super VL

This guide explains how to post-train the SuperVL3p5 vision-language checkpoint (120B total parameters, 12B active) with GRPO using NeMo RL, Megatron-Core, and vLLM. It covers the synchronous V1 CLEVR-CoGenT and MMPR-Tiny image recipes.

## Multimodal payload deduplication

The recipes enable `grpo.deduplicate_multimodal_data` and leave payload-size diagnostics disabled. This shares immutable media across logical GRPO generations. Deduplication uses the vLLM backend with `data_plane.enabled: false`. See [the Nano Omni guide](nemotron-3-nano-omni.md#multimodal-payload-deduplication).

## Megatron backend

The setup uses the HF checkpoint's tokenizer, processor, and `chat_template.jinja`, together with the `super-v3.5-posttraining` branch's pinned Megatron Bridge and nested Megatron-LM submodules. NeMo RL must be mounted recursively and visible on every worker. Provision compatible worker environments and build the pinned MCore dataset helpers before launching a driver. Confirm that Python imports Bridge and MCore from this checkout.

The commands below run inside the head container of an existing multi-node Ray allocation with four GPUs per node. They use the existing mounts; they do not start an allocation or Ray cluster.

### Checkpoint compatibility

Set `MM_TRAINER_MODEL_PATH` to the local SuperVL3p5 HF checkpoint. Reuse a Megatron conversion cache only with the same input weights, model integration, and parallel layout. A fresh cache can require HF-to-Megatron conversion before training. Do not reuse a Nano checkpoint or a legacy model-layout cache for SuperVL3p5.

### Maintained recipes

| Workload | Recipe | Topology |
|---|---|---|
| CLEVR-CoGenT | [16-node CLEVR recipe](../../../../examples/configs/recipes/vlm/vlm_grpo-supervl3p5-clevr-16n4g-megatron-tp8ep8.v1.yaml) | 16 × 4 GPUs; policy TP8 / EP8 / CP1; colocated vLLM TP4 / EP4 |
| MMPR-Tiny | [32-node MMPR-Tiny recipe](../../../../examples/configs/recipes/vlm/vlm_grpo-supervl3p5-mmpr-32n4g-megatron-tp8ep16.v1.yaml) | 32 × 4 GPUs; policy TP8 / EP16 / CP1; colocated vLLM TP4 / EP4 |

The image recipes inherit the corresponding Nano Omni task recipes with more nodes to fit SuperVL3p5. They retain R3 disabled, FP32 LM heads, frozen vision/audio modules, optimizer offload during logprob calculation, and synchronous checkpoint writes. `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:False` is set in their policy and generation worker environments for CUDA IPC compatibility.

### Common launch environment

Replace the placeholder paths with shared paths already visible inside every worker container. Keep source data read-only and caches/results writable. Set shared cache and worker environment values before Ray starts. Choose a new W&B ID and output directory for each independent experiment; provide credentials through the environment or existing login.

```bash
export RL_DIR=/opt/nemo-rl
export DRIVER_PYTHON=/opt/nemo_rl_venv/bin/python
export MM_TRAINER_MODEL_PATH=/path/to/supervl3p5/hf
export SUPER_CACHE_DIR=/path/to/shared-cache/supervl3p5
export MM_TRAINER_WANDB_ID=supervl3p5-clevr-prod-unique
export SUPER_WANDB_PROJECT=grpo-supervl3p5
export MM_TRAINER_WANDB_NAME=$MM_TRAINER_WANDB_ID
export MM_TRAINER_RESULTS_DIR=/path/to/experiments/$MM_TRAINER_WANDB_ID
export HF_HOME=$SUPER_CACHE_DIR/huggingface
export HF_DATASETS_CACHE=$HF_HOME/datasets
export UV_CACHE_DIR=$SUPER_CACHE_DIR/uv
export BRIDGE_DIR=$RL_DIR/3rdparty/Megatron-Bridge-workspace/Megatron-Bridge
export PYTHONPATH=$RL_DIR:$BRIDGE_DIR/src:$BRIDGE_DIR/3rdparty/Megatron-LM${PYTHONPATH:+:$PYTHONPATH}
export RAY_ADDRESS=auto
export RAY_ENABLE_UV_RUN_RUNTIME_ENV=0
export NEMO_RL_VENV_DIR=/opt/ray_venvs
export NRL_FORCE_REBUILD_VENVS=false
export CUDA_DEVICE_MAX_CONNECTIONS=1
export NCCL_NVLS_ENABLE=0
export VLLM_TRITON_FORCE_FIRST_CONFIG=1
unset WANDB_MODE
mkdir -p "$MM_TRAINER_RESULTS_DIR" "$SUPER_CACHE_DIR"
cd "$RL_DIR"
```

Before training, check that the external Ray cluster has the selected recipe's node/GPU count:

```bash
uv run --no-sync --python "$DRIVER_PYTHON" python -c \
  'import ray; ray.init(address="auto"); print(len([n for n in ray.nodes() if n["Alive"]]), ray.cluster_resources().get("GPU", 0))'
```

### Recipe 1 — CLEVR-CoGenT

| Field | Value |
|---|---|
| `data.train.dataset_name` / split | `clevr-cogent` / `train` |
| `data.validation.dataset_name` / split | `clevr-cogent` / `valA` |
| `env.clevr-cogent.reward_functions` | `format` (0.2) + `exact_alnum` (0.8) |
| Prompts × generations / training global batch | 8 × 16 = 128 rollouts / 8 |
| Maximum response / total context | 4096 / 8192 tokens |
| Sequence-error threshold | Unset; no sequence-error masking |

The dataset loader downloads CLEVR-CoGenT on first use; no manual preparation is required. Validation and checkpoints run every 10 steps.

### Launch (16-node container allocation)

```bash
export SUPER_MEGATRON_CACHE=$SUPER_CACHE_DIR/megatron-supervl3p5-tp8-ep8-cp1
export NRL_MEGATRON_CHECKPOINT_DIR=$SUPER_MEGATRON_CACHE
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:False
uv run --no-sync --python "$DRIVER_PYTHON" python examples/run_vlm_grpo.py \
  --config examples/configs/recipes/vlm/vlm_grpo-supervl3p5-clevr-16n4g-megatron-tp8ep8.v1.yaml
```

### Recipe 2 — MMPR-Tiny

| Field | Value |
|---|---|
| `data.train.dataset_name` | `mmpr-tiny` |
| `data.train.download_dir` | `${oc.env:SUPER_MMPR_CACHE}` |
| `data.train.split_validation_size` | `0.008`; validation is carved from MMPR-Tiny |
| `data.validation` | `null`; avoids inheriting CLEVR `valA` |
| `env.mmpr-tiny.reward_functions` | `geo3k` (1.0), `format_score: 0.1` |
| Prompts × generations / training global batch | 512 × 16 = 8192 rollouts / 2048 |
| Maximum response / total context | 8192 / 8192 tokens |
| `grpo.seq_logprob_error_threshold` | `2.0` |

The loader downloads/extracts OpenGVLab/MMPR-Tiny under `SUPER_MMPR_CACHE`; share this cache across retries. Overlong filtering is enabled. Validation and checkpoints run every 10 steps.

### Launch (32-node container allocation)

```bash
export MM_TRAINER_WANDB_ID=supervl3p5-mmpr-prod-unique
export MM_TRAINER_WANDB_NAME=$MM_TRAINER_WANDB_ID
export MM_TRAINER_RESULTS_DIR=/path/to/experiments/$MM_TRAINER_WANDB_ID
export SUPER_MMPR_CACHE=$SUPER_CACHE_DIR/datasets/mmpr-tiny
export SUPER_MEGATRON_CACHE=$SUPER_CACHE_DIR/megatron-supervl3p5-tp8-ep16-cp1
export NRL_MEGATRON_CHECKPOINT_DIR=$SUPER_MEGATRON_CACHE
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:False
uv run --no-sync --python "$DRIVER_PYTHON" python examples/run_vlm_grpo.py \
  --config examples/configs/recipes/vlm/vlm_grpo-supervl3p5-mmpr-32n4g-megatron-tp8ep16.v1.yaml
```

Append Hydra-style overrides to either command, for example `checkpointing.checkpoint_must_save_by=00:03:00:00` to fit the remaining allocation time.


## Checkpoints and qualification

Use a fresh conversion cache when the Bridge/MCore pins or model layout change. Legacy `llava_model` Megatron checkpoints do not match the dedicated Nemotron Omni model; convert from the HF checkpoint instead. Do not resume checkpoints from the previous worktree until compatibility has been checked.

Both recipes save every 10 steps, retain the best two checkpoints by `val:accuracy`, and save optimizer state synchronously. A checkpoint deadline can trigger a final save between regular checkpoint steps. Set the deadline to fit the remaining allocation time and leave room for the final save.

Historical qualification on the previous checkout reached CLEVR step 18 (74.12% validation accuracy at step 10) and completed full-size MMPR-Tiny updates. These results do not qualify the new dependency pins. Recheck full-size rollout, logprobs, optimizer updates, reward, TMPE, and checkpoint restore on this branch before accepting convergence results.
