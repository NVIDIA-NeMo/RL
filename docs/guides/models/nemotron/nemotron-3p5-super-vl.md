# Nemotron 3.5 Super VL

This guide explains how to post-train the SuperVL3p5 vision-language checkpoint (120B total parameters, 12B active) with GRPO using NeMo RL, Megatron-Core, and vLLM. It covers CLEVR-CoGenT, MMPR-Tiny, and the unified-teacher NeMo Gym recipe, plus single-controller ablations of the two image workloads.

## Multimodal payload deduplication

The recipes enable `grpo.deduplicate_multimodal_data` and leave payload-size diagnostics disabled. This shares immutable media across logical GRPO generations. On the TransferQueue data plane, deduplication saves driver RAM; it does not remove the per-logical-row wire payload. The initial NeMo Gym image payload path has separate limitations, described in [the Nano Omni guide](nemotron-3-nano-omni.md#multimodal-payload-deduplication).

## Megatron backend

The qualified setup uses the HF checkpoint's tokenizer, processor, and `chat_template.jinja`, together with the project's pinned Megatron Bridge, Megatron-LM, and Gym submodules. NeMo RL must be mounted recursively and visible on every worker. Complete worker-environment, Lens, MCore-helper, media-library, and vLLM setup before launching a driver.

The tested HSG container is `/home/rohitkumarj/data/enroot-containers/rl.nightly.sep30.2026.sqsh`. The commands below run inside its head container on an existing multi-node Ray allocation with four GPUs per node. They use the existing mounts; they do not start an allocation or Ray cluster.

### Checkpoint compatibility

Set `MM_TRAINER_MODEL_PATH` to the local SuperVL3p5 HF checkpoint. Reuse a Megatron conversion cache only with the same input weights, model integration, and parallel layout. A fresh cache can require HF-to-Megatron conversion before training. Do not reuse a Nano checkpoint or a legacy model-layout cache for SuperVL3p5.

### Maintained recipes

| Workload | Recipe | Topology |
|---|---|---|
| CLEVR-CoGenT | [16-node CLEVR recipe](../../../../examples/configs/recipes/vlm/vlm_grpo-supervl3p5-clevr-16n4g-megatron-tp8ep8.v1.yaml) | 16 × 4 GPUs; policy TP8 / EP8 / CP1; colocated vLLM TP4 / EP4 |
| MMPR-Tiny | [32-node MMPR-Tiny recipe](../../../../examples/configs/recipes/vlm/vlm_grpo-supervl3p5-mmpr-32n4g-megatron-tp8ep16.v1.yaml) | 32 × 4 GPUs; policy TP8 / EP16 / CP1; colocated vLLM TP4 / EP4 |
| Unified teachers | [32-node unified V2 recipe](../../../../examples/configs/recipes/vlm/super_vl_35_mixed_teachers_production.yaml) | 32 × 4 GPUs; policy TP2 / EP16 / CP2; separate 16-node vLLM TP4 / EP4 fleet |

The image recipes inherit the corresponding Nano Omni task recipes with more nodes to fit SuperVL3p5. They retain R3 disabled, FP32 LM heads, frozen vision/audio modules, optimizer offload during logprob calculation, and synchronous checkpoint writes. `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:False` is set in their policy and generation worker environments to avoid a reproduced CUDA IPC allocator failure in this container.

### Common launch environment

Replace the placeholder paths with shared paths already visible inside every worker container. Keep source data read-only and caches/results writable. Set shared cache and worker environment values before Ray starts. Choose a new W&B ID and output directory for each independent experiment; provide credentials through the environment or existing login.

```bash
export RL_DIR=/opt/nemo-rl
export DRIVER_PYTHON=/opt/nemo_rl_venv/bin/python
export MM_TRAINER_MODEL_PATH=/path/to/supervl3p5/hf
export SUPER_CACHE_DIR=/path/to/shared-cache/supervl3p5
export MM_TRAINER_WANDB_ID=supervl3p5-clevr-prod-unique
export MM_TRAINER_WANDB_NAME=$MM_TRAINER_WANDB_ID
export MM_TRAINER_RESULTS_DIR=/path/to/experiments/$MM_TRAINER_WANDB_ID
export HF_HOME=$SUPER_CACHE_DIR/huggingface
export HF_DATASETS_CACHE=$HF_HOME/datasets
export UV_CACHE_DIR=$SUPER_CACHE_DIR/uv
export BRIDGE_DIR=$RL_DIR/3rdparty/Megatron-Bridge-workspace/Megatron-Bridge
export NEMO_GYM_EXTRA_ROOTS=$RL_DIR/3rdparty/Gym-workspace/Gym:$RL_DIR/examples/nemo_gym/supervl3p5
export PYTHONPATH=$RL_DIR:$NEMO_GYM_EXTRA_ROOTS:$BRIDGE_DIR/src:$BRIDGE_DIR/3rdparty/Megatron-LM${PYTHONPATH:+:$PYTHONPATH}
export RAY_ADDRESS=auto
export RAY_ENABLE_UV_RUN_RUNTIME_ENV=0
export NEMO_RL_VENV_DIR=/opt/ray_venvs
export NRL_VENVS_TRUST_EXISTING=1
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

## Single-controller ablations

These overlays inherit the production image recipes and keep their datasets, rewards, prompt counts, 16 generations per prompt, model parallelism, context limits, and R3/sequence-error settings. They use native environment rollouts with `examples/run_grpo_single_controller.py`; NeMo Gym token capture is disabled.

| Workload | Ablation recipe | Policy / generation nodes | Training global batch |
|---|---|---|---|
| CLEVR-CoGenT | [24-node single-controller recipe](../../../../examples/configs/recipes/vlm/vlm_grpo-supervl3p5-clevr-24n4g-megatron-tp8ep8-single-controller.v1.yaml) | 16 / 8, four GPUs per node | 128 |
| MMPR-Tiny | [40-node single-controller recipe](../../../../examples/configs/recipes/vlm/vlm_grpo-supervl3p5-mmpr-40n4g-megatron-tp8ep16-single-controller.v1.yaml) | 32 / 8, four GPUs per node | 8192 |

The runtime requires three changes beyond the entry point:

- vLLM uses an async engine on a separate eight-node fleet; the original policy GPU capacity is retained.
- One controller step consumes the full rollout batch in one optimizer step. Global batch increases from 8 to 128 for CLEVR and from 2048 to 8192 for MMPR-Tiny. Account for this changed optimizer-update frequency when comparing reward curves.
- Validation is disabled because the controller has no validation loop. Checkpoint selection uses `metric_name: null`, with data-plane recovery enabled and regular saves every 10 steps.

Both overlays use `grpo.async_grpo: null`, `data_plane.enabled: true`, and an in-order sampler with `max_lookahead_versions: 0`. Inflight/buffer capacity and the streaming threshold each equal the prompt count, so each step waits for its complete batch and generation stays on the current policy version. Simple data-plane storage-unit counts remain 32 for CLEVR and 64 for MMPR-Tiny.

Use a **new** run ID and output directory. After selecting the matching CLEVR or MMPR cache environment above, launch once from the corresponding 24- or 40-node allocation head:

```bash
# CLEVR: 24 nodes / 96 GPUs.
uv run --no-sync --python "$DRIVER_PYTHON" python examples/run_grpo_single_controller.py \
  --config examples/configs/recipes/vlm/vlm_grpo-supervl3p5-clevr-24n4g-megatron-tp8ep8-single-controller.v1.yaml
```

```bash
# MMPR-Tiny: 40 nodes / 160 GPUs; use the MMPR environment and a separate run ID.
uv run --no-sync --python "$DRIVER_PYTHON" python examples/run_grpo_single_controller.py \
  --config examples/configs/recipes/vlm/vlm_grpo-supervl3p5-mmpr-40n4g-megatron-tp8ep16-single-controller.v1.yaml
```

The overlays are ablations awaiting full GPU qualification; the production image-run results below came from the colocated entry point.

## NeMo Gym unified-teacher GRPO

The unified recipe uses task-routed `training.jsonl` rows with `agent_ref` and `responses_create_params`. Preserve task answers/labels, the complete cached video-frame list, and `_is_video_frame` / `_video_source` metadata. Bare media paths and once-decoded file URIs must resolve on every worker. The committed Gym overlay supplies the SA-V tracking verifier; image tools write crops under the experiment directory.

The routes include GUI coordinates, math, MCQA, string matching, SA-V tracking, and image tools. The current math route has `should_use_judge: false`; this recipe does not start separate teacher LMs.

```bash
export MM_TRAINER_WANDB_ID=supervl3p5-unified-v2-prod-unique
export MM_TRAINER_WANDB_NAME=$MM_TRAINER_WANDB_ID
export MM_TRAINER_RESULTS_DIR=/path/to/experiments/$MM_TRAINER_WANDB_ID
export MM_TRAINER_DATA_PATH=/path/to/mm-trainer-unified/training.jsonl
export MM_TRAINER_MEDIA_ROOT=/path/to/media-root
export MM_TRAINER_GYM_VENV_DIR=/opt/gym_venvs
export NEMO_GYM_VENV_DIR=$MM_TRAINER_GYM_VENV_DIR
export NRL_MEGATRON_CHECKPOINT_DIR=$SUPER_CACHE_DIR/megatron-supervl3p5-tp2-ep16-cp2
export NRL_VIDEO_BACKEND=torchcodec
export VLLM_VIDEO_LOADER_BACKEND=nemotron_vl
uv run --no-sync --python "$DRIVER_PYTHON" python examples/run_grpo_single_controller.py \
  --config examples/configs/recipes/vlm/super_vl_35_mixed_teachers_production.yaml
```

The effective policy layout is TP2 / EP16 / CP2, despite `cp1` in an inherited filename. The recipe uses 128 × 16 = 2048 samples, 32768 output tokens, 65536 total context, and 64 video frames with temporal patch size 2 and target patches 1024. Its in-order sampler has lookahead 1, inflight 128, buffer 256, and streaming minimum 128 groups. R3 is disabled, sequence-error masking is 2.0, saves run every 10 steps, and validation is disabled. Maximum training steps is 125.

## Checkpoints and observed results

All recipes write checkpoints to `$MM_TRAINER_RESULTS_DIR/checkpoints` and logs to `$MM_TRAINER_RESULTS_DIR/logs`, with W&B project `nvidia/rohit-unified-teacher-supervl3p5` by default. The checkpoint deadline defaults to 3h15m; adjust it for the remaining allocation time and leave room for startup and the final save. The deadline can save a step outside the regular ten-step cadence.

CLEVR/MMPR production retain the best two checkpoints by `val:accuracy`. Unified teachers and the controller ablations retain recent checkpoints and save data-plane state. Resume with the same YAML/output directory, and confirm that weights, optimizer, step, and any data-plane state restore before accepting new metrics. Historical unified checkpoints occupied roughly 1.60–1.72 TiB each; budget space for retained checkpoints plus a new save.

Recorded qualification on 2026-10-05:

- CLEVR passed two full-size smoke steps, reached production step 11, and saved step 10. Regular production steps took about eight minutes.
- MMPR-Tiny passed one full 8192-rollout smoke update and one production update. A production update took about 23 minutes. Its threshold-2.0 setting and the controller overlays still need convergence qualification.
- Historical unified V2 runs reached steps 111 and 125. Accepted sequences had masked TMPE around 1.02; high-error sequences were still rejected.

Monitor reward and held-out accuracy over multiple steps, generation KL error, raw/filtered TMPE, rejected sequence counts, response lengths, and rollout/train/save time. Low masked TMPE describes accepted sequences and does not establish that raw mismatches disappeared. Fixed-weight native vLLM decode/fresh-prefill tests did not reproduce the severe mismatch; live rollout assembly, media expansion, and changing-weight paths remain separate questions.
