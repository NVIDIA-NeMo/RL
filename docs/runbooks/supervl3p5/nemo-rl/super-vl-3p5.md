# SuperVL3.5: NeMo-RL production recipes

Train the SuperVL3p5 checkpoint (120B total parameters, 12B active) with Megatron-Core and vLLM. The three production YAMLs use environment variables for model, data, cache, and output paths.

| Recipe | Allocation | Policy TP / EP / CP | Generation | Prompts × generations | Training global batch |
|---|---|---|---|---|---|
| [CLEVR](../../../../examples/configs/recipes/vlm/vlm_grpo-supervl3p5-clevr-16n4g-megatron-tp8ep8.v1.yaml) | 16 nodes × 4 GPUs | 8 / 8 / 1 | Colocated, TP4 / EP4 | 8 × 16 = 128 | 8 |
| [MMPR-Tiny](../../../../examples/configs/recipes/vlm/vlm_grpo-supervl3p5-mmpr-32n4g-megatron-tp8ep16.v1.yaml) | 32 nodes × 4 GPUs | 8 / 16 / 1 | Colocated, TP4 / EP4 | 512 × 16 = 8192 | 2048 |
| [Unified teachers, V2](../../../../examples/configs/recipes/vlm/super_vl_35_mixed_teachers_production.yaml) | 32 nodes × 4 GPUs | 2 / 16 / 2 | Separate 16-node fleet, TP4 / EP4 | 128 × 16 = 2048 | 2048 |

CLEVR and MMPR-Tiny inherit the corresponding Nano Omni task recipes, with more nodes to fit SuperVL3p5. They use the synchronous `examples/run_vlm_grpo.py` entry point. Unified teachers uses `examples/run_grpo_single_controller.py` with the in-order async sampler. Effective CP is 2 for unified teachers, despite `cp1` in an inherited filename.

## Prerequisites

- Start inside the provisioned head container of an existing Ray allocation. Use the node counts above, four GPUs per node, and the existing project mounts. NeMo-RL and its nested Bridge/Megatron/Gym submodules must be visible on every node.
- The tested HSG image is `/home/rohitkumarj/data/enroot-containers/rl.nightly.sep30.2026.sqsh`. Driver Python is `/opt/nemo_rl_venv/bin/python`; worker environments are under `/opt/ray_venvs`.
- Complete worker-environment, Lens, MCore-helper, media-library, and vLLM setup before training. The commands below assume that this setup and Ray startup are complete.
- Supply the SuperVL3p5 HF checkpoint, including its tokenizer, processor, and `chat_template.jinja`. The first use of a new Megatron cache can require HF-to-Megatron conversion. Reuse a compatible cache for retries.
- Model and data paths must be readable on all workers. Cache and output paths must be writable. Keep source data read-only.
- Provide W&B credentials through the environment or existing login. Use a separate run ID and output directory for each new experiment.

## Common environment

Replace `/path/to/...` with paths visible inside the existing containers. Set these shared cache/environment values on all nodes before starting Ray; set the run ID and output directory on the head before launching the driver.

```bash
export RL_DIR=/opt/nemo-rl
export DRIVER_PYTHON=/opt/nemo_rl_venv/bin/python
export MM_TRAINER_MODEL_PATH=/path/to/supervl3p5/hf
export SUPER_CACHE_DIR=/path/to/shared-cache/supervl3p5
export MM_TRAINER_WANDB_ID=supervl3p5-clevr-prod-unique
export MM_TRAINER_WANDB_NAME=$MM_TRAINER_WANDB_ID
export MM_TRAINER_RESULTS_DIR=/path/to/experiments/$MM_TRAINER_WANDB_ID
export MM_TRAINER_GYM_VENV_DIR=/opt/gym_venvs
export NEMO_GYM_VENV_DIR=$MM_TRAINER_GYM_VENV_DIR
export NEMO_GYM_EXTRA_ROOTS=$RL_DIR/3rdparty/Gym-workspace/Gym:$RL_DIR/examples/nemo_gym/supervl3p5
export BRIDGE_DIR=$RL_DIR/3rdparty/Megatron-Bridge-workspace/Megatron-Bridge
export PYTHONPATH=$RL_DIR:$NEMO_GYM_EXTRA_ROOTS:$BRIDGE_DIR/src:$BRIDGE_DIR/3rdparty/Megatron-LM${PYTHONPATH:+:$PYTHONPATH}
export UV_CACHE_DIR=$SUPER_CACHE_DIR/uv
export HF_HOME=$SUPER_CACHE_DIR/huggingface
export HF_HUB_CACHE=$HF_HOME/hub
export HF_MODULES_CACHE=$HF_HOME/modules
export HF_DATASETS_CACHE=$HF_HOME/datasets
export TORCH_HOME=$SUPER_CACHE_DIR/torch
export TRITON_CACHE_DIR=$SUPER_CACHE_DIR/triton
export XDG_CACHE_HOME=$SUPER_CACHE_DIR/xdg
export RAY_ADDRESS=auto
export RAY_ENABLE_UV_RUN_RUNTIME_ENV=0
export NEMO_RL_VENV_DIR=/opt/ray_venvs
export NRL_VENVS_TRUST_EXISTING=1
export NRL_FORCE_REBUILD_VENVS=false
export CUDA_DEVICE_MAX_CONNECTIONS=1
export NCCL_NVLS_ENABLE=0
export NVTE_FWD_LAYERNORM_SM_MARGIN=16
export NVTE_BWD_LAYERNORM_SM_MARGIN=16
export VLLM_TRITON_FORCE_FIRST_CONFIG=1
export FLASHINFER_DISABLE_VERSION_CHECK=1
export TORCH_CUDA_ARCH_LIST=10.0
unset WANDB_MODE
mkdir -p "$MM_TRAINER_RESULTS_DIR" "$SUPER_CACHE_DIR"
cd "$RL_DIR"
```

Confirm the external cluster before launching. Expect 16 nodes / 64 GPUs for CLEVR or 32 nodes / 128 GPUs for MMPR-Tiny and unified teachers:

```bash
uv run --no-sync --python "$DRIVER_PYTHON" python -c \
  'import ray; ray.init(address="auto"); print(len([n for n in ray.nodes() if n["Alive"]]), ray.cluster_resources().get("GPU", 0))'
```

## CLEVR production

Use the CLEVR run ID/output directory from the common environment, or choose a new pair. The dataset loader prepares CLEVR-CoGenT; training uses `train` and validation uses `valA`. Rewards combine format (0.2) and exact alphanumeric answer matching (0.8).

```bash
export SUPER_MEGATRON_CACHE=$SUPER_CACHE_DIR/megatron-supervl3p5-tp8-ep8-cp1
export NRL_MEGATRON_CHECKPOINT_DIR=$SUPER_MEGATRON_CACHE
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:False
uv run --no-sync --python "$DRIVER_PYTHON" python examples/run_vlm_grpo.py \
  --config examples/configs/recipes/vlm/vlm_grpo-supervl3p5-clevr-16n4g-megatron-tp8ep8.v1.yaml
```

- Maximum response: 4096 tokens; total context: 8192. Validation and checkpointing run every 10 steps.
- Policy and generation use FP32 LM heads, frozen vision/audio modules, and R3 disabled. Megatron optimizer offload for logprobs is enabled; checkpoint writes are synchronous.
- The YAML leaves sequence-error masking unset. Generation/training mismatch metrics are still reported.
- `expandable_segments:False` avoids the CUDA IPC allocator failure reproduced with this container. Preserve both the policy and generation YAML environment settings.

## MMPR-Tiny production

Select a new run ID and output directory. The loader downloads/extracts OpenGVLab/MMPR-Tiny under `SUPER_MMPR_CACHE`; reuse the shared cache for retries. A validation split of 0.008 is taken from MMPR-Tiny; `data.validation: null` prevents inheriting CLEVR's `valA` dataset. Reward uses `geo3k`, including format score 0.1.

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

- The production batch contains **8192 rollouts**, with training global batch 2048. This is larger than CLEVR's batch; it is not a reduced smoke configuration.
- Maximum response and total context are 8192 tokens. Overlong filtering is enabled.
- `grpo.seq_logprob_error_threshold: 2.0` is committed in the YAML. It masks sequences with excessive generation/training logprob error before training.
- R3 is disabled. FP32 heads, logprob optimizer offload, synchronous saves, and the allocator setting match CLEVR. Validation and checkpoints run every 10 steps.

## Unified-teacher V2 production

Provide the mixed-task training JSONL and a media root visible on every node. Rows retain `agent_ref`, `responses_create_params`, answers/labels, and any cached video-frame metadata. The committed Gym overlay supplies the SA-V tracking verifier.

```bash
export MM_TRAINER_WANDB_ID=supervl3p5-unified-v2-prod-unique
export MM_TRAINER_WANDB_NAME=$MM_TRAINER_WANDB_ID
export MM_TRAINER_RESULTS_DIR=/path/to/experiments/$MM_TRAINER_WANDB_ID
export MM_TRAINER_DATA_PATH=/path/to/mm-trainer-unified/training.jsonl
export MM_TRAINER_MEDIA_ROOT=/path/to/media-root
export NRL_MEGATRON_CHECKPOINT_DIR=$SUPER_CACHE_DIR/megatron-supervl3p5-tp2-ep16-cp2
export NRL_VIDEO_BACKEND=torchcodec
export NRL_VIDEO_SAMPLING_STYLE=nemotron_vl
export NRL_VIDEO_TEMPORAL_PATCH_SIZE=2
export VLLM_VIDEO_LOADER_BACKEND=nemotron_vl
export NEMO_RL_VIDEO_MEDIA_ROOT=$MM_TRAINER_MEDIA_ROOT
export NEMO_RL_VIDEO_TRAIN_JSONL=$MM_TRAINER_DATA_PATH
export NEMO_RL_VIDEO_VAL_JSONL=$MM_TRAINER_DATA_PATH
uv run --no-sync --python "$DRIVER_PYTHON" python examples/run_grpo_single_controller.py \
  --config examples/configs/recipes/vlm/super_vl_35_mixed_teachers_production.yaml
```

- In-order sampler: lookahead 1, inflight prompts 128, buffered rollouts 256, streaming minimum 128 groups. Maximum training steps: 125.
- Maximum response: 32768 tokens; total context: 65536. Video sampling uses 64 frames, temporal patch size 2, and target patches 1024.
- R3 is disabled; sequence-error threshold is 2.0; saves are synchronous and run every 10 steps. Validation is disabled.
- Routes include GUI coordinates, math, MCQA, string matching, SA-V tracking, and image tools. The current math route has `should_use_judge: false`; this recipe does not launch separate teacher LMs.

## Checkpoints and monitoring

- All recipes write checkpoints under `$MM_TRAINER_RESULTS_DIR/checkpoints`, logs under `$MM_TRAINER_RESULTS_DIR/logs`, and W&B metrics to `nvidia/rohit-unified-teacher-supervl3p5` by default.
- Regular checkpoint saves run every 10 steps; the allocation deadline can trigger a final save at another step. `checkpointing.checkpoint_must_save_by` defaults to 3h15m. Override it to fit the remaining allocation time, leaving room for setup and the final save.
- CLEVR/MMPR retain the best two checkpoints by `val:accuracy`; unified teachers retains recent checkpoints with `metric_name: null`. Check available disk space before a new save.
- Resume with the same YAML and output directory. Confirm restoration of weights, optimizer, and step; unified V2 also restores data-plane state. Choose a new W&B ID/output directory for an independent run.
- Monitor reward and validation accuracy over multiple steps, generation KL error, masked TMPE, rejected sequence counts, response lengths, step time, and memory. A low masked TMPE describes accepted sequences; it does not establish that raw mismatches disappeared.

Validation evidence as of 2026-10-05: CLEVR passed two production-sized smoke steps, reached production step 11, and saved step 10. MMPR-Tiny passed one full 8192-rollout smoke update and one production update; the original run was cancelled, and the threshold-2.0 replacement was queued. Historical unified V2 runs reached steps 111 and 125, with masked TMPE around 1.02. The new MMPR threshold setting still needs convergence validation.

Guide structure follows the [Nano Omni RL guide](https://docs.nvidia.com/nemo/rl/nightly/guides/models/nemotron/nemotron-3-nano-omni.html). Commands use the committed recipes directly.
