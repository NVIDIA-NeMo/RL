# Nemotron Omni video GRPO launchers

This directory contains the reproducible Slurm launchers used to validate
video Gym training:

- `run_2n_sync.sh`: 2-node synchronous GRPO on the `interactive` partition
  and `nemotron_edge_omni` account.
- `run_16n_async.sh`: 16-node asynchronous GRPO on either the
  `nemotron_edge_omni` or `nemotron_omni_vision` account.

The canonical training and data configuration is kept with the NeMo RL
recipes:

- `examples/nemo_gym/grpo_nemotron_omni_30ba3b_video_sync.yaml`
- `examples/nemo_gym/grpo_nemotron_omni_30ba3b_video_async.yaml`
- `examples/nemo_gym/prepare_video_dataset.py`

Both launchers use cached-video JSONL input. They intentionally keep
`grpo.max_num_steps=-1` and `grpo.seq_logprob_error_threshold=null`; the
four-hour Slurm allocation, rather than a GRPO step cap, ends a validation
run. They also enable W&B and checkpointing. No credential, user-specific
path, temporary source overlay, or version-mismatch bypass is embedded in
the scripts.

The 16-node asynchronous recipe uses four prompts with sixteen generations
per prompt (global batch 64). Its policy uses the validated TP4/EP4 two-node
topology and the other fourteen nodes run generation. Thus only 16/128 GPUs
are idle during the initial rollout, below the batch scheduler's 25-percent
idle reaper threshold. This does not cap generations or GRPO training steps.

The Megatron policy uses MBridge's canonical `NemotronOmniModel` expanded-
sequence contract. V2-labeled Nano Omni MoE checkpoints are routed through the
canonical bridge while dense V2 checkpoints retain their legacy behavior. The
launchers key the converted Megatron checkpoint cache by the exact MBridge
commit so a checkpoint produced by the retired LLaVA path cannot be reused.

## Prepare the cached-video dataset

The recipe expects training and validation JSONL files plus a media root.
The JSONL may be generated from a supported source dataset with:

```bash
uv run examples/nemo_gym/prepare_video_dataset.py --help
```

Set `NEMO_RL_VIDEO_MEDIA_ROOT` to the filesystem prefix from which the video
paths in the JSONL can be resolved. No dataset content is committed to this
repository.

## Required environment

Set these variables before launching either job:

```bash
export CONTAINER=/path/to/nemo-rl-vllm-0.25.1.sqsh
export MOUNTS=/lustre:/lustre
export NEMO_RL_MODEL=/path/to/nemotron-omni-checkpoint
export NEMO_RL_CHAT_TEMPLATE="${NEMO_RL_MODEL}/chat_template.jinja"
export NEMO_RL_VIDEO_TRAIN_JSONL=/path/to/cached_video_train.jsonl
export NEMO_RL_VIDEO_VAL_JSONL=/path/to/cached_video_validation.jsonl
export NEMO_RL_VIDEO_MEDIA_ROOT=/lustre
export NEMO_RL_RUN_ROOT=/path/to/training-output
export WANDB_API_KEY=...
export WANDB_ENTITY=...
export WANDB_PROJECT=...
```

The launchers install the stock vLLM `0.25.1` wheel declared by the project
into a node-local overlay and verify its version before training. The base
container must provide `/opt/rl_main_vg_runtime.env`, the NeMo RL
environment at `/opt/nemo_rl_venv`, the project metadata at `/opt/nemo-rl`,
and the video decoding dependencies. They bootstrap the exact `uv` release
declared by `docker/Dockerfile`, then install the interpreter declared by
`.python-version` into shared, versioned directories before Ray starts. This
keeps actor environments aligned with the checked-out branch when a compatible
base container has older `uv` or Python patch releases. A branch-locked main
environment is also materialized node-locally so the Ray cluster, driver, and
actor environments all use that same Python release. A runtime Ray dispatcher
in the output directory ensures the launcher's initial cleanup can use the base
image while cluster startup uses the branch-locked environment; it is not added
to the repository. After the initial cleanup, node setup installs that dispatcher
at the container-local Ray CLI path and preserves the image CLI as its fallback.
This ensures `ray.sub` starts the cluster with the same interpreter as the driver
even when the base image has an older Python patch release. Runtime dependency
synchronization uses the committed lock file in locked mode, so setup fails on
dependency drift instead of rewriting `uv.lock` in the source tree. Gym child
services use a run-scoped node-local venv root keyed by the commit and Python
release, preventing the recipe's venv reuse from selecting image-baked services
created with a different Python patch release.
The large uv package cache is also commit-keyed and node-local, avoiding shared
Lustre quota exhaustion while CUDA and PyTorch wheels are extracted on each node.
Credentials must come from the caller's environment or an approved secret
mechanism; do not add them to these scripts.
The launchers create a UTC timestamp run ID by default. Set
`NEMO_RL_RUN_ID` explicitly only when a stable W&B/checkpoint run name is
required, such as for an intentional resume.

## Launch 2-node synchronous validation

```bash
bash ehsan_scripts/run_2n_sync.sh
```

The launcher submits a four-hour job. To change only the Slurm duration,
set `SBATCH_TIME` before invoking it. Do not use the duration to introduce a
GRPO step limit.

## Launch 16-node asynchronous validation

Select one approved account and launch:

```bash
export SBATCH_ACCOUNT=nemotron_edge_omni
bash ehsan_scripts/run_16n_async.sh
```

`SBATCH_ACCOUNT=nemotron_omni_vision` is also supported. If jobs are raced
between accounts, cancel the duplicate immediately after one starts.
