#!/bin/bash
# Launch the Qwen3-30B-A3B 20-step trace-collection run (grpo_qwen3_30ba3b_8n4g_gym_trace20.yaml)
# on the shared Slurm cluster's GB200 NVL72 partitions.
#
# Run from any shell (bash script, so tcsh heredoc issues don't apply):
#   ./launch_trace20.sh
# Override the partition if desired:
#   PARTITION=gb200nvl72 ./launch_trace20.sh
set -eu

# Use the real path, not the /home/sechoi/scratch symlink: the symlink does not
# exist inside the container, so MOUNTS and the COMMAND cd must both use this.
REPO=/home/scratch.sechoi_coreai/nrl/RL-traces

PARTITION=${PARTITION:-gb200nvl72_preprod}
ACCOUNT=blackwell
# Prefer the pre-imported squashfs (fast start, no per-node registry pull).
# It must be imported on an aarch64 (GB200) node — see enroot-import job.
# Fallback pyxis registry syntax is 'REGISTRY#IMAGE:TAG': a plain 'nvcr.io/...'
# path makes enroot look for the image on Docker Hub and fail with 401.
# Internal nightly (matches recent main): the v0.6.0 release container is too
# old for this checkout — its CUDA 12.9 stack can't build the lock's CUDA 13.0
# deps. Pulling nvidian requires NGC creds in ~/.config/enroot/.credentials.
SQSH=/home/scratch.sechoi_coreai/nrl/nemo-rl-nightly.sqsh
if [[ -f $SQSH ]]; then
  CONTAINER_IMAGE=$SQSH
else
  CONTAINER_IMAGE='nvcr.io#nvidian/nemo-rl:nightly'
fi
# Overridable: the cluster enforces a singleton dependency per job name, so a
# second submission under the same name queues behind the first instead of
# racing it on another partition.
EXP_NAME=${EXP_NAME:-qwen3-30ba3b-trace20}

# The site prolog points TMPDIR at NFS scratch, where enroot's image unpacking
# is both glacial and broken (overlayfs whiteouts hit EPERM on NFS). Force
# enroot onto node-local NVMe. Exported here so sbatch propagates it to every
# node's pyxis srun.
export ENROOT_TEMP_PATH=/tmp
export TMPDIR=/tmp

cd "$REPO"
# Per-experiment trace dir: two runs (e.g. a queued retry racing a live run)
# must never append to the same rollout-level JSONL — that writer is only
# single-writer-safe within one run.
TRACE_DIR=$REPO/traces/$EXP_NAME
mkdir -p "$TRACE_DIR"

# Optional: LENGTH_ORDER_JSON=<file from tools/build_length_ordering.py>
# reorders training prompts shortest-output-first (forces shuffle off).
# Optional: LPT_ADMISSION_JSON=<same file format> enables longest-first rollout
# admission in the async trajectory collector (async_grpo runs only).
EXTRA_OVERRIDES=""
if [[ -n "${LENGTH_ORDER_JSON:-}" ]]; then
  EXTRA_OVERRIDES="++data.length_order_json=$LENGTH_ORDER_JSON"
fi
if [[ -n "${LPT_ADMISSION_JSON:-}" ]]; then
  EXTRA_OVERRIDES="$EXTRA_OVERRIDES ++grpo.async_grpo.lpt_admission_json=$LPT_ADMISSION_JSON"
fi
# Async runs must not exceed steps-per-epoch (19 here): the async collector
# doesn't wrap the dataloader at epoch end, so a 20th step hangs forever.
if [[ -n "${MAX_STEPS:-}" ]]; then
  EXTRA_OVERRIDES="$EXTRA_OVERRIDES ++grpo.max_num_steps=$MAX_STEPS"
fi

read -r -d '' COMMAND <<EOF || true
cd $REPO
HF_HOME=$REPO/.cache \
HF_HUB_OFFLINE=1 \
uv run python examples/nemo_gym/run_grpo_nemo_gym.py \
    --config examples/nemo_gym/grpo_qwen3_30ba3b_8n4g_gym_trace20.yaml \
    ++policy.generation.mocker_request_trace_jsonl=$TRACE_DIR/qwen3_30ba3b_8n4g_mocker_requests.jsonl \
    ++policy.generation.vllm_cfg.mocker_request_server_trace_jsonl=$TRACE_DIR/qwen3_30ba3b_8n4g_vllm_server_requests.jsonl \
    ++logger.log_dir=results/$EXP_NAME \
    ++checkpointing.checkpoint_dir=results/$EXP_NAME $EXTRA_OVERRIDES
EOF

echo -e "Running command:\n$COMMAND"

# This checkout's uv.lock pins ray 2.55.1 but the v0.6.0 container venv ships
# ray 2.54.0. The driver's `uv run` syncs the lock, so the ray daemons must be
# synced too or ray.init() fails on a version mismatch. SETUP_COMMAND runs on
# every node before `ray start`. Do NOT set UV_CACHE_DIR_OVERRIDE to shared
# NFS here: uv's cache locking is per-host, and 8 nodes syncing concurrently
# corrupt the shared cache (seen as ModuleNotFoundError: packaging.version).
# Node-local caches re-download the small lock delta instead.

# The container also ships pre-built actor venvs (/opt/ray_venvs) with the old
# ray; force-rebuild them from the lock so vLLM/mcore workers match the cluster.
# Rebuild happens per node on local disk — adds a few minutes at startup.
# VLLM_USE_FLASHINFER_MOE_FP16=0: on Blackwell, vLLM's default FlashInfer-TRTLLM
# MoE kernel re-blocks w13/w2 into a 4D layout at engine init (via
# process_weights_after_loading under load_format=dummy). The ZMQ refit then
# feeds per-expert 2D weights into FusedMoE.weight_loader, which chokes
# ("shard_dim=0 ... 3D tensor"). Forcing the Triton MoE kernel keeps the
# canonical 3D (E, 2I/tp, H) layout so refit works. Slightly slower decode;
# trace content (requests/responses) is unaffected.
# RAY_CGRAPH_get_timeout: vLLM's Ray compiled-graph executor kills the engine
# (EngineDeadError) if a TP rank produces no output for 300s. Async runs with
# LPT admission front-load the longest 32k-context chains, and engines have
# stalled past 300s in that opening wave (observed twice, incl. on the head
# node). A dead engine's in-flight rollouts are lost and the collector then
# deadlocks, so tolerate slow stalls instead of dying.
COMMAND="export NRL_FORCE_REBUILD_VENVS=true NRL_REFIT_ERROR_LOG=$REPO/traces/refit_errors.log VLLM_USE_FLASHINFER_MOE_FP16=0 RAY_CGRAPH_get_timeout=1800 && $COMMAND" \
SETUP_COMMAND="cd $REPO && uv sync --locked" \
CONTAINER=$CONTAINER_IMAGE \
MOUNTS=$REPO:$REPO \
GPUS_PER_NODE=4 \
sbatch \
    --nodes=8 \
    --account=$ACCOUNT \
    --partition=$PARTITION \
    --gres=gpu:4 \
    --time=8:0:0 \
    --job-name=$EXP_NAME \
    --exclude="${EXCLUDE_NODES:-gb-nvl-059-compute09,gb-nvl-057-compute06,gb-nvl-057-compute09}" \
    ray.sub
