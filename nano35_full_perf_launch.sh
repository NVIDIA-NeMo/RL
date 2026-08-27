#!/bin/bash
# =============================================================================
# nano35_full_perf_launch.sh — full-scale (68-node) SC perf A/B launcher.
#
# Campaign: reports/auto_research/full_perf_run — SC + token capture (arm A)
# vs SC without token capture (arm B), both at the full rlvr_dolphin_sc.yaml
# pipeclean shape. Same code (60f4569a1 / Gym d9ad9d2e), same blend, same
# judges; the arms differ only in token_capture.enabled.
#
# This reuses the launch plumbing proven by nano35_ledger_smoke.sh (jobs
# 6490451 / 6497027 / 6498229, all PASS on this container + env stack), but
# execs the FULL launcher nano35_dolphin_launch_sc.sh instead of the smoke
# wrapper: 8 train + 40 gen + 8 gym in hetgroup 0, plus in-job judges
# (EXTERNAL_JUDGES=1: GenRM 2xTP8 = 4 nodes, NL2Bash 8xTP4 = 8 nodes) = 68
# nodes, 128 prompts x 16 generations, GBS 2048, lookahead 4, nccl_reshard.
#
# CHECKPOINTING MUST BE OFF ON THIS TREE. setup_single_controller (setup.py:388)
# raises NotImplementedError when checkpointing.enabled is true — the yaml's
# `enabled: true` documents the recovery branch, not this HEAD. So there is no
# resume: NRL_MAX_STEPS must fit one 4 h `batch` allocation. Default 10 steps
# (~50 min startup + ~12 min/step steady state at the historical 0.18 groups/s).
#
# Usage (networked shell, repo root):
#   ARM=capture   DRY_RUN=1 bash nano35_full_perf_launch.sh   # inspect
#   ARM=capture   DRY_RUN=0 bash nano35_full_perf_launch.sh   # submit arm A
#   ARM=nocapture DRY_RUN=0 bash nano35_full_perf_launch.sh   # submit arm B
# =============================================================================
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

ARM="${ARM:?Set ARM=capture or ARM=nocapture}"
case "${ARM}" in
  capture)   _ARM_OVERRIDES=() ;;
  nocapture) _ARM_OVERRIDES=("token_capture.enabled=false") ;;
  *) echo "ARM must be capture or nocapture, got '${ARM}'" >&2; exit 1 ;;
esac

_DRY_RUN_IN="${DRY_RUN:-}"

set -a
# Secrets (WANDB_API_KEY, HF_TOKEN): untracked mode-600 file, never in git.
# shellcheck disable=SC1091
source "${HERE}/swe_nano.secrets.env"

# ---- container / shared read-only assets (proven by the ledger smokes) -------
CONTAINER=/lustre/fsw/portfolios/llmservice/users/zhiyul/enroot-images/nvcr.io+nvidian+nemo-rl+nightly-gym.2026-08-10.squashfs
SANDBOX_CONTAINER=/lustre/fsw/portfolios/coreai/users/cye/enroot/nemo-rl:skills-sandbox-latest.squashfs
HF_HUB_CACHE=/lustre/fsw/portfolios/llmservice/users/zhiyul/hf_cache/hub

# ---- per-user write paths ----------------------------------------------------
# PERSISTENT_CACHE / HF_HOME reuse the smoke campaign's, so the 62 GB
# HF->Megatron conversion is already cached and does not rerun per job.
# Fresh workspace for the stack-pinned tree: the stack's fingerprint encoding
# must never read fork-era capture/ledger files (same version constant,
# different bytes), so this must not reuse RL-pr3456-fullperf-workspace.
WORKSPACE_DIR=/lustre/fsw/portfolios/llmservice/users/pthombre/sweRun/RL-pr3837-gymstack-fullperf-workspace
HF_HOME=/lustre/fsw/portfolios/llmservice/users/pthombre/hf_cache
PERSISTENT_CACHE=/lustre/fsw/portfolios/llmservice/users/pthombre/persistent_cache
NRL_MEGATRON_CHECKPOINT_DIR=${PERSISTENT_CACHE}/megatron_ckpt_cache

# SHAPE_TAG must default before EXP_NAME expands it (set -u), not down in the
# Slurm-shape block where the other shape knobs live.
SHAPE_TAG="${SHAPE_TAG:-}"
EXP_NAME="pthombre-nano35-fullperf-sc-${ARM}-stack-r10${SHAPE_TAG}"
RESULTS_DIR=${WORKSPACE_DIR}/results/${EXP_NAME}
BASE_LOG_DIR=${WORKSPACE_DIR}/ray_logs/${EXP_NAME}
WANDB_PROJ=nano-35-rlvr

# ---- Slurm shape: 64 nodes on short QOS --------------------------------------
# short QOS: node cap 64, MaxWall 2h, priority 200 (vs normal's 100) — trades
# run depth for scheduling speed. The 4 nodes come out of the NL2Bash judge
# pool (8xTP4 -> 4xTP4): generation is the throughput bottleneck and stays at
# 40 nodes; NL2Bash is a latency resource whose queueing the 640-group
# oversubscription hides unless DP4 saturates (watch: steps 2+ slow while
# engine num_requests_waiting is 0 -> rerun with NL2BASH_REPLICAS=6).
# Total: 56 hetgroup0 (8 train + 40 gen + 8 gym) + 8 hetgroup1 (GenRM 2xTP8
# + NL2Bash 4xTP4) = 64.
SLURM_PARTITION=batch
SLURM_ACCOUNT=nemotron_sw_post
# Defaults give the 64-node short-QOS shape. For the original 68-node shape:
#   SLURM_QOS=normal WALLTIME=4:00:00 NL2BASH_REPLICAS=8 NRL_MAX_STEPS=10 \
#   SHAPE_TAG=-68n JOB_REAPER_EXEMPT_MINS=360 ARM=... bash nano35_full_perf_launch.sh
SLURM_QOS="${SLURM_QOS:-short}"
GPUS_PER_NODE=4          # GB200 NVL72
WALLTIME="${WALLTIME:-2:00:00}"
EXTERNAL_JUDGES=1
NL2BASH_REPLICAS="${NL2BASH_REPLICAS:-4}"

# Reaper exemption must exceed the FULL walltime: the gym nodes hold 31 idle
# GPUs for the whole run and the train nodes idle through every rollout-bound
# phase, so any idle threshold below the wall kills the job mid-step. r2 of
# this campaign (job 6521428) died exactly this way at 86 min: ultra_launch.sh
# hardcoded exemptIdleTimeMins=60 and ignored this variable until the
# campaign-branch patch that makes the batch sbatch paths read it.
JOB_REAPER_EXEMPT_MINS="${JOB_REAPER_EXEMPT_MINS:-240}"

# Whole run must fit one allocation (no checkpointing on this tree). 5 is
# greedy against the 2h wall on purpose: a wall-kill mid-step-5 loses nothing
# (per-step metrics are already logged; steps 2-4 carry the steady-state A/B).
NRL_MAX_STEPS="${NRL_MAX_STEPS:-5}"

USE_SNAPSHOT=0           # live worktree, same as the smoke A/B

# ---- launch plumbing proven by the 0820/0824 ledger smokes --------------------
NRL_FORCE_REBUILD_VENVS=false
NRL_FORCE_REBUILD_VENVS_LIST="nemo_rl.environments.nemo_gym.NemoGym,nemo_rl.models.generation.vllm.vllm_worker_async.VllmAsyncGenerationWorker"
NRL_DRIVER_PYTHONPATH="/opt/nemo-rl/3rdparty/Gym-workspace/Gym"
NRL_DRIVER_PIP_INSTALL="orjson"
NRL_DRIVER_UV_RUN_FLAGS="--locked --no-sync"
NRL_TQ_SKIP_RUNTIME_ENV_PIN=1
NRL_VENV_SYNC_FROZEN=1
NRL_WG_USE_RAY_REF=1
NRL_REFIT_ERRORS_FATAL=1
# Baked node-local cache, NOT the shared Lustre cache: uv revalidates and
# rewrites the local project's (file:///opt/nemo-rl) cache entry on every
# invocation, so a Lustre cache shared by 56 nodes x 2 builders races on
# that one entry regardless of pre-warming (falsified by job 6596222's
# serial pre-warm; jobs 6578674..6596222 all died here). Explicit because
# ultra_launch.sh otherwise falls back to an EMPTY /tmp cache per job.
UV_CACHE_DIR=/root/.cache/uv

DRY_RUN="${_DRY_RUN_IN:-1}"
set +a

echo "================================================================"
echo "  full_perf_run arm: ${ARM}  (steps=${NRL_MAX_STEPS}, exp=${EXP_NAME})"
echo "================================================================"

# checkpointing.enabled=false: mandatory on this tree (setup.py:388 raises).
# rollout_checkpointing.restore_mode=none: extra-allowed/inert block here,
# kept identical to the smoke invocation for command fidelity.
exec bash "${HERE}/examples/nemo_gym/nemotron-3.5-nano/nano35_dolphin_launch_sc.sh" \
  rollout_checkpointing.restore_mode=none \
  checkpointing.enabled=false \
  "${_ARM_OVERRIDES[@]}" \
  "$@"
