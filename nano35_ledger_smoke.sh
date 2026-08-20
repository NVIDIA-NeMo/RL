#!/bin/bash
# =============================================================================
# nano35_ledger_smoke.sh — BATCH rerun of the nano-35-rlvr-sc-tc-smoke shape
# (failed job 6300221) on the lineage-ledger token-capture code.
#
# Purpose: regression validation of the multi-worker capture path. Job 6300221
# died at the first rollout batch because the old per-process gate registry
# (register-before-dispatch) split across policy_model num_workers=16 /
# policy_model_reasoning_off num_workers=4 — 409 Conflict on unregistered
# workers, then UnknownRolloutError from ingest_coords mid-flight. The ledger
# replaced that with a process-shared FileLineageStore, so the SAME shape with
# num_workers deliberately left at 16/4 must now pass.
#
# Deviations from the original run, both deliberate:
#   - container: zhiyul nightly-gym.2026-08-10 (the bake this branch's launch
#     fixes were validated against), not amahishi's nightly-gym squashfs.
#   - paths/W&B/secrets: pthombre-owned.
# Everything else (node shape, judges, batch geometry, walltime, QOS, model,
# blend, num_workers defaults) mirrors job 6300221.
#
# Run from a NETWORKED shell at the repo root (fsw twin path):
#     DRY_RUN=1 bash nano35_ledger_smoke.sh          # inspect only
#     DRY_RUN=0 bash nano35_ledger_smoke.sh          # submit
# =============================================================================
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

_DRY_RUN_IN="${DRY_RUN:-}"

set -a
# Secrets (WANDB_API_KEY, HF_TOKEN): untracked mode-600 file, never in git.
# shellcheck disable=SC1091
source "${HERE}/swe_nano.secrets.env"

# ---- container / shared read-only assets ------------------------------------
CONTAINER=/lustre/fsw/portfolios/llmservice/users/zhiyul/enroot-images/nvcr.io+nvidian+nemo-rl+nightly-gym.2026-08-10.squashfs
SANDBOX_CONTAINER=/lustre/fsw/portfolios/coreai/users/cye/enroot/nemo-rl:skills-sandbox-latest.squashfs
# Model/blend/judge paths are baked into the dolphin launcher + recipe
# defaults (all verified readable); HF_HUB_CACHE keeps reading zhiyul's
# already-downloaded hub shards.
HF_HUB_CACHE=/lustre/fsw/portfolios/llmservice/users/zhiyul/hf_cache/hub

# ---- per-user write paths ----------------------------------------------------
WORKSPACE_DIR=/lustre/fsw/portfolios/llmservice/users/pthombre/sweRun/RL-pr3456-delta-smoke-workspace
HF_HOME=/lustre/fsw/portfolios/llmservice/users/pthombre/hf_cache
PERSISTENT_CACHE=/lustre/fsw/portfolios/llmservice/users/pthombre/persistent_cache
NRL_MEGATRON_CHECKPOINT_DIR=${PERSISTENT_CACHE}/megatron_ckpt_cache

EXP_NAME="${SC_EXP_NAME:-nano35-rlvr-sc-tc-smoke-ledger}"
RESULTS_DIR=${WORKSPACE_DIR}/results/${EXP_NAME}
BASE_LOG_DIR=${WORKSPACE_DIR}/ray_logs/${EXP_NAME}
WANDB_PROJ=nano-35-rlvr

# ---- Slurm shape: exactly job 6300221 ----------------------------------------
SLURM_PARTITION=batch
SLURM_ACCOUNT=nemotron_sw_post
SLURM_QOS=short
GPUS_PER_NODE=4          # GB200 NVL72
WALLTIME=2:00:00
# Smoke wrapper defaults reproduce the rest: 4 train + 1 gen + 1 gym nodes,
# EXTERNAL_JUDGES=1 (GenRM 1xTP4 + NL2Bash 1xTP4 hetgroup), 8 prompts x 16
# generations, NRL_MAX_STEPS=3, STREAM_MIN_GROUPS=2, NUM_STORAGE_UNITS=2.
# num_workers stays at the recipe's 16/4 — the regression trigger under test.

USE_SNAPSHOT=0           # live worktree, same as the original run

# ---- launch plumbing proven by the 0820 ledger A/B smokes --------------------
# (swe_nano.env / swe_nano_sc_capture.sh; see session/20260820_004547)
NRL_FORCE_REBUILD_VENVS=false
NRL_FORCE_REBUILD_VENVS_LIST="nemo_rl.environments.nemo_gym.NemoGym,nemo_rl.models.generation.vllm.vllm_worker_async.VllmAsyncGenerationWorker"
NRL_DRIVER_PYTHONPATH="/opt/nemo-rl/3rdparty/Gym-workspace/Gym"
NRL_DRIVER_PIP_INSTALL="orjson"
NRL_DRIVER_UV_RUN_FLAGS="--locked --no-sync"
NRL_TQ_SKIP_RUNTIME_ENV_PIN=1
NRL_VENV_SYNC_FROZEN=1
NRL_WG_USE_RAY_REF=1
NRL_REFIT_ERRORS_FATAL=1
# Shared prewarmed uv cache — NEVER /tmp; and never UV_CACHE_DIR_OVERRIDE with
# prefetch-venvs containers (it severs the baked venvs' hardlinks).
UV_CACHE_DIR=/lustre/fsw/portfolios/llmservice/users/pthombre/uv

DRY_RUN="${_DRY_RUN_IN:-1}"
set +a

# Same trailing override as the original invocation. rollout_checkpointing is
# an extra-allowed (ignored) block in this tree; kept for command fidelity.
# checkpointing.enabled=false: this tree's SingleController raises
# NotImplementedError on trainer checkpointing (the recovery-refresh branch
# supports it); irrelevant to the multi-worker capture path under test.
exec bash "${HERE}/examples/nemo_gym/nemotron-3.5-nano/nano35_dolphin_launch_sc_smoke.sh" \
  rollout_checkpointing.restore_mode=none \
  checkpointing.enabled=false \
  "$@"
