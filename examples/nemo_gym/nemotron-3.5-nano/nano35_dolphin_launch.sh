#!/bin/bash
set -euo pipefail

# =============================================================================
# nano35_dolphin_launch.sh
#
# Nemotron 3.5 Nano — RLVR, legacy async-1, honest-dolphin warm start.
# Reproduces the internal reference run
#   geshen-ultra-rl-nano-honest-dolphin-v10-iter6000-mopd-rlvr
# (launch_nano_honest_dolphin.sh + grpo_ultra_512n4g_bf16.yaml on
#  nemo-rl-internal @ 97c55ee2) on public NeMo-RL main.
#
# This is a thin wrapper over examples/nemo_gym/nemotron-3-ultra/ultra_launch.sh.
# That launcher is fully parameterised and already handles code snapshotting,
# persistent-cache seeding, container mounts, Ray/Gym orchestration, and the
# OccupiedIdleGPUsJobReaper --comment exemption — so we set environment and
# delegate rather than forking 800+ lines.
#
# Usage (legacy async-1 with an already-running GenRM):
#   GENRM_BASE_URL=http://<lb-host>:9213/v1 \
#     bash examples/nemo_gym/nemotron-3.5-nano/nano35_dolphin_launch.sh
#
#   DRY_RUN=1 GENRM_BASE_URL=... bash .../nano35_dolphin_launch.sh   # inspect only
#
# Extra Hydra overrides are forwarded verbatim:
#   GENRM_BASE_URL=... bash .../nano35_dolphin_launch.sh grpo.max_num_steps=2
#
# Set EXTERNAL_JUDGES=1 to let PR 3511 launch GenRM and NL2Bash in a separate
# Slurm heterogeneous component instead. In that mode GENRM_BASE_URL must be
# unset; the launcher discovers both load-balancer URLs inside the allocation.
# =============================================================================

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "${REPO_ROOT}"   # ultra_launch.sh derives PROJECT_ROOT from $PWD

# -----------------------------------------------------------------------------
# GenRM deployment.
#
# Legacy async-1 consumes an already-running GenRM through GENRM_BASE_URL.
# SingleController V2 sets EXTERNAL_JUDGES=1 and uses PR 3511's heterogeneous
# service component for both GenRM and NL2Bash.
#
# Stand it up first (copy the dir — it holds .lb_pid_*, logs/ and a flock'd
# registry, so running geshen's in place would collide with their pool):
#
#   cp -r /lustre/fs1/portfolios/llmservice/projects/llmservice_modelalignment_ppo/\
# users/geshen/mopd_nano_fast/genrm_serving  <your-dir>/genrm_serving
#   cd <your-dir>/genrm_serving
#   MODEL=/lustre/fsw/portfolios/llmservice/users/ansubramania/models/qwen235b_principle_comparison_genrm_step1230 \
#   ACCOUNT=nemotron_sw_post PARTITION=batch_long TIME=1-12:00:00 \
#   LB_PORT=9213 GENRM_GROUP_ID=nano35_dolphin \
#     ./genrm_server_manager.sh launch N
#   ./genrm_server_manager.sh url
#
# NOTE: that script's default MODEL is the *ultra* GenRM (step_720) — override it.
# Each worker is 2 nodes x 4 GPUs at TP=8, separate from this job's 64 nodes.
#
# The `model` field must equal the pool's --served-model-name ("model" in
# genrm_worker.sh). ultra_launch.sh sets base_url XOR model, never both, so the
# name is pinned in rlvr_dolphin.yaml instead of passed here.
# -----------------------------------------------------------------------------
export EXTERNAL_JUDGES="${EXTERNAL_JUDGES:-0}"
if [[ "${EXTERNAL_JUDGES}" == "1" ]]; then
  if [[ -n "${GENRM_BASE_URL:-}" ]]; then
    echo "ERROR: unset GENRM_BASE_URL when EXTERNAL_JUDGES=1; PR 3511 launches a fresh GenRM pool." >&2
    exit 1
  fi
  export GENRM_MODEL="${GENRM_MODEL:-/lustre/fsw/portfolios/llmservice/users/ansubramania/models/qwen235b_principle_comparison_genrm_step1230}"
else
  : "${GENRM_BASE_URL:?GENRM_BASE_URL must point to the external GenRM /v1 endpoint}"
  export GENRM_BASE_URL
  unset GENRM_MODEL
fi

# -----------------------------------------------------------------------------
# Experiment identity
# EXP_NAME drives the W&B run name, the singleton job name, and the checkpoint
# and log dirs — so changing it starts a *new* run rather than resuming.
# -----------------------------------------------------------------------------
export EXP_NAME="${EXP_NAME:-akamehra-nano35-v1-async1-n64-t8-g40-gym16-tp4_cp4_ep8-gpp16-pps128-gbs2048}"
export CONFIG_PATH="${CONFIG_PATH:-examples/nemo_gym/nemotron-3.5-nano/rlvr_dolphin.yaml}"

# Pin the legacy async-1 driver explicitly. The SC V2 wrapper overrides this
# before delegating to this shared Nano site launcher.
export TRAIN_ENTRYPOINT="${TRAIN_ENTRYPOINT:-./examples/nemo_gym/run_grpo_nemo_gym.py}"

# -----------------------------------------------------------------------------
# Model and data
# -----------------------------------------------------------------------------
# honest-dolphin SFT v10 (closethink unmask, from midtrain 100B LC), iter_0006000.
export MODEL_PATH="${MODEL_PATH:-/lustre/fsw/portfolios/llmservice/users/venkats/training_actual_0603/nano_n3_post/checkpoints/nano-3.5-sft-v10-closethink-unmask-orig6k-from-midtrain-100B-lc-lr2e-5/eval/iter_0006000/hf}"

# trusty_viper: 199,680 prompts / 24 agent families, carrying agent_ref per row.
# VAL_PATH intentionally equals TRAIN_PATH, as in the reference — validation is
# effectively disabled (grpo.val_period is very large) because the genrm cohort
# envs are train-only and would hang under eval.
_BLEND="${_BLEND:-/lustre/fs1/portfolios/llmservice/projects/llmservice_modelalignment_ppo/users/geshen/rl-data-tools/blends/curriculum_honest_dolphin_v41_trusty_viper.train.jsonl}"
export TRAIN_PATH="${TRAIN_PATH:-${_BLEND}}"
export VAL_PATH="${VAL_PATH:-${_BLEND}}"

# -----------------------------------------------------------------------------
# Judge models. With EXTERNAL_JUDGES=1, PR 3511 serves GenRM and NL2Bash in the
# external-service hetgroup; Safety remains in the Gym pool.
# -----------------------------------------------------------------------------
export NL2BASH_JUDGE_MODEL="${NL2BASH_JUDGE_MODEL:-/lustre/fsw/portfolios/llmservice/users/ansubramania/models/Qwen3-235B-A22B-Instruct-2507-FP8}"
export SAFETY_JUDGE_MODEL="${SAFETY_JUDGE_MODEL:-/lustre/fsw/portfolios/llmservice/users/ansubramania/super_v3/model_checkpoints/Nemotron-Content-Safety-Reasoning-4B}"

# -----------------------------------------------------------------------------
# Containers
# PR 3511 requires the vLLM 0.25.1 environment in this image. Using the older
# V1 July image with the PR's mounted nemo_rl code fails at import time before
# either the legacy or SC driver can start.
# -----------------------------------------------------------------------------
export CONTAINER="${CONTAINER:-/lustre/fsw/portfolios/coreai/users/yifuw/enroot-images/gitlab-master.nvidia.com/yifuw/images/nemo-rl:nightly-20260806-sandbox.squashfs}"

# -----------------------------------------------------------------------------
# Sandbox process DISABLED — this is what killed jobs 5726250 and 5732681.
#
# ray.sub launches the nemo-skills sandbox on every node with
# `--kill-on-bad-exit=1` (ray.sub:967), unlike the Ray worker step which uses
# `--kill-on-bad-exit=0` (ray.sub:1016). So a single node failing during sandbox
# startup makes srun tear down all 64 sandbox tasks; ray.sub then sees its
# background sandbox srun die and exits. At 64 nodes that happened on 3 of 3
# attempts, on different nodes each time (nvl72133-T01, nvl72141-T07,
# nvl72126-T05/T17) — node exclusion cannot fix it.
#
# Only three Gym servers use the sandbox: competitive_coding_challenges (not in
# our blend), math_formal_lean and ns_tools. Both of the latter are in
# config_paths and are LEFT THERE deliberately: they construct their sandbox
# client lazily (math_formal_lean/app.py:387 builds it, :444 uses it inside a
# request handler; ns_tools only holds sandbox_host/port as config), so they
# boot fine without a sandbox and only contact it if a request routes to them.
# The dolphin blend routes to neither, so the sandbox is never needed.
#
# PR 3511 exposes NO_COLOCATED_SANDBOX for exactly this case. It clears both
# SANDBOX_CONTAINER and SANDBOX_COMMAND before ray.sub builds the allocation,
# skipping the ports directory, readiness gate, and sandbox srun.
#
# Side benefit: no 16 GB sandbox image extracted on 64 nodes, so faster startup.
# To re-enable (e.g. if a future blend uses Lean4), set
# NO_COLOCATED_SANDBOX=0 and supply SANDBOX_CONTAINER.
# -----------------------------------------------------------------------------
export NO_COLOCATED_SANDBOX="${NO_COLOCATED_SANDBOX:-1}"

# -----------------------------------------------------------------------------
# Caches
# PERSISTENT_CACHE must be set explicitly: ultra_launch.sh requires it, and the
# internal reference's derivation (/lustre/fsw/portfolios/${ACCOUNT%%_*}/users/$USER)
# would resolve to /lustre/fsw/portfolios/nemotron/... which is read-only for us.
# HF_HOME is a *sibling* of the cache, not inside it, because the launcher purges
# vllm_compile_cache* under PERSISTENT_CACHE on every submission.
# HF_HOME also decides where the HF->Megatron conversion of the 62 GB checkpoint
# lands (get_megatron_checkpoint_dir falls back to $HF_HOME/nemo_rl), so keeping
# it on Lustre means the conversion is done once, not once per job.
# -----------------------------------------------------------------------------
export PERSISTENT_CACHE="${PERSISTENT_CACHE:-/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_llm/users/akamehra/.cache/nano35-dolphin}"
export HF_HOME="${HF_HOME:-/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_llm/users/akamehra/hf_home}"

# -----------------------------------------------------------------------------
# Container mounts — REQUIRED.
# ultra_launch.sh starts MOUNTS empty and only appends three source overlays
# (nemo_rl, examples/configs, Gym). It never mounts /lustre, so without this the
# container cannot see the checkpoint, the blend jsonl, the judge models,
# HF_HOME or PERSISTENT_CACHE. The internal reference hardcoded this mount.
# -----------------------------------------------------------------------------
export MOUNTS="${MOUNTS:-/lustre:/lustre}"

# -----------------------------------------------------------------------------
# Do not write bytecode into the Lustre-mounted source.
#
# nemo_rl is bind-mounted from Lustre into every container. Without this, all 64
# nodes write .pyc back into that shared tree (194 files appeared during earlier
# runs, including generation/__pycache__/interfaces.cpython-313.pyc). That is
# metadata churn on the exact directories every node is importing from.
#
# Job 5742619 died when Ray unpickled VllmAsyncGenerationWorker on one node:
# __init__.py executed from the mount, then its sibling interfaces.py was not
# found — a per-node directory-view inconsistency, not a missing file. Reads
# alone are far safer than reads plus concurrent writes.
# -----------------------------------------------------------------------------
export PYTHONDONTWRITEBYTECODE="${PYTHONDONTWRITEBYTECODE:-1}"

# -----------------------------------------------------------------------------
# examples/nemo_gym mount — REQUIRED, and the reason job 5730369 died with
#   FileNotFoundError: /opt/nemo-rl/examples/nemo_gym/nemotron-3.5-nano/rlvr_dolphin.yaml
#
# ultra_launch.sh overlays only nemo_rl, examples/configs and Gym. Everything
# else under /opt/nemo-rl — including examples/nemo_gym, where this recipe and
# the nemotron-3-ultra base config live — comes from the container image.
# Two things are therefore invisible without this mount:
#   1. rlvr_dolphin.yaml itself (written here, never in any image), and
#   2. nemotron-3-ultra/student_rlvr1.yaml, which it inherits — that landed on
#      main in 64cb9f985 (Jul 29-30) but this image was built Jul 26, so the
#      image genuinely does not contain it (verified by inspecting the image).
# Mounting the directory fixes both, and also removes a version skew: nemo_rl
# would otherwise come from this checkout while examples/nemo_gym came from a
# Jul-26 image.
#
# Scoped to this one directory rather than mounting the repo root over
# /opt/nemo-rl, to avoid shadowing anything the image builds in place.
# -----------------------------------------------------------------------------
_NEMO_GYM_MOUNT="${REPO_ROOT}/examples/nemo_gym:/opt/nemo-rl/examples/nemo_gym"
if [[ -n "${EXTRA_MOUNTS:-}" ]]; then
  export EXTRA_MOUNTS="${EXTRA_MOUNTS},${_NEMO_GYM_MOUNT}"
else
  export EXTRA_MOUNTS="${_NEMO_GYM_MOUNT}"
fi

# -----------------------------------------------------------------------------
# Snapshotting is OFF because tools/code_snapshot.sh copies only *git-tracked*
# files, and this recipe is untracked (`?? examples/nemo_gym/nemotron-3.5-nano/`).
# With snapshotting on, the mount above would point into a snapshot that does
# not contain the config. To restore frozen provenance, `git add` the recipe and
# set USE_SNAPSHOT=1.
# -----------------------------------------------------------------------------
export USE_SNAPSHOT="${USE_SNAPSHOT:-0}"

# -----------------------------------------------------------------------------
# Results root — MUST be absolute.
# ultra_launch.sh defaults RESULTS_DIR to the relative "results/${EXP_NAME}".
# The host mkdir would land it in the repo, but TRAIN_CMD does `cd /opt/nemo-rl`
# inside the container, so the same relative string resolves to
# /opt/nemo-rl/results/... — the container's ephemeral overlay. Checkpoints
# would vanish at job end and the singleton auto-resume would never find them,
# so on a 4 h wall the run would restart from the SFT checkpoint forever.
# An absolute Lustre path fixes checkpoints, logs, ray_logs and slurm output.
# -----------------------------------------------------------------------------
export RESULTS_DIR="${RESULTS_DIR:-/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_llm/users/akamehra/runs/${EXP_NAME}}"

# -----------------------------------------------------------------------------
# SLURM
# Validated V1 shape: 8 train + 40 gen + 16 gym = 64 GB200 nodes
# (4 GPUs each). SEGMENT_SIZE=2 is the nano value; ultra defaults to 16.
# Partition `batch` caps at 4 h (batch_long is 7 d), so WALLTIME is 4 h and
# CHECKPOINTING_SAVE_BY keeps the reference's 25-minute teardown margin.
# -----------------------------------------------------------------------------
export SLURM_ACCOUNT="${SLURM_ACCOUNT:-nemotron_sw_post}"
export SLURM_PARTITION="${SLURM_PARTITION:-batch}"
export WALLTIME="${WALLTIME:-4:00:00}"
# Async rollout collection can leave the training GPUs idle while judges work.
# Match the launcher's 60-minute idle-GPU reaper exemption.
export JOB_REAPER_EXEMPT_IDLE_MINS="${JOB_REAPER_EXEMPT_IDLE_MINS:-60}"
export CHECKPOINTING_SAVE_BY="${CHECKPOINTING_SAVE_BY:-00:03:35:00}"
export NUM_TRAIN_NODES="${NUM_TRAIN_NODES:-8}"
export NUM_GEN_NODES="${NUM_GEN_NODES:-40}"
export NUM_GYM_NODES="${NUM_GYM_NODES:-16}"
export SEGMENT_SIZE="${SEGMENT_SIZE:-2}"

# -----------------------------------------------------------------------------
# W&B. WANDB_API_KEY must already be in the environment — ultra_launch.sh needs
# it. If it is exported only from ~/.zshrc, submit from zsh; a bash context
# will not see it.
# -----------------------------------------------------------------------------
export WANDB_PROJ="${WANDB_PROJ:-ultra-streaming}"
export WANDB_ENTITY="${WANDB_ENTITY:-joc}"

# MTP: head *training* is on via the config (5 repeated layers, loss 0.3,
# detached heads), matching the reference. MTP *speculative decoding* for vLLM
# is a separate, independent switch and is off, also matching the reference.
export ENABLE_MTP_INFERENCE="${ENABLE_MTP_INFERENCE:-0}"

echo "================================================================"
echo "  Nemotron 3.5 Nano — RLVR async-1 (honest-dolphin)"
echo "================================================================"
echo "  Experiment : ${EXP_NAME}"
echo "  Config     : ${CONFIG_PATH}"
echo "  Model      : ${MODEL_PATH}"
echo "  Blend      : ${TRAIN_PATH}"
echo "  Container  : ${CONTAINER}"
echo "  Cache      : ${PERSISTENT_CACHE}"
echo "  HF_HOME    : ${HF_HOME}"
if [[ "${EXTERNAL_JUDGES}" == "1" ]]; then
echo "  Judges     : GenRM + NL2Bash in PR 3511 external-service hetgroup"
else
echo "  GenRM      : ${GENRM_BASE_URL} (existing external service; model: model)"
fi
echo "  SLURM      : ${SLURM_ACCOUNT} / ${SLURM_PARTITION} / ${WALLTIME}"
echo "  Reaper     : ${JOB_REAPER_EXEMPT_IDLE_MINS} min idle exemption"
echo "  Nodes      : ${NUM_TRAIN_NODES} train + ${NUM_GEN_NODES} gen + ${NUM_GYM_NODES} gym"
echo "  W&B        : ${WANDB_ENTITY}/${WANDB_PROJ}"
echo "================================================================"
echo ""

exec bash examples/nemo_gym/nemotron-3-ultra/ultra_launch.sh "$@"
