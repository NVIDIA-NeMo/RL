#!/bin/bash
# =============================================================================
# swe_nano_sc.sh — BATCH (non-interactive) launch of the nano SWE
# SingleController + TransferQueue recipe, e2e, on `main`. Adapted from
# branch zhiyul/swe_tq's swe_nano_sc.sh. ray.sub runs the driver directly
# (no interactive idle/attach).
#
# Mirrors docs/guides/nano-swe-transferqueue.md's "Unattended reproduction,
# one command" path. See NANO_SWE_TQ_MAINPORT_NOTES.md for the full writeup
# of what changed vs. the branch and why.
#
# Usage (run from a real/networked shell — NOT through a sandboxed tool):
#   bash swe_nano_sc.sh                                   # DRY_RUN=1 default: inspect first
#   DRY_RUN=0 bash swe_nano_sc.sh                         # submit
#   DRY_RUN=0 bash swe_nano_sc.sh grpo.num_prompts_per_step=2 \
#     policy.train_global_batch_size=8 grpo.max_num_steps=5
#
# Known landmine: main's tracked ray.sub hardcodes `#SBATCH --time=1:0:0` at
# line 7, which silently overrides a CLI --time= passed to sbatch on this
# cluster (verified: a job died at 60 min despite --time=4:00:00). ray.sub is
# shared/tracked, so we do NOT edit it in place — instead this launcher
# points RAY_SUB at ray.sub.nano-swe, a copy with only that one line changed
# (matches WALLTIME below).
#
# Logs land in ${WORKSPACE_DIR}/results/${EXP_NAME}/runs/latest/slurm/.
# =============================================================================
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Capture DRY_RUN passed on the command line BEFORE sourcing swe_nano.env,
# which exports DRY_RUN=1 and would otherwise clobber the caller's value.
_DRY_RUN_IN="${DRY_RUN:-}"

set -a
# shellcheck disable=SC1091
source "${HERE}/swe_nano.env"

# --- SingleController + TransferQueue overrides ------------------------------
EXP_NAME=nano-swe-sc-tq-mainport
NRL_ENTRYPOINT="${CODE_DIR}/examples/run_grpo_single_controller.py"
CONFIG_PATH="${CODE_DIR}/examples/configs/ultra/nano_swe_teacher_sc.yaml"
RESULTS_DIR="${WORKSPACE_DIR}/results/${EXP_NAME}"
BASE_LOG_DIR="${WORKSPACE_DIR}/ray_logs/${EXP_NAME}"

# ray.sub's own #SBATCH --time is unreliable via CLI override on this cluster
# (see header note above) -- point at the time-fixed copy instead of
# ultra_launch.sh's default (${PROJECT_ROOT}/ray.sub).
RAY_SUB="${HERE}/ray.sub.nano-swe"

[ -n "${_DRY_RUN_IN}" ] && DRY_RUN="${_DRY_RUN_IN}"
set +a

# DELIBERATELY NOT overriding env.nemo_gym.uv_venv_dir (tried it, reverted --
# see below). Gym's own default (nemo_gym/global_config.py:714,
# `global_config_dict.setdefault(UV_VENV_DIR_KEY_NAME, str(WORKING_DIR))`,
# where WORKING_DIR == PARENT_DIR for an editable install -- confirmed via
# nemo_gym/__init__.py) places each server's venv at
# <gym-repo>/<server_dir>/.venv, e.g.
# 3rdparty/Gym-workspace/Gym/responses_api_agents/swe_agents/.venv --
# INSIDE the Gym submodule tree, which EXTRA_MOUNTS already mounts
# read-write from shared Lustre onto every node identically. So the
# cross-node-visibility problem this override was meant to solve doesn't
# exist in this setup to begin with -- no override needed, and this matches
# the branch's own guide/launcher, which never sets uv_venv_dir either.
#
# History, for whoever reads this next: job 6278584 crashed with
# `ConfigAttributeError: Key 'uv_venv_dir' is not in struct` on a plain
# `env.nemo_gym.uv_venv_dir=...` override (fixed with `+env....=`), and job
# 6279915 then crashed on a stale venv at a leftover June-30 cache dir
# (fixed by pointing at a fresh dir, job 6282910). 6282910 then stalled 15+
# min with 0/3 Gym servers ready building into that fresh custom dir. Rather
# than keep patching a custom uv_venv_dir, dropping the override entirely
# removes the whole problem class -- confirmed no stale content sitting at
# the default path either (checked, doesn't exist yet).
exec bash "${HERE}/ultra_launch.sh" \
  "$@"
