#!/bin/bash
# =============================================================================
# swe_nano_sc.sh — BATCH launch of the nano SWE SingleController + TransferQueue
# recipe (ray.sub runs the driver directly; no interactive idle, no attach).
#
# Same 6-node shape, entrypoint and config as swe_nano_sc_interactive.sh — the
# driver command ray.sub runs is byte-for-byte the one the interactive path
# writes to <jobid>-run-cmd.sh. Use this for an unattended reproduction; use the
# interactive script when you expect to iterate on the config, since a cold
# start pays the ~60 GB checkpoint download and Megatron conversion every time.
#
# Run from a NETWORKED shell at the repo root:
#     bash swe_nano_sc.sh                     # DRY_RUN inherited from swe_nano.env (=1): inspect first
#     DRY_RUN=0 bash swe_nano_sc.sh           # submit
#     DRY_RUN=0 bash swe_nano_sc.sh grpo.num_prompts_per_step=2 policy.train_global_batch_size=8
#
# Logs land in ${WORKSPACE_DIR}/results/${EXP_NAME}/runs/latest/slurm/.
# =============================================================================
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Capture DRY_RUN passed on the command line BEFORE sourcing swe_nano.env, which
# exports DRY_RUN=1 and would otherwise clobber the caller's value.
_DRY_RUN_IN="${DRY_RUN:-}"

set -a
# shellcheck disable=SC1091
source "${HERE}/swe_nano.env"

# --- SingleController + TransferQueue overrides ------------------------------
EXP_NAME="${SC_EXP_NAME:-nano-swe-sc-tq}"
NRL_ENTRYPOINT="${CODE_DIR}/examples/run_grpo_single_controller.py"
CONFIG_PATH="${CODE_DIR}/examples/configs/ultra/nano_swe_teacher_sc.yaml"
RESULTS_DIR="${WORKSPACE_DIR}/results/${EXP_NAME}"
BASE_LOG_DIR="${WORKSPACE_DIR}/ray_logs/${EXP_NAME}"
[ -n "${_DRY_RUN_IN}" ] && DRY_RUN="${_DRY_RUN_IN}"
set +a

bash "${HERE}/ultra_launch.sh" "$@"
