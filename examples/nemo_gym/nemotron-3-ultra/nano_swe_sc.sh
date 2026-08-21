#!/bin/bash
# =============================================================================
# nano_swe_sc.sh — BATCH launch of the nano SWE SingleController + TransferQueue
# recipe (ray.sub runs the driver directly; no interactive idle, no attach).
#
# Same 6-node shape, entrypoint and config as nano_swe_sc_interactive.sh — the
# driver command ray.sub runs is the one the interactive path writes to
# <jobid>-run-cmd.sh. Use this for an unattended reproduction; use the
# interactive script when you expect to iterate on the config, since a cold
# start pays the ~60 GB checkpoint download and Megatron conversion every time.
#
# Run from a NETWORKED shell at the repo root:
#     bash examples/nemo_gym/nemotron-3-ultra/nano_swe_sc.sh              # DRY_RUN=1: print the driver command and exit
#     DRY_RUN=0 bash examples/nemo_gym/nemotron-3-ultra/nano_swe_sc.sh    # submit
#     DRY_RUN=0 bash examples/nemo_gym/nemotron-3-ultra/nano_swe_sc.sh \
#         grpo.num_prompts_per_step=2 policy.train_global_batch_size=8    # reach a train step fast
#
# Keep the invariant num_prompts_per_step × num_generations_per_prompt ==
# train_global_batch_size — the SingleController split path enforces it.
#
# Logs land in ${WORKSPACE_DIR}/results/${EXP_NAME}/runs/latest/slurm/.
# =============================================================================
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Capture DRY_RUN from the command line BEFORE sourcing nano_swe.env.
_DRY_RUN_IN="${DRY_RUN:-}"

set -a
# shellcheck disable=SC1091
source "${HERE}/nano_swe.env"

# --- SingleController + TransferQueue overrides ------------------------------
EXP_NAME="${EXP_NAME:-nano-swe-sc-tq}"
# Absolute paths: ultra_launch.sh overlays only nemo_rl/ and examples/configs/,
# so neither this config nor this entrypoint exists in the container image.
NRL_ENTRYPOINT="${CODE_DIR}/examples/run_grpo_single_controller.py"
CONFIG_PATH="${CODE_DIR}/examples/nemo_gym/nemotron-3-ultra/nano_swe_teacher_sc.yaml"
RESULTS_DIR="${WORKSPACE_DIR}/results/${EXP_NAME}"
BASE_LOG_DIR="${WORKSPACE_DIR}/ray_logs/${EXP_NAME}"
[ -n "${_DRY_RUN_IN}" ] && DRY_RUN="${_DRY_RUN_IN}"
set +a

_nano_swe_preflight

bash "${HERE}/ultra_launch.sh" "$@"
