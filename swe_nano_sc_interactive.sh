#!/bin/bash
# =============================================================================
# swe_nano_sc_interactive.sh — nano SWE via SingleController async GRPO with a
# HONOURED TransferQueue data plane. This is the VERIFIED recipe (see
# docs/guides/nano-swe-transferqueue.md).
#
# Same 6-node nano shape as swe_nano_interactive.sh (train 4 / gen 2), but:
#   entrypoint  examples/run_grpo_single_controller.py   (NOT run_grpo_nemo_gym.py)
#   config      examples/configs/ultra/nano_swe_teacher_sc.yaml
#
# run_grpo_nemo_gym.py's async path (async_grpo_train) drives an in-memory
# ReplayBuffer and ignores data_plane; only grpo_train_sync and the
# SingleController path put rollouts through TransferQueue.
#
# The entrypoint is spawned by ABSOLUTE path from ${CODE_DIR}: the container
# only has nemo_rl/ and examples/configs mounted over it, so the baked
# examples/ has no run_grpo_single_controller.py.
#
# Run from a NETWORKED shell (the sandbox cannot reach slurmctld):
#     bash swe_nano_sc_interactive.sh
#
# On submit it prints:
#     bash <jobid>-attach.sh        # shell on the head node (Ray already up)
#     source <jobid>-run-cmd.sh     # run the driver; edit + re-source to iterate
# Cancel with: scancel <jobid>
#
# Extra hydra overrides pass through, e.g. a fast first training step:
#     bash swe_nano_sc_interactive.sh grpo.num_prompts_per_step=2 policy.train_global_batch_size=8
# Keep the invariant num_prompts_per_step × num_generations_per_prompt ==
# train_global_batch_size — the SingleController split path enforces it.
# =============================================================================
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

set -a
# shellcheck disable=SC1091
source "${HERE}/swe_nano.env"

# --- override for the SingleController + TransferQueue variant ---------------
EXP_NAME=nano-swe-sc-tq-zhiyul
NRL_ENTRYPOINT="${CODE_DIR}/examples/run_grpo_single_controller.py"
CONFIG_PATH="${CODE_DIR}/examples/configs/ultra/nano_swe_teacher_sc.yaml"
RESULTS_DIR="${WORKSPACE_DIR}/results/${EXP_NAME}"
BASE_LOG_DIR="${WORKSPACE_DIR}/ray_logs/${EXP_NAME}"
set +a

INTERACTIVE=1 DRY_RUN=0 INTERACTIVE_WAIT="${INTERACTIVE_WAIT:-1}" \
  bash "${HERE}/ultra_launch.sh" "$@"
