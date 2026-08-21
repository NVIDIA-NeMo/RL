#!/bin/bash
# =============================================================================
# nano_swe_sc_interactive.sh — nano SWE via SingleController async GRPO with a
# HONOURED TransferQueue data plane, in an interactive allocation.
#
# Allocates the same 6 nodes as nano_swe_sc.sh, starts Ray, then idles so you
# can attach and run (and re-run) the driver by hand. Iterating inside one
# allocation is the point: a cold start pays the ~60 GB checkpoint download plus
# Megatron conversion and vLLM graph capture before the first rollout.
#
# Run from a NETWORKED shell at the repo root:
#     bash examples/nemo_gym/nemotron-3-ultra/nano_swe_sc_interactive.sh
#
# On submit it prints:
#     bash <jobid>-attach.sh        # shell on the head node (Ray already up)
#     source <jobid>-run-cmd.sh     # run the driver; edit + re-source to iterate
# Cancel with: scancel <jobid>
#
# Extra hydra overrides pass through, e.g. a fast first training step:
#     bash ... /nano_swe_sc_interactive.sh grpo.num_prompts_per_step=2 policy.train_global_batch_size=8
# Keep the invariant num_prompts_per_step × num_generations_per_prompt ==
# train_global_batch_size — the SingleController split path enforces it.
# =============================================================================
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

set -a
# shellcheck disable=SC1091
source "${HERE}/nano_swe.env"

# --- SingleController + TransferQueue overrides ------------------------------
EXP_NAME="${EXP_NAME:-nano-swe-sc-tq}"
NRL_ENTRYPOINT="${CODE_DIR}/examples/run_grpo_single_controller.py"
CONFIG_PATH="${CODE_DIR}/examples/nemo_gym/nemotron-3-ultra/nano_swe_teacher_sc.yaml"
RESULTS_DIR="${WORKSPACE_DIR}/results/${EXP_NAME}"
BASE_LOG_DIR="${WORKSPACE_DIR}/ray_logs/${EXP_NAME}"
set +a

_nano_swe_preflight

INTERACTIVE=1 DRY_RUN=0 INTERACTIVE_WAIT="${INTERACTIVE_WAIT:-1}" \
  bash "${HERE}/ultra_launch.sh" "$@"
