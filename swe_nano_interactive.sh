#!/bin/bash
# =============================================================================
# swe_nano_interactive.sh — interactive launch of the minimal Nano SWE smoke.
#
# Nano v3 30B-A3B, 6-node Qwen3-mesh recipe: async GRPO through
# run_grpo_nemo_gym.py. This is the NO-TransferQueue baseline — that path drives
# an in-memory ReplayBuffer and ignores data_plane. For a honoured data plane
# use swe_nano_sc_interactive.sh (async) or swe_nano_tq_interactive.sh (sync).
#
# Allocates the nodes, starts Ray, and idles so you can attach and run/edit/
# re-run the driver WITHOUT requeueing.
#
# Run from a NETWORKED shell (the sandbox can't reach slurmctld):
#     bash swe_nano_interactive.sh
#     bash swe_nano_interactive.sh grpo.max_num_steps=1        # extra hydra overrides pass through
#
# On submit it prints:
#     bash <jobid>-attach.sh        # shell on the head node (Ray already up)
#     source <jobid>-run-cmd.sh     # run the recipe inside that shell; edit + re-source to iterate
# Cancel with: scancel <jobid>
#
# First run downloads ~60 GB (Nemotron-3-Nano-30B-A3B-BF16) + converts to
# Megatron, so give the alloc room, e.g.:
#     INTERACTIVE_WALLTIME=2:00:00 bash swe_nano_interactive.sh
# =============================================================================
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

set -a
# shellcheck disable=SC1091
source "${HERE}/swe_nano.env"
set +a

# INTERACTIVE_WAIT=1 blocks until Ray is up then prints the attach command;
# set INTERACTIVE_WAIT=0 to submit and return immediately.
INTERACTIVE=1 DRY_RUN=0 INTERACTIVE_WAIT="${INTERACTIVE_WAIT:-1}" \
  bash "${HERE}/ultra_launch.sh" "$@"
