#!/bin/bash
# Score-centering capture smoke with top-k/top-p sampling: the trainer replays
# the sampler's support, renormalizing over the finite entries of its captured
# head. top_logprobs (16) is at least top_k (12), so the head holds the
# sampler's kept set. IS off runs the unpacked
# loss path and TIS the packed one.
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)

SC_TOP_P=0.9 SC_TOP_K=12 SC_IS_MODES="off tis" \
    SC_EXP_PREFIX=grpo_score_centering_filtered_gym_single_controller \
    bash "$SCRIPT_DIR/grpo_score_centering_gym_single_controller.sh"
