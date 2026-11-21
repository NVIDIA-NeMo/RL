#!/bin/bash
# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd)

# ===== BEGIN CONFIG =====
# Mirrors grpo-nano-v2-12b-1n8g-megatron-single-controller-token_capture.sh (delegated base).
NUM_NODES=1
GPUS_PER_NODE=8
STEPS_PER_RUN=30
MAX_STEPS=30
NUM_RUNS=$(( (MAX_STEPS + STEPS_PER_RUN - 1) / STEPS_PER_RUN ))  # Round up
NUM_MINUTES=120
# ===== END CONFIG =====

# Medians of the dense arm, in seconds, taken from its metrics.json:
#   DENSE_LOGPROB_SECONDS   median(timing/train/policy_and_reference_logprobs)
#   DENSE_TRAINING_SECONDS  median(timing/train/policy_training)
# While one is empty, the script fails after training, before it checks the timings.
DENSE_LOGPROB_SECONDS=
DENSE_TRAINING_SECONDS=
# The shared arm must stay below this fraction of each dense median.
MAX_FRACTION_OF_DENSE=0.8

set -eou pipefail

# Run the dense arm's script under this script's name, so it uses this script's
# recipe, which differs only in policy.sequence_packing.share_prefixes. The dense
# script runs the reward, token_mult_prob_error and capture checks.
BASE_RECIPE=grpo-nano-v2-12b-1n8g-megatron-single-controller-token_capture
export EXP_NAME=$(basename "$0" .sh)
bash "$SCRIPT_DIR/$BASE_RECIPE.sh" "$@"

# The delegated script runs in a subshell, so common.env's TEST_DRYRUN exit does
# not reach here. `if` form, not `&& exit`: a false `[[ ]]` returns 1 and set -e
# would abort.
if [[ -n "${TEST_DRYRUN:-}" ]]; then exit 0; fi

for name in DENSE_LOGPROB_SECONDS DENSE_TRAINING_SECONDS; do
    if [[ -z "${!name}" ]]; then
        echo "[ERROR] $name is empty in $(basename "$0"). Set it from the dense arm's median."
        exit 1
    fi
done

# Sharing acts on the logprob passes and on training, so both must be faster
# than in the dense arm. Total step time would hide the gain behind rollout time.
cd "$SCRIPT_DIR/../../.."
uv run tests/check_metrics.py "$SCRIPT_DIR/$EXP_NAME/metrics.json" \
    "median(data[\"timing/train/policy_and_reference_logprobs\"]) < $MAX_FRACTION_OF_DENSE * $DENSE_LOGPROB_SECONDS" \
    "median(data[\"timing/train/policy_training\"]) < $MAX_FRACTION_OF_DENSE * $DENSE_TRAINING_SECONDS"
