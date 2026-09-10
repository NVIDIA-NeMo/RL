#!/bin/bash
# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
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

# =============================================================================
# Nemotron 3.5 Nano — 8-node SingleController pipeclean
#
# Scaled-down nano35_dolphin_launch_sc.sh for validating the stack end to end
# before spending a 68-node allocation: SingleController, the TransferQueue
# data plane, nccl_reshard refit with MTP-head gating, and the in-job judge
# hetgroup this recipe gained from the 6K side.
#
# Usage:
#   bash examples/nemo_gym/nemotron-3.5-nano/nano35_dolphin_launch_sc_smoke.sh
#   DRY_RUN=1 bash .../nano35_dolphin_launch_sc_smoke.sh        # inspect only
#
# Training parallelism is deliberately NOT scaled. Four nodes is the floor that
# preserves it exactly: TP=4 x CP=4 x PP=1 fills 16 GPUs, and EP=8 still divides
# 16 at ETP=1, so every rank owns the same 16 of the 128 routed experts as the
# full-scale run. The only difference is DP, which drops from 2 to 1.
# Going to 2 training nodes would force CP below 4, and CP=4 is what makes the
# 73728-token context tractable, so the comparison would stop being meaningful.
#
# Sequence length is left at the production 73728 on purpose. Truncating it
# would risk failures in the blend rather than in the code under test, and
# generation length is bounded by EOS in practice, not by the cap.
#
# What gets scaled instead is the cohort, from 128 prompts to 8, and the step
# count. NRL_MAX_STEPS must stay >= 2: the first weight sync only happens
# between steps, and refit is the main thing this run exists to exercise.
#
# The default shape is 8 nodes total. Raise NUM_GEN_NODES / NUM_GYM_NODES to 2
# each (10 nodes) if rollouts are throughput-bound or Gym's env servers are
# cramped on a single node.
# =============================================================================

set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)

# 4 + 1 + 1 in hetgroup 0, plus 2 service nodes below = 8. Both counts have to
# stay divisible by their segment size, which is 2 on nano.
export NUM_TRAIN_NODES="${NUM_TRAIN_NODES:-4}"
export NUM_GEN_NODES="${NUM_GEN_NODES:-1}"
export NUM_GYM_NODES="${NUM_GYM_NODES:-1}"
export SEGMENT_SIZE="${SEGMENT_SIZE:-2}"

# Judges in-job, which is the path that has never been run and the main reason
# this profile exists. One replica each at the production TP, so the servers
# are sharded exactly as they would be at full scale -- there are just fewer.
# This also frees Gym down to a single node, since only the safety judge
# (TP=1, DP=1) still runs there.
export EXTERNAL_JUDGES="${EXTERNAL_JUDGES:-1}"
export GENRM_REPLICAS="${GENRM_REPLICAS:-1}"
export GENRM_TENSOR_PARALLEL_SIZE="${GENRM_TENSOR_PARALLEL_SIZE:-4}"
export NL2BASH_REPLICAS="${NL2BASH_REPLICAS:-1}"
export NL2BASH_TENSOR_PARALLEL_SIZE="${NL2BASH_TENSOR_PARALLEL_SIZE:-4}"
export EXTERNAL_VLLM_SEGMENT_SIZE="${EXTERNAL_VLLM_SEGMENT_SIZE:-2}"

# 8 prompts x 16 generations. Over DP=1 that is 128 sequences per rank against
# 1024 at full scale, so this is strictly lighter per GPU, not heavier.
NUM_PROMPTS_PER_STEP="${NUM_PROMPTS_PER_STEP:-8}"
TRAIN_GLOBAL_BATCH_SIZE="${TRAIN_GLOBAL_BATCH_SIZE:-$((NUM_PROMPTS_PER_STEP * 16))}"

# Production lookahead, so the sampler and its capacity checks behave as they
# would at scale; the buffer floor scales with the cohort.
MAX_LOOKAHEAD_VERSIONS="${MAX_LOOKAHEAD_VERSIONS:-4}"
export MAX_LOOKAHEAD_VERSIONS
export _NUM_PROMPTS_PER_STEP="${NUM_PROMPTS_PER_STEP}"

# Start the optimizer on a quarter cohort so streaming is actually exercised
# rather than trivially satisfied by the full cohort arriving at once.
export STREAM_MIN_GROUPS="${STREAM_MIN_GROUPS:-$((NUM_PROMPTS_PER_STEP / 4))}"
export NUM_STORAGE_UNITS="${NUM_STORAGE_UNITS:-2}"

# The full-scale recipe inherits akamehra's results and cache directories from
# the reference run, and they are not group-writable. Point both at the
# submitter's own scratch so this profile runs as-is for whoever launches it.
_USER_SCRATCH="/lustre/fsw/portfolios/coreai/users/${USER}"
export RESULTS_DIR="${RESULTS_DIR:-${_USER_SCRATCH}/runs/nano35-sc-pipeclean-n8}"
export PERSISTENT_CACHE="${PERSISTENT_CACHE:-${_USER_SCRATCH}/.cache/nano35-dolphin}"

export EXP_NAME="${EXP_NAME:-${USER}-nano35-sc-pipeclean-n8}"
export NRL_MAX_STEPS="${NRL_MAX_STEPS:-3}"
export WALLTIME="${WALLTIME:-2:00:00}"
export SLURM_QOS="${SLURM_QOS:-short}"
# Run the live worktree rather than a submission-time copy, so a fix can be
# retried without re-snapshotting between attempts.
export USE_SNAPSHOT="${USE_SNAPSHOT:-0}"

if (( NRL_MAX_STEPS < 2 )); then
  echo "ERROR: NRL_MAX_STEPS=${NRL_MAX_STEPS} never reaches a weight sync, which" >&2
  echo "       leaves nccl_reshard refit -- the point of this run -- untested." >&2
  exit 1
fi

echo "Nemotron 3.5 Nano — SingleController pipeclean profile"
echo "  hetgroup 0: train=${NUM_TRAIN_NODES}, generation=${NUM_GEN_NODES}, Gym=${NUM_GYM_NODES}"
echo "  hetgroup 1: GenRM=${GENRM_REPLICAS}xTP${GENRM_TENSOR_PARALLEL_SIZE}, NL2Bash=${NL2BASH_REPLICAS}xTP${NL2BASH_TENSOR_PARALLEL_SIZE}"
echo "  parallelism: TP4 x CP4 x EP8 x PP1 (ETP1) — unchanged, DP 2 -> 1"
echo "  batch: prompts=${NUM_PROMPTS_PER_STEP}, generations=16, global=${TRAIN_GLOBAL_BATCH_SIZE}"
echo "  steps: ${NRL_MAX_STEPS}"

exec bash "${SCRIPT_DIR}/nano35_dolphin_launch_sc.sh" \
  "grpo.num_prompts_per_step=${NUM_PROMPTS_PER_STEP}" \
  "policy.train_global_batch_size=${TRAIN_GLOBAL_BATCH_SIZE}" \
  "$@"
