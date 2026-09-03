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
# Nemotron 3.5 Nano — RLVR with the Ultra 550B GenRM on a GB300 static tier
#
# Swaps the Qwen3-235B GenRM of nano35_dolphin_launch.sh for
# NVIDIA-Nemotron-3-Ultra-550B-A55B-GenRM and re-sizes the static-model tier to
# 108 GPUs, so the same three services can be run here and on NVCF and the
# resulting throughput compared. Policy, blend, parallelism and the 5:1
# generation-to-training split are untouched — they are the constant.
#
# Usage:
#   bash examples/nemo_gym/nemotron-3.5-nano/nano35_ultra_genrm_launch.sh
#   DRY_RUN=1 bash .../nano35_ultra_genrm_launch.sh        # inspect only
#
# Allocation, 82 nodes / 328 GPUs. All GB300: 4 GPUs/node, ~278 GiB per GPU.
#   hetgroup 0   64 nodes    8 train + 40 generation + 16 gym
#                            Gym's 64 GPUs host the NL2Bash judge (32) and the
#                            safety judge (4); 28 remain for env servers.
#   hetgroup 1   18 nodes    Ultra GenRM, 9 replicas x TP=8
#
# Static tier: GenRM 72 + NL2Bash 32 + safety 4 = 108 GPUs. Only GenRM grows
# relative to the 68-node recipe (4 nodes -> 18): the Ultra checkpoint is 1.1 TB
# where the Qwen GenRM is 438 GiB.
#
# TP=8 is not a preference, it is the floor. At 1.1 TB of bf16 weights a TP=4
# replica needs ~1126 GiB and a GB300 quad yields ~1056 GiB at 0.95 utilization,
# so TP=4 does not fit on GB300 either. At TP=8 the weights leave ~987 GiB for
# KV, which is ample for this architecture: nemotron_h is hybrid Mamba with 2 KV
# heads, and the pool already serves it with --kv-cache-dtype fp8.
#
# Runs on batch_long rather than batch. Nine replicas each read 1.1 TB at
# startup — about 10 TB of Lustre traffic before the first step — so a 4 h job
# would spend a large fraction of its wall clock loading weights.
# GENRM_STARTUP_TIMEOUT is raised from the 3600 s default for the same reason,
# and checkpointing.checkpoint_must_save_by in the YAML tracks WALLTIME here.
# =============================================================================

set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)

export CONFIG_PATH="${CONFIG_PATH:-examples/nemo_gym/nemotron-3.5-nano/rlvr_dolphin_ultra_genrm.yaml}"

# hetgroup 0. Train and generation match the 68-node recipe exactly; Gym is back
# to 16 because both judges are served there (see NL2BASH_IN_GYM below).
export NUM_TRAIN_NODES="${NUM_TRAIN_NODES:-8}"
export NUM_GEN_NODES="${NUM_GEN_NODES:-40}"
export NUM_GYM_NODES="${NUM_GYM_NODES:-16}"

# hetgroup 1 holds GenRM alone: 9 x TP=8 / 4 GPUs per node = 18 nodes, which is
# divisible by the nano EXTERNAL_VLLM_SEGMENT_SIZE of 2. NL2BASH_IN_GYM keeps the
# NL2Bash judge out of the pool so it can run at TP=2, which a pool replica
# cannot — pool replicas own whole nodes.
export EXTERNAL_JUDGES="${EXTERNAL_JUDGES:-1}"
export NL2BASH_IN_GYM="${NL2BASH_IN_GYM:-1}"
export GENRM_MODEL="${GENRM_MODEL:-/lustre/fsw/portfolios/llmservice/users/jiaqiz/results/ultra-genrm-training/pipeclean-ultra-rl-rerun-ultra-genrm-training-from-wdai-step2600_tp8_cp8_ep32_pp1_gpp8_pps128_gbs1024-20260323-jiaqi/eval/step_720/hf}"
export GENRM_REPLICAS="${GENRM_REPLICAS:-9}"
export GENRM_TENSOR_PARALLEL_SIZE="${GENRM_TENSOR_PARALLEL_SIZE:-8}"
export GENRM_STARTUP_TIMEOUT="${GENRM_STARTUP_TIMEOUT:-7200}"

# batch caps at 4 h, which the GenRM load alone can rival. checkpoint_must_save_by
# in rlvr_dolphin_ultra_genrm.yaml is set for this wall clock; change both together.
export SLURM_PARTITION="${SLURM_PARTITION:-batch_long}"
export WALLTIME="${WALLTIME:-12:00:00}"

# The 68-node recipe inherits the reference run's results and cache directories,
# which are not group-writable. Point both at the submitter's own scratch.
_USER_SCRATCH="/lustre/fsw/portfolios/llmservice/users/${USER}"
export RESULTS_DIR="${RESULTS_DIR:-${_USER_SCRATCH}/runs/nano35-ultra-genrm}"
export PERSISTENT_CACHE="${PERSISTENT_CACHE:-${_USER_SCRATCH}/.cache/nano35-dolphin}"

export EXP_NAME="${EXP_NAME:-${USER}-nano35-ultra-genrm-n82}"

# A typo in a 1.1 TB path otherwise surfaces only after an 82-node allocation has
# started and each replica has tried to open it.
if [[ ! -r "${GENRM_MODEL}/config.json" ]]; then
  echo "ERROR: GENRM_MODEL is not readable: ${GENRM_MODEL}/config.json" >&2
  echo "       Point GENRM_MODEL at an Ultra GenRM HF checkpoint directory." >&2
  exit 1
fi

_GENRM_NODES=$((GENRM_REPLICAS * GENRM_TENSOR_PARALLEL_SIZE / 4))
_RAY_NODES=$((NUM_TRAIN_NODES + NUM_GEN_NODES + NUM_GYM_NODES))

echo "Nemotron 3.5 Nano — RLVR with the Ultra 550B GenRM (GB300)"
echo "  hetgroup 0 : ${_RAY_NODES} nodes — ${NUM_TRAIN_NODES} train + ${NUM_GEN_NODES} gen + ${NUM_GYM_NODES} gym"
echo "  hetgroup 1 : ${_GENRM_NODES} nodes — GenRM ${GENRM_REPLICAS} x TP=${GENRM_TENSOR_PARALLEL_SIZE}"
echo "  total      : $((_RAY_NODES + _GENRM_NODES)) nodes / $(((_RAY_NODES + _GENRM_NODES) * 4)) GPUs"
echo "  static tier: GenRM $((GENRM_REPLICAS * GENRM_TENSOR_PARALLEL_SIZE)) + NL2Bash 32 + safety 4 GPUs"
echo "  judges     : NL2Bash 16 x TP=2 and safety 4 x TP=1, both in the Gym pool"

exec bash "${SCRIPT_DIR}/nano35_dolphin_launch.sh" "$@"
