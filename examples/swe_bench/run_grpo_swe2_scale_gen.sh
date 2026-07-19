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

# ============================================================================
# GENERATION-SCALING launcher for async SWE GRPO (derived from
# run_grpo_repro_bihu_swe2.sh / bihu dc3m70us).
#
# Single knob:  NUM_VLLM_REPLICAS (R)  -> number of vLLM generation replicas.
# Everything else is auto-derived to hold these invariants constant so that
# runs at different R are directly comparable:
#   - per generation-replica workload : samples/replica/step = 2
#   - train:gen node ratio            : 1:1 by default
#
# Nano 3 default derivation
# (REPLICAS_PER_NODE = gpus_per_node / VLLM_TP = 8/4 = 2):
#   GEN_NODES   = R / 2
#   TRAIN_NODES = R / 2                 (linear follow; override with TRAIN_NODES=)
#   TOTAL_NODES = TRAIN_NODES+GEN_NODES = R     -> sbatch --nodes & cluster.num_nodes
#   PPS         = 2*R / GPP             = R/4
#   GBS         = PPS*GPP               = 2*R   (force_on_policy_ratio requires ==)
#   CONCURRENCY = max(768, GBS*age)
# With the default Nano training layout (TP=2, CP=4, PP=2, EP=8), R must be a
# multiple of 4 so both training parallelism and full generation nodes divide.
#
# All runs of this sweep share one wandb group (WANDB_GROUP) under project
# swe-benchmark for easy comparison.
#
# Usage:
#   NUM_VLLM_REPLICAS=64 bash examples/swe_bench/run_grpo_swe2_scale_gen.sh
#   NUM_VLLM_REPLICAS=64 DRY_RUN=1 bash examples/swe_bench/run_grpo_swe2_scale_gen.sh   # print config, no submit
#   SKIP_TRAINING=1 NUM_VLLM_REPLICAS=4 bash examples/swe_bench/run_grpo_swe2_scale_gen.sh  # generation-only (no-op train, R%4)
# Optional env: SKIP_TRAINING, TRAJECTORY_COLLECTION, ROLLOUT_ONLY_GPP,
#               TRAJECTORY_COLLECTION_BATCH_SIZE, TRAIN_DATA_PATH, VAL_DATA_PATH,
#               TRAIN_NODES, TRAIN_TP, VLLM_TP, CONFIG_FILE, WANDB_GROUP,
#               EXP_SUFFIX, MODEL_PATH, TOKENIZER_PATH,
#               CONTAINER, MAX_NUM_STEPS, SBATCH_TIME, SBATCH_DEPENDENCY,
#               CACHE_NAMESPACE, PERSISTENT_CACHE,
#               UV_CACHE_SEED_MODE,
#               SUBMIT_MODE,
#               BASE_LOG_DIR, LOGGER_LOG_DIR, STREAMING_TOOL_CALL,
#               LOG_GYM_RESPONSES, TEMPERATURE, TOP_P,
#               SNAPSHOT_POLL_INTERVAL_SECONDS,
#               SNAPSHOT_LONG_POLL_TIMEOUT_SECONDS,
#               STREAMING_MIN_CHUNK_CHARS,
#               STREAMING_INITIAL_CHUNK_CHARS,
#               STREAMING_INCREMENTAL_TOKENIZER_ONLY,
#               EXACT_INCREMENTAL_TOKENIZER,
#               FINAL_ONLY_INCREMENTAL_TOKENIZER,
#               FINAL_ONLY_PREFILL,
#               FINAL_ONLY_PREFILL_COMPLETION_GRACE_SECONDS,
#               PREFIX_SEEDED_START,
#               PREFILL_AFTER_ADMISSION,
#               BACKGROUND_PREFILL_COMPLETION,
#               STABLE_FIRST_SNAPSHOT_PREFILL,
#               COMPACT_REQUEST_CONTEXT,
#               INCREMENTAL_TOKENIZER_CHECKPOINT_INTERVAL,
#               COUNTERFACTUAL_FULL_TOKENIZER_TIMING,
#               DETAILED_RUNTIME_METRICS, BASE_CONCURRENCY,
#               SWE_BENCH_ARTIFACT_CACHE_OFFLINE
# Credentials are NOT sourced here — export HF_HOME / HF_TOKEN / WANDB_API_KEY yourself.
# ============================================================================

set -e

# ============================ Paths ============================
# Auto-detected from this script's location (examples/swe_bench/), so it works from
# any clone of the repo. Override by exporting REPO_ROOT.
REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
CONFIG_FILE="${CONFIG_FILE:-${REPO_ROOT}/examples/swe_bench/grpo_nemotron_nano3_30b_async_swe.yaml}"
CHECKPOINT_ROOT="${REPO_ROOT}/results"
DEFAULT_DATA_PATH="/lustre/fsw/portfolios/llmservice/projects/llmservice_modelalignment_ppo/users/sdevare/repos/nano/dataset/rl/swe_all_datasets_train_w_agent_ref_r2e_gym_subset.jsonl"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH:-${DEFAULT_DATA_PATH}}"
VAL_DATA_PATH="${VAL_DATA_PATH:-${TRAIN_DATA_PATH}}"
# Training starts from the official Nano 3 Base checkpoint, while rollout-only
# evaluation must use the instruction checkpoint so the OpenHands policy can
# emit function calls instead of filling the context window with raw reasoning.
if [ "${TRAJECTORY_COLLECTION:-0}" = "1" ]; then
  DEFAULT_MODEL_PATH="nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16"
else
  DEFAULT_MODEL_PATH="nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-Base-BF16"
fi
MODEL_PATH="${1:-${MODEL_PATH:-${DEFAULT_MODEL_PATH}}}"
# Keep the recipe tokenizer by default because Base-model training may
# intentionally pair a Base checkpoint with an instruction tokenizer. Set this
# explicitly when evaluating a different model family or checkpoint whose
# tokenizer/chat template must follow MODEL_PATH.
TOKENIZER_PATH="${TOKENIZER_PATH:-}"

# ================ Container and mount config ================
# NeMo RL nightly-gym used for Nano 3 streaming-prefill development.
export CONTAINER="${CONTAINER:-/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/joyang/RL/results/images/nemo-rl-nightly-gym-20260718.squashfs}"
GYM_CODE="${REPO_ROOT}/3rdparty/Gym-workspace/Gym"
export MOUNTS="/lustre:/lustre,${REPO_ROOT}:${REPO_ROOT},${GYM_CODE}:/opt/nemo-rl/3rdparty/Gym-workspace/Gym"

# ======================= Cluster / resources =======================
NUM_GPU=8
export GPUS_PER_NODE=${NUM_GPU}
export CPUS_PER_WORKER=114

# ============================ Parallelism ============================
# SKIP_TRAINING=1 -> generation-only benchmark: training is a no-op on a SINGLE node
# (no optimizer, weights frozen, refit every step + keep-alive matmul). Training
# parallelism must fit 1 node, so model_parallel = TP*CP*PP must divide gpus_per_node(=8).
TRAJECTORY_COLLECTION="${TRAJECTORY_COLLECTION:-0}"
TRAJECTORY_COLLECTION_BATCH_SIZE="${TRAJECTORY_COLLECTION_BATCH_SIZE:-64}"
SKIP_TRAINING="${SKIP_TRAINING:-0}"
if [ "${TRAJECTORY_COLLECTION}" = "1" ]; then
  SKIP_TRAINING=1
fi
STREAMING_TOOL_CALL="${STREAMING_TOOL_CALL:-0}"
STREAMING_INCREMENTAL_TOKENIZER_ONLY="${STREAMING_INCREMENTAL_TOKENIZER_ONLY:-0}"
EXACT_INCREMENTAL_TOKENIZER="${EXACT_INCREMENTAL_TOKENIZER:-0}"
FINAL_ONLY_INCREMENTAL_TOKENIZER="${FINAL_ONLY_INCREMENTAL_TOKENIZER:-0}"
FINAL_ONLY_PREFILL="${FINAL_ONLY_PREFILL:-0}"
FINAL_ONLY_PREFILL_COMPLETION_GRACE_SECONDS="${FINAL_ONLY_PREFILL_COMPLETION_GRACE_SECONDS:-0.0}"
PREFIX_SEEDED_START="${PREFIX_SEEDED_START:-0}"
PREFILL_AFTER_ADMISSION="${PREFILL_AFTER_ADMISSION:-0}"
BACKGROUND_PREFILL_COMPLETION="${BACKGROUND_PREFILL_COMPLETION:-0}"
SAME_REQUEST_FINAL_DECODE="${SAME_REQUEST_FINAL_DECODE:-0}"
STABLE_FIRST_SNAPSHOT_PREFILL="${STABLE_FIRST_SNAPSHOT_PREFILL:-0}"
COMPACT_REQUEST_CONTEXT="${COMPACT_REQUEST_CONTEXT:-0}"
INCREMENTAL_TOKENIZER_CHECKPOINT_INTERVAL="${INCREMENTAL_TOKENIZER_CHECKPOINT_INTERVAL:-8}"
COUNTERFACTUAL_FULL_TOKENIZER_TIMING="${COUNTERFACTUAL_FULL_TOKENIZER_TIMING:-0}"
DETAILED_RUNTIME_METRICS="${DETAILED_RUNTIME_METRICS:-0}"
STREAMING_EVENT_DRIVEN_SNAPSHOT_WAIT="${STREAMING_EVENT_DRIVEN_SNAPSHOT_WAIT:-1}"
if [ "${STREAMING_TOOL_CALL}" != "0" ] && [ "${STREAMING_TOOL_CALL}" != "1" ]; then
  echo "ERROR: STREAMING_TOOL_CALL must be 0 or 1." >&2
  exit 1
fi
if [ "${STREAMING_INCREMENTAL_TOKENIZER_ONLY}" != "0" ] && [ "${STREAMING_INCREMENTAL_TOKENIZER_ONLY}" != "1" ]; then
  echo "ERROR: STREAMING_INCREMENTAL_TOKENIZER_ONLY must be 0 or 1." >&2
  exit 1
fi
if [ "${STREAMING_INCREMENTAL_TOKENIZER_ONLY}" = "1" ] && [ "${STREAMING_TOOL_CALL}" != "1" ]; then
  echo "ERROR: STREAMING_INCREMENTAL_TOKENIZER_ONLY requires STREAMING_TOOL_CALL=1." >&2
  exit 1
fi
if [ "${EXACT_INCREMENTAL_TOKENIZER}" != "0" ] && [ "${EXACT_INCREMENTAL_TOKENIZER}" != "1" ]; then
  echo "ERROR: EXACT_INCREMENTAL_TOKENIZER must be 0 or 1." >&2
  exit 1
fi
if [ "${EXACT_INCREMENTAL_TOKENIZER}" = "1" ] && [ "${STREAMING_INCREMENTAL_TOKENIZER_ONLY}" != "1" ]; then
  echo "ERROR: EXACT_INCREMENTAL_TOKENIZER requires STREAMING_INCREMENTAL_TOKENIZER_ONLY=1." >&2
  exit 1
fi
if [ "${FINAL_ONLY_INCREMENTAL_TOKENIZER}" != "0" ] && [ "${FINAL_ONLY_INCREMENTAL_TOKENIZER}" != "1" ]; then
  echo "ERROR: FINAL_ONLY_INCREMENTAL_TOKENIZER must be 0 or 1." >&2
  exit 1
fi
if [ "${FINAL_ONLY_INCREMENTAL_TOKENIZER}" = "1" ] && [ "${EXACT_INCREMENTAL_TOKENIZER}" != "1" ]; then
  echo "ERROR: FINAL_ONLY_INCREMENTAL_TOKENIZER requires EXACT_INCREMENTAL_TOKENIZER=1." >&2
  exit 1
fi
if [ "${FINAL_ONLY_PREFILL}" != "0" ] && [ "${FINAL_ONLY_PREFILL}" != "1" ]; then
  echo "ERROR: FINAL_ONLY_PREFILL must be 0 or 1." >&2
  exit 1
fi
if [ "${FINAL_ONLY_PREFILL}" = "1" ] && [ "${FINAL_ONLY_INCREMENTAL_TOKENIZER}" != "1" ]; then
  echo "ERROR: FINAL_ONLY_PREFILL requires FINAL_ONLY_INCREMENTAL_TOKENIZER=1." >&2
  exit 1
fi
if ! [[ "${FINAL_ONLY_PREFILL_COMPLETION_GRACE_SECONDS}" =~ ^[0-9]+([.][0-9]+)?$ ]]; then
  echo "ERROR: FINAL_ONLY_PREFILL_COMPLETION_GRACE_SECONDS must be non-negative." >&2
  exit 1
fi
if [ "${PREFIX_SEEDED_START}" != "0" ] && [ "${PREFIX_SEEDED_START}" != "1" ]; then
  echo "ERROR: PREFIX_SEEDED_START must be 0 or 1." >&2
  exit 1
fi
if [ "${PREFIX_SEEDED_START}" = "1" ] && [ "${FINAL_ONLY_PREFILL}" != "1" ]; then
  echo "ERROR: PREFIX_SEEDED_START requires FINAL_ONLY_PREFILL=1." >&2
  exit 1
fi
if [ "${PREFILL_AFTER_ADMISSION}" != "0" ] && [ "${PREFILL_AFTER_ADMISSION}" != "1" ]; then
  echo "ERROR: PREFILL_AFTER_ADMISSION must be 0 or 1." >&2
  exit 1
fi
if [ "${PREFILL_AFTER_ADMISSION}" = "1" ] && [ "${FINAL_ONLY_PREFILL}" != "1" ]; then
  echo "ERROR: PREFILL_AFTER_ADMISSION requires FINAL_ONLY_PREFILL=1." >&2
  exit 1
fi
if [ "${BACKGROUND_PREFILL_COMPLETION}" != "0" ] && [ "${BACKGROUND_PREFILL_COMPLETION}" != "1" ]; then
  echo "ERROR: BACKGROUND_PREFILL_COMPLETION must be 0 or 1." >&2
  exit 1
fi
if [ "${BACKGROUND_PREFILL_COMPLETION}" = "1" ] && [ "${PREFILL_AFTER_ADMISSION}" != "1" ]; then
  echo "ERROR: BACKGROUND_PREFILL_COMPLETION requires PREFILL_AFTER_ADMISSION=1." >&2
  exit 1
fi
if [ "${SAME_REQUEST_FINAL_DECODE}" != "0" ] && [ "${SAME_REQUEST_FINAL_DECODE}" != "1" ]; then
  echo "ERROR: SAME_REQUEST_FINAL_DECODE must be 0 or 1." >&2
  exit 1
fi
if [ "${SAME_REQUEST_FINAL_DECODE}" = "1" ] && [ "${BACKGROUND_PREFILL_COMPLETION}" != "1" ]; then
  echo "ERROR: SAME_REQUEST_FINAL_DECODE requires BACKGROUND_PREFILL_COMPLETION=1." >&2
  exit 1
fi
if [ "${STABLE_FIRST_SNAPSHOT_PREFILL}" != "0" ] && [ "${STABLE_FIRST_SNAPSHOT_PREFILL}" != "1" ]; then
  echo "ERROR: STABLE_FIRST_SNAPSHOT_PREFILL must be 0 or 1." >&2
  exit 1
fi
if [ "${STABLE_FIRST_SNAPSHOT_PREFILL}" = "1" ] && [ "${PREFILL_AFTER_ADMISSION}" != "1" ]; then
  echo "ERROR: STABLE_FIRST_SNAPSHOT_PREFILL requires PREFILL_AFTER_ADMISSION=1." >&2
  exit 1
fi
if [ "${STABLE_FIRST_SNAPSHOT_PREFILL}" = "1" ] && [ "${PREFIX_SEEDED_START}" != "1" ]; then
  echo "ERROR: STABLE_FIRST_SNAPSHOT_PREFILL requires PREFIX_SEEDED_START=1." >&2
  exit 1
fi
if [ "${COMPACT_REQUEST_CONTEXT}" != "0" ] && [ "${COMPACT_REQUEST_CONTEXT}" != "1" ]; then
  echo "ERROR: COMPACT_REQUEST_CONTEXT must be 0 or 1." >&2
  exit 1
fi
if [ "${COMPACT_REQUEST_CONTEXT}" = "1" ] && [ "${EXACT_INCREMENTAL_TOKENIZER}" != "1" ]; then
  echo "ERROR: COMPACT_REQUEST_CONTEXT requires EXACT_INCREMENTAL_TOKENIZER=1." >&2
  exit 1
fi
if ! [[ "${INCREMENTAL_TOKENIZER_CHECKPOINT_INTERVAL}" =~ ^[1-9][0-9]*$ ]]; then
  echo "ERROR: INCREMENTAL_TOKENIZER_CHECKPOINT_INTERVAL must be a positive integer." >&2
  exit 1
fi
if [ "${COUNTERFACTUAL_FULL_TOKENIZER_TIMING}" != "0" ] && [ "${COUNTERFACTUAL_FULL_TOKENIZER_TIMING}" != "1" ]; then
  echo "ERROR: COUNTERFACTUAL_FULL_TOKENIZER_TIMING must be 0 or 1." >&2
  exit 1
fi
if [ "${COUNTERFACTUAL_FULL_TOKENIZER_TIMING}" = "1" ] && [ "${EXACT_INCREMENTAL_TOKENIZER}" != "1" ]; then
  echo "ERROR: COUNTERFACTUAL_FULL_TOKENIZER_TIMING requires EXACT_INCREMENTAL_TOKENIZER=1." >&2
  exit 1
fi
if [ "${DETAILED_RUNTIME_METRICS}" != "0" ] && [ "${DETAILED_RUNTIME_METRICS}" != "1" ]; then
  echo "ERROR: DETAILED_RUNTIME_METRICS must be 0 or 1." >&2
  exit 1
fi
if [ "${STREAMING_EVENT_DRIVEN_SNAPSHOT_WAIT}" != "0" ] && [ "${STREAMING_EVENT_DRIVEN_SNAPSHOT_WAIT}" != "1" ]; then
  echo "ERROR: STREAMING_EVENT_DRIVEN_SNAPSHOT_WAIT must be 0 or 1." >&2
  exit 1
fi
if [ "${DETAILED_RUNTIME_METRICS}" = "1" ]; then
  DETAILED_RUNTIME_METRICS_ENABLED=True
else
  DETAILED_RUNTIME_METRICS_ENABLED=False
fi
if [ "${STREAMING_TOOL_CALL}" = "1" ]; then
  STREAMING_TOOL_CALL_ENABLED=True
  VLLM_STREAMING_TOOL_CALL_ENABLED=True
  STREAMING_TOOL_CALL_TAG="-streamtool"
else
  STREAMING_TOOL_CALL_ENABLED=False
  VLLM_STREAMING_TOOL_CALL_ENABLED=False
  STREAMING_TOOL_CALL_TAG=""
fi
if [ "${STREAMING_INCREMENTAL_TOKENIZER_ONLY}" = "1" ]; then
  VLLM_STREAMING_TOOL_CALL_ENABLED=False
  STREAMING_INCREMENTAL_TOKENIZER_ONLY_ENABLED=True
  STREAMING_TOOL_CALL_TAG="-streamtokenizer"
else
  STREAMING_INCREMENTAL_TOKENIZER_ONLY_ENABLED=False
fi
if [ "${EXACT_INCREMENTAL_TOKENIZER}" = "1" ]; then
  EXACT_INCREMENTAL_TOKENIZER_ENABLED=True
else
  EXACT_INCREMENTAL_TOKENIZER_ENABLED=False
fi
if [ "${FINAL_ONLY_INCREMENTAL_TOKENIZER}" = "1" ]; then
  FINAL_ONLY_INCREMENTAL_TOKENIZER_ENABLED=True
  STREAMING_TOOL_CALL_TAG="-streamtokenizer-final"
else
  FINAL_ONLY_INCREMENTAL_TOKENIZER_ENABLED=False
fi
if [ "${FINAL_ONLY_PREFILL}" = "1" ]; then
  FINAL_ONLY_PREFILL_ENABLED=True
  VLLM_STREAMING_TOOL_CALL_ENABLED=True
  STREAMING_TOOL_CALL_TAG="-streamtokenizer-final-prefill"
else
  FINAL_ONLY_PREFILL_ENABLED=False
fi
if [ "${PREFIX_SEEDED_START}" = "1" ]; then
  PREFIX_SEEDED_START_ENABLED=True
  STREAMING_TOOL_CALL_TAG="-streamtokenizer-final-prefill-seeded"
else
  PREFIX_SEEDED_START_ENABLED=False
fi
if [ "${PREFILL_AFTER_ADMISSION}" = "1" ]; then
  PREFILL_AFTER_ADMISSION_ENABLED=True
  STREAMING_TOOL_CALL_TAG="${STREAMING_TOOL_CALL_TAG}-continuation"
else
  PREFILL_AFTER_ADMISSION_ENABLED=False
fi
if [ "${BACKGROUND_PREFILL_COMPLETION}" = "1" ]; then
  BACKGROUND_PREFILL_COMPLETION_ENABLED=True
  STREAMING_TOOL_CALL_TAG="${STREAMING_TOOL_CALL_TAG}-background"
else
  BACKGROUND_PREFILL_COMPLETION_ENABLED=False
fi
if [ "${SAME_REQUEST_FINAL_DECODE}" = "1" ]; then
  SAME_REQUEST_FINAL_DECODE_ENABLED=True
  VLLM_STREAMING_TOOL_CALL_ENABLED=True
  STREAMING_TOOL_CALL_TAG="${STREAMING_TOOL_CALL_TAG}-same-request"
else
  SAME_REQUEST_FINAL_DECODE_ENABLED=False
fi
if [ "${STABLE_FIRST_SNAPSHOT_PREFILL}" = "1" ]; then
  STABLE_FIRST_SNAPSHOT_PREFILL_ENABLED=True
  STREAMING_TOOL_CALL_TAG="${STREAMING_TOOL_CALL_TAG}-stablefirst"
else
  STABLE_FIRST_SNAPSHOT_PREFILL_ENABLED=False
fi
if [ "${COMPACT_REQUEST_CONTEXT}" = "1" ]; then
  COMPACT_REQUEST_CONTEXT_ENABLED=True
  STREAMING_TOOL_CALL_TAG="${STREAMING_TOOL_CALL_TAG}-compact"
else
  COMPACT_REQUEST_CONTEXT_ENABLED=False
fi
if [ "${STREAMING_EVENT_DRIVEN_SNAPSHOT_WAIT}" = "1" ]; then
  STREAMING_EVENT_DRIVEN_SNAPSHOT_WAIT_ENABLED=True
  STREAMING_TOOL_CALL_TAG="${STREAMING_TOOL_CALL_TAG}-eventwait"
else
  STREAMING_EVENT_DRIVEN_SNAPSHOT_WAIT_ENABLED=False
  STREAMING_TOOL_CALL_TAG="${STREAMING_TOOL_CALL_TAG}-pollwait"
fi
if [ "${COUNTERFACTUAL_FULL_TOKENIZER_TIMING}" = "1" ]; then
  COUNTERFACTUAL_FULL_TOKENIZER_TIMING_ENABLED=True
else
  COUNTERFACTUAL_FULL_TOKENIZER_TIMING_ENABLED=False
fi
if [ "${SKIP_TRAINING}" = "1" ]; then
  TP=8; EP=8; CP=1; PP=1; ETP=1     # model_parallel = 8 (fits 1 node), train_DP=1
else
  TP="${TRAIN_TP:-2}"; EP=8; CP=4; PP=2; ETP=1
fi
VLLM_TP="${VLLM_TP:-4}"
if ! [[ "${VLLM_TP}" =~ ^[1-9][0-9]*$ ]] || [ $(( NUM_GPU % VLLM_TP )) -ne 0 ]; then
  echo "ERROR: VLLM_TP must be a positive divisor of ${NUM_GPU} (got ${VLLM_TP})." >&2
  exit 1
fi
MIN_PAD=1
if [ ${CP} -gt 1 ]; then MIN_PAD=$((MIN_PAD * CP * 2)); fi
if [ ${TP} -gt 1 ]; then MIN_PAD=$((MIN_PAD * TP)); fi
MAKE_SEQ_DIVISIBLE_BY=${MIN_PAD}

# ================= Generation-scaling: derive all sizes from R =================
# Training sweeps keep eight generations per prompt. Trajectory collection
# exits through the validation-only collector before GRPO and emits one result
# per manifest row, so a smaller setup-only GPP may be used to scale a
# diagnostic below four replicas without changing collected trajectories.
GPP=8
if [ "${TRAJECTORY_COLLECTION}" = "1" ]; then
  GPP="${ROLLOUT_ONLY_GPP:-8}"
fi
if ! [[ "${GPP}" =~ ^[1-9][0-9]*$ ]]; then
  echo "ERROR: ROLLOUT_ONLY_GPP must be a positive integer (got ${GPP})." >&2
  exit 1
fi
SAMPLES_PER_REPLICA=2                             # invariant: samples/replica/step
BASE_CONCURRENCY="${BASE_CONCURRENCY:-768}"       # nemo-gym fan-out floor
if ! [[ "${BASE_CONCURRENCY}" =~ ^[1-9][0-9]*$ ]]; then
  echo "ERROR: BASE_CONCURRENCY must be a positive integer." >&2
  exit 1
fi
REPLICAS_PER_NODE=$(( NUM_GPU / VLLM_TP ))
MODEL_PARALLEL=$(( TP * CP * PP ))                # = 32
EXPERT_TMP=$(( ETP * EP * PP ))                   # = 16

NUM_VLLM_REPLICAS="${NUM_VLLM_REPLICAS:-}"
if [ -z "${NUM_VLLM_REPLICAS}" ]; then
  echo "ERROR: NUM_VLLM_REPLICAS is required (number of vLLM replicas). e.g. NUM_VLLM_REPLICAS=64" >&2
  exit 1
fi

# Smallest valid step for R.
gcd() { local a=$1 b=$2 t; while [ ${b} -ne 0 ]; do t=${b}; b=$(( a % b )); a=${t}; done; echo ${a}; }
lcm() { echo $(( $1 / $(gcd $1 $2) * $2 )); }
R_STEP_PPS=$(( GPP / $(gcd ${SAMPLES_PER_REPLICA} ${GPP}) ))
if [ "${SKIP_TRAINING}" = "1" ]; then
  # train fixed at 1 node (train_world=8, divisible by model_parallel=8); only gen
  # must fill whole nodes and produce an integral, nonzero PPS.
  R_STEP=$(lcm ${REPLICAS_PER_NODE} ${R_STEP_PPS})
else
  # With the default 1:1 train:gen node ratio, train_world=VLLM_TP*R.
  # It must be divisible by both model-parallel and expert-parallel sizes.
  L=$(lcm ${MODEL_PARALLEL} ${EXPERT_TMP})          # train-world divisor
  R_STEP_TRAIN=$(( L / $(gcd ${VLLM_TP} ${L}) ))
  R_STEP=$(lcm $(lcm ${R_STEP_TRAIN} ${REPLICAS_PER_NODE}) ${R_STEP_PPS})
fi
if [ $(( NUM_VLLM_REPLICAS % R_STEP )) -ne 0 ] || [ ${NUM_VLLM_REPLICAS} -lt ${R_STEP} ]; then
  echo "ERROR: NUM_VLLM_REPLICAS must be a positive multiple of ${R_STEP} (got ${NUM_VLLM_REPLICAS})." >&2
  exit 1
fi

GEN_NODES=$(( NUM_VLLM_REPLICAS / REPLICAS_PER_NODE ))
if [ "${SKIP_TRAINING}" = "1" ]; then
  TRAIN_NODES="${TRAIN_NODES:-1}"                 # no-op training: single node
else
  TRAIN_NODES="${TRAIN_NODES:-${GEN_NODES}}"      # linear 1:1 follow by default
fi
TOTAL_NODES=$(( TRAIN_NODES + GEN_NODES ))
PPS=$(( SAMPLES_PER_REPLICA * NUM_VLLM_REPLICAS / GPP ))
GBS=$(( PPS * GPP ))
CONCURRENCY=$(( GBS * 1 ))                         # GBS * max_trajectory_age_steps(=1)
if [ ${CONCURRENCY} -lt ${BASE_CONCURRENCY} ]; then CONCURRENCY=${BASE_CONCURRENCY}; fi

# Sanity: training divisibility (also re-checks any TRAIN_NODES override).
TRAIN_WORLD=$(( TRAIN_NODES * NUM_GPU ))
if [ $(( TRAIN_WORLD % MODEL_PARALLEL )) -ne 0 ] || [ $(( TRAIN_WORLD % EXPERT_TMP )) -ne 0 ]; then
  echo "ERROR: train world ${TRAIN_WORLD} (TRAIN_NODES=${TRAIN_NODES}) not divisible by model-parallel ${MODEL_PARALLEL} / expert ${EXPERT_TMP}." >&2
  exit 1
fi
TRAIN_DP=$(( TRAIN_WORLD / MODEL_PARALLEL ))
if [ $(( GBS % TRAIN_DP )) -ne 0 ]; then
  echo "ERROR: GBS ${GBS} not divisible by train DP ${TRAIN_DP}." >&2
  exit 1
fi
PER_GPU_BATCH=$(( GBS / TRAIN_DP ))
PER_REPLICA_SAMPLES=$(( GBS / NUM_VLLM_REPLICAS ))

# ===================== Sequence length & packing =====================
SEQLEN=131072
SEQUENCE_PACKING=True

# ================= Sync/Async mode & async GRPO settings =================
ASYNC_GRPO_ENABLED=True
MAX_TRAJECTORY_AGE_STEPS=1
FORCE_ON_POLICY_RATIO=True
INFLIGHT_WEIGHT_UPDATE=True
RECOMPUTE_KV_CACHE_AFTER_WEIGHT_UPDATES=False
SEQ_LOGPROB_ERROR_THRESHOLD=null
if [ "${ASYNC_GRPO_ENABLED}" = "True" ]; then
  COLOCATED_ENABLED=False
  VLLM_GPU_UTIL=0.8
  OVERLAP_GRAD_REDUCE=False
  ADVANTAGE_CLIP_LOW=-100
  ADVANTAGE_CLIP_HIGH=100
  TIS_THRESHOLD=5
else
  COLOCATED_ENABLED=True
  VLLM_GPU_UTIL=0.5
  OVERLAP_GRAD_REDUCE=True
fi

# ========================= GRPO / sampling =========================
NORMALIZE_REWARDS=True
OVERLONG_FILTERING=True

# ========================== Loss function ==========================
KL=0
CLIP_MIN=0.2
CLIP_MAX=0.28
USE_ON_POLICY_KL_APPROXIMATION=True
IMPORTANCE_SAMPLING_CORRECTION=True
SEQ_LEVEL_IS=False
TOKEN_LEVEL_LOSS=True

# ============================ Optimizer ============================
LR="1e-06"

# =============================== MoE ===============================
MOE_FREEZE_ROUTER=True
MOE_PERMUTE_FUSION=True
MOE_ENABLE_DEEPEP=False
MOE_TOKEN_DISPATCHER_TYPE="alltoall"
MOE_AUX_LOSS_COEFF=0
MOE_ROUTER_LOAD_BALANCING_TYPE="none"
MOE_ROUTER_BIAS_UPDATE_RATE="1e-3"

# ======================= Generation / vLLM =======================
TEMPERATURE="${TEMPERATURE:-1.0}"
TOP_P="${TOP_P:-1.0}"
SNAPSHOT_POLL_INTERVAL_SECONDS="${SNAPSHOT_POLL_INTERVAL_SECONDS:-0.1}"
SNAPSHOT_LONG_POLL_TIMEOUT_SECONDS="${SNAPSHOT_LONG_POLL_TIMEOUT_SECONDS:-1.0}"
STREAMING_MIN_CHUNK_CHARS="${STREAMING_MIN_CHUNK_CHARS:-}"
STREAMING_INITIAL_CHUNK_CHARS="${STREAMING_INITIAL_CHUNK_CHARS:-}"
SWE_BENCH_ARTIFACT_CACHE_OFFLINE="${SWE_BENCH_ARTIFACT_CACHE_OFFLINE:-0}"

if [ -n "${STREAMING_MIN_CHUNK_CHARS}" ] && ! [[ "${STREAMING_MIN_CHUNK_CHARS}" =~ ^[1-9][0-9]*$ ]]; then
  echo "ERROR: STREAMING_MIN_CHUNK_CHARS must be a positive integer when set." >&2
  exit 1
fi
if [ -n "${STREAMING_INITIAL_CHUNK_CHARS}" ] && ! [[ "${STREAMING_INITIAL_CHUNK_CHARS}" =~ ^[1-9][0-9]*$ ]]; then
  echo "ERROR: STREAMING_INITIAL_CHUNK_CHARS must be a positive integer when set." >&2
  exit 1
fi

# =================== Checkpointing & validation ===================
SAVE_PERIOD=5
VAL_PERIOD=1000
KEEP_TOP_K=2

# ============================ SWE agent ============================
AGENT_MAX_TURNS=200
AGENT_TIMEOUT=1800

# ============================== Logging ==============================
WANDB_PROJ="swe-benchmark"
# Shared group for the whole generation-scaling sweep (compare runs by R).
WANDB_GROUP="${WANDB_GROUP:-nemotron-nano3-swe-gen-scale-linear}"
# Log full trajectories to wandb so we can verify function_call items appear.
LOG_GYM_RESPONSES="${LOG_GYM_RESPONSES:-true}"

# ========================= SLURM submission =========================
SBATCH_ACCOUNT="${SBATCH_ACCOUNT:-nemotron_sw_post}"
SBATCH_PARTITION="${SBATCH_PARTITION:-batch}"
SBATCH_TIME="${SBATCH_TIME:-4:0:0}"
SBATCH_DEPENDENCY="${SBATCH_DEPENDENCY:-singleton}"
SUBMIT_MODE="${SUBMIT_MODE:-sbatch}"
if [ "${SUBMIT_MODE}" != "sbatch" ] && [ "${SUBMIT_MODE}" != "direct" ]; then
  echo "ERROR: SUBMIT_MODE must be sbatch or direct." >&2
  exit 1
fi
# Optional smoke-test knob: cap training steps (appended as ++grpo.max_num_steps). Empty = use YAML default.
MAX_NUM_STEPS="${MAX_NUM_STEPS:-}"

# ========================= Experiment naming =========================
if [ "${ASYNC_GRPO_ENABLED}" = "True" ]; then
  SYNC_MODE="async-age${MAX_TRAJECTORY_AGE_STEPS}"
else
  SYNC_MODE="sync"
fi
EXP_SUFFIX="${EXP_SUFFIX:-nano3-swe-genscale-${SYNC_MODE}-genrep${NUM_VLLM_REPLICAS}-nodes${TOTAL_NODES}-pps${PPS}-gpp${GPP}-gbs${GBS}-lr${LR}${STREAMING_TOOL_CALL_TAG}}"
WANDB_NAME="${EXP_SUFFIX}"
CHECKPOINT_DIR="${CHECKPOINT_ROOT}/${EXP_SUFFIX}"
SNAPSHOT_DIR="${REPO_ROOT}"

mkdir -p "${CHECKPOINT_DIR}"

# ============= Unified SLURM/Ray log location =============
export BASE_LOG_DIR="${BASE_LOG_DIR:-${SNAPSHOT_DIR}/logs/swe_bench_scale}"
mkdir -p "${BASE_LOG_DIR}"
LOGGER_LOG_DIR="${LOGGER_LOG_DIR:-logs}"

# ========================= Environment variables =========================
# Credentials are NOT sourced here. Export these yourself before submitting:
#   HF_HOME, HF_TOKEN, WANDB_API_KEY  (and GITHUB_TOKEN / GITLAB_TOKEN if needed)
export HUGGINGFACE_TOKEN="${HUGGINGFACE_TOKEN:-${HF_TOKEN}}"
export GITLAB_TOKEN="${GITLAB_TOKEN:-}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${HF_HOME}/datasets}"
export UV_CACHE_DIR=/tmp/uv_cache
export UV_LOCK_TIMEOUT=3600
export RAY_DEDUP_LOGS=1
export SSL_CERT_FILE=/etc/ssl/certs/ca-certificates.crt
export REQUESTS_CA_BUNDLE=/etc/ssl/certs/ca-certificates.crt
export CURL_CA_BUNDLE=/etc/ssl/certs/ca-certificates.crt
export OMP_NUM_THREADS=16

# ========================= Node-local cache config =========================
# Keep incompatible model/container/lock artifacts in separate namespaces.
CACHE_MODEL_KEY="$(printf '%s' "$(basename "${MODEL_PATH}")" | tr -c '[:alnum:]_.-' '_')"
CACHE_CONTAINER_KEY="$(printf '%s' "$(basename "${CONTAINER}")" | tr -c '[:alnum:]_.-' '_')"
CACHE_LOCK_KEY="$(sha256sum "${REPO_ROOT}/uv.lock" | cut -c1-12)"
CACHE_NAMESPACE="${CACHE_NAMESPACE:-${CACHE_MODEL_KEY}-${CACHE_CONTAINER_KEY}-${CACHE_LOCK_KEY}}"
PERSISTENT_CACHE="${PERSISTENT_CACHE:-${REPO_ROOT}/results/cache/swe_scale/${CACHE_NAMESPACE}}"
export PERSISTENT_CACHE
export LUSTRE_UV_CACHE_SEED="${LUSTRE_UV_CACHE_SEED:-${PERSISTENT_CACHE}/uv_seed}"
UV_CACHE_SEED_MODE="${UV_CACHE_SEED_MODE:-persistent}"
if [ "${UV_CACHE_SEED_MODE}" != "persistent" ] && [ "${UV_CACHE_SEED_MODE}" != "image-only" ]; then
  echo "ERROR: UV_CACHE_SEED_MODE must be persistent or image-only." >&2
  exit 1
fi
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-${PERSISTENT_CACHE}/pip}"
# Never let `uv sync` reuse a worktree `.venv` created under another image.
# The setup and driver run in the same allocation and rebuild this node-local
# environment from the image/lock-scoped package cache.
export UV_PROJECT_ENVIRONMENT="${UV_PROJECT_ENVIRONMENT:-/tmp/nemo_rl_driver_venv}"
# Gym launches server processes on arbitrary Ray nodes. Keep their venvs on the
# shared filesystem rather than a node-local /opt path so every actor sees the
# environment created by the Gym setup process.
export NEMO_GYM_VENV_DIR="${PERSISTENT_CACHE}/gym_venvs"
UV_BIN="${UV_BIN:-${PERSISTENT_CACHE}/bin/uv}"
UV_BIN_DIR="$(dirname "${UV_BIN}")"
export NEMO_RL_VENV_DIR="${NEMO_RL_VENV_DIR:-/opt/ray_venvs}"
export UV_LINK_MODE=copy
VLLM_SHARED_VENV="${PERSISTENT_CACHE}/ray_venvs/vllm_shared"
VLLM_VENV_TARGETS=(
  nemo_rl.models.generation.vllm.vllm_worker_async.VllmAsyncGenerationWorker
  nemo_rl.algorithms.async_utils.ReplayBuffer
  nemo_rl.algorithms.async_utils.AsyncTrajectoryCollector
)
mkdir -p "${VLLM_SHARED_VENV}"
for venv_target in "${VLLM_VENV_TARGETS[@]}"; do
  MOUNTS+=",${VLLM_SHARED_VENV}:${NEMO_RL_VENV_DIR}/${venv_target}"
done
export MOUNTS
export LUSTRE_VLLM_CACHE="${PERSISTENT_CACHE}/vllm_compile_cache"
export LUSTRE_INDUCTOR_CACHE="${PERSISTENT_CACHE}/inductor_cache"
export LUSTRE_TRITON_CACHE="${PERSISTENT_CACHE}/triton_cache"
export NRL_VLLM_CACHE_ROOT_BASE="/tmp/nemo_rl_vllm_cache"
export NRL_VLLM_CACHE_WRITEBACK_DIR="${LUSTRE_VLLM_CACHE}"
export INDUCTOR_CACHE_DIR="/tmp/nemo_rl_inductor_cache"
export TORCHINDUCTOR_CACHE_DIR="${INDUCTOR_CACHE_DIR}"
export TRITON_CACHE_DIR="/tmp/nemo_rl_triton_cache"
mkdir -p "${LUSTRE_UV_CACHE_SEED}" "${PIP_CACHE_DIR}" "${NEMO_GYM_VENV_DIR}" "${UV_BIN_DIR}" "${LUSTRE_VLLM_CACHE}" "${LUSTRE_INDUCTOR_CACHE}" "${LUSTRE_TRITON_CACHE}"

# ============================== Summary ==============================
echo "=========================================="
echo "SWE generation-scaling | Experiment: ${EXP_SUFFIX}"
echo "Mode: ${SYNC_MODE}, Colocated: ${COLOCATED_ENABLED}"
echo "wandb: project=${WANDB_PROJ}, group=${WANDB_GROUP}, name=${WANDB_NAME}"
echo "------------------------------------------"
echo "Scaling input:  NUM_VLLM_REPLICAS = ${NUM_VLLM_REPLICAS}  (R-step=${R_STEP})"
echo "  replicas/node = ${REPLICAS_PER_NODE} (vllm_tp=${VLLM_TP})"
echo "  GEN_NODES     = ${GEN_NODES}"
echo "  TRAIN_NODES   = ${TRAIN_NODES}   (train_DP=${TRAIN_DP})"
echo "  TOTAL_NODES   = ${TOTAL_NODES}"
echo "  PPS           = ${PPS}"
echo "  GPP           = ${GPP}"
echo "  GBS           = ${GBS}"
echo "  CONCURRENCY   = ${CONCURRENCY}"
echo "  invariants    : samples/replica=${PER_REPLICA_SAMPLES}, batch/train-GPU=${PER_GPU_BATCH}"
echo "Parallelism: TP=${TP}, EP=${EP}, CP=${CP}, PP=${PP}, vLLM_TP=${VLLM_TP}, pad=${MAKE_SEQ_DIVISIBLE_BY}"
echo "Model: ${MODEL_PATH}"
echo "Tokenizer: ${TOKENIZER_PATH:-recipe default}"
echo "Container: ${CONTAINER}"
echo "Cache namespace: ${CACHE_NAMESPACE}"
echo "Streaming tool call: ${STREAMING_TOOL_CALL_ENABLED}"
echo "Streaming tokenizer-only: ${STREAMING_INCREMENTAL_TOKENIZER_ONLY_ENABLED}"
echo "Detailed OpenHands runtime metrics: ${DETAILED_RUNTIME_METRICS_ENABLED}"
echo "vLLM streaming prefill: ${VLLM_STREAMING_TOOL_CALL_ENABLED}"
echo "Exact incremental tokenizer: ${EXACT_INCREMENTAL_TOKENIZER_ENABLED}"
echo "Final-only incremental tokenizer: ${FINAL_ONLY_INCREMENTAL_TOKENIZER_ENABLED}"
echo "Final-only prefill: ${FINAL_ONLY_PREFILL_ENABLED}"
echo "Final-only prefill completion grace: ${FINAL_ONLY_PREFILL_COMPLETION_GRACE_SECONDS}s"
echo "Background prefill completion: ${BACKGROUND_PREFILL_COMPLETION_ENABLED}"
echo "Same-request final decode: ${SAME_REQUEST_FINAL_DECODE_ENABLED}"
echo "Authoritative-prefix seeded start: ${PREFIX_SEEDED_START_ENABLED}"
echo "Prefill after admission: ${PREFILL_AFTER_ADMISSION_ENABLED}"
echo "Stable first-snapshot prefill: ${STABLE_FIRST_SNAPSHOT_PREFILL_ENABLED}"
echo "Compact request context: ${COMPACT_REQUEST_CONTEXT_ENABLED}"
echo "Incremental tokenizer checkpoint interval: ${INCREMENTAL_TOKENIZER_CHECKPOINT_INTERVAL}"
echo "Counterfactual full-tokenizer timing: ${COUNTERFACTUAL_FULL_TOKENIZER_TIMING_ENABLED}"
echo "Streaming snapshot poll interval: ${SNAPSHOT_POLL_INTERVAL_SECONDS}s"
echo "Streaming snapshot long-poll timeout: ${SNAPSHOT_LONG_POLL_TIMEOUT_SECONDS}s"
if [ -n "${STREAMING_MIN_CHUNK_CHARS}" ]; then
  echo "Streaming min chunk chars: ${STREAMING_MIN_CHUNK_CHARS}"
fi
if [ -n "${STREAMING_INITIAL_CHUNK_CHARS}" ]; then
  echo "Streaming initial chunk chars: ${STREAMING_INITIAL_CHUNK_CHARS}"
fi
echo "SWE-bench artifact cache offline: ${SWE_BENCH_ARTIFACT_CACHE_OFFLINE}"
echo "Package caches: uv-local=${UV_CACHE_DIR} uv-seed=${LUSTRE_UV_CACHE_SEED:-none} pip=${PIP_CACHE_DIR} driver-venv=${UV_PROJECT_ENVIRONMENT}"
echo "uv cache seed mode: ${UV_CACHE_SEED_MODE}"
echo "Gym shared venv: ${NEMO_GYM_VENV_DIR}"
echo "uv executable: ${UV_BIN}"
echo "Ray vLLM venv: ${VLLM_SHARED_VENV} -> ${NEMO_RL_VENV_DIR}/{async-worker,replay,collector} (link-mode=${UV_LINK_MODE})"
echo "HF caches: home=${HF_HOME} datasets=${HF_DATASETS_CACHE}"
echo "Build caches: vllm-local=${NRL_VLLM_CACHE_ROOT_BASE} vllm-seed=${LUSTRE_VLLM_CACHE} inductor-local=${TORCHINDUCTOR_CACHE_DIR} triton-local=${TRITON_CACHE_DIR}"
echo "Checkpoint: ${CHECKPOINT_DIR}"
echo "Submission mode: ${SUBMIT_MODE}"
echo "=========================================="

cd "${SNAPSHOT_DIR}"

# ================ SETUP_COMMAND (bihu's: install apptainer + seed caches + uv sync) ================
read -r -d '' SETUP_COMMAND <<SETUPEOF || true
if [ ! -x "${UV_BIN}" ]; then
  echo "[SETUP] ERROR: compatible uv binary is missing: ${UV_BIN}" >&2
  echo "[SETUP] Prewarm the nightly container cache before launching." >&2
  exit 1
fi
export PATH="${UV_BIN_DIR}:\$PATH"
echo "[SETUP] uv version: \$(uv --version)"

RET=1
RETRIES=3
if command -v apptainer >/dev/null 2>&1 || command -v singularity >/dev/null 2>&1; then
  echo "[SETUP] singularity/apptainer already available; skipping apt setup"
  RET=0
else
  echo "[SETUP] Installing apptainer for SWE sandbox..."
  apt-get update && apt-get install -y git build-essential gcc wget 2>/dev/null || true
  for attempt in \$(seq 1 \$RETRIES); do
    cd /tmp && \
    wget --no-check-certificate -q https://github.com/apptainer/apptainer/releases/download/v1.3.1/apptainer_1.3.1_amd64.deb && \
    apt install -y ./apptainer_1.3.1_amd64.deb && \
    ln -sf /usr/bin/apptainer /usr/bin/singularity
    if command -v apptainer >/dev/null 2>&1; then
      echo "[SETUP] apptainer installed successfully"
      RET=0
      break
    fi
    echo "[SETUP] apptainer install attempt \$attempt failed, retrying..."
    sleep 10
  done
  if [ \$RET -ne 0 ]; then
    echo "[SETUP] WARNING: apptainer installation failed after \$RETRIES attempts"
  fi
fi

echo "[CACHE SEED] Clearing stale /tmp caches and seeding from Lustre..."
rm -rf "${NRL_VLLM_CACHE_ROOT_BASE}"
rm -rf "${INDUCTOR_CACHE_DIR}" "${TRITON_CACHE_DIR}"
mkdir -p "${NRL_VLLM_CACHE_ROOT_BASE}" "${INDUCTOR_CACHE_DIR}" "${TRITON_CACHE_DIR}"

find "${LUSTRE_INDUCTOR_CACHE}" -maxdepth 1 -name '.tmp_*' -mmin +30 -exec rm -rf {} + 2>/dev/null || true
find "${LUSTRE_TRITON_CACHE}" -maxdepth 1 -name '.tmp_*' -mmin +30 -exec rm -rf {} + 2>/dev/null || true

_seed_cache() {
  local lustre="\$1" local_dir="\$2" name="\$3"
  if [ -d "\$lustre" ] && [ "\$(ls -A "\$lustre" 2>/dev/null)" ]; then
    rsync -a --exclude '.tmp_*' "\$lustre/" "\$local_dir/" 2>/dev/null \
      && echo "[CACHE SEED] \$name: seeded from Lustre" \
      || echo "[CACHE SEED] \$name: seed failed (non-fatal)"
  else
    echo "[CACHE SEED] \$name: no warm cache on Lustre yet"
  fi
}

_seed_cache "${LUSTRE_VLLM_CACHE}" "${NRL_VLLM_CACHE_ROOT_BASE}" "vLLM compile"
_seed_cache "${LUSTRE_INDUCTOR_CACHE}" "${INDUCTOR_CACHE_DIR}" "Inductor"
_seed_cache "${LUSTRE_TRITON_CACHE}" "${TRITON_CACHE_DIR}" "Triton"
mkdir -p /tmp/uv_cache
if [ -d /root/.cache/uv ] && [ "\$(ls -A /root/.cache/uv 2>/dev/null)" ]; then
  rsync -a --exclude '.tmp*' /root/.cache/uv/ /tmp/uv_cache/ 2>/dev/null \
    && echo "[CACHE SEED] uv (image): seeded from baked cache" \
    || echo "[CACHE SEED] uv (image): seed failed (non-fatal)"
else
  echo "[CACHE SEED] uv (image): no baked cache"
fi
if [ "${UV_CACHE_SEED_MODE}" = "persistent" ]; then
  if [ -f "${LUSTRE_UV_CACHE_SEED}/.seed-complete-mcore-v1" ] \
    || [ -f "${LUSTRE_UV_CACHE_SEED}/.seed-complete-mcore-v2" ]; then
    _seed_cache "${LUSTRE_UV_CACHE_SEED}" "/tmp/uv_cache" "uv (prebuilt transformer-engine)"
  elif [ -d "${LUSTRE_UV_CACHE_SEED}" ] && [ "\$(ls -A "${LUSTRE_UV_CACHE_SEED}" 2>/dev/null)" ]; then
    echo "[CACHE SEED] uv (persistent): incomplete seed has no completion marker; ignored"
  else
    echo "[CACHE SEED] uv (persistent): no warm cache on Lustre yet"
  fi
else
  echo "[CACHE SEED] uv (persistent): image-only mode; using baked image cache"
fi
echo "[CACHE SEED] Done."

UV_CACHE_DIR=/tmp/uv_cache \
UV_HTTP_TIMEOUT=3600 \
TORCH_CUDA_ARCH_LIST='9.0 10.0' \
NVTE_CUDA_ARCHS='90;100' \
UV_PROJECT_ENVIRONMENT="${UV_PROJECT_ENVIRONMENT}" \
  uv sync --frozen --extra mcore

if [ "${UV_CACHE_SEED_MODE}" = "persistent" ]; then
  # Persist complete uv cache entries after every successful locked sync. A
  # marker records the sync contract but must not suppress future merges: the
  # driver and Ray environment builders can add native wheels after setup.
  (
    if flock -n 9; then
      rsync -a --exclude '.tmp*' /tmp/uv_cache/ "${LUSTRE_UV_CACHE_SEED}/"
      touch "${LUSTRE_UV_CACHE_SEED}/.seed-complete-mcore-v2"
      echo "[CACHE WRITEBACK] uv setup merge: complete"
    else
      echo "[CACHE WRITEBACK] uv setup merge: another node owns the seed lock; skipped"
    fi
  ) 9>"${LUSTRE_UV_CACHE_SEED}.lock"
else
  echo "[CACHE WRITEBACK] uv setup merge: skipped in image-only mode"
fi
SETUPEOF
export SETUP_COMMAND

# ================ Training command (bihu-style: uv run --frozen, no --extra mcore) ================
export COMMAND="PATH=${UV_BIN_DIR}:\${PATH} \
  PYTHONPATH=${REPO_ROOT} \
  NRL_EXPECTED_REPO_ROOT=${REPO_ROOT} \
  NRL_VLLM_USE_V1=1 \
  NRL_WG_USE_RAY_REF=1 \
  WANDB_API_KEY=${WANDB_API_KEY} \
  HUGGINGFACE_TOKEN=${HUGGINGFACE_TOKEN} \
  GITHUB_TOKEN=${GITHUB_TOKEN} \
  GITLAB_TOKEN=${GITLAB_TOKEN} \
  SWE_BENCH_ARTIFACT_CACHE_OFFLINE=${SWE_BENCH_ARTIFACT_CACHE_OFFLINE} \
  HF_HOME=${HF_HOME} \
  HF_DATASETS_CACHE=${HF_DATASETS_CACHE} \
  UV_CACHE_DIR=${UV_CACHE_DIR} \
  UV_PROJECT_ENVIRONMENT=${UV_PROJECT_ENVIRONMENT} \
  UV_LINK_MODE=${UV_LINK_MODE} \
  PIP_CACHE_DIR=${PIP_CACHE_DIR} \
  NEMO_RL_VENV_DIR=${NEMO_RL_VENV_DIR} \
  NEMO_GYM_VENV_DIR=${NEMO_GYM_VENV_DIR} \
  VLLM_ATTENTION_BACKEND=FLASH_ATTN \
  NRL_VLLM_CACHE_ROOT_BASE=${NRL_VLLM_CACHE_ROOT_BASE} \
  NRL_VLLM_CACHE_WRITEBACK_DIR=${NRL_VLLM_CACHE_WRITEBACK_DIR} \
  INDUCTOR_CACHE_DIR=${INDUCTOR_CACHE_DIR} \
  TORCHINDUCTOR_CACHE_DIR=${TORCHINDUCTOR_CACHE_DIR} \
  TRITON_CACHE_DIR=${TRITON_CACHE_DIR} \
  DG_JIT_CACHE_DIR=${NRL_VLLM_CACHE_ROOT_BASE}/deep_gemm \
  VLLM_DEEP_GEMM_WARMUP=skip \
  NRL_FORCE_REBUILD_VENVS=false \
  NRL_IGNORE_VERSION_MISMATCH=1 \
  RAY_ENABLE_UV_RUN_RUNTIME_ENV=0 \
  UV_HTTP_TIMEOUT=3600 \
  UV_LOCK_TIMEOUT=900 \
  TORCH_CUDA_ARCH_LIST='9.0 10.0' \
  NVTE_CUDA_ARCHS='90;100' \
  NEMO_GYM_SKIP_VENV_IF_PRESENT=1 \
  uv run --frozen --extra mcore ./examples/nemo_gym/run_grpo_nemo_gym.py \
  --config=${CONFIG_FILE} \
  cluster.num_nodes=${TOTAL_NODES} \
  cluster.gpus_per_node=${NUM_GPU} \
  ++data.train.data_path=${TRAIN_DATA_PATH} \
  ++data.validation.data_path=${VAL_DATA_PATH} \
  grpo.num_prompts_per_step=${PPS} \
  grpo.num_generations_per_prompt=${GPP} \
  grpo.val_at_start=False \
  grpo.normalize_rewards=${NORMALIZE_REWARDS} \
  grpo.overlong_filtering=${OVERLONG_FILTERING} \
  grpo.val_period=${VAL_PERIOD} \
  grpo.seq_logprob_error_threshold=${SEQ_LOGPROB_ERROR_THRESHOLD} \
  grpo.async_grpo.enabled=${ASYNC_GRPO_ENABLED} \
  grpo.async_grpo.in_flight_weight_updates=${INFLIGHT_WEIGHT_UPDATE} \
  grpo.async_grpo.recompute_kv_cache_after_weight_updates=${RECOMPUTE_KV_CACHE_AFTER_WEIGHT_UPDATES} \
  grpo.async_grpo.max_trajectory_age_steps=${MAX_TRAJECTORY_AGE_STEPS} \
  env.should_log_nemo_gym_responses=${LOG_GYM_RESPONSES} \
  policy.generation.colocated.enabled=${COLOCATED_ENABLED} \
  policy.model_name=${MODEL_PATH} \
  policy.max_total_sequence_length=${SEQLEN} \
  policy.dynamic_batching.enabled=False \
  policy.train_global_batch_size=${GBS} \
  policy.make_sequence_length_divisible_by=${MAKE_SEQ_DIVISIBLE_BY} \
  policy.offload_optimizer_for_logprob=true \
  policy.sequence_packing.enabled=${SEQUENCE_PACKING} \
  policy.megatron_cfg.tensor_model_parallel_size=${TP} \
  policy.megatron_cfg.expert_model_parallel_size=${EP} \
  policy.megatron_cfg.context_parallel_size=${CP} \
  policy.megatron_cfg.pipeline_model_parallel_size=${PP} \
  policy.megatron_cfg.sequence_parallel=True \
  policy.megatron_cfg.bias_activation_fusion=False \
  policy.megatron_cfg.distributed_data_parallel_config.overlap_grad_reduce=${OVERLAP_GRAD_REDUCE} \
  policy.megatron_cfg.moe_permute_fusion=${MOE_PERMUTE_FUSION} \
  policy.megatron_cfg.moe_enable_deepep=${MOE_ENABLE_DEEPEP} \
  policy.megatron_cfg.moe_token_dispatcher_type=${MOE_TOKEN_DISPATCHER_TYPE} \
  policy.megatron_cfg.moe_aux_loss_coeff=${MOE_AUX_LOSS_COEFF} \
  policy.megatron_cfg.moe_router_load_balancing_type=${MOE_ROUTER_LOAD_BALANCING_TYPE} \
  policy.megatron_cfg.moe_router_bias_update_rate=${MOE_ROUTER_BIAS_UPDATE_RATE} \
  policy.megatron_cfg.freeze_moe_router=${MOE_FREEZE_ROUTER} \
  policy.megatron_cfg.optimizer.lr=${LR} \
  policy.megatron_cfg.optimizer.min_lr=${LR} \
  policy.megatron_cfg.optimizer.weight_decay=0 \
  policy.megatron_cfg.empty_unused_memory_level=2 \
  policy.megatron_cfg.activation_checkpointing=True \
  policy.generation.temperature=${TEMPERATURE} \
  policy.generation.top_p=${TOP_P} \
  policy.generation.vllm_cfg.tensor_parallel_size=${VLLM_TP} \
  policy.generation.vllm_cfg.gpu_memory_utilization=${VLLM_GPU_UTIL} \
  policy.generation.vllm_cfg.skip_tokenizer_init=False \
  policy.generation.vllm_cfg.streaming_tool_call.enabled=${VLLM_STREAMING_TOOL_CALL_ENABLED} \
  policy.generation.vllm_cfg.streaming_tool_call.tokenizer_only=${STREAMING_INCREMENTAL_TOKENIZER_ONLY_ENABLED} \
  policy.generation.vllm_cfg.streaming_tool_call.exact_incremental_tokenizer=${EXACT_INCREMENTAL_TOKENIZER_ENABLED} \
  policy.generation.vllm_cfg.streaming_tool_call.final_only_incremental_tokenizer=${FINAL_ONLY_INCREMENTAL_TOKENIZER_ENABLED} \
  policy.generation.vllm_cfg.streaming_tool_call.final_only_prefill=${FINAL_ONLY_PREFILL_ENABLED} \
  policy.generation.vllm_cfg.streaming_tool_call.final_only_prefill_completion_grace_seconds=${FINAL_ONLY_PREFILL_COMPLETION_GRACE_SECONDS} \
  policy.generation.vllm_cfg.streaming_tool_call.prefix_seeded_start=${PREFIX_SEEDED_START_ENABLED} \
  policy.generation.vllm_cfg.streaming_tool_call.prefill_after_admission=${PREFILL_AFTER_ADMISSION_ENABLED} \
  policy.generation.vllm_cfg.streaming_tool_call.background_prefill_completion=${BACKGROUND_PREFILL_COMPLETION_ENABLED} \
  policy.generation.vllm_cfg.streaming_tool_call.same_request_final_decode=${SAME_REQUEST_FINAL_DECODE_ENABLED} \
  policy.generation.vllm_cfg.streaming_tool_call.stable_first_snapshot_prefill=${STABLE_FIRST_SNAPSHOT_PREFILL_ENABLED} \
  policy.generation.vllm_cfg.streaming_tool_call.compact_request_context=${COMPACT_REQUEST_CONTEXT_ENABLED} \
  policy.generation.vllm_cfg.streaming_tool_call.incremental_tokenizer_checkpoint_interval=${INCREMENTAL_TOKENIZER_CHECKPOINT_INTERVAL} \
  policy.generation.vllm_cfg.streaming_tool_call.counterfactual_full_tokenizer_timing=${COUNTERFACTUAL_FULL_TOKENIZER_TIMING_ENABLED} \
  policy.generation.vllm_cfg.streaming_tool_call.snapshot_poll_interval_seconds=${SNAPSHOT_POLL_INTERVAL_SECONDS} \
  policy.generation.vllm_cfg.streaming_tool_call.snapshot_long_poll_timeout_seconds=${SNAPSHOT_LONG_POLL_TIMEOUT_SECONDS} \
  policy.generation.vllm_cfg.streaming_tool_call.event_driven_snapshot_wait=${STREAMING_EVENT_DRIVEN_SNAPSHOT_WAIT_ENABLED} \
  env.nemo_gym.streaming_tool_call.enabled=${STREAMING_TOOL_CALL_ENABLED} \
  env.nemo_gym.streaming_tool_call.tokenizer_only=${STREAMING_INCREMENTAL_TOKENIZER_ONLY_ENABLED} \
  env.nemo_gym.streaming_tool_call.exact_incremental_tokenizer=${EXACT_INCREMENTAL_TOKENIZER_ENABLED} \
  env.nemo_gym.streaming_tool_call.final_only_incremental_tokenizer=${FINAL_ONLY_INCREMENTAL_TOKENIZER_ENABLED} \
  env.nemo_gym.streaming_tool_call.final_only_prefill=${FINAL_ONLY_PREFILL_ENABLED} \
  env.nemo_gym.streaming_tool_call.final_only_prefill_completion_grace_seconds=${FINAL_ONLY_PREFILL_COMPLETION_GRACE_SECONDS} \
  env.nemo_gym.streaming_tool_call.prefix_seeded_start=${PREFIX_SEEDED_START_ENABLED} \
  env.nemo_gym.streaming_tool_call.prefill_after_admission=${PREFILL_AFTER_ADMISSION_ENABLED} \
  env.nemo_gym.streaming_tool_call.background_prefill_completion=${BACKGROUND_PREFILL_COMPLETION_ENABLED} \
  env.nemo_gym.streaming_tool_call.same_request_final_decode=${SAME_REQUEST_FINAL_DECODE_ENABLED} \
  env.nemo_gym.streaming_tool_call.stable_first_snapshot_prefill=${STABLE_FIRST_SNAPSHOT_PREFILL_ENABLED} \
  env.nemo_gym.streaming_tool_call.compact_request_context=${COMPACT_REQUEST_CONTEXT_ENABLED} \
  env.nemo_gym.streaming_tool_call.incremental_tokenizer_checkpoint_interval=${INCREMENTAL_TOKENIZER_CHECKPOINT_INTERVAL} \
  env.nemo_gym.streaming_tool_call.snapshot_poll_interval_seconds=${SNAPSHOT_POLL_INTERVAL_SECONDS} \
  env.nemo_gym.streaming_tool_call.snapshot_long_poll_timeout_seconds=${SNAPSHOT_LONG_POLL_TIMEOUT_SECONDS} \
  env.nemo_gym.streaming_tool_call.event_driven_snapshot_wait=${STREAMING_EVENT_DRIVEN_SNAPSHOT_WAIT_ENABLED} \
  ++env.nemo_gym.detailed_runtime_metrics=${DETAILED_RUNTIME_METRICS_ENABLED} \
  loss_fn.reference_policy_kl_penalty=${KL} \
  loss_fn.ratio_clip_min=${CLIP_MIN} \
  loss_fn.ratio_clip_max=${CLIP_MAX} \
  loss_fn.use_on_policy_kl_approximation=${USE_ON_POLICY_KL_APPROXIMATION} \
  loss_fn.use_importance_sampling_correction=${IMPORTANCE_SAMPLING_CORRECTION} \
  loss_fn.sequence_level_importance_ratios=${SEQ_LEVEL_IS} \
  loss_fn.token_level_loss=${TOKEN_LEVEL_LOSS} \
  loss_fn.force_on_policy_ratio=${FORCE_ON_POLICY_RATIO} \
  checkpointing.checkpoint_dir=${CHECKPOINT_DIR} \
  checkpointing.save_period=${SAVE_PERIOD} \
  checkpointing.keep_top_k=${KEEP_TOP_K} \
  ++checkpointing.metric_name=train:total_reward/mean \
  ++checkpointing.checkpoint_must_save_by=00:03:35:00 \
  logger.wandb_enabled=True \
  logger.log_dir=${LOGGER_LOG_DIR} \
  logger.wandb.name=${WANDB_NAME} \
  logger.wandb.project=${WANDB_PROJ} \
  ++logger.wandb.group=${WANDB_GROUP}"

if [ -n "${TOKENIZER_PATH}" ]; then
  export COMMAND="${COMMAND} policy.tokenizer.name=${TOKENIZER_PATH}"
fi

if [ -n "${STREAMING_MIN_CHUNK_CHARS}" ]; then
  export COMMAND="${COMMAND} \
  policy.generation.vllm_cfg.streaming_tool_call.min_chunk_chars=${STREAMING_MIN_CHUNK_CHARS} \
  env.nemo_gym.streaming_tool_call.min_chunk_chars=${STREAMING_MIN_CHUNK_CHARS}"
fi

if [ -n "${STREAMING_INITIAL_CHUNK_CHARS}" ]; then
  export COMMAND="${COMMAND} \
  policy.generation.vllm_cfg.streaming_tool_call.initial_chunk_chars=${STREAMING_INITIAL_CHUNK_CHARS} \
  env.nemo_gym.streaming_tool_call.initial_chunk_chars=${STREAMING_INITIAL_CHUNK_CHARS}"
fi

if [ "${ASYNC_GRPO_ENABLED}" = "True" ]; then
  export COMMAND="${COMMAND} \
  policy.generation.colocated.resources.num_nodes=${GEN_NODES} \
  policy.generation.colocated.resources.gpus_per_node=${NUM_GPU} \
  grpo.advantage_clip_low=${ADVANTAGE_CLIP_LOW} \
  grpo.advantage_clip_high=${ADVANTAGE_CLIP_HIGH} \
  loss_fn.truncated_importance_sampling_ratio=${TIS_THRESHOLD} \
  env.nemo_gym.swe_agents_train.responses_api_agents.swe_agents.agent_max_turns=${AGENT_MAX_TURNS} \
  env.nemo_gym.swe_agents_train.responses_api_agents.swe_agents.swebench_agent_timeout=${AGENT_TIMEOUT} \
  env.nemo_gym.swe_agents_train.responses_api_agents.swe_agents.concurrency=${CONCURRENCY} \
  env.nemo_gym.swe_agents_val.responses_api_agents.swe_agents.agent_max_turns=${AGENT_MAX_TURNS} \
  env.nemo_gym.swe_agents_val.responses_api_agents.swe_agents.swebench_agent_timeout=${AGENT_TIMEOUT} \
  env.nemo_gym.swe_agents_val.responses_api_agents.swe_agents.concurrency=${CONCURRENCY}"
fi

# Optional: cap training steps (smoke test).
if [ -n "${MAX_NUM_STEPS}" ]; then
  export COMMAND="${COMMAND} grpo.max_num_steps=${MAX_NUM_STEPS}"
fi

# Generation-only benchmark: no-op training (no optimizer) + disable checkpoint saving.
if [ "${SKIP_TRAINING}" = "1" ]; then
  export COMMAND="${COMMAND} ++grpo.gen_benchmark_skip_training=true checkpointing.enabled=false"
fi

# Rollout-only evaluation. Completed batches are appended to
# trajectory_collection.jsonl under logger.log_dir. Collection consumes the
# validation dataloader, whose SWE rows target swe_agents_val. Do not start the
# unused train server: train and val otherwise race to install the same shared
# Gym venv before either server can observe that it is ready.
if [ "${TRAJECTORY_COLLECTION}" = "1" ]; then
  export COMMAND="${COMMAND} \
  env.nemo_gym.is_trajectory_collection=true \
  env.nemo_gym.trajectory_collection_batch_size=${TRAJECTORY_COLLECTION_BATCH_SIZE} \
  env.nemo_gym.swe_agents_train=null"
fi

# The driver can build native wheels that setup did not need. Merge only
# complete cache entries after it exits, and preserve the driver's exit code so
# cache persistence never turns a failed workload into a successful job.
export COMMAND="${COMMAND}
_nrl_driver_exit=\$?
if [ '${UV_CACHE_SEED_MODE}' = 'persistent' ]; then
  (
    if flock -w 600 9; then
      if rsync -a --exclude '.tmp*' /tmp/uv_cache/ '${LUSTRE_UV_CACHE_SEED}/'; then
        touch '${LUSTRE_UV_CACHE_SEED}/.driver-writeback-mcore-v1'
        echo '[CACHE WRITEBACK] uv driver merge: complete'
      else
        echo '[CACHE WRITEBACK] WARNING: uv driver merge failed' >&2
      fi
    else
      echo '[CACHE WRITEBACK] WARNING: timed out waiting for uv seed lock' >&2
    fi
  ) 9>'${LUSTRE_UV_CACHE_SEED}.lock'
else
  echo '[CACHE WRITEBACK] uv driver merge: skipped in image-only mode'
fi
exit \${_nrl_driver_exit}"

# Validate the fully assembled driver without printing it; the command contains
# credentials and must never be echoed for diagnostics.
if ! bash -n <(printf '%s\n' "${COMMAND}"); then
  echo "ERROR: generated driver command is not valid Bash" >&2
  exit 1
fi

# ================ Submit job (skipped under DRY_RUN=1) ================
if [ "${DRY_RUN:-0}" = "1" ]; then
  echo ""
  if [ "${SUBMIT_MODE}" = "direct" ]; then
    echo "[DRY_RUN] Not launching. Would run ray.sub directly inside an existing ${TOTAL_NODES}-node, ${NUM_GPU}-GPU/node Slurm allocation."
  else
    echo "[DRY_RUN] Not submitting. Would run:"
    echo "[DRY_RUN]   sbatch --nodes=${TOTAL_NODES} --account=${SBATCH_ACCOUNT} --partition=${SBATCH_PARTITION} --time=${SBATCH_TIME} --dependency=${SBATCH_DEPENDENCY} --gpus-per-node=${NUM_GPU} ... ray.sub"
  fi
  cd - > /dev/null
  exit 0
fi

if [ "${SUBMIT_MODE}" = "direct" ]; then
  if [ -z "${SLURM_JOB_ID:-}" ]; then
    echo "ERROR: SUBMIT_MODE=direct requires an existing Slurm allocation." >&2
    exit 1
  fi
  ALLOCATED_NODES="${SLURM_NNODES:-${SLURM_JOB_NUM_NODES:-}}"
  if [ "${ALLOCATED_NODES}" != "${TOTAL_NODES}" ]; then
    echo "ERROR: direct allocation has ${ALLOCATED_NODES:-unknown} nodes; launcher requires ${TOTAL_NODES}." >&2
    exit 1
  fi
  if [ -n "${SLURM_GPUS_ON_NODE:-}" ] && [[ "${SLURM_GPUS_ON_NODE}" =~ ^[0-9]+$ ]] && [ "${SLURM_GPUS_ON_NODE}" -ne "${NUM_GPU}" ]; then
    echo "ERROR: direct allocation has ${SLURM_GPUS_ON_NODE} GPUs/node; launcher requires ${NUM_GPU}." >&2
    exit 1
  fi
  if [ -n "${SLURM_JOB_ACCOUNT:-}" ] && [ "${SLURM_JOB_ACCOUNT}" != "${SBATCH_ACCOUNT}" ]; then
    echo "ERROR: direct allocation account is ${SLURM_JOB_ACCOUNT}; expected ${SBATCH_ACCOUNT}." >&2
    exit 1
  fi
  if [ -n "${SLURM_JOB_PARTITION:-}" ]; then
    IFS=',' read -r -a ALLOCATED_PARTITIONS <<< "${SLURM_JOB_PARTITION}"
    for ALLOCATED_PARTITION in "${ALLOCATED_PARTITIONS[@]}"; do
      if [[ ",${SBATCH_PARTITION}," != *",${ALLOCATED_PARTITION},"* ]]; then
        echo "ERROR: direct allocation partition is ${SLURM_JOB_PARTITION}; expected one of ${SBATCH_PARTITION}." >&2
        exit 1
      fi
    done
  fi

  # ray.sub normally runs as the batch step. Under an attached srun, its Ray
  # head/worker steps must explicitly overlap the outer launcher step.
  export RAY_SUB_OVERLAP_EXISTING_STEP=1
  export NRL_CONTAINER_WORKDIR="${NRL_CONTAINER_WORKDIR:-${SNAPSHOT_DIR}}"
  echo "Launching directly in Slurm job ${SLURM_JOB_ID}: account=${SLURM_JOB_ACCOUNT:-unknown} partition=${SLURM_JOB_PARTITION:-unknown} nodes=${ALLOCATED_NODES} gpus/node=${SLURM_GPUS_ON_NODE:-unknown}"
  bash ray.sub
  echo "Direct Slurm job ${SLURM_JOB_ID} completed: ${EXP_SUFFIX}"
  cd - > /dev/null
  exit 0
fi

# A caller may launch this helper from an existing srun allocation. Do not
# propagate any of that allocation's SLURM job, step, task, CPU-binding, or GPU
# variables into the new independent batch job. The scheduler will populate a
# fresh SLURM_* environment for the submitted job. Keep the client-side cluster
# configuration needed by sbatch itself.
SUBMIT_SLURM_CONF="${SLURM_CONF:-}"
SUBMIT_SLURM_CLUSTER_NAME="${SLURM_CLUSTER_NAME:-}"
unset "${!SLURM_@}"
if [ -n "${SUBMIT_SLURM_CONF}" ]; then
  export SLURM_CONF="${SUBMIT_SLURM_CONF}"
fi
if [ -n "${SUBMIT_SLURM_CLUSTER_NAME}" ]; then
  export SLURM_CLUSTER_NAME="${SUBMIT_SLURM_CLUSTER_NAME}"
fi
unset CUDA_VISIBLE_DEVICES ROCR_VISIBLE_DEVICES HIP_VISIBLE_DEVICES
unset GPU_DEVICE_ORDINAL

sbatch \
  --nodes="${TOTAL_NODES}" \
  --account="${SBATCH_ACCOUNT}" \
  --job-name="${WANDB_NAME}" \
  --partition="${SBATCH_PARTITION}" \
  --time="${SBATCH_TIME}" \
  --gpus-per-node="${NUM_GPU}" \
  --output="${BASE_LOG_DIR}/slurm-%j.out" \
  --exclusive \
  --dependency="${SBATCH_DEPENDENCY}" \
  --comment='{"OccupiedIdleGPUsJobReaper":{"exemptIdleTimeMins":"180","reason":"data_loading","description":"Async GRPO SWE generation-scaling benchmark"}}' \
  ray.sub | tee /dev/stderr | grep -o '[0-9]\+' > latest_scale_gen_job_id.txt

JOB_ID="$(cat latest_scale_gen_job_id.txt)"
echo "=========================================="
echo "Job submitted: ${EXP_SUFFIX}"
echo "Job ID: ${JOB_ID}"
echo "wandb group: ${WANDB_GROUP}"
echo "Monitor with: squeue -j ${JOB_ID}"
echo "Ray/SLURM logs: ${BASE_LOG_DIR}/${JOB_ID}-logs/"
echo "Checkpoints: ${CHECKPOINT_DIR}/"
echo "=========================================="

cd - > /dev/null
