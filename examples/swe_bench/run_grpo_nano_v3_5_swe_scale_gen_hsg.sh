#!/bin/bash
# ============================================================================
# nano V3.5 SWE-e2e GRPO reproduction launcher (my conventions).
#
# Adapted from sdevare's repro_nano.sh
#   (.../code_snapshots/sdd-swe-e2e-nano-...-gen-nodes-32/scripts/repro_nano.sh)
# but rewritten to match this checkout's launch pattern (see
# test_assets/ultra_SWE/repro_ultra_launch.sh):
#   - SLURM account nemotron_sw_post, my container / results / logs paths.
#   - Secrets sourced from ~/script/export_env_vars.sh (no hardcoded keys).
#   - DRY_RUN=1 default with secret redaction; require_path preflight checks.
#   - SIF container_formatter + apptainer setup + compile-cache sidecar.
#
# Run shape (matches the nano gen-nodes-32 snapshot):
#   - Model:   mopd_ultrav3_to_nanov3_5_repro_v5 step_18 (KD opt full).
#   - Geometry: train TP4/CP16/EP32/PP1/ETP1, vLLM TP4/PP1, max_len 196608.
#   - Nodes:   32 train + 32 generation (non-colocated, yaml default), gym=0.
#   - Batch:   PPS=32, GPP=16, GBS=512.
#   - Precision: bf16 by default; mxfp8 recipes via PRECISION_RECIPE (see below).
#
# Examples:
#   DRY_RUN=1 bash test_assets/nanoV3_5/repro_nano.sh
#   MAX_NUM_STEPS=4 DRY_RUN=1 bash test_assets/nanoV3_5/repro_nano.sh
#   PRECISION_RECIPE=mxfp8-rollout DRY_RUN=1 bash test_assets/nanoV3_5/repro_nano.sh
#   bash test_assets/nanoV3_5/repro_nano.sh                       # real submit
#   SLURM_QOS= SLURM_PARTITION=batch bash test_assets/nanoV3_5/repro_nano.sh
# ============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# nanoV3_5 lives two levels under the repo root (test_assets/nanoV3_5/).
REPO_ROOT="${REPO_ROOT:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"

# ====================== Source-commit validation ======================
# This launcher reproduces the W&B run logged in nvidia/ultra-v3-swe-e2e
# (run name == EXP_SUFFIX below: sdd-swe-e2e-nano-ultra-prod-nano-swe-e2e-...
# -gen-nodes-32), which ran from this commit. The submodule revisions are pinned
# deterministically by this superproject commit, so verifying HEAD is sufficient:
#   Automodel        1d42deb98169fd94b54c714c0fe4bf308fe7115a
#   Gym              bd1050cf07b0a0aceb4a63614781f973e74dcdfe
#   Megatron-Bridge  4e0209d36597f91026ed5ba42967ad3d9e8ea705
#   Megatron-LM      ceb31e61faed0a2e844c1495a877163ddb253694
# Set ALLOW_COMMIT_MISMATCH=1 to run intentionally from different code. Set
# EXPECTED_COMMIT= (empty) to skip the check entirely.
#
# NOTE (current-repo reproduction): this copy lives in a different checkout/branch
# than the upstream run above (HEAD here is not 9d08c1b2...), so the check defaults
# to empty (skipped). The upstream commit is kept above for provenance. Export
# EXPECTED_COMMIT=<sha> to re-enable pinning against a specific commit.
EXPECTED_COMMIT="${EXPECTED_COMMIT-}"
if [ -n "${EXPECTED_COMMIT}" ]; then
  CURRENT_COMMIT="$(git -C "${REPO_ROOT}" -c safe.directory="${REPO_ROOT}" rev-parse HEAD 2>/dev/null || echo unknown)"
  if [ "${CURRENT_COMMIT}" != "${EXPECTED_COMMIT}" ]; then
    echo "ERROR: this run reproduces commit ${EXPECTED_COMMIT}," >&2
    echo "       but ${REPO_ROOT} HEAD is ${CURRENT_COMMIT}." >&2
    echo "  Fix: git -C ${REPO_ROOT} checkout ${EXPECTED_COMMIT} \\" >&2
    echo "         && git -C ${REPO_ROOT} submodule update --init --recursive" >&2
    echo "  Or:  ALLOW_COMMIT_MISMATCH=1 (proceed) / EXPECTED_COMMIT= (skip check)." >&2
    if [ "${ALLOW_COMMIT_MISMATCH:-0}" != "1" ]; then
      exit 1
    fi
    echo "WARNING: proceeding despite commit mismatch (ALLOW_COMMIT_MISMATCH=1)." >&2
  fi
fi

# ============================ Precision recipe ============================
# bf16 (default) or one of: mxfp8-rollout, mxfp8-train, mxfp8-e2e.
# These emit extra Hydra overrides appended last so they win over the bf16
# defaults set in the command below.
get_precision_config() {
  local PRECISION_RECIPE="$1"
  local DISABLE_FP8_LINEAR="$2"
  local DISABLE_FP8_MOE="$3"
  local ENABLE_FP8_PARAM_IN_TRAIN="$4"
  local PRECISION_EXTRA_ARGS=""

  local MXFP8_GEN_EXTRA_ARGS="policy.generation.vllm_cfg.precision=fp8 \
++policy.generation.vllm_cfg.fp8_cfg.is_mx=true \
policy.generation.vllm_cfg.gpu_memory_utilization=0.8 \
policy.generation.vllm_cfg.tensor_parallel_size=4 \
policy.generation.vllm_cfg.expert_parallel_size=4"

  local IGNORED_LAYER_KWS="\"conv1d\",\"mtp\""
  if [ "$DISABLE_FP8_MOE" == "1" ]; then
    IGNORED_LAYER_KWS="$IGNORED_LAYER_KWS,\".experts.\""
  fi
  if [ "$DISABLE_FP8_LINEAR" == "1" ]; then
    IGNORED_LAYER_KWS="$IGNORED_LAYER_KWS,\"in_proj\",\"out_proj\",\"q_proj\",\"k_proj\",\"v_proj\",\"o_proj\",\"fc1_latent_proj\",\"fc2_latent_proj\",\"shared_experts\""
  fi
  MXFP8_GEN_EXTRA_ARGS="$MXFP8_GEN_EXTRA_ARGS +policy.generation.vllm_cfg.quantization_ignored_layer_kws=[$IGNORED_LAYER_KWS]"

  local MXFP8_TRAIN_EXTRA_ARGS="policy.megatron_cfg.fp8_cfg.enabled=true \
policy.megatron_cfg.fp8_cfg.fp8=\"e4m3\" \
policy.megatron_cfg.fp8_cfg.fp8_recipe=\"mxfp8\" \
++policy.megatron_cfg.fp8_cfg.fp8_param=false \
policy.megatron_cfg.moe_router_dtype=fp32 \
policy.megatron_cfg.expert_model_parallel_size=64 \
"

  local MXFP8_PARAM_EXTRA_ARGS="++policy.megatron_cfg.fp8_cfg.fp8_param=true \
+policy.megatron_cfg.optimizer.reuse_grad_buf_for_mxfp8_param_ag=true \
+policy.megatron_cfg.optimizer.fp8_recipe=mxfp8 \
+policy.megatron_cfg.optimizer.overlap_param_gather=true \
++policy.megatron_cfg.distributed_data_parallel_config.overlap_param_gather=true \
++policy.megatron_cfg.distributed_data_parallel_config.overlap_grad_reduce=true \
"

  if [ "$ENABLE_FP8_PARAM_IN_TRAIN" == "1" ]; then
    MXFP8_TRAIN_EXTRA_ARGS="$MXFP8_TRAIN_EXTRA_ARGS $MXFP8_PARAM_EXTRA_ARGS"
  fi

  if [ "$PRECISION_RECIPE" == "mxfp8-rollout" ]; then
    PRECISION_EXTRA_ARGS="$MXFP8_GEN_EXTRA_ARGS"
  elif [ "$PRECISION_RECIPE" == "mxfp8-train" ]; then
    PRECISION_EXTRA_ARGS="$MXFP8_TRAIN_EXTRA_ARGS"
  elif [ "$PRECISION_RECIPE" == "mxfp8-e2e" ]; then
    PRECISION_EXTRA_ARGS="$MXFP8_GEN_EXTRA_ARGS $MXFP8_TRAIN_EXTRA_ARGS"
  else
    PRECISION_EXTRA_ARGS=""
  fi

  echo "${PRECISION_EXTRA_ARGS}"
}

PRECISION_RECIPE="${PRECISION_RECIPE:-bf16}"
DISABLE_FP8_LINEAR="${DISABLE_FP8_LINEAR:-0}"
DISABLE_FP8_MOE="${DISABLE_FP8_MOE:-0}"
ENABLE_FP8_PARAM_IN_TRAIN="${ENABLE_FP8_PARAM_IN_TRAIN:-0}"
PRECISION_EXTRA_ARGS="$(get_precision_config "${PRECISION_RECIPE}" "${DISABLE_FP8_LINEAR}" "${DISABLE_FP8_MOE}" "${ENABLE_FP8_PARAM_IN_TRAIN}")"
# bf16 and mxfp8 compile different subgraphs under the same torch.compile hash,
# so they MUST use separate vLLM compile-cache trees on Lustre.
case "${PRECISION_RECIPE}" in
  mxfp8-rollout|mxfp8-e2e) VLLM_CACHE_PRECISION="mxfp8" ;;
  *)                       VLLM_CACHE_PRECISION="bf16"  ;;
esac

# =========== Scaling modes: ALIGN_BASELINE | default (R) | SKIP_TRAINING ===========
# One knob NUM_VLLM_REPLICAS (R) = vLLM gen replicas = gen nodes. The mode picks the
# TRAINING geometry (CP/EP -> nodes-per-DP); nodes/batch/segment are then derived.
#
#   ALIGN_BASELINE=1  -> reproduce the validated 32-node baseline: CP=16/EP=32, R
#                        defaults to 16 -> 16 train + 16 gen = 32 nodes, GBS=256.
#   (default, R set)  -> real training; scale BOTH train and gen from the validated
#                        2-node/DP base CP=2/EP=8 (NODES_PER_DP=2). train grows with R
#                        (DP=ceil(R/NODES_PER_DP)), gen=R. NOTE: the CP=1/EP=4 1-node
#                        base OOMs on both GPU and host, so 2 nodes/DP is the stable base.
#   SKIP_TRAINING=1   -> generation benchmark (NVIDIA-NeMo/RL#2930). Training pinned to
#                        a minimal 1-node policy cluster (CP=1/EP=4, no optimizer built ->
#                        no OOM); train does NOT scale, only gen (=R). Sets
#                        NRL_GEN_BENCHMARK_SKIP_TRAINING=1.
#
# FIXED across modes: TP=4, PP=1, ETP=1, GPP=16, VLLM_TP=4, MAX_LENGTH=196608. Any of
# CP/EP/NUM_TRAIN_NODES/GBS/PPS/SBATCH_SEGMENT can still be overridden explicitly.
#
# Examples:
#   ALIGN_BASELINE=1 bash examples/swe_bench/run_grpo_nano_v3_5_swe_scale_gen_hsg.sh
#   NUM_VLLM_REPLICAS=8 bash examples/swe_bench/run_grpo_nano_v3_5_swe_scale_gen_hsg.sh
#   SKIP_TRAINING=1 NUM_VLLM_REPLICAS=8 bash examples/swe_bench/run_grpo_nano_v3_5_swe_scale_gen_hsg.sh
NUM_GPU=4
TP="${TP:-4}"
PP="${PP:-1}"
ETP="${ETP:-1}"
GPP="${GPP:-16}"
VLLM_TP="${VLLM_TP:-4}"
VLLM_PP="${VLLM_PP:-1}"
MAX_LENGTH="${MAX_LENGTH:-196608}"
PER_GEN_REPLICA_BATCH="${PER_GEN_REPLICA_BATCH:-16}"

ALIGN_BASELINE="${ALIGN_BASELINE:-0}"
SKIP_TRAINING="${SKIP_TRAINING:-0}"
if [ "${ALIGN_BASELINE}" = "1" ] && [ "${SKIP_TRAINING}" = "1" ]; then
  echo "ERROR: set only one of ALIGN_BASELINE / SKIP_TRAINING." >&2; exit 1
fi

if [ "${ALIGN_BASELINE}" = "1" ]; then
  CP="${CP:-16}"; EP="${EP:-32}"                 # Mode 1: 32-node baseline geometry
  NUM_VLLM_REPLICAS="${NUM_VLLM_REPLICAS:-16}"   # -> 16 gen + 16 train = 32 nodes
elif [ "${SKIP_TRAINING}" = "1" ]; then
  CP="${CP:-1}"; EP="${EP:-4}"                    # Mode 3: gen benchmark, minimal train
  export NRL_GEN_BENCHMARK_SKIP_TRAINING=1
else
  CP="${CP:-2}"; EP="${EP:-8}"                    # Mode 2: validated 2-node/DP stable base
fi

NUM_VLLM_REPLICAS="${NUM_VLLM_REPLICAS:-}"
if [ -z "${NUM_VLLM_REPLICAS}" ]; then
  echo "ERROR: NUM_VLLM_REPLICAS is required (= vLLM gen replicas = gen nodes)." >&2
  echo "       e.g. NUM_VLLM_REPLICAS=8, or ALIGN_BASELINE=1 for the 32-node baseline." >&2
  exit 1
fi
if ! printf '%s' "${NUM_VLLM_REPLICAS}" | grep -qE '^[0-9]+$' || [ "${NUM_VLLM_REPLICAS}" -lt 1 ]; then
  echo "ERROR: NUM_VLLM_REPLICAS must be a positive integer, got ${NUM_VLLM_REPLICAS}" >&2
  exit 1
fi
R="${NUM_VLLM_REPLICAS}"

NUM_GEN_NODES="${NUM_GEN_NODES:-${R}}"
NODES_PER_DP=$(( (TP * CP * PP) / NUM_GPU ))       # nodes for one train DP replica
[ "${NODES_PER_DP}" -lt 1 ] && NODES_PER_DP=1
if [ "${SKIP_TRAINING}" = "1" ]; then
  # gen benchmark: train pinned to a single minimal DP replica; does NOT scale with R.
  NUM_TRAIN_NODES="${NUM_TRAIN_NODES:-${NODES_PER_DP}}"
else
  # scale train data-parallelism with R: DP = ceil(R / NODES_PER_DP).
  TRAIN_DP_DERIVED=$(( (R + NODES_PER_DP - 1) / NODES_PER_DP ))
  [ "${TRAIN_DP_DERIVED}" -lt 1 ] && TRAIN_DP_DERIVED=1
  NUM_TRAIN_NODES="${NUM_TRAIN_NODES:-$(( NODES_PER_DP * TRAIN_DP_DERIVED ))}"
fi
NUM_GYM_NODES="${NUM_GYM_NODES:-0}"
TOTAL_NODES=$(( NUM_TRAIN_NODES + NUM_GEN_NODES + NUM_GYM_NODES ))

TRAIN_GPUS=$(( NUM_TRAIN_NODES * NUM_GPU ))
ATTN_BASE=$(( TP * CP * PP ))
EXPERT_BASE=$(( ETP * EP * PP ))
if [ $(( TRAIN_GPUS % ATTN_BASE )) -ne 0 ]; then
  echo "ERROR: train GPUs ${TRAIN_GPUS} not divisible by TP*CP*PP=${ATTN_BASE}." >&2; exit 1
fi
if [ $(( TRAIN_GPUS % EXPERT_BASE )) -ne 0 ]; then
  echo "ERROR: train GPUs ${TRAIN_GPUS} not divisible by ETP*EP*PP=${EXPERT_BASE}." >&2; exit 1
fi
TRAIN_DP=$(( TRAIN_GPUS / ATTN_BASE ))

GBS="${GBS:-$(( PER_GEN_REPLICA_BATCH * R ))}"
if [ $(( GBS % GPP )) -ne 0 ]; then
  echo "ERROR: GBS=${GBS} not divisible by GPP=${GPP}." >&2; exit 1
fi
PPS="${PPS:-$(( GBS / GPP ))}"
if [ $(( PPS % TRAIN_DP )) -ne 0 ]; then
  echo "ERROR: PPS=${PPS} must be divisible by train DP=${TRAIN_DP} (prompts split across DP)." >&2
  echo "       Pick R so that R is a multiple of ${TRAIN_DP}." >&2; exit 1
fi
CONCURRENCY="${CONCURRENCY:-$(( 2 * PPS * GPP ))}"

# Topology segment: largest of {16,8,4,2,1} that DIVIDES *each* worker group's node
# count (train, gen, gym) -- not just the total. nemo-rl's RayVirtualCluster segments
# each group separately and asserts num_nodes % segment_size == 0 per group, so a
# segment that divides the total but not a group (e.g. total=2 but train=1/gen=1)
# fails with "num_nodes (1) must be divisible by segment_size (2)". Override with
# SBATCH_SEGMENT=<n>.
if [ -z "${SBATCH_SEGMENT:-}" ]; then
  for s in 16 8 4 2 1; do
    if [ $(( NUM_TRAIN_NODES % s )) -eq 0 ] && [ $(( NUM_GEN_NODES % s )) -eq 0 ] \
       && { [ "${NUM_GYM_NODES}" -eq 0 ] || [ $(( NUM_GYM_NODES % s )) -eq 0 ]; }; then
      SBATCH_SEGMENT="$s"; break
    fi
  done
fi
CLUSTER_SEGMENT_SIZE="${CLUSTER_SEGMENT_SIZE:-${SBATCH_SEGMENT}}"

_MODE="scale train+gen (real training)"
[ "${ALIGN_BASELINE}" = "1" ] && _MODE="ALIGN_BASELINE (32-node baseline)"
[ "${SKIP_TRAINING}" = "1" ] && _MODE="SKIP_TRAINING (gen benchmark; NRL_GEN_BENCHMARK_SKIP_TRAINING=1)"
echo "=========================================="
echo "nano V3.5 SWE scale-gen  |  mode: ${_MODE}  |  R=${R}"
echo "  nodes:   train=${NUM_TRAIN_NODES} (DP=${TRAIN_DP}), gen=${NUM_GEN_NODES}, gym=${NUM_GYM_NODES}, total=${TOTAL_NODES}"
echo "  batch:   PPS=${PPS}, GPP=${GPP}, GBS=${GBS}, per-replica=$(( GBS / R )), concurrency=${CONCURRENCY}"
echo "  parall:  TP=${TP}, CP=${CP}, EP=${EP}, PP=${PP}, ETP=${ETP} (nodes/DP=${NODES_PER_DP}); vLLM_TP=${VLLM_TP}, vLLM_PP=${VLLM_PP}; max_length=${MAX_LENGTH}"
echo "  segment: slurm=${SBATCH_SEGMENT}, cluster=${CLUSTER_SEGMENT_SIZE}  (auto: largest {16..1} dividing each worker group)"
echo "=========================================="

# ============================ Paths ============================
EXP_NAME="${EXP_NAME:-nano-v3-5-swe-${USER}-r${R}}"
# In-repo authoritative config (nano_v3 reasoning parser + 6-family container_formatter baked in).
CONFIG_PATH="${CONFIG_PATH:-${REPO_ROOT}/examples/swe_bench/grpo_nano_v3_5_swe_hsg.yaml}"
# nano V3.5 student: mopd ultrav3 -> nanov3_5 repro v5 KD opt full, step_18.
MODEL_PATH="${MODEL_PATH:-/lustre/fsw/portfolios/llmservice/users/pjin/devel/nemo-rl-ultra-v3-nano-opd-dev-20260513/results/mopd_ultrav3_to_nanov3_5_repro_v5_kd_opt_full-hsg-20260524-r1/step_18/hf}"
TRAIN_PATH="${TRAIN_PATH:-/lustre/fsw/portfolios/llmservice/users/sdevare/repos/ultra/datasets/swe/blends/large_root_cause_curriculum_with_mercor_ots_plus_singlefile_swerebench_overlap_fix.jsonl}"
VAL_PATH="${VAL_PATH:-/lustre/fsw/portfolios/llmservice/users/sdevare/repos/ultra/datasets/swe/swe_public_datasets_val_swebench.jsonl}"
SIF_DIR="${SIF_DIR:-/lustre/fsw/portfolios/llmservice/users/sdevare/images}"

# Results/logs live under the repo this job is submitted from (REPO_ROOT),
# so a run's artifacts stay with the checkout that produced them.
RESULTS_ROOT="${RESULTS_ROOT:-${REPO_ROOT}/results}"
LOG_ROOT="${LOG_ROOT:-${REPO_ROOT}/logs}"
RESULTS_DIR="${RESULTS_DIR:-${RESULTS_ROOT}/${EXP_NAME}}"
RUN_LOG_DIR="${RUN_LOG_DIR:-${LOG_ROOT}/${EXP_NAME}}"
NEMO_LOG_DIR="${NEMO_LOG_DIR:-${RUN_LOG_DIR}/nemo}"

RAY_SUB="${RAY_SUB:-${REPO_ROOT}/ray.sub}"
ENTRYPOINT="${ENTRYPOINT:-${REPO_ROOT}/examples/nemo_gym/run_grpo_nemo_gym.py}"
GYM_CODE="${GYM_CODE:-${REPO_ROOT}/3rdparty/Gym-workspace/Gym}"
LATEST_JOB_ID_FILE="${LATEST_JOB_ID_FILE:-${SCRIPT_DIR}/latest_nano_v3_5_job_id.txt}"

# ========================= Container / mounts =========================
# Container with gym venvs baked in at /opt/gym_venvs and Ray/Python matching this
# checkout (built from nightly-063026). Override CONTAINER=... for a different image.
CONTAINER="${CONTAINER:-/lustre/fsw/portfolios/nemotron/users/ruit/enroot-images/nemo-rl:nightly-063026-gymvenvs.squashfs}"
SANDBOX_CONTAINER="${SANDBOX_CONTAINER:-/lustre/fsw/portfolios/llmservice/users/igitman/images/nemo-skills-sandbox-b620e79.sqsh}"

EXTRA_MOUNTS="${EXTRA_MOUNTS:-/lustre:/lustre}"
MOUNTS="${EXTRA_MOUNTS}"
append_mount() {
  local src="$1"
  local dst="$2"
  if [ -d "${src}" ] || [ -f "${src}" ]; then
    MOUNTS="${MOUNTS},${src}:${dst}"
  else
    echo "WARNING: mount source missing, using container built-in if available: ${src}" >&2
  fi
}
append_mount "${REPO_ROOT}" "${REPO_ROOT}"
append_mount "${GYM_CODE}" "/opt/nemo-rl/3rdparty/Gym-workspace/Gym"
export MOUNTS
export CONTAINER
export SANDBOX_CONTAINER

# ======================= Cluster / resources =======================
NUM_GPU=4
NUM_TRAIN_NODES="${NUM_TRAIN_NODES:-32}"
NUM_GEN_NODES="${NUM_GEN_NODES:-32}"
NUM_GYM_NODES="${NUM_GYM_NODES:-0}"
TOTAL_NODES=$((NUM_TRAIN_NODES + NUM_GEN_NODES + NUM_GYM_NODES))

export GPUS_PER_NODE="${NUM_GPU}"
export CPUS_PER_WORKER="${CPUS_PER_WORKER:-144}"

# ============================ nano geometry ============================
TP="${TP:-4}"
CP="${CP:-16}"
EP="${EP:-32}"
PP="${PP:-1}"
ETP="${ETP:-1}"
GPP="${GPP:-16}"
PPS="${PPS:-32}"
GBS="${GBS:-512}"
VLLM_TP="${VLLM_TP:-4}"
VLLM_PP="${VLLM_PP:-1}"
VLLM_GPU_UTIL="${VLLM_GPU_UTIL:-0.85}"
MAX_LENGTH="${MAX_LENGTH:-196608}"
MAX_NUM_STEPS="${MAX_NUM_STEPS:-1000000}"
CONCURRENCY="${CONCURRENCY:-$((2 * PPS * GPP))}"
USE_MULTIPLE_DATALOADER="${USE_MULTIPLE_DATALOADER:-False}"
# logprob_chunk_size + fuse_loss: aligned to the source nano script (yaml defaults
# to 2048; the prod nano run overrode to 1024 with fused loss).
LOGPROB_CHUNK_SIZE="${LOGPROB_CHUNK_SIZE:-1024}"
FUSE_LOSS="${FUSE_LOSS:-true}"
# Empty = use the yaml value. Set only when a newer code path requires overriding.
GRADIENT_ACCUMULATION_FUSION="${GRADIENT_ACCUMULATION_FUSION:-}"
REASONING_PARSER_PLUGIN="${REASONING_PARSER_PLUGIN:-}"
MAKE_SEQUENCE_LENGTH_DIVISIBLE_BY="${MAKE_SEQUENCE_LENGTH_DIVISIBLE_BY:-}"
if [ -n "${MAKE_SEQUENCE_LENGTH_DIVISIBLE_BY}" ] && ! printf '%s' "${MAKE_SEQUENCE_LENGTH_DIVISIBLE_BY}" | grep -qE '^[0-9]+$'; then
  echo "ERROR: MAKE_SEQUENCE_LENGTH_DIVISIBLE_BY must be an integer, got ${MAKE_SEQUENCE_LENGTH_DIVISIBLE_BY}" >&2
  exit 1
fi

CHECKPOINTING_SAVE_BY="${CHECKPOINTING_SAVE_BY:-00:03:30:00}"
SAVE_PERIOD="${SAVE_PERIOD:-5}"
SAVE_OPTIMIZER="${SAVE_OPTIMIZER:-true}"
WANDB_PROJ="${WANDB_PROJ:-ruit-nano-v3-5-swe}"
WANDB_GROUP="${WANDB_GROUP:-nano-v3-5-swe}"
# Tag the wandb name as the internal-repo reproduction.
WANDB_NAME="${WANDB_NAME:-${EXP_NAME}-${PRECISION_RECIPE}-internal-repo}"

# ========================= SLURM submission =========================
SLURM_ACCOUNT="${SLURM_ACCOUNT-nemotron_sw_post}"
SLURM_PARTITION="${SLURM_PARTITION-batch_long}"
# Default: no explicit QOS -> partition's default QOS (hero-res needs a matching
# reservation/permission, which raises "Invalid qos specification" otherwise).
# Opt in with SLURM_QOS=hero-res SLURM_RESERVATION=<res> when you have one.
SLURM_QOS="${SLURM_QOS-}"
SLURM_RESERVATION="${SLURM_RESERVATION-}"
# Idle-GPU reaper exemption. Async GRPO leaves the training GPUs idle while rollouts
# are collected (~30min for SWE) and during per-step validation, which trips the
# default OccupiedIdleGPUsJobReaper (cancels at 30min idle). This --comment JSON
# raises the exemption to 60min. Override with SLURM_COMMENT=..., disable with
# SLURM_COMMENT= (empty).
DEFAULT_SLURM_COMMENT='{"OccupiedIdleGPUsJobReaper":{"exemptIdleTimeMins":"60","reason":"data_loading","description":"Async GRPO RL training: training GPUs idle during rollout collection (~30min) and validation each step"}}'
SLURM_COMMENT="${SLURM_COMMENT-$DEFAULT_SLURM_COMMENT}"
WALLTIME="${WALLTIME:-4:00:00}"
# SBATCH_SEGMENT / CLUSTER_SEGMENT_SIZE are derived above from the total node count
# (largest divisor of {16,8,4,2,1}); not re-defaulted here to avoid pinning 16.

# ========================= Environment variables =========================
# Source the user's env file by default (provides WANDB_API_KEY, HF_TOKEN and
# GITLAB_PAT, the latter needed for the flashinfer internal PyPI index). Set
# SOURCE_USER_ENV=0 to skip -- note this file may also mutate global git config.
if [ "${SOURCE_USER_ENV:-1}" = "1" ] && [ -f "${HOME}/script/export_env_vars.sh" ]; then
  # shellcheck disable=SC1090
  source "${HOME}/script/export_env_vars.sh"
fi

HF_HOME="${HF_HOME:-/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/${USER}/hf_home}"
HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${HF_HOME}/datasets}"
PERSISTENT_CACHE="${PERSISTENT_CACHE:-${REPO_ROOT}/.cache/nemotron_nano_v3_5}"
# Default points at the venvs baked into the CONTAINER above (/opt/gym_venvs); with
# skip_venv_if_present=true the gym reuses them and skips the build entirely.
# Do NOT default this to Lustre: building the editable nemo-gym venv there hangs on
# uv's flock. If you use a container WITHOUT baked venvs, set this to a node-local
# path (e.g. /tmp/nemo_gym_venvs) so the runtime build has working flock + fast IO.
NEMO_GYM_VENV_DIR="${NEMO_GYM_VENV_DIR:-/opt/gym_venvs}"
# The uv distribution-cache lock (editable nemo-gym build) is what hangs on Lustre,
# and it lives in the *uv cache*, not the venv dir. So the gym's uv cache must also
# be node-local, else moving only the venv doesn't help. Kept separate from the main
# NeMo-RL UV_CACHE_DIR (which stays on Lustre for package persistence).
NEMO_GYM_UV_CACHE="${NEMO_GYM_UV_CACHE:-/tmp/nemo_gym_uv_cache}"

# wandb run files + artifact *staging* default to XDG_DATA_HOME (~/.local/share/wandb),
# which on this setup points at the coreai Lustre quota -- that filled up and aborted a
# run mid-artifact-write with "Disk quota exceeded". Redirect wandb to the roomy
# nemotron project space instead.
WANDB_STAGE_ROOT="${WANDB_STAGE_ROOT:-/lustre/fs1/portfolios/nemotron/projects/nemotron_sw_post/users/ruit/wandb_stage}"
WANDB_DIR="${WANDB_DIR:-${WANDB_STAGE_ROOT}/dir}"
WANDB_CACHE_DIR="${WANDB_CACHE_DIR:-${WANDB_STAGE_ROOT}/cache}"
WANDB_DATA_DIR="${WANDB_DATA_DIR:-${WANDB_STAGE_ROOT}/data}"
export WANDB_DIR WANDB_CACHE_DIR WANDB_DATA_DIR

export HF_HOME
export HF_DATASETS_CACHE
export PERSISTENT_CACHE
export NEMO_GYM_VENV_DIR
export BASE_LOG_DIR="${RUN_LOG_DIR}"
export RAY_LOG_SYNC_FREQUENCY="${RAY_LOG_SYNC_FREQUENCY:-60}"
export CACHE_SYNC_FREQUENCY="${CACHE_SYNC_FREQUENCY:-1800}"

# vLLM compile cache is precision-scoped (bf16 vs mxfp8 must not share a tree).
LUSTRE_VLLM_CACHE="${PERSISTENT_CACHE}/cache_write/vllm_compile_cache_${VLLM_CACHE_PRECISION}"
LUSTRE_INDUCTOR_CACHE="${PERSISTENT_CACHE}/cache_write/inductor_cache"
LUSTRE_TRITON_CACHE="${PERSISTENT_CACHE}/cache_write/triton_cache"
INDUCTOR_CACHE_DIR="/tmp/nemo_rl_inductor_cache"
TRITON_CACHE_DIR="/tmp/nemo_rl_triton_cache"
VLLM_PRECOMPILED_WHEEL_LOCATION="${VLLM_PRECOMPILED_WHEEL_LOCATION:-https://github.com/vllm-project/vllm/releases/download/v0.17.0/vllm-0.17.0-cp38-abi3-manylinux_2_31_aarch64.whl}"

# NOTE: do NOT mkdir NEMO_GYM_VENV_DIR here. It is a *container-internal* path
# (e.g. /opt/gym_venvs when baked into the sqsh, or node-local /tmp created at
# runtime inside the container) -- creating it from the login node either fails
# (/opt not writable) or is pointless (node-local /tmp differs per node).
mkdir -p "${RESULTS_DIR}" "${RUN_LOG_DIR}" "${NEMO_LOG_DIR}" \
  "${LUSTRE_VLLM_CACHE}" "${LUSTRE_INDUCTOR_CACHE}" "${LUSTRE_TRITON_CACHE}" \
  "${HF_HOME}" "${HF_DATASETS_CACHE}" \
  "${WANDB_DIR}" "${WANDB_CACHE_DIR}" "${WANDB_DATA_DIR}"

# Record the exact superproject + submodule revisions actually used, for repro.
{
  echo "expected_commit: ${EXPECTED_COMMIT:-<skipped>}"
  echo "superproject:    $(git -C "${REPO_ROOT}" -c safe.directory="${REPO_ROOT}" rev-parse HEAD 2>/dev/null || echo unknown)"
  echo "submodules (status --recursive):"
  git -C "${REPO_ROOT}" -c safe.directory="${REPO_ROOT}" submodule status --recursive 2>/dev/null || echo "  <unavailable>"
} > "${RUN_LOG_DIR}/git-revision.txt" 2>/dev/null || true

require_path() {
  local path="$1"
  local label="$2"
  if [ ! -e "${path}" ]; then
    echo "ERROR: missing ${label}: ${path}" >&2
    exit 1
  fi
}

# vLLM editable source tree: pyproject declares `vllm = {path="3rdparty/vllm", editable}`,
# but that dir is created by the docker build and only exists inside the container at
# /opt/nemo-rl/3rdparty/vllm. In overlay mode we cd into the (mounted) REPO_ROOT and run
# uv there, so REPO_ROOT/3rdparty/vllm must resolve to the container's copy. Create a
# symlink (dangling on the host, resolves inside the container).
if [ ! -e "${REPO_ROOT}/3rdparty/vllm" ] && [ ! -L "${REPO_ROOT}/3rdparty/vllm" ]; then
  mkdir -p "${REPO_ROOT}/3rdparty"
  ln -s /opt/nemo-rl/3rdparty/vllm "${REPO_ROOT}/3rdparty/vllm" \
    && echo "[INFO] linked ${REPO_ROOT}/3rdparty/vllm -> /opt/nemo-rl/3rdparty/vllm (resolves inside container)"
fi

require_path "${CONFIG_PATH}" "config"
require_path "${MODEL_PATH}" "model"
require_path "${TRAIN_PATH}" "train data"
require_path "${VAL_PATH}" "validation data"
require_path "${SIF_DIR}" "SIF root"
require_path "${CONTAINER}" "container"
require_path "${RAY_SUB}" "ray.sub"
require_path "${ENTRYPOINT}" "training entrypoint"
require_path "${GYM_CODE}/nemo_gym/__init__.py" "Gym checkout"

for sif_subdir in swerebench nv_internal r2e_gym swegym swebench mercor/swebenchpro_ots; do
  require_path "${SIF_DIR}/${sif_subdir}" "SIF subdir ${sif_subdir}"
done

if [ "${DRY_RUN:-0}" != "1" ] && [ -z "${WANDB_API_KEY:-}" ]; then
  echo "ERROR: WANDB_API_KEY must be set for a real submission." >&2
  exit 1
fi

# Per-instance .sif resolution: swe_agents tries each pattern in order and uses the
# first existing file. Mirrors test_assets/ultra_SWE/repro_ultra_launch.sh.
SIF_FORMATTERS="[\"${SIF_DIR}/swerebench/{instance_id}.sif\",\"${SIF_DIR}/nv_internal/{instance_id}.sif\",\"${SIF_DIR}/r2e_gym/{instance_id}.sif\",\"${SIF_DIR}/swegym/sweb.eval.arm64.{instance_id}.sif\",\"${SIF_DIR}/swebench/swe-bench.eval.arm64.{instance_id}.sif\",\"${SIF_DIR}/mercor/swebenchpro_ots/{instance_id}.sif\"]"

echo "=========================================="
echo "nano V3.5 SWE-e2e reproduction: ${EXP_NAME}"
echo "Repo:        ${REPO_ROOT}"
echo "Config:      ${CONFIG_PATH}"
echo "Model:       ${MODEL_PATH}"
echo "Data train:  ${TRAIN_PATH}"
echo "Data val:    ${VAL_PATH}"
echo "SIF root:    ${SIF_DIR}"
echo "Precision:   ${PRECISION_RECIPE} (vLLM cache tree: ${VLLM_CACHE_PRECISION})"
echo "Nodes:       train=${NUM_TRAIN_NODES}, gen=${NUM_GEN_NODES}, gym=${NUM_GYM_NODES}, total=${TOTAL_NODES}"
echo "Parallelism: TP=${TP}, CP=${CP}, EP=${EP}, PP=${PP}, ETP=${ETP}, vLLM_TP=${VLLM_TP}, vLLM_PP=${VLLM_PP}"
echo "Batch:       PPS=${PPS}, GPP=${GPP}, GBS=${GBS}, concurrency=${CONCURRENCY}"
echo "SeqLen:      ${MAX_LENGTH}"
echo "Padding:     make_sequence_length_divisible_by=${MAKE_SEQUENCE_LENGTH_DIVISIBLE_BY:-<yaml default (=TP)>}"
echo "LogprobChunk:${LOGPROB_CHUNK_SIZE}, fuse_loss=${FUSE_LOSS}"
echo "Checkpoints: ${RESULTS_DIR}"
echo "Logs:        ${RUN_LOG_DIR}"
echo "Cache:       vLLM=${LUSTRE_VLLM_CACHE}; Inductor/Triton=/tmp with ${CACHE_SYNC_FREQUENCY}s rsync to Lustre"
echo "Container:   ${CONTAINER}"
echo "Slurm:       account=${SLURM_ACCOUNT}, partition=${SLURM_PARTITION}, qos=${SLURM_QOS}, reservation=${SLURM_RESERVATION:-<none>}"
if [ -n "${PRECISION_EXTRA_ARGS}" ]; then
  echo "PrecisionArgs: ${PRECISION_EXTRA_ARGS}"
fi
echo "=========================================="

cd "${REPO_ROOT}"

read -r -d '' SETUP_COMMAND <<SETUPEOF || true
set -euo pipefail

echo "[SETUP] Ensuring apptainer/singularity and zstd are available..."
({ command -v zstd >/dev/null 2>&1 && command -v rsync >/dev/null 2>&1; } || { apt-get update -qq && apt-get install -y -qq zstd rsync; }) 2>/dev/null || true
if ! command -v apptainer >/dev/null 2>&1 && ! command -v singularity >/dev/null 2>&1; then
  apt-get update -qq && apt-get install -y -qq git build-essential gcc wget 2>/dev/null || true
  cd /tmp
  wget --no-check-certificate -q -nc https://github.com/apptainer/apptainer/releases/download/v1.3.1/apptainer_1.3.1_arm64.deb || true
  apt install -y ./apptainer_1.3.1_arm64.deb 2>/dev/null || true
  ln -sf /usr/bin/apptainer /usr/bin/singularity 2>/dev/null || true
fi

echo "[CACHE SEED] Clearing stale node-local compile caches..."
rm -rf "${INDUCTOR_CACHE_DIR}" "${TRITON_CACHE_DIR}"
mkdir -p "${INDUCTOR_CACHE_DIR}" "${TRITON_CACHE_DIR}" \
  "${LUSTRE_VLLM_CACHE}" "${LUSTRE_INDUCTOR_CACHE}" "${LUSTRE_TRITON_CACHE}"

_seed_cache() {
  local src="\$1"
  local dst="\$2"
  local name="\$3"
  if [ -d "\$src" ] && [ "\$(ls -A "\$src" 2>/dev/null)" ]; then
    rsync -a --exclude '.tmp_*' "\$src/" "\$dst/" 2>/dev/null \
      && echo "[CACHE SEED] \$name: seeded from Lustre" \
      || echo "[CACHE SEED] \$name: seed failed (non-fatal)"
  else
    echo "[CACHE SEED] \$name: no warm cache yet"
  fi
}

_seed_cache "${LUSTRE_INDUCTOR_CACHE}" "${INDUCTOR_CACHE_DIR}" "Inductor"
_seed_cache "${LUSTRE_TRITON_CACHE}" "${TRITON_CACHE_DIR}" "Triton"
echo "[CACHE SEED] Done."

_sync_cache_one() {
  local src="\$1"
  local dst="\$2"
  local name="\$3"
  mkdir -p "\$dst"
  if [ -d "\$src" ] && [ "\$(ls -A "\$src" 2>/dev/null)" ]; then
    rsync -a --ignore-existing --exclude '.tmp_*' --exclude 'tmp*' "\$src/" "\$dst/" 2>/dev/null \
      && echo "[CACHE SYNC] \$name: synced node-local cache to Lustre" \
      || echo "[CACHE SYNC] \$name: sync failed (non-fatal)"
  else
    echo "[CACHE SYNC] \$name: no node-local entries yet"
  fi
}

_sync_compile_caches_to_lustre() {
  _sync_cache_one "${INDUCTOR_CACHE_DIR}" "${LUSTRE_INDUCTOR_CACHE}" "Inductor"
  _sync_cache_one "${TRITON_CACHE_DIR}" "${LUSTRE_TRITON_CACHE}" "Triton"
}

_start_cache_sync_sidecar() {
  local pidfile="/tmp/nemo_rl_compile_cache_sync_sidecar.pid"
  if [ -f "\$pidfile" ]; then
    local old_pid
    old_pid="\$(cat "\$pidfile" 2>/dev/null || true)"
    if [ -n "\$old_pid" ] && kill -0 "\$old_pid" 2>/dev/null; then
      echo "[CACHE SYNC] Sidecar already running on this node. pid=\${old_pid}"
      return
    fi
  fi

  local setup_log_dir
  setup_log_dir="\$(cd "\$(dirname "\$0")" && pwd)"
  local sidecar_log="/tmp/nemo_rl_compile_cache_sync_sidecar.log"
  local frequency="${CACHE_SYNC_FREQUENCY}"

  (
    set +e
    trap '_sync_compile_caches_to_lustre; exit 0' TERM INT
    echo "[CACHE SYNC] Sidecar started. frequency=\${frequency}s log_dir=\${setup_log_dir}"
    while true; do
      if [ -f "\${setup_log_dir}/ENDED" ]; then
        echo "[CACHE SYNC] ENDED detected; final sync."
        _sync_compile_caches_to_lustre
        exit 0
      fi
      sleep "\${frequency}"
      _sync_compile_caches_to_lustre
    done
  ) > "\$sidecar_log" 2>&1 &
  echo "\$!" > "\$pidfile"
  echo "[CACHE SYNC] Sidecar pid=\$! log=\${sidecar_log}"
}

if [ "${CACHE_SYNC_FREQUENCY}" -gt 0 ] 2>/dev/null; then
  _start_cache_sync_sidecar
else
  echo "[CACHE SYNC] Disabled because CACHE_SYNC_FREQUENCY=${CACHE_SYNC_FREQUENCY}"
fi
SETUPEOF
export SETUP_COMMAND

GITLAB_PASSWORD="${GITLAB_PAT:-${GITLAB_TOKEN:-}}"
export COMMAND="cd ${REPO_ROOT} && \
trap 'touch ${BASE_LOG_DIR}/\${SLURM_JOB_ID}-logs/ENDED 2>/dev/null || true' EXIT && \
date && \
OMP_NUM_THREADS=16 \
RAY_DEDUP_LOGS=1 \
NRL_VLLM_USE_V1=1 \
VLLM_CACHE_ROOT=${LUSTRE_VLLM_CACHE} \
DG_JIT_CACHE_DIR=${LUSTRE_VLLM_CACHE}/deep_gemm \
TORCHINDUCTOR_CACHE_DIR=${INDUCTOR_CACHE_DIR} \
TRITON_CACHE_DIR=${TRITON_CACHE_DIR} \
UV_CACHE_DIR=${PERSISTENT_CACHE}/uv \
RAY_ENABLE_UV_RUN_RUNTIME_ENV=0 \
UV_HTTP_TIMEOUT=10 \
UV_LOCK_TIMEOUT=${UV_LOCK_TIMEOUT:-1200} \
NEMO_GYM_VENV_DIR=${NEMO_GYM_VENV_DIR} \
VLLM_USE_PRECOMPILED=1 \
VLLM_PRECOMPILED_WHEEL_LOCATION=${VLLM_PRECOMPILED_WHEEL_LOCATION} \
VLLM_USE_FLASHINFER_MOE_FP8=1 \
VLLM_FLASHINFER_MOE_BACKEND=latency \
NRL_VLLM_ASYNC_TIMEOUT_SECONDS=1800 \
NRL_WG_USE_RAY_REF=1 \
NRL_USE_FASTOKENS=1 \
NRL_GEN_BENCHMARK_SKIP_TRAINING=${NRL_GEN_BENCHMARK_SKIP_TRAINING:-0} \
HF_HOME=${HF_HOME} \
HF_TOKEN=${HF_TOKEN:-} \
WANDB_API_KEY=${WANDB_API_KEY:-} \
WANDB_DIR=${WANDB_DIR} \
WANDB_CACHE_DIR=${WANDB_CACHE_DIR} \
WANDB_DATA_DIR=${WANDB_DATA_DIR} \
UV_INDEX_FLASHINFER_INTERNAL_PYPI_USERNAME=__token__ \
UV_INDEX_FLASHINFER_INTERNAL_PYPI_PASSWORD=${GITLAB_PASSWORD} \
uv run ${ENTRYPOINT} \
  --config ${CONFIG_PATH} \
  policy.model_name=${MODEL_PATH} \
  cluster.gpus_per_node=${NUM_GPU} \
  cluster.num_nodes=${TOTAL_NODES} \
  cluster.segment_size=${CLUSTER_SEGMENT_SIZE} \
  grpo.num_prompts_per_step=${PPS} \
  grpo.num_generations_per_prompt=${GPP} \
  policy.train_global_batch_size=${GBS} \
  policy.max_total_sequence_length=${MAX_LENGTH} \
  policy.logprob_chunk_size=${LOGPROB_CHUNK_SIZE} \
  ++policy.sequence_packing.fuse_loss=${FUSE_LOSS} \
  ${MAKE_SEQUENCE_LENGTH_DIVISIBLE_BY:+policy.make_sequence_length_divisible_by=${MAKE_SEQUENCE_LENGTH_DIVISIBLE_BY}} \
  policy.megatron_cfg.tensor_model_parallel_size=${TP} \
  policy.megatron_cfg.context_parallel_size=${CP} \
  policy.megatron_cfg.expert_model_parallel_size=${EP} \
  policy.megatron_cfg.pipeline_model_parallel_size=${PP} \
  policy.megatron_cfg.expert_tensor_parallel_size=${ETP} \
  ${GRADIENT_ACCUMULATION_FUSION:+++policy.megatron_cfg.gradient_accumulation_fusion=${GRADIENT_ACCUMULATION_FUSION}} \
  policy.generation.vllm_cfg.tensor_parallel_size=${VLLM_TP} \
  policy.generation.vllm_cfg.pipeline_parallel_size=${VLLM_PP} \
  policy.generation.vllm_cfg.gpu_memory_utilization=${VLLM_GPU_UTIL} \
  policy.generation.vllm_cfg.max_model_len=${MAX_LENGTH} \
  ${REASONING_PARSER_PLUGIN:+++policy.generation.vllm_cfg.reasoning_parser_plugin=${REASONING_PARSER_PLUGIN}} \
  ${REASONING_PARSER_PLUGIN:+'~policy.generation.vllm_cfg.http_server_serving_chat_kwargs.reasoning_parser_plugin'} \
  policy.generation.colocated.enabled=False \
  policy.generation.colocated.resources.num_nodes=${NUM_GEN_NODES} \
  policy.generation.colocated.resources.gpus_per_node=${NUM_GPU} \
  env.nemo_gym.num_gpu_nodes=${NUM_GYM_NODES} \
  ++env.nemo_gym.uv_venv_dir=${NEMO_GYM_VENV_DIR} \
  ++env.nemo_gym.uv_cache_dir=${NEMO_GYM_UV_CACHE} \
  env.nemo_gym.swe_agents_train.responses_api_agents.swe_agents.dataset_path=${TRAIN_PATH} \
  env.nemo_gym.swe_agents_val.responses_api_agents.swe_agents.dataset_path=${VAL_PATH} \
  env.nemo_gym.swe_agents_train.responses_api_agents.swe_agents.concurrency=${CONCURRENCY} \
  env.nemo_gym.swe_agents_val.responses_api_agents.swe_agents.concurrency=${CONCURRENCY} \
  'env.nemo_gym.swe_agents_train.responses_api_agents.swe_agents.container_formatter=${SIF_FORMATTERS}' \
  'env.nemo_gym.swe_agents_val.responses_api_agents.swe_agents.container_formatter=${SIF_FORMATTERS}' \
  data.train.data_path=${TRAIN_PATH} \
  data.validation.data_path=${VAL_PATH} \
  ++data.use_multiple_dataloader=${USE_MULTIPLE_DATALOADER} \
  checkpointing.checkpoint_dir=${RESULTS_DIR} \
  checkpointing.checkpoint_must_save_by=${CHECKPOINTING_SAVE_BY} \
  checkpointing.save_period=${SAVE_PERIOD} \
  ++checkpointing.save_optimizer=${SAVE_OPTIMIZER} \
  logger.log_dir=${NEMO_LOG_DIR} \
  logger.wandb_enabled=True \
  logger.wandb.name=${WANDB_NAME} \
  logger.wandb.project=${WANDB_PROJ} \
  ++logger.wandb.group=${WANDB_GROUP} \
  grpo.max_num_steps=${MAX_NUM_STEPS} \
  ${PRECISION_EXTRA_ARGS} \
  ${EXTRA_ARGS:-}"

SBATCH_ARGS=(
  --nodes="${TOTAL_NODES}"
  --account="${SLURM_ACCOUNT}"
  --job-name="${WANDB_NAME}"
  --partition="${SLURM_PARTITION}"
  --time="${WALLTIME}"
  --gres="gpu:${NUM_GPU}"
  --exclusive
  --mem=0
  --dependency=singleton
  --segment="${SBATCH_SEGMENT}"
  --output="${RUN_LOG_DIR}/slurm-%j.out"
)

if [ -n "${SLURM_QOS}" ]; then
  SBATCH_ARGS+=(--qos="${SLURM_QOS}")
fi
if [ -n "${SLURM_RESERVATION}" ]; then
  SBATCH_ARGS+=(--reservation="${SLURM_RESERVATION}")
fi
if [ -n "${SLURM_COMMENT}" ]; then
  SBATCH_ARGS+=(--comment="${SLURM_COMMENT}")
fi

if [ "${DRY_RUN:-0}" = "1" ]; then
  echo ""
  echo "[DRY_RUN] Not submitting. Would run:"
  printf '[DRY_RUN] sbatch'
  printf ' %q' "${SBATCH_ARGS[@]}"
  printf ' %q\n' "${RAY_SUB}"
  echo ""
  echo "[DRY_RUN] COMMAND:"
  echo "${COMMAND}" | sed -E \
    -e 's/(WANDB_API_KEY=)[^ ]*/\1<redacted>/g' \
    -e 's/(HF_TOKEN=)[^ ]*/\1<redacted>/g' \
    -e 's/(UV_INDEX_FLASHINFER_INTERNAL_PYPI_PASSWORD=)[^ ]*/\1<redacted>/g'
  exit 0
fi

SBATCH_OUTPUT="$(sbatch "${SBATCH_ARGS[@]}" "${RAY_SUB}")"
echo "${SBATCH_OUTPUT}" >&2

JOB_ID="$(printf '%s\n' "${SBATCH_OUTPUT}" | grep -o '[0-9]\+' | tail -n 1)"
if [ -z "${JOB_ID}" ]; then
  echo "ERROR: failed to parse Slurm job id from sbatch output: ${SBATCH_OUTPUT}" >&2
  exit 1
fi
printf '%s\n' "${JOB_ID}" > "${LATEST_JOB_ID_FILE}"

echo "=========================================="
echo "Job submitted: ${EXP_NAME}"
echo "Job ID: ${JOB_ID}"
echo "Monitor with: squeue -j ${JOB_ID}"
echo "Ray/Slurm logs: ${RUN_LOG_DIR}/${JOB_ID}-logs/"
echo "Slurm output: ${RUN_LOG_DIR}/slurm-${JOB_ID}.out"
echo "Checkpoints: ${RESULTS_DIR}/"
echo "=========================================="
