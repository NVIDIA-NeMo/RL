#!/usr/bin/env bash
# Fixed R1 Nemotron Nano V3.5 SWE E2E on Ray-managed Dynamo.
#
# Shape: two 4-GPU training nodes (TP4/CP2/EP8) plus one 4-GPU
# Dynamo/vLLM node (TP4), one asynchronous GRPO step.

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=${REPO_ROOT:-$(cd "${SCRIPT_DIR}/../.." && pwd)}
ROOT=${ROOT:-/lustre/fsw/portfolios/coreai/users/jothomson/nemo-rl-dynamo-slurm-swe}

CONFIG_PATH=${CONFIG_PATH:-${REPO_ROOT}/examples/swe_bench/grpo_nano_v3_5_swe_dynamo_hsg.yaml}
ENTRYPOINT=${ENTRYPOINT:-${REPO_ROOT}/examples/nemo_gym/run_grpo_nemo_gym.py}
RAY_SUB=${RAY_SUB:-${REPO_ROOT}/ray.sub}
GYM_CODE=${GYM_CODE:-${REPO_ROOT}/3rdparty/Gym-workspace/Gym}

MODEL_SOURCE_PATH=${MODEL_SOURCE_PATH:-/lustre/fsw/portfolios/llmservice/users/pjin/devel/nemo-rl-ultra-v3-nano-opd-dev-20260513/results/mopd_ultrav3_to_nanov3_5_repro_v5_kd_opt_full-hsg-20260524-r1/step_18/hf}
# Dynamo's Rust model-registration path requires strict JSON, while the source
# Transformers config uses the non-standard JSON token `Infinity`. Keep the
# shared checkpoint immutable and present a lightweight, strict-JSON overlay.
MODEL_PATH=${MODEL_PATH:-${ROOT}/models/nemotron-nano-v3.5-swe-dynamo}
TRAIN_PATH=${TRAIN_PATH:-/lustre/fsw/portfolios/llmservice/users/sdevare/repos/ultra/datasets/swe/blends/large_root_cause_curriculum_with_mercor_ots_plus_singlefile_swerebench_overlap_fix.jsonl}
VAL_PATH=${VAL_PATH:-/lustre/fsw/portfolios/llmservice/users/sdevare/repos/ultra/datasets/swe/swe_public_datasets_val_swebench.jsonl}
SIF_DIR=${SIF_DIR:-/lustre/fsw/portfolios/llmservice/users/sdevare/images}
SANDBOX_CONTAINER=${SANDBOX_CONTAINER:-/lustre/fsw/portfolios/llmservice/users/igitman/images/nemo-skills-sandbox-b620e79.sqsh}
CONTAINER=${CONTAINER:-${ROOT}/images/nemo-rl-dynamo-swe-hsg.sqsh}

NUM_GPU=4
NUM_TRAIN_NODES=2
NUM_GEN_NODES=1
TOTAL_NODES=3
TP=4
CP=2
EP=8
PP=1
ETP=1
VLLM_TP=4
VLLM_PP=1
PPS=1
GPP=2
GBS=2
MAX_LENGTH=${MAX_LENGTH:-196608}
MAX_NUM_STEPS=${MAX_NUM_STEPS:-1}
CONCURRENCY=${CONCURRENCY:-4}
LOGPROB_CHUNK_SIZE=${LOGPROB_CHUNK_SIZE:-1024}

if [[ "${NUM_TRAIN_NODES}" != "2" || "${NUM_GEN_NODES}" != "1" ]]; then
  echo "This milestone is fixed to two training nodes and one generation node." >&2
  exit 2
fi

SLURM_ACCOUNT=${SLURM_ACCOUNT:-coreai_tritoninference_triton3}
SLURM_PARTITION=${SLURM_PARTITION:-batch_long}
WALLTIME=${WALLTIME:-04:00:00}
SBATCH_SEGMENT=1
CLUSTER_SEGMENT_SIZE=1
DEFAULT_SLURM_COMMENT='{"OccupiedIdleGPUsJobReaper":{"exemptIdleTimeMins":"60","reason":"data_loading","description":"Async SWE GRPO rollout and validation"}}'
SLURM_COMMENT=${SLURM_COMMENT-$DEFAULT_SLURM_COMMENT}

EXP_NAME=${EXP_NAME:-nano-v3-5-swe-dynamo-r1-${USER}}
RESULTS_DIR=${RESULTS_DIR:-${ROOT}/results/${EXP_NAME}}
RUN_LOG_DIR=${RUN_LOG_DIR:-${ROOT}/logs/${EXP_NAME}}
NEMO_LOG_DIR=${NEMO_LOG_DIR:-${RUN_LOG_DIR}/nemo}
HF_HOME=${HF_HOME:-${ROOT}/cache/huggingface}
HF_DATASETS_CACHE=${HF_DATASETS_CACHE:-${HF_HOME}/datasets}
PERSISTENT_CACHE=${PERSISTENT_CACHE:-${ROOT}/cache/nemotron_nano_v3_5}
WANDB_STAGE_ROOT=${WANDB_STAGE_ROOT:-${ROOT}/cache/wandb}
NEMO_GYM_VENV_DIR=${NEMO_GYM_VENV_DIR:-/opt/gym_venvs}
NEMO_GYM_UV_CACHE=${NEMO_GYM_UV_CACHE:-/tmp/nemo_gym_uv_cache}
INDUCTOR_CACHE_DIR=/tmp/nemo_rl_inductor_cache
TRITON_CACHE_DIR=/tmp/nemo_rl_triton_cache
LUSTRE_VLLM_CACHE=${PERSISTENT_CACHE}/cache_write/vllm_compile_cache_bf16
LUSTRE_INDUCTOR_CACHE=${PERSISTENT_CACHE}/cache_write/inductor_cache
LUSTRE_TRITON_CACHE=${PERSISTENT_CACHE}/cache_write/triton_cache
CACHE_SYNC_FREQUENCY=${CACHE_SYNC_FREQUENCY:-1800}

require_path() {
  local path=$1
  local label=$2
  if [[ ! -e "${path}" ]]; then
    echo "Missing ${label}: ${path}" >&2
    exit 1
  fi
}

prepare_model_overlay() {
  if [[ "${MODEL_PATH}" == "${MODEL_SOURCE_PATH}" ]]; then
    return
  fi

  mkdir -p "${MODEL_PATH}"
  while IFS= read -r -d '' source_file; do
    ln -sfn "${source_file}" "${MODEL_PATH}/$(basename "${source_file}")"
  done < <(find "${MODEL_SOURCE_PATH}" -mindepth 1 -maxdepth 1 ! -name config.json -print0)

  local config_tmp="${MODEL_PATH}/config.json.tmp.$$"
  sed 's/Infinity/1.7976931348623157e308/g' \
    "${MODEL_SOURCE_PATH}/config.json" > "${config_tmp}"
  mv "${config_tmp}" "${MODEL_PATH}/config.json"

  python3 - "${MODEL_PATH}/config.json" <<'PY'
import json
import sys

with open(sys.argv[1], encoding="utf-8") as config_file:
    json.load(
        config_file,
        parse_constant=lambda value: (_ for _ in ()).throw(
            ValueError(f"non-standard JSON constant: {value}")
        ),
    )
PY
}

require_path "${CONFIG_PATH}" config
require_path "${ENTRYPOINT}" entrypoint
require_path "${RAY_SUB}" ray.sub
require_path "${GYM_CODE}/nemo_gym/__init__.py" "Gym checkout"
require_path "${MODEL_SOURCE_PATH}/config.json" "source model"
require_path "${MODEL_SOURCE_PATH}/chat_template.jinja" "source model chat template"
prepare_model_overlay
require_path "${MODEL_PATH}/config.json" model
require_path "${MODEL_PATH}/chat_template.jinja" "model chat template"
require_path "${TRAIN_PATH}" "training dataset"
require_path "${VAL_PATH}" "validation dataset"
require_path "${SANDBOX_CONTAINER}" "SWE sandbox container"
if [[ "${DRY_RUN:-0}" != "1" ]]; then
  require_path "${CONTAINER}" "combined Dynamo image"
fi
for sif_subdir in swerebench nv_internal r2e_gym swegym swebench mercor/swebenchpro_ots; do
  require_path "${SIF_DIR}/${sif_subdir}" "SIF directory ${sif_subdir}"
done

GYM_COMMIT=$(git -C "${GYM_CODE}" rev-parse HEAD)
EXPECTED_GYM_COMMIT=eddd5e98a541cc90e0ee41f1b5e9bd146b5be665
if [[ "${GYM_COMMIT}" != "${EXPECTED_GYM_COMMIT}" ]]; then
  echo "Gym must be pinned to ${EXPECTED_GYM_COMMIT}; got ${GYM_COMMIT}." >&2
  exit 1
fi

mkdir -p \
  "${RESULTS_DIR}" "${RUN_LOG_DIR}" "${NEMO_LOG_DIR}" \
  "${HF_HOME}" "${HF_DATASETS_CACHE}" \
  "${PERSISTENT_CACHE}/uv" \
  "${LUSTRE_VLLM_CACHE}" "${LUSTRE_INDUCTOR_CACHE}" "${LUSTRE_TRITON_CACHE}" \
  "${WANDB_STAGE_ROOT}/dir" "${WANDB_STAGE_ROOT}/cache" "${WANDB_STAGE_ROOT}/data"

cat > "${RUN_LOG_DIR}/git-revision.txt" <<EOF
superproject: $(git -C "${REPO_ROOT}" rev-parse HEAD)
gym:          ${GYM_COMMIT}
dynamo:       59358c26d0aeed19300706462b63ada25a0a6d7c
EOF

SIF_FORMATTERS="[\"${SIF_DIR}/swerebench/{instance_id}.sif\",\"${SIF_DIR}/nv_internal/{instance_id}.sif\",\"${SIF_DIR}/r2e_gym/{instance_id}.sif\",\"${SIF_DIR}/swegym/sweb.eval.arm64.{instance_id}.sif\",\"${SIF_DIR}/swebench/swe-bench.eval.arm64.{instance_id}.sif\",\"${SIF_DIR}/mercor/swebenchpro_ots/{instance_id}.sif\"]"

read -r -d '' SETUP_COMMAND <<SETUPEOF || true
set -euo pipefail

({ command -v zstd >/dev/null 2>&1 && command -v rsync >/dev/null 2>&1; } || {
  apt-get update -qq && apt-get install -y -qq zstd rsync
}) 2>/dev/null || true

if ! command -v apptainer >/dev/null 2>&1 && ! command -v singularity >/dev/null 2>&1; then
  apt-get update -qq
  apt-get install -y -qq git build-essential gcc wget 2>/dev/null || true
  cd /tmp
  wget --no-check-certificate -q -nc \
    https://github.com/apptainer/apptainer/releases/download/v1.3.1/apptainer_1.3.1_arm64.deb
  apt install -y ./apptainer_1.3.1_arm64.deb 2>/dev/null || true
  ln -sf /usr/bin/apptainer /usr/bin/singularity 2>/dev/null || true
fi

rm -rf "${INDUCTOR_CACHE_DIR}" "${TRITON_CACHE_DIR}"
mkdir -p "${INDUCTOR_CACHE_DIR}" "${TRITON_CACHE_DIR}"
for pair in \
  "${LUSTRE_INDUCTOR_CACHE}:${INDUCTOR_CACHE_DIR}" \
  "${LUSTRE_TRITON_CACHE}:${TRITON_CACHE_DIR}"; do
  src=\${pair%%:*}
  dst=\${pair#*:}
  if [[ -d "\${src}" ]]; then
    rsync -a --exclude '.tmp_*' "\${src}/" "\${dst}/" 2>/dev/null || true
  fi
done

if [[ "${CACHE_SYNC_FREQUENCY}" -gt 0 ]]; then
  (
    set +e
    while [[ ! -f "${BASE_LOG_DIR:-${RUN_LOG_DIR}}/\${SLURM_JOB_ID}-logs/ENDED" ]]; do
      sleep "${CACHE_SYNC_FREQUENCY}"
      rsync -a --ignore-existing --exclude '.tmp_*' "${INDUCTOR_CACHE_DIR}/" "${LUSTRE_INDUCTOR_CACHE}/" 2>/dev/null || true
      rsync -a --ignore-existing --exclude '.tmp_*' "${TRITON_CACHE_DIR}/" "${LUSTRE_TRITON_CACHE}/" 2>/dev/null || true
    done
  ) >/tmp/nemo_rl_compile_cache_sync.log 2>&1 &
fi
SETUPEOF

export ROOT REPO_ROOT CONTAINER SANDBOX_CONTAINER SETUP_COMMAND
export GPUS_PER_NODE=${NUM_GPU}
export CPUS_PER_WORKER=${CPUS_PER_WORKER:-144}
export BASE_LOG_DIR=${RUN_LOG_DIR}
export RAY_LOG_SYNC_FREQUENCY=${RAY_LOG_SYNC_FREQUENCY:-60}
export HF_HOME HF_DATASETS_CACHE
export NEMO_GYM_VENV_DIR
export NEMO_RL_PY_EXECUTABLES_SYSTEM=1
export DYNAMO_PYTHON=/opt/dynamo_venv/bin/python
export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export WANDB_DIR=${WANDB_STAGE_ROOT}/dir
export WANDB_CACHE_DIR=${WANDB_STAGE_ROOT}/cache
export WANDB_DATA_DIR=${WANDB_STAGE_ROOT}/data
export MOUNTS="/lustre:/lustre,${GYM_CODE}:/opt/nemo-rl/3rdparty/Gym-workspace/Gym"

export COMMAND="cd ${REPO_ROOT} && \
trap 'touch ${RUN_LOG_DIR}/\${SLURM_JOB_ID}-logs/ENDED 2>/dev/null || true' EXIT && \
OMP_NUM_THREADS=16 \
RAY_DEDUP_LOGS=1 \
RAY_ENABLE_UV_RUN_RUNTIME_ENV=0 \
NRL_WG_USE_RAY_REF=1 \
NRL_USE_FASTOKENS=1 \
NEMO_RL_PY_EXECUTABLES_SYSTEM=1 \
DYNAMO_PYTHON=/opt/dynamo_venv/bin/python \
VLLM_CACHE_ROOT=${LUSTRE_VLLM_CACHE} \
DG_JIT_CACHE_DIR=${LUSTRE_VLLM_CACHE}/deep_gemm \
TORCHINDUCTOR_CACHE_DIR=${INDUCTOR_CACHE_DIR} \
TRITON_CACHE_DIR=${TRITON_CACHE_DIR} \
UV_CACHE_DIR=${PERSISTENT_CACHE}/uv \
NEMO_GYM_VENV_DIR=${NEMO_GYM_VENV_DIR} \
NEMO_GYM_UV_CACHE=${NEMO_GYM_UV_CACHE} \
HF_HOME=${HF_HOME} \
HF_DATASETS_CACHE=${HF_DATASETS_CACHE} \
PYTHONPATH=${REPO_ROOT} \
/opt/nemo_rl_venv/bin/python -u ${ENTRYPOINT} \
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
  policy.megatron_cfg.tensor_model_parallel_size=${TP} \
  policy.megatron_cfg.context_parallel_size=${CP} \
  policy.megatron_cfg.expert_model_parallel_size=${EP} \
  policy.megatron_cfg.pipeline_model_parallel_size=${PP} \
  policy.megatron_cfg.expert_tensor_parallel_size=${ETP} \
  policy.generation.vllm_cfg.tensor_parallel_size=${VLLM_TP} \
  policy.generation.vllm_cfg.pipeline_parallel_size=${VLLM_PP} \
  policy.generation.vllm_cfg.gpu_memory_utilization=0.85 \
  policy.generation.vllm_cfg.max_model_len=${MAX_LENGTH} \
  policy.generation.dynamo_cfg.engine_world_size=4 \
  policy.generation.colocated.enabled=False \
  policy.generation.colocated.resources.num_nodes=${NUM_GEN_NODES} \
  policy.generation.colocated.resources.gpus_per_node=${NUM_GPU} \
  env.nemo_gym.num_gpu_nodes=0 \
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
  checkpointing.enabled=False \
  logger.log_dir=${NEMO_LOG_DIR} \
  logger.wandb_enabled=False \
  grpo.max_num_steps=${MAX_NUM_STEPS}"

echo "Nemotron Nano SWE Dynamo E2E"
echo "  nodes: train=${NUM_TRAIN_NODES}, inference=${NUM_GEN_NODES}, total=${TOTAL_NODES}"
echo "  train: TP=${TP}, CP=${CP}, EP=${EP}; inference: TP=${VLLM_TP}"
echo "  batch: PPS=${PPS}, GPP=${GPP}, GBS=${GBS}; max_length=${MAX_LENGTH}"
echo "  model: ${MODEL_PATH}"
echo "  source model: ${MODEL_SOURCE_PATH}"
echo "  image: ${CONTAINER}"
echo "  logs:  ${RUN_LOG_DIR}"

SBATCH_ARGS=(
  --nodes=${TOTAL_NODES}
  --account=${SLURM_ACCOUNT}
  --job-name=${EXP_NAME}
  --partition=${SLURM_PARTITION}
  --time=${WALLTIME}
  --gres=gpu:${NUM_GPU}
  --exclusive
  --mem=0
  --dependency=singleton
  --segment=${SBATCH_SEGMENT}
  --output=${RUN_LOG_DIR}/slurm-%j.out
)
if [[ -n "${SLURM_COMMENT}" ]]; then
  SBATCH_ARGS+=(--comment="${SLURM_COMMENT}")
fi

if [[ "${DRY_RUN:-0}" == "1" ]]; then
  printf 'sbatch'
  printf ' %q' "${SBATCH_ARGS[@]}"
  printf ' %q\n' "${RAY_SUB}"
  printf '%s\n' "${COMMAND}" | sed -E \
    -e 's/(HF_TOKEN=)[^ ]*/\1<redacted>/g' \
    -e 's/(WANDB_API_KEY=)[^ ]*/\1<redacted>/g'
  exit 0
fi

SBATCH_OUTPUT=$(sbatch "${SBATCH_ARGS[@]}" "${RAY_SUB}")
echo "${SBATCH_OUTPUT}"
JOB_ID=$(grep -o '[0-9]\+' <<<"${SBATCH_OUTPUT}" | tail -1)
printf '%s\n' "${JOB_ID}" > "${RUN_LOG_DIR}/latest_job_id.txt"
echo "Monitor: squeue -j ${JOB_ID}"
echo "Driver:  ${RUN_LOG_DIR}/${JOB_ID}-logs/ray-driver.log"
