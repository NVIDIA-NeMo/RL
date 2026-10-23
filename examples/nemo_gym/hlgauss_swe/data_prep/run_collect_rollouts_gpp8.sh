#!/bin/bash
# Generation-only companion to run_ppo_qwen35_4b_actor_9b_critic_pretrain_formal.sh.
# MODE=smoke uses one interactive node; MODE=formal submits 8 independent
# one-node batch collectors. No critic/policy training is started here.
set -euo pipefail

WS=/lustre/fsw/portfolios/coreai/users/bihu/repos/multi-turn-swe
NEMO_RL=${NEMO_RL:-${WS}/nemo-rl}
GYM_CODE=${WS}/gym
MBRIDGE_CODE=${WS}/nemo-rl/3rdparty/Megatron-Bridge-workspace/Megatron-Bridge
CONFIG_FILE=${NEMO_RL}/examples/nemo_gym/hlgauss_swe/configs/collect_qwen35_4b_jiaqiz3145_gpp8.yaml
CONTAINER_IMAGE=/lustre/fsw/portfolios/coreai/users/terryk/enroot-images/nvcr.io/nvidian/nemo-rl:15f91e0-60262614.squashfs
TRAIN_DATA=${WS}/data/jiaqiz_intersection_openhands_images/train.jsonl
RESULT_DIR=/lustre/fsw/portfolios/coreai/users/bihu/rl-results/qwen35-4b-jiaqiz3145-openhands-gpp8-rollouts
MODE=${MODE:-smoke}
NUM_SHARDS=${NUM_SHARDS:-8}
INDEX_START=${INDEX_START:-0}
INDEX_END=${INDEX_END:-}
MAX_GROUPS=${MAX_GROUPS:-null}
DEPENDS_CSV=${DEPENDS_CSV:-}

if [[ ${MODE} == smoke ]]; then
  PARTITION=interactive
  WALLTIME=04:00:00
  NUM_SHARDS=1
  INDEX_END=${INDEX_END:-1}
elif [[ ${MODE} == formal ]]; then
  PARTITION=batch
  WALLTIME=04:00:00
  INDEX_END=${INDEX_END:-3145}
else
  echo "MODE must be smoke or formal" >&2
  exit 1
fi
if [[ -n ${DEPENDS_CSV} ]]; then
  IFS=, read -r -a DEPENDS <<< "${DEPENDS_CSV}"
  [[ ${#DEPENDS[@]} -eq ${NUM_SHARDS} ]] || {
    echo "DEPENDS_CSV needs ${NUM_SHARDS} job IDs in shard order" >&2
    exit 1
  }
  for id in "${DEPENDS[@]}"; do
    [[ ${id} =~ ^[0-9]+$ ]] || { echo "Invalid dependency job ID: ${id}" >&2; exit 1; }
  done
fi

for path in "${CONFIG_FILE}" "${CONTAINER_IMAGE}" "${TRAIN_DATA}" \
  "${NEMO_RL}/examples/nemo_gym/run_swe_rollout_collection.py" \
  "${NEMO_RL}/nemo_rl/algorithms/rollout_collection.py" \
  "${GYM_CODE}/responses_api_agents/swe_agents/swe_openhands_setup/OpenHands"; do
  [[ -e ${path} ]] || { echo "Missing required path: ${path}" >&2; exit 1; }
done
[[ $(git -C "${GYM_CODE}" rev-parse HEAD) == 5fb61e871a7e7e4a722a236430e8056628355537 ]] || {
  echo "Gym commit changed; this recipe requires the non-compaction OpenHands Gym" >&2
  exit 1
}
[[ -z "${EXPECTED_RL_COMMIT:-}" || $(git -C "${NEMO_RL}" rev-parse HEAD) == "${EXPECTED_RL_COMMIT}" ]] || {
  echo "NeMo-RL base commit changed; inspect the collector port before submission" >&2
  exit 1
}

source /lustre/fsw/portfolios/coreai/users/bihu/rl-scaling/env.sh
export CONTAINER=${CONTAINER_IMAGE}
export MOUNTS="/lustre:/lustre,${NEMO_RL}:${NEMO_RL},${GYM_CODE}:/opt/nemo-rl/3rdparty/Gym-workspace/Gym,${GYM_CODE}:${NEMO_RL}/3rdparty/Gym-workspace/Gym,${MBRIDGE_CODE}:/opt/nemo-rl/3rdparty/Megatron-Bridge-workspace/Megatron-Bridge,${MBRIDGE_CODE}:${NEMO_RL}/3rdparty/Megatron-Bridge-workspace/Megatron-Bridge"
export GPUS_PER_NODE=8
export BASE_LOG_DIR=${RESULT_DIR}
read -r -d '' SETUP_COMMAND <<'SETUPEOF' || true
if ! command -v apptainer >/dev/null 2>&1 && ! command -v singularity >/dev/null 2>&1; then
  apt-get update && apt-get install -y git build-essential gcc wget 2>/dev/null || true
  cd /tmp && wget --no-check-certificate -q https://github.com/apptainer/apptainer/releases/download/v1.3.1/apptainer_1.3.1_amd64.deb \
    && apt install -y ./apptainer_1.3.1_amd64.deb && ln -sf /usr/bin/apptainer /usr/bin/singularity || true
fi
SETUPEOF
export SETUP_COMMAND

echo "Mode: ${MODE}; partition: ${PARTITION}; shards: ${NUM_SHARDS}; range: [${INDEX_START}, ${INDEX_END}); max groups: ${MAX_GROUPS}"
echo "Output: ${RESULT_DIR}"
mkdir -p "${RESULT_DIR}"
cd "${NEMO_RL}"
for ((shard_id = 0; shard_id < NUM_SHARDS; shard_id++)); do
  # ray.sub clears SLURM_ARRAY_TASK_ID in the driver environment, so bake the
  # shard number into COMMAND before each independent sbatch submission.
  printf -v COMMAND \
    'cd %q && ulimit -c 0 && mkdir -p /tmp/nemo_rl_triton_cache /tmp/nemo_rl_inductor_cache /tmp/nemo_rl_vllm_cache/deep_gemm /tmp/uv_cache && export PYTHONPATH=%q TRITON_CACHE_DIR=/tmp/nemo_rl_triton_cache TORCHINDUCTOR_CACHE_DIR=/tmp/nemo_rl_inductor_cache VLLM_CACHE_ROOT=/tmp/nemo_rl_vllm_cache DG_JIT_CACHE_DIR=/tmp/nemo_rl_vllm_cache/deep_gemm UV_CACHE_DIR=/tmp/uv_cache UV_LOCK_TIMEOUT=1200 NRL_VLLM_USE_V1=1 NRL_WG_USE_RAY_REF=1 NRL_FORCE_REBUILD_VENVS=false NRL_IGNORE_VERSION_MISMATCH=1 RAY_ENABLE_UV_RUN_RUNTIME_ENV=0 VLLM_ATTENTION_BACKEND=FLASH_ATTN VLLM_DEEP_GEMM_WARMUP=skip TORCH_CUDA_ARCH_LIST="9.0 10.0" NEMO_GYM_SKIP_VENV_IF_PRESENT=1 && /opt/nemo_rl_venv/bin/python examples/nemo_gym/run_swe_rollout_collection.py --config %q ++collection.shard_id=%q ++collection.num_shards=%q ++collection.index_start=%q ++collection.index_end=%q ++collection.max_groups=%q' \
    "${NEMO_RL}" "${NEMO_RL}" "${CONFIG_FILE}" "${shard_id}" "${NUM_SHARDS}" "${INDEX_START}" "${INDEX_END}" "${MAX_GROUPS}"
  export COMMAND
  dependency_arg=()
  if [[ -n ${DEPENDS_CSV} ]]; then
    dependency_arg=("--dependency=afterany:${DEPENDS[shard_id]}")
  fi
  if [[ ${DRY_RUN:-0} == 1 ]]; then
    echo "shard ${shard_id}: ${dependency_arg[*]} ${COMMAND}"
  else
    job_id=$(sbatch --parsable --nodes=1 --account=coreai_dlalgo_nemorl \
      --job-name="qwen35-4b-rollouts-s${shard_id}" --partition="${PARTITION}" \
      --time="${WALLTIME}" --gres=gpu:8 --exclusive --requeue \
      "${dependency_arg[@]}" ray.sub)
    echo "shard ${shard_id}: job ${job_id}"
  fi
done
