#!/usr/bin/env bash

set -euo pipefail

ACTION=${ACTION:-render}
MODE=${MODE:-sync}
MODEL=${MODEL:-nano}
ARM=${ARM:-bf16}
MAX_STEPS=${MAX_STEPS:-20}
RUN_SUFFIX=${RUN_SUFFIX:-$(date +%Y%m%d-%H%M%S)}
BRANCH=${BRANCH:-sna/exp-gb200-deck-refresh-20260818}
EXPECTED_HEAD=${EXPECTED_HEAD:-}

case "${MODEL}:${MODE}:${ARM}" in
  nano:sync:bf16|nano:sync:mxfp8)
    CONFIG=examples/configs/recipes/llm/grpo-nanov3-30BA3B-2n8g-megatron-pack-cp.yaml
    TOTAL_NODES=4
    GEN_NODES=4
    SEGMENT_SIZE=4
    TRAIN_GLOBAL_BATCH_SIZE=16
    REFIT_TRANSPORT=null
    COLOCATED=true
    ASYNC_GRPO=false
    VLLM_TP=1
    ;;
  nano:async:bf16|nano:async:mxfp8)
    CONFIG=examples/configs/recipes/llm/grpo-nanov3-30BA3B-2n8g-megatron-pack-cp.yaml
    TOTAL_NODES=8
    GEN_NODES=4
    SEGMENT_SIZE=4
    TRAIN_GLOBAL_BATCH_SIZE=16
    REFIT_TRANSPORT=nccl_reshard
    COLOCATED=false
    ASYNC_GRPO=true
    VLLM_TP=1
    ;;
  qwen235:sync:mxfp8_legacy)
    CONFIG=examples/configs/recipes/llm/performance/grpo-qwen3-235b-32n4g-async-1off-mxfp8-rollout.yaml
    TOTAL_NODES=32
    GEN_NODES=16
    SEGMENT_SIZE=16
    TRAIN_GLOBAL_BATCH_SIZE=512
    REFIT_TRANSPORT=null
    COLOCATED=false
    ASYNC_GRPO=false
    VLLM_TP=4
    ;;
  qwen235:sync:mxfp8_nccl)
    CONFIG=examples/configs/recipes/llm/performance/grpo-qwen3-235b-32n4g-async-1off-mxfp8-rollout.yaml
    TOTAL_NODES=32
    GEN_NODES=16
    SEGMENT_SIZE=16
    TRAIN_GLOBAL_BATCH_SIZE=512
    REFIT_TRANSPORT=nccl_reshard
    COLOCATED=false
    ASYNC_GRPO=false
    VLLM_TP=4
    ;;
  *)
    echo "Combination MODEL=${MODEL}, MODE=${MODE}, ARM=${ARM} is not supported" >&2
    exit 2
    ;;
esac

case "${ARM}" in
  bf16)
    ROLLOUT_PRECISION=bfloat16
    IS_MX=false
    QUANTIZATION_SCOPE=none
    ;;
  mxfp8|mxfp8_legacy|mxfp8_nccl)
    ROLLOUT_PRECISION=fp8
    IS_MX=true
    QUANTIZATION_SCOPE=routed_expert_fc1_fc2_only
    ;;
esac

RUN_ARGS=(
  --config "${CONFIG}"
  "cluster.num_nodes=${TOTAL_NODES}"
  "cluster.gpus_per_node=4"
  "cluster.segment_size=${SEGMENT_SIZE}"
  "policy.precision=bfloat16"
  "grpo.async_grpo.enabled=${ASYNC_GRPO}"
  "data_plane.enabled=false"
  "policy.generation.colocated.enabled=${COLOCATED}"
  "policy.generation.colocated.resources.num_nodes=${GEN_NODES}"
  "policy.generation.colocated.resources.gpus_per_node=4"
  "policy.generation.refit_transport=${REFIT_TRANSPORT}"
  "policy.generation.real_quant_export_cpu_offload=false"
  "policy.generation.vllm_cfg.tensor_parallel_size=${VLLM_TP}"
  "policy.generation.vllm_cfg.pipeline_parallel_size=1"
  "policy.generation.vllm_cfg.expert_parallel_size=1"
  "policy.generation.vllm_cfg.async_engine=false"
  "policy.generation.vllm_cfg.enforce_eager=false"
  "policy.generation.vllm_cfg.use_tqdm=false"
  "policy.generation.vllm_cfg.precision=${ROLLOUT_PRECISION}"
  "++policy.generation.vllm_kwargs.moe_backend=flashinfer_trtllm"
  "++policy.generation.vllm_kwargs.distributed_timeout_seconds=2400"
  "policy.megatron_cfg.expert_tensor_parallel_size=1"
  "policy.train_global_batch_size=${TRAIN_GLOBAL_BATCH_SIZE}"
  "loss_fn.force_on_policy_ratio=false"
  "loss_fn.use_importance_sampling_correction=true"
  "grpo.skip_reference_policy_logprobs_calculation=false"
  "grpo.max_num_steps=${MAX_STEPS}"
  "grpo.seed=42"
  "grpo.val_at_start=false"
  "++grpo.val_at_end=false"
  "checkpointing.enabled=false"
)
if [[ "${IS_MX}" == true ]]; then
  RUN_ARGS+=("policy.generation.vllm_cfg.is_mx=true")
fi

printf -v RUN_COMMAND '%q ' uv run --frozen examples/run_grpo.py "${RUN_ARGS[@]}"
if [[ "${ACTION}" == render ]]; then
  printf '%s\n' "${RUN_COMMAND}"
  exit 0
fi

case "${ACTION}" in
  test-only) SBATCH_ACTION=(--test-only) ;;
  submit) SBATCH_ACTION=() ;;
  *) echo "ACTION must be render, test-only, or submit" >&2; exit 2 ;;
esac

BASE=${BASE:-/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/sna}
REPO=${REPO:-${BASE}/RL-gb200-deck-refresh-20260818}
CONTAINER=${CONTAINER:-/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/mkar/containers/nemo-rl-nightly-ngc-20260815_212622.sqsh}
HF_HOME=${HF_HOME:-${BASE}/hf_home}
ACCOUNT=${SLURM_ACCOUNT:-nemotron_sw_post}
PARTITION=${PARTITION:-batch}
WALLTIME=${WALLTIME:-04:00:00}
RESULT_ROOT=${RESULT_ROOT:-${BASE}/experiments/gb200-deck-refresh}
EXPERIMENT_ROOT=${EXPERIMENT_ROOT:-${RESULT_ROOT}/${MODEL}/${MODE}/${ARM}/${RUN_SUFFIX}}
CACHE_ROOT=${CACHE_ROOT:-${BASE}/.cache/gb200-deck-refresh/${MODEL}/${MODE}/${ARM}/${RUN_SUFFIX}}
WORKER_VENV_ROOT=${WORKER_VENV_ROOT:-/tmp/nemo_rl_worker_venvs/gb200-deck-refresh/${MODEL}/${MODE}/${ARM}/${RUN_SUFFIX}}
RAY_RUNTIME_VENV=${RAY_RUNTIME_VENV:-${BASE}/.cache/gb200-deck-refresh/ray-runtime-py31314}
WANDB_PROJECT=${WANDB_PROJECT:-sna-gb200-deck-refresh}
WANDB_NAME=${WANDB_NAME:-${MODEL}-${MODE}-${ARM}-${MAX_STEPS}step-${RUN_SUFFIX}}

git -C "${REPO}" fetch origin "${BRANCH}"
git -C "${REPO}" pull --ff-only origin "${BRANCH}"
git -C "${REPO}" submodule update --init --recursive
LOCAL_HEAD=$(git -C "${REPO}" rev-parse HEAD)
REMOTE_HEAD=$(git -C "${REPO}" rev-parse "origin/${BRANCH}")
test "${LOCAL_HEAD}" = "${REMOTE_HEAD}"
if [[ -n "${EXPECTED_HEAD}" ]]; then
  test "${LOCAL_HEAD}" = "${EXPECTED_HEAD}"
fi
test -z "$(git -C "${REPO}" status --porcelain --untracked-files=no)"
if git -C "${REPO}" submodule status --recursive | grep -q '^-'; then
  echo "All pinned submodules must be initialized" >&2
  exit 2
fi
for path in "${REPO}/${CONFIG}" "${REPO}/ray.sub" "${CONTAINER}" "${HF_HOME}" \
  "${RAY_RUNTIME_VENV}/bin/python" "${RAY_RUNTIME_VENV}/bin/ray" \
  "${RAY_RUNTIME_VENV}/READY"; do
  test -e "${path}"
done

export SETUP_COMMAND=$(cat <<EOF
set -euo pipefail
rm -f /opt/nemo_rl_venv/bin/ray
ln -s '${RAY_RUNTIME_VENV}/bin/ray' /opt/nemo_rl_venv/bin/ray
'${RAY_RUNTIME_VENV}/bin/python' -c \
  'import ray, requests, urllib3; assert ray.__version__ == "2.56.1"'
ray --version
EOF
)

mkdir -p "${EXPERIMENT_ROOT}" "${CACHE_ROOT}"
WANDB_KEY_FILE=${CACHE_ROOT}/.wandb_key
if [[ -f "${HOME}/.netrc" ]]; then
  umask 077
  awk '
    { for (i = 1; i <= NF; i++) {
        if ($i == "machine" && $(i + 1) == "api.wandb.ai") found = 1
        if (found && $i == "password") { print $(i + 1); exit }
    }}
  ' "${HOME}/.netrc" >"${WANDB_KEY_FILE}"
elif [[ -n "${WANDB_API_KEY:-}" ]]; then
  umask 077
  printf '%s\n' "${WANDB_API_KEY}" >"${WANDB_KEY_FILE}"
fi
test -s "${WANDB_KEY_FILE}"

cat >"${EXPERIMENT_ROOT}/metadata.env" <<EOF
repo_sha=${LOCAL_HEAD}
branch=${BRANCH}
config=${CONFIG}
container=${CONTAINER}
hardware=GB200
model=${MODEL}
mode=${MODE}
arm=${ARM}
nodes=${TOTAL_NODES}
gpus_per_node=4
generation_nodes=${GEN_NODES}
training_precision=bf16
rollout_precision=${ROLLOUT_PRECISION}
quantization_scope=${QUANTIZATION_SCOPE}
refit_transport=${REFIT_TRANSPORT}
colocated=${COLOCATED}
cuda_graphs=enabled
moe_backend=flashinfer_trtllm
logprob_work=previous_policy_and_reference_policy
generation_logprob_reuse=disabled
ray_runtime=${RAY_RUNTIME_VENV}
aggregate_steps=3-20
max_steps=${MAX_STEPS}
seed=42
wandb_project=${WANDB_PROJECT}
wandb_name=${WANDB_NAME}
EOF

COMMAND=$(cat <<EOF
set -euo pipefail
cd ${REPO}
export HF_HOME=${HF_HOME}
export HF_DATASETS_CACHE=${HF_HOME}/cache
export HUGGINGFACE_HUB_CACHE=${HF_HOME}/hub
export NCCL_NVLS_ENABLE=0
export RAY_CGRAPH_get_timeout=2400
export NRL_FORCE_REBUILD_VENVS=true
export NEMO_RL_VENV_DIR=${WORKER_VENV_ROOT}
export NVTE_CUDA_ARCHS=100
export TORCH_CUDA_ARCH_LIST=10.0
export PYTHONPATH=${REPO}
export UV_CACHE_DIR=${CACHE_ROOT}/uv-cache
export UV_PROJECT_ENVIRONMENT=${CACHE_ROOT}/driver-venv
export UV_PYTHON_INSTALL_DIR=${CACHE_ROOT}/uv-python
export UV_LOCK_TIMEOUT=7200
export WANDB_API_KEY="\$(cat ${WANDB_KEY_FILE})"
printf 'NEMO_RL_SOURCE_COMMIT=%s\n' "\$(git rev-parse HEAD)"
${RUN_COMMAND} \
  logger.log_dir=${EXPERIMENT_ROOT}/logs \
  logger.wandb_enabled=true \
  logger.tensorboard_enabled=true \
  logger.monitor_gpus=true \
  ++logger.wandb.entity=nvidia \
  logger.wandb.project=${WANDB_PROJECT} \
  logger.wandb.name=${WANDB_NAME}
uv run --frozen tests/json_dump_tb_logs.py ${EXPERIMENT_ROOT}/logs \
  --output_path ${EXPERIMENT_ROOT}/metrics.json || true
EOF
)

export BASE_LOG_DIR=${EXPERIMENT_ROOT}
export COMMAND
export CONTAINER
export CONTAINER_REMAP_ROOT=1
export GPUS_PER_NODE=4
export MOUNTS=/lustre:/lustre
export UV_CACHE_DIR_OVERRIDE=${CACHE_ROOT}/uv-cache

SBATCH_ARGS=(
  --nodes="${TOTAL_NODES}"
  --exclusive
  --account="${ACCOUNT}"
  --partition="${PARTITION}"
  --time="${WALLTIME}"
  --segment="${SEGMENT_SIZE}"
  --job-name="${ACCOUNT}-gb200-deck.${MODEL}-${MODE}-${ARM}"
  --output="${EXPERIMENT_ROOT}/slurm-%j.out"
  --comment='{"OccupiedIdleGPUsJobReaper":{"exemptIdleTimeMins":"180","reason":"model_loading","description":"NeMo-RL environment build, model load, FlashInfer autotuning, and CUDA Graph capture"}}'
)
if sinfo -p "${PARTITION}" -h -o '%G' | grep -q 'gpu:'; then
  SBATCH_ARGS+=(--gres=gpu:4)
fi

printf 'action=%s\nmodel=%s\nmode=%s\narm=%s\nsha=%s\nresult=%s\n' \
  "${ACTION}" "${MODEL}" "${MODE}" "${ARM}" "${LOCAL_HEAD}" "${EXPERIMENT_ROOT}"
exec sbatch "${SBATCH_ACTION[@]}" "${SBATCH_ARGS[@]}" "${REPO}/ray.sub"
