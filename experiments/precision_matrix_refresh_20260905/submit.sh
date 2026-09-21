#!/usr/bin/env bash

set -euo pipefail

ACTION=${ACTION:-render}
CLUSTER=${CLUSTER:-oci}
MODEL=${MODEL:-qwen30}
MODE=${MODE:-async}
ARM=${ARM:-bf16-bf16}
TOPOLOGY=${TOPOLOGY:-default}
PERFORMANCE_RECIPE=${PERFORMANCE_RECIPE:-0}
SUPER_GPU_MEMORY_UTILIZATION=${SUPER_GPU_MEMORY_UTILIZATION:-}
GPU_MEMORY_UTILIZATION=${GPU_MEMORY_UTILIZATION:-${SUPER_GPU_MEMORY_UTILIZATION}}
MODEL_SNAPSHOT_OVERRIDE=${MODEL_SNAPSHOT_OVERRIDE:-}
SOURCE_ARCHIVE_OVERRIDE=${SOURCE_ARCHIVE_OVERRIDE:-}
SOURCE_ARCHIVE_SHA256=${SOURCE_ARCHIVE_SHA256:-}
SOURCE_PAYLOAD_SHA=${SOURCE_PAYLOAD_SHA:-}
MAX_STEPS=${MAX_STEPS:-20}
RUN_GROUP=${RUN_GROUP:-$(date +%Y%m%d-%H%M%S)}
WALLTIME=${WALLTIME:-04:00:00}
PARTITION=${PARTITION:-}
AFTEROK_JOB_ID=${AFTEROK_JOB_ID:-}
EXPERIMENT=experiments/precision_matrix_refresh_20260905

case "${ACTION}" in
  render|test-only|submit) ;;
  *) echo "ACTION must be render, test-only, or submit" >&2; exit 2 ;;
esac
case "${MODEL}" in
  qwen30|qwen235|lightning|qwen35|super) ;;
  *) echo "MODEL must be qwen30, qwen235, lightning, qwen35, or super" >&2; exit 2 ;;
esac
case "${MODE}" in
  sync|async) ;;
  *) echo "MODE must be sync or async" >&2; exit 2 ;;
esac
case "${ARM}" in
  bf16-bf16|bf16-mxfp8|mxfp8-false-mxfp8|mxfp8-true-mxfp8|mxfp8-mxfp8) ;;
  *) echo "ARM must be bf16-bf16, bf16-mxfp8, mxfp8-false-mxfp8, or mxfp8-true-mxfp8" >&2; exit 2 ;;
esac
case "${TOPOLOGY}" in
  default|ep32-alltoall|ep32-hybridep) ;;
  *) echo "TOPOLOGY must be default, ep32-alltoall, or ep32-hybridep" >&2; exit 2 ;;
esac
if [[ "${TOPOLOGY}" != default && "${MODEL}:${MODE}" != qwen35:sync ]]; then
  echo "TOPOLOGY=${TOPOLOGY} is only defined for MODEL=qwen35 MODE=sync" >&2
  exit 2
fi

case "${CLUSTER}" in
  oci)
    REPO=${REPO:-/home/${USER}/RL-precision-matrix-refresh-20260905}
    CONTAINER=${CONTAINER:-/lustre/fsw/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/${USER}/containers/nemo_rl_nightly.sqsh}
    HF_HOME_SOURCE=${HF_HOME_SOURCE:-/lustre/fsw/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/${USER}/hf_home}
    RESULT_ROOT=${RESULT_ROOT:-/lustre/fsw/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/${USER}/precision-matrix-refresh-20260905}
    LOCAL_ROOT=${LOCAL_ROOT:-/raid/scratch/${USER}/precision-matrix-refresh-20260905}
    GPU_REQUEST=(--gres=gpu:4)
    ;;
  ptyche)
    REPO=${REPO:-/home/${USER}/RL-precision-matrix-refresh-20260905}
    CONTAINER=${CONTAINER:-/lustre/fsw/coreai_dlalgo_llm/users/${USER}/containers/nemo_rl_nightly.sqsh}
    HF_HOME_SOURCE=${HF_HOME_SOURCE:-/lustre/fsw/coreai_dlalgo_llm/users/${USER}/hf_home}
    RESULT_ROOT=${RESULT_ROOT:-/lustre/fsw/coreai_dlalgo_llm/users/${USER}/precision-matrix-refresh-20260905}
    LOCAL_ROOT=${LOCAL_ROOT:-/tmp/${USER}/precision-matrix-refresh-20260905}
    GPU_REQUEST=()
    ;;
  lyris)
    REPO=${REPO:-/home/${USER}/RL-precision-matrix-refresh-20260905}
    CONTAINER=${CONTAINER:-/lustre/fsw/coreai_dlalgo_llm/users/${USER}/containers/nemo_rl_nightly.sqsh}
    HF_HOME_SOURCE=${HF_HOME_SOURCE:-/lustre/fsw/coreai_dlalgo_llm/users/${USER}/hf_home}
    RESULT_ROOT=${RESULT_ROOT:-/lustre/fsw/coreai_dlalgo_llm/users/${USER}/precision-matrix-refresh-20260905}
    LOCAL_ROOT=${LOCAL_ROOT:-/raid/scratch/${USER}/precision-matrix-refresh-20260905}
    GPU_REQUEST=()
    ;;
  *) echo "CLUSTER must be oci, ptyche, or lyris" >&2; exit 2 ;;
esac

: "${SLURM_ACCOUNT:?Set SLURM_ACCOUNT after checking FairShare}"
: "${WANDB_HOME:=/home/${USER}}"
: "${NRL_DISABLE_NUMA_MEMBIND:=1}"
: "${NRL_FORCE_REBUILD_VENVS:=false}"
: "${NRL_IGNORE_VERSION_MISMATCH:=1}"
: "${ACTOR_VENV_ROOT:=/opt/ray_venvs}"

case "${MODEL}:${MODE}" in
  super:sync|super:async)
    CONFIG=examples/configs/recipes/llm/performance/grpo-nemotron3-super-120BA12B-32n4g.yaml
    NUM_NODES=32
    SEGMENT_SIZE=8
    MODEL_CACHE=models--nvidia--NVIDIA-Nemotron-3-Super-120B-A12B-BF16
    FIRST_BF16=2
    LAST_BF16=6
    ;;
  qwen30:sync)
    CONFIG=${EXPERIMENT}/qwen30-sync.yaml
    NUM_NODES=8
    SEGMENT_SIZE=8
    MODEL_CACHE=models--Qwen--Qwen3-30B-A3B
    FIRST_BF16=0
    LAST_BF16=0
    ;;
  qwen30:async)
    CONFIG=${EXPERIMENT}/qwen30-async.yaml
    NUM_NODES=4
    SEGMENT_SIZE=2
    MODEL_CACHE=models--Qwen--Qwen3-30B-A3B
    FIRST_BF16=0
    LAST_BF16=0
    ;;
  qwen235:sync)
    CONFIG=${EXPERIMENT}/qwen235-sync.yaml
    NUM_NODES=16
    SEGMENT_SIZE=16
    MODEL_CACHE=models--Qwen--Qwen3-235B-A22B
    FIRST_BF16=0
    LAST_BF16=0
    ;;
  qwen235:async)
    CONFIG=${EXPERIMENT}/qwen235-async.yaml
    NUM_NODES=32
    SEGMENT_SIZE=16
    MODEL_CACHE=models--Qwen--Qwen3-235B-A22B
    FIRST_BF16=0
    LAST_BF16=0
    ;;
  lightning:sync)
    CONFIG=${EXPERIMENT}/lightning-sync.yaml
    NUM_NODES=8
    SEGMENT_SIZE=8
    MODEL_CACHE=models--nvidia--NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16
    FIRST_BF16=2
    LAST_BF16=6
    ;;
  lightning:async)
    CONFIG=${EXPERIMENT}/lightning-async.yaml
    NUM_NODES=8
    SEGMENT_SIZE=4
    MODEL_CACHE=models--nvidia--NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16
    FIRST_BF16=2
    LAST_BF16=6
    ;;
  qwen35:sync)
    case "${TOPOLOGY}" in
      default)
        CONFIG=${EXPERIMENT}/qwen35-sync.yaml
        NUM_NODES=8
        SEGMENT_SIZE=8
        ;;
      ep32-alltoall)
        CONFIG=${EXPERIMENT}/qwen35-sync-ep32-alltoall.yaml
        NUM_NODES=8
        SEGMENT_SIZE=8
        ;;
      ep32-hybridep)
        CONFIG=${EXPERIMENT}/qwen35-sync-ep32-hybridep.yaml
        NUM_NODES=8
        SEGMENT_SIZE=8
        ;;
    esac
    MODEL_CACHE=models--Qwen--Qwen3.5-35B-A3B-Base
    FIRST_BF16=2
    LAST_BF16=6
    ;;
  qwen35:async)
    CONFIG=${EXPERIMENT}/qwen35-async.yaml
    NUM_NODES=8
    SEGMENT_SIZE=4
    MODEL_CACHE=models--Qwen--Qwen3.5-35B-A3B-Base
    FIRST_BF16=2
    LAST_BF16=6
    ;;
esac

if [[ "${PERFORMANCE_RECIPE}" == 1 ]]; then
  if [[ "${TOPOLOGY}" != default ]]; then
    echo "Performance recipes do not allow topology overrides" >&2
    exit 2
  fi
  PERF_DIR=examples/configs/recipes/llm/performance
  case "${MODEL}:${MODE}" in
    qwen30:sync) CONFIG=${PERF_DIR}/grpo-qwen3-30ba3b-4n4g.yaml; NUM_NODES=4; SEGMENT_SIZE=4 ;;
    qwen30:async) CONFIG=${PERF_DIR}/grpo-qwen3-30ba3b-4n4g-async-1off.yaml ;;
    qwen235:sync) CONFIG=${EXPERIMENT}/qwen235-performance-sync.yaml ;;
    qwen235:async) CONFIG=${PERF_DIR}/grpo-qwen3-235b-32n4g-async-1off.yaml ;;
    qwen35:sync) CONFIG=${EXPERIMENT}/qwen35-performance-sync.yaml ;;
    qwen35:async) CONFIG=${EXPERIMENT}/qwen35-performance-async.yaml ;;
    super:sync) CONFIG=${PERF_DIR}/grpo-nemotron3-super-120BA12B-32n4g.yaml ;;
    super:async) CONFIG=${PERF_DIR}/grpo-nemotron3-super-120BA12B-32n4g-async-1off.yaml ;;
    *) echo "No audited performance recipe for ${MODEL}:${MODE}" >&2; exit 2 ;;
  esac
fi

SOURCE_SHA=$(git -C "${REPO}" rev-parse HEAD 2>/dev/null || printf unknown)
if [[ -z "${SOURCE_PAYLOAD_SHA}" ]]; then
  SOURCE_PAYLOAD_SHA=${SOURCE_SHA}
fi
RUN_NAME="pmx-${CLUSTER}-${MODEL}-${MODE}-${ARM}-${TOPOLOGY}-${RUN_GROUP}"
JOB_NAME="${SLURM_ACCOUNT}-pmx.${CLUSTER}-${MODEL}-${MODE}-${ARM}-${TOPOLOGY}-${RUN_GROUP}"
RUN_ROOT="${RESULT_ROOT}/${RUN_NAME}"
LOCAL_JOB_ROOT="${LOCAL_ROOT}/${RUN_NAME}"
RAY_LOCAL_ROOT=${RAY_LOCAL_ROOT:-/raid/scratch/${USER}/r}
RUN_REPO="${LOCAL_JOB_ROOT}/source"
DATASETS_CACHE="${LOCAL_JOB_ROOT}/hf/datasets"
DATASET_STAGE_COMMAND="if [ -d ${HF_HOME_SOURCE}/datasets ]; then rsync -a --ignore-existing ${HF_HOME_SOURCE}/datasets/ ${LOCAL_JOB_ROOT}/hf/datasets/; fi"
if [[ "${MODE}" == async ]]; then
  # The driver sends a memory-mapped HF dataloader to a Ray actor that may run
  # on another node. Keep dataset Arrow files at one shared, reusable path.
  DATASETS_CACHE="${HF_HOME_SOURCE}/datasets"
  DATASET_STAGE_COMMAND=""
fi
USE_SHARED_MODEL=${USE_SHARED_MODEL:-$([[ ${CLUSTER}:${MODEL} == lyris:qwen235 ]] && printf 1 || printf 0)}
MOE_BACKEND=flashinfer_trtllm

COMMON_OVERRIDES=(
  "grpo.max_num_steps=${MAX_STEPS}"
  "grpo.val_at_start=false"
  "grpo.val_period=0"
  "++grpo.val_at_end=false"
  "++grpo.skip_reference_policy_logprobs_calculation=false"
  "++grpo.seq_logprob_error_threshold=null"
  "loss_fn.force_on_policy_ratio=false"
  "loss_fn.reference_policy_kl_penalty=0.01"
  "cluster.num_nodes=${NUM_NODES}"
  "cluster.gpus_per_node=4"
  "++policy.generation.refit_timeout_s=300.0"
  "checkpointing.enabled=false"
  "policy.generation.vllm_cfg.use_tqdm=false"
  "policy.generation.vllm_cfg.refit_cache_loader_routes=true"
  "policy.generation.vllm_kwargs.moe_backend=${MOE_BACKEND}"
  "policy.generation.vllm_kwargs.expert_placement_strategy=linear"
  "logger.log_dir=${RUN_ROOT}/logs"
  "logger.wandb_enabled=true"
  "logger.wandb.project=nemo-rl-mxfp8-training"
  "logger.wandb.name=${RUN_NAME}"
  "logger.monitor_gpus=true"
)

if [[ "${MODEL}" == qwen35 ]]; then
  COMMON_OVERRIDES+=(
    "grpo.num_prompts_per_step=${QWEN35_NUM_PROMPTS_PER_STEP:-128}"
    "grpo.num_generations_per_prompt=${QWEN35_NUM_GENERATIONS_PER_PROMPT:-16}"
    "policy.train_global_batch_size=${QWEN35_TRAIN_GLOBAL_BATCH_SIZE:-2048}"
  )
fi

case "${ARM}" in
  bf16-bf16)
    PRECISION_OVERRIDES=(
      "policy.megatron_cfg.fp8_cfg.enabled=false"
      "policy.megatron_cfg.fp8_cfg.fp8_param=false"
      "policy.generation.vllm_cfg.precision=bfloat16"
      "++policy.generation.vllm_cfg.is_mx=false"
      "policy.generation.vllm_cfg.refit_prequantize=false"
      "policy.generation.vllm_cfg.num_first_layers_in_bf16=0"
      "policy.generation.vllm_cfg.num_last_layers_in_bf16=0"
    )
    ;;
  bf16-mxfp8)
    PRECISION_OVERRIDES=(
      "policy.megatron_cfg.fp8_cfg.enabled=false"
      "policy.megatron_cfg.fp8_cfg.fp8_param=false"
      "policy.generation.vllm_cfg.precision=fp8"
      "++policy.generation.vllm_cfg.is_mx=true"
      "policy.generation.vllm_cfg.refit_prequantize=$([[ ${MODE} == sync ]] && printf true || printf false)"
      "policy.generation.vllm_cfg.num_first_layers_in_bf16=${FIRST_BF16}"
      "policy.generation.vllm_cfg.num_last_layers_in_bf16=${LAST_BF16}"
    )
    ;;
  mxfp8-false-mxfp8)
    PRECISION_OVERRIDES=(
      "policy.megatron_cfg.fp8_cfg.enabled=true"
      "policy.megatron_cfg.fp8_cfg.fp8=e4m3"
      "policy.megatron_cfg.fp8_cfg.fp8_recipe=mxfp8"
      "policy.megatron_cfg.fp8_cfg.fp8_param=false"
      "++policy.megatron_cfg.moe_router_dtype=fp32"
      "++policy.megatron_cfg.te_precision_config_file=${RUN_REPO}/${EXPERIMENT}/te_routed.yaml"
      "++policy.megatron_cfg.first_last_layers_bf16=true"
      "++policy.megatron_cfg.num_layers_at_start_in_bf16=${FIRST_BF16}"
      "++policy.megatron_cfg.num_layers_at_end_in_bf16=${LAST_BF16}"
      "policy.megatron_cfg.distributed_data_parallel_config.overlap_param_gather=true"
      "policy.megatron_cfg.distributed_data_parallel_config.overlap_grad_reduce=true"
      "policy.generation.vllm_cfg.precision=fp8"
      "++policy.generation.vllm_cfg.is_mx=true"
      "policy.generation.vllm_cfg.refit_prequantize=$([[ ${MODE} == sync ]] && printf true || printf false)"
      "policy.generation.vllm_cfg.num_first_layers_in_bf16=${FIRST_BF16}"
      "policy.generation.vllm_cfg.num_last_layers_in_bf16=${LAST_BF16}"
    )
    ;;
  mxfp8-true-mxfp8|mxfp8-mxfp8)
    PRECISION_OVERRIDES=(
      "policy.megatron_cfg.fp8_cfg.enabled=true"
      "policy.megatron_cfg.fp8_cfg.fp8=e4m3"
      "policy.megatron_cfg.fp8_cfg.fp8_recipe=mxfp8"
      "policy.megatron_cfg.fp8_cfg.fp8_param=true"
      "++policy.megatron_cfg.moe_router_dtype=fp32"
      "++policy.megatron_cfg.te_precision_config_file=${RUN_REPO}/${EXPERIMENT}/te_routed_fp8param.yaml"
      "++policy.megatron_cfg.first_last_layers_bf16=true"
      "++policy.megatron_cfg.num_layers_at_start_in_bf16=${FIRST_BF16}"
      "++policy.megatron_cfg.num_layers_at_end_in_bf16=${LAST_BF16}"
      "policy.megatron_cfg.distributed_data_parallel_config.overlap_param_gather=true"
      "policy.megatron_cfg.distributed_data_parallel_config.overlap_grad_reduce=true"
      "policy.generation.vllm_cfg.precision=fp8"
      "++policy.generation.vllm_cfg.is_mx=true"
      "policy.generation.vllm_cfg.refit_prequantize=false"
      "policy.generation.vllm_cfg.num_first_layers_in_bf16=${FIRST_BF16}"
      "policy.generation.vllm_cfg.num_last_layers_in_bf16=${LAST_BF16}"
    )
    ;;
esac

if [[ "${PERFORMANCE_RECIPE}" == 1 ]]; then
  # Keep workload and topology from the audited recipe; add only matched
  # precision, refit, and metric controls.
  NORMALIZED_OVERRIDES=()
  for override in "${COMMON_OVERRIDES[@]}" "${PRECISION_OVERRIDES[@]}"; do
    case "${override}" in
      *overlap_param_gather=*|*overlap_grad_reduce=*) continue ;;
    esac
    override=${override#++}
    NORMALIZED_OVERRIDES+=("++${override}")
  done
  COMMON_OVERRIDES=("${NORMALIZED_OVERRIDES[@]}")
  PRECISION_OVERRIDES=("++loss_fn.use_importance_sampling_correction=true")

  if [[ "${MODE}" == async ]]; then
    PRECISION_OVERRIDES+=("++policy.generation.refit_transport=nccl_reshard")
  fi

  if [[ -n "${GPU_MEMORY_UTILIZATION}" ]]; then
    PRECISION_OVERRIDES+=(
      "++policy.generation.vllm_cfg.gpu_memory_utilization=${GPU_MEMORY_UTILIZATION}"
    )
  fi

  # Qwen3.5 carries its model-specific vision, attention, GDN, and shared
  # expert exclusions in the wrapper YAML. Other performance recipes need
  # their routed-expert-only rollout scope supplied here.
  if [[ "${ARM}" == *-mxfp8 && "${MODEL}" != qwen35 ]]; then
    if [[ "${MODEL}" == super ]]; then
      IGNORE_PATTERNS='["*layers.*.mixer.qkv_proj","*layers.*.mixer.o_proj","*layers.*.mixer.in_proj","*layers.*.mixer.out_proj","*layers.*.mixer.up_proj","*layers.*.mixer.down_proj","*layers.*.mixer.gate","*layers.*.mixer.shared_experts.*","*layers.*.mixer.fc1_latent_proj","*layers.*.mixer.fc2_latent_proj","*mtp.*","lm_head"]'
    else
      IGNORE_PATTERNS='["*layers.*.self_attn.*","*layers.*.mlp.gate","*layers.*.mlp.shared_experts.*","*mtp.*","lm_head"]'
    fi
    PRECISION_OVERRIDES+=("++policy.generation.vllm_cfg.quantization_ignore_patterns=${IGNORE_PATTERNS}")
  fi
fi

printf 'cluster=%s\nmodel=%s\nmode=%s\narm=%s\ntopology=%s\nconfig=%s\nnodes=%s\nsegment=%s\nsteps=%s\nshared_model=%s\nmoe_backend=%s\ngpu_memory_utilization=%s\ndatasets_cache=%s\nray_local_root=%s\nnuma_membind_disabled=%s\nforce_rebuild_venvs=%s\nactor_venv_root=%s\nsha=%s\nsource_payload_sha=%s\nsource_archive_override=%s\nsource_archive_sha256=%s\nray_memory_usage_threshold=%s\nrun=%s\n' \
  "${CLUSTER}" "${MODEL}" "${MODE}" "${ARM}" "${TOPOLOGY}" "${CONFIG}" "${NUM_NODES}" \
  "${SEGMENT_SIZE}" "${MAX_STEPS}" "${USE_SHARED_MODEL}" "${MOE_BACKEND}" \
  "${GPU_MEMORY_UTILIZATION}" "${DATASETS_CACHE}" "${RAY_LOCAL_ROOT}" \
  "${NRL_DISABLE_NUMA_MEMBIND}" "${NRL_FORCE_REBUILD_VENVS}" "${ACTOR_VENV_ROOT}" "${SOURCE_SHA}" \
  "${SOURCE_PAYLOAD_SHA}" "${SOURCE_ARCHIVE_OVERRIDE}" "${SOURCE_ARCHIVE_SHA256}" \
  "${RAY_memory_usage_threshold:-}" "${RUN_NAME}"
printf 'overrides:'
printf ' %q' "${COMMON_OVERRIDES[@]}" "${PRECISION_OVERRIDES[@]}"
printf '\n'

if [[ "${ACTION}" == render ]]; then
  exit 0
fi

MODEL_SOURCE="${HF_HOME_SOURCE}/hub/${MODEL_CACHE}"
if [[ -n "${MODEL_SNAPSHOT_OVERRIDE}" ]]; then
  MODEL_SOURCE="${MODEL_SNAPSHOT_OVERRIDE}"
fi

for path in "${REPO}/${CONFIG}" "${REPO}/ray.sub" "${CONTAINER}" \
  "${MODEL_SOURCE}" "${WANDB_HOME}/.netrc"; do
  if [[ ! -e "${path}" ]]; then
    echo "Missing required path: ${path}" >&2
    exit 2
  fi
done

MODEL_STAGE_COMMAND="rsync -a --ignore-existing ${MODEL_SOURCE}/ ${LOCAL_JOB_ROOT}/hf/hub/${MODEL_CACHE}/;"
if [[ -n "${MODEL_SNAPSHOT_OVERRIDE}" ]]; then
  COMMON_OVERRIDES+=("policy.model_name=${MODEL_SNAPSHOT_OVERRIDE}")
  MODEL_STAGE_COMMAND=""
elif [[ "${USE_SHARED_MODEL}" == 1 ]]; then
  MODEL_REF_FILE="${HF_HOME_SOURCE}/hub/${MODEL_CACHE}/refs/main"
  if [[ ! -f "${MODEL_REF_FILE}" ]]; then
    echo "Missing model ref: ${MODEL_REF_FILE}" >&2
    exit 2
  fi
  MODEL_SNAPSHOT="${HF_HOME_SOURCE}/hub/${MODEL_CACHE}/snapshots/$(<"${MODEL_REF_FILE}")"
  if [[ ! -d "${MODEL_SNAPSHOT}" ]]; then
    echo "Missing model snapshot: ${MODEL_SNAPSHOT}" >&2
    exit 2
  fi
  COMMON_OVERRIDES+=("policy.model_name=${MODEL_SNAPSHOT}")
  MODEL_STAGE_COMMAND=""
fi

if [[ "${ACTION}" == submit ]]; then
  git -C "${REPO}" -c fetch.recurseSubmodules=false pull --ff-only
  if [[ -z "${SOURCE_ARCHIVE_OVERRIDE}" ]]; then
    git -C "${REPO}" submodule update --init --recursive --checkout
    if [[ -n "$(git -C "${REPO}" status --porcelain --untracked-files=no --ignore-submodules=none)" ]]; then
      echo "Repository and pinned submodules must be clean before submission" >&2
      exit 2
    fi
  elif [[ -n "$(git -C "${REPO}" status --porcelain --untracked-files=no --ignore-submodules=all)" ]]; then
    echo "Repository must be clean before submission" >&2
    exit 2
  fi
fi

SOURCE_SHA=$(git -C "${REPO}" rev-parse HEAD)
if [[ -n "${EXPECTED_SOURCE_SHA:-}" && "${SOURCE_SHA}" != "${EXPECTED_SOURCE_SHA}" ]]; then
  echo "Source changed after preflight: expected ${EXPECTED_SOURCE_SHA}, got ${SOURCE_SHA}" >&2
  exit 2
fi
if [[ -n "${SOURCE_ARCHIVE_OVERRIDE}" ]]; then
  : "${SOURCE_ARCHIVE_SHA256:?Set SOURCE_ARCHIVE_SHA256 with SOURCE_ARCHIVE_OVERRIDE}"
  : "${SOURCE_PAYLOAD_SHA:?Set SOURCE_PAYLOAD_SHA with SOURCE_ARCHIVE_OVERRIDE}"
  SOURCE_ARCHIVE=${SOURCE_ARCHIVE_OVERRIDE}
  if [[ ! -f "${SOURCE_ARCHIVE}" ]]; then
    echo "Missing source archive override: ${SOURCE_ARCHIVE}" >&2
    exit 2
  fi
  ACTUAL_SOURCE_ARCHIVE_SHA256=$(sha256sum "${SOURCE_ARCHIVE}" | cut -d ' ' -f 1)
  if [[ "${ACTUAL_SOURCE_ARCHIVE_SHA256}" != "${SOURCE_ARCHIVE_SHA256}" ]]; then
    echo "Source archive SHA256 mismatch: expected ${SOURCE_ARCHIVE_SHA256}, got ${ACTUAL_SOURCE_ARCHIVE_SHA256}" >&2
    exit 2
  fi
else
  SOURCE_PAYLOAD_SHA=${SOURCE_SHA}
  SOURCE_STATE=$(git -C "${REPO}" submodule status --recursive)
  SOURCE_ID=$(printf '%s\n%s\n' "${SOURCE_SHA}" "${SOURCE_STATE}" | sha256sum | cut -c1-16)
  # Compute nodes do not necessarily share the login node's /home filesystem.
  # Keep one immutable tar on shared storage, then expand it into node-local
  # scratch so source files never create metadata traffic on Lustre at runtime.
  SOURCE_ARCHIVE_ROOT=${SOURCE_ARCHIVE_ROOT:-${RESULT_ROOT}/source-archives}
  SOURCE_ARCHIVE="${SOURCE_ARCHIVE_ROOT}/nemo-rl-${SOURCE_ID}.tar"

  if [[ "${ACTION}" == submit && ! -f "${SOURCE_ARCHIVE}" ]]; then
    mkdir -p "${SOURCE_ARCHIVE_ROOT}"
    SOURCE_MANIFEST=$(mktemp "${TMPDIR:-/tmp}/nemo-rl-source-manifest.XXXXXX")
    SOURCE_ARCHIVE_TMP=$(mktemp "${TMPDIR:-/tmp}/nemo-rl-source.XXXXXX.tar")
    trap 'rm -f "${SOURCE_MANIFEST:-}" "${SOURCE_ARCHIVE_TMP:-}"' EXIT
    git -C "${REPO}" ls-files -z --recurse-submodules --cached --full-name > "${SOURCE_MANIFEST}"
    tar --null -cf "${SOURCE_ARCHIVE_TMP}" -C "${REPO}" -T "${SOURCE_MANIFEST}"
    if [[ ! -f "${SOURCE_ARCHIVE}" ]]; then
      mv "${SOURCE_ARCHIVE_TMP}" "${SOURCE_ARCHIVE}"
    fi
    rm -f "${SOURCE_MANIFEST}" "${SOURCE_ARCHIVE_TMP}"
    trap - EXIT
  fi

  if [[ "${ACTION}" == submit && ! -f "${SOURCE_ARCHIVE}" ]]; then
    echo "Failed to create source archive: ${SOURCE_ARCHIVE}" >&2
    exit 2
  fi
fi

mkdir -p "${RUN_ROOT}/logs"

COMMAND=$(printf '%q ' /opt/nemo_rl_venv/bin/python examples/run_grpo.py \
  --config "${CONFIG}" "${COMMON_OVERRIDES[@]}" "${PRECISION_OVERRIDES[@]}")
COMMAND="set -euo pipefail; cd ${RUN_REPO}; \
export HOME=/root; \
export HF_HOME=${LOCAL_JOB_ROOT}/hf; \
export HF_DATASETS_CACHE=${DATASETS_CACHE}; \
export HUGGINGFACE_HUB_CACHE=${LOCAL_JOB_ROOT}/hf/hub; \
export NRL_MEGATRON_CHECKPOINT_DIR=${HF_HOME_SOURCE}/nemo_rl; \
export NEMO_RL_VENV_DIR=${ACTOR_VENV_ROOT}; \
export VLLM_CACHE_ROOT=${LOCAL_JOB_ROOT}/vllm; \
export TORCHINDUCTOR_CACHE_DIR=${LOCAL_JOB_ROOT}/inductor; \
export TRITON_CACHE_DIR=${LOCAL_JOB_ROOT}/triton; \
export UV_CACHE_DIR=${LOCAL_JOB_ROOT}/uv; \
export RAY_TMPDIR=${RAY_LOCAL_ROOT}/\${SLURM_JOB_ID}; \
export PYTHONPATH=${RUN_REPO}:${RUN_REPO}/3rdparty/Megatron-Bridge-workspace/Megatron-Bridge/src:${RUN_REPO}/3rdparty/Megatron-Bridge-workspace/Megatron-Bridge/3rdparty/Megatron-LM; \
export FLA_TILELANG=0; \
export NRL_DISABLE_NUMA_MEMBIND=${NRL_DISABLE_NUMA_MEMBIND}; \
export NRL_FORCE_REBUILD_VENVS=${NRL_FORCE_REBUILD_VENVS}; \
export NRL_IGNORE_VERSION_MISMATCH=${NRL_IGNORE_VERSION_MISMATCH}; \
${COMMAND}"

SETUP_COMMAND="set -euo pipefail; \
rm -rf ${LOCAL_JOB_ROOT} ${RAY_LOCAL_ROOT}/\${SLURM_JOB_ID}; \
mkdir -p ${RUN_REPO} ${LOCAL_JOB_ROOT}/hf/hub ${LOCAL_JOB_ROOT}/hf/datasets ${LOCAL_JOB_ROOT}/vllm ${LOCAL_JOB_ROOT}/inductor ${LOCAL_JOB_ROOT}/triton ${LOCAL_JOB_ROOT}/uv ${RAY_LOCAL_ROOT}/\${SLURM_JOB_ID}; \
tar -xf ${SOURCE_ARCHIVE} -C ${RUN_REPO}; \
${MODEL_STAGE_COMMAND} \
${DATASET_STAGE_COMMAND}"

export CONTAINER
export MOUNTS="/lustre:/lustre,/home:/home,${WANDB_HOME}/.netrc:/root/.netrc"
if [[ -n "${VLLM_PADDING_SOURCE:-}" ]]; then
  : "${VLLM_PADDING_SHA:?Pin the tested vLLM commit}"
  if [[ "${NRL_FORCE_REBUILD_VENVS}" != false || "${NEMO_RL_PY_EXECUTABLES_SYSTEM:-0}" != 0 ]]; then
    echo "The tested vLLM overlay requires the prebuilt actor environment" >&2
    exit 2
  fi
  VLLM_RESOLVED_SHA=$(git -C "${VLLM_PADDING_SOURCE}" rev-parse "${VLLM_PADDING_SHA}^{commit}")
  if [[ "${VLLM_RESOLVED_SHA}" != "${VLLM_PADDING_SHA}" ]]; then
    echo "VLLM_PADDING_SHA must be a full commit SHA" >&2
    exit 2
  fi
  VLLM_FILES=(
    vllm/model_executor/layers/quantization/utils/flashinfer_utils.py
    vllm/model_executor/layers/fused_moe/oracle/unquantized.py
  )
  VLLM_SNAPSHOT_ROOT=/home/${USER}/.cache/nemo-rl-vllm-overlays
  VLLM_SNAPSHOT=${VLLM_SNAPSHOT_ROOT}/${VLLM_RESOLVED_SHA}
  mkdir -p "${VLLM_SNAPSHOT_ROOT}"
  if [[ ! -d "${VLLM_SNAPSHOT}" ]]; then
    VLLM_STAGE=$(mktemp -d "${VLLM_SNAPSHOT_ROOT}/.stage.XXXXXX")
    git -C "${VLLM_PADDING_SOURCE}" archive "${VLLM_RESOLVED_SHA}" "${VLLM_FILES[@]}" | tar -xf - -C "${VLLM_STAGE}"
    mv -T "${VLLM_STAGE}" "${VLLM_SNAPSHOT}"
  fi
  if [[ "${MODE}" == sync ]]; then
    VLLM_WORKER_VENV=nemo_rl.models.generation.vllm.vllm_worker.VllmGenerationWorker
  else
    VLLM_WORKER_VENV=nemo_rl.models.generation.vllm.vllm_worker_async.VllmAsyncGenerationWorker
  fi
  VLLM_PACKAGE=${ACTOR_VENV_ROOT}/${VLLM_WORKER_VENV}/lib/python3.13/site-packages
  for file in "${VLLM_FILES[@]}"; do
    git -C "${VLLM_PADDING_SOURCE}" show "${VLLM_RESOLVED_SHA}:${file}" | cmp -s - "${VLLM_SNAPSHOT}/${file}"
    MOUNTS+=",${VLLM_SNAPSHOT}/${file}:${VLLM_PACKAGE}/${file}:ro"
  done
  printf 'vllm_overlay_commit=%s\n' "${VLLM_RESOLVED_SHA}"
  sha256sum "${VLLM_FILES[@]/#/${VLLM_SNAPSHOT}/}"
  COMMAND="export PYTHONDONTWRITEBYTECODE=1 PYTHONPYCACHEPREFIX=${LOCAL_JOB_ROOT}/pycache; ${COMMAND}"
fi
if [[ "${CLUSTER}" == oci || "${CLUSTER}" == lyris ]]; then
  MOUNTS="${MOUNTS},/raid/scratch:/raid/scratch"
fi
export CONTAINER_REMAP_ROOT=1
export COMMAND
export SETUP_COMMAND
export GPUS_PER_NODE=4
export CPUS_PER_WORKER=${CPUS_PER_WORKER:-144}
export BASE_LOG_DIR="${RUN_ROOT}"
export RAY_TMPDIR_ROOT="${RAY_LOCAL_ROOT}"

SBATCH_MODE=()
if [[ "${ACTION}" == test-only ]]; then
  SBATCH_MODE=(--test-only)
fi

SBATCH_PARTITION=()
if [[ -n "${PARTITION}" ]]; then
  SBATCH_PARTITION=(--partition="${PARTITION}")
fi

SBATCH_DEPENDENCY=(--dependency=)
if [[ -n "${AFTEROK_JOB_ID}" ]]; then
  SBATCH_DEPENDENCY=(--dependency="afterok:${AFTEROK_JOB_ID}")
fi

# Resolve through login-node wrappers while they are available. ray.sub also
# probes versioned Slurm directories because wrapper paths may not be mounted
# on compute nodes.
if [[ -z "${SLURM_COMMAND_PATH:-}" ]] && command -v scontrol >/dev/null 2>&1; then
  SLURM_COMMAND_PATH=$(dirname "$(readlink -f "$(command -v scontrol)")")
fi
export SLURM_COMMAND_PATH

exec sbatch "${SBATCH_MODE[@]}" \
  --nodes="${NUM_NODES}" \
  "${GPU_REQUEST[@]}" \
  --exclusive \
  --account="${SLURM_ACCOUNT}" \
  "${SBATCH_PARTITION[@]}" \
  --time="${WALLTIME}" \
  --segment="${SEGMENT_SIZE}" \
  "${SBATCH_DEPENDENCY[@]}" \
  --job-name="${JOB_NAME}" \
  --output="${RUN_ROOT}/slurm-%j.out" \
  --comment='{"OccupiedIdleGPUsJobReaper":{"exemptIdleTimeMins":"120","reason":"model_loading","description":"precision matrix startup"}}' \
  "${REPO}/ray.sub"
