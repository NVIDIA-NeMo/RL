#!/usr/bin/env bash
set -euo pipefail

# Slurm launcher for Nemotron Super Omni multimodal Gym GRPO.
#
# Standalone by design. The text-only Super stages share
# examples/nemo_gym/nemotron-3-super/super_launch.sh, which hardcodes the
# text training entrypoint. The Omni path needs the multimodal driver and
# Megatron sources on PYTHONPATH, so rather than add a mode to a launcher with
# several other consumers, the Slurm plumbing is duplicated here.
#
# Every path below is a placeholder; export the real values:
#   MODEL_PATH=... TRAIN_PATH=... CONTAINER=... ./super_omni_launch.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CODE_DIR="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
cd "${CODE_DIR}"

export EXP_NAME="${EXP_NAME:-grpo-super-omni-async-gym}"
export MODEL_PATH="${MODEL_PATH:-/path/to/nemotron-super-omni-hf-checkpoint}"
export TRAIN_PATH="${TRAIN_PATH:-/path/to/super-omni-gym-train.jsonl}"
export VAL_PATH="${VAL_PATH:-${TRAIN_PATH}}"
export CONTAINER="${CONTAINER:-/path/to/nemo-rl.sqsh}"
export SANDBOX_CONTAINER="${SANDBOX_CONTAINER:-}"
export PERSISTENT_CACHE="${PERSISTENT_CACHE:-/path/to/cache/nemo-rl-super-omni}"
export SLURM_ACCOUNT="${SLURM_ACCOUNT:-your_slurm_account}"
export SLURM_PARTITION="${SLURM_PARTITION:-batch}"

# Transformers derives trust_remote_code module names from local path basenames.
# A trailing slash gives an empty basename and can collide in the import cache.
while [[ "${MODEL_PATH}" == */ && "${MODEL_PATH}" != "/" ]]; do
    MODEL_PATH="${MODEL_PATH%/}"
done

CONFIG_PATH="${CONFIG_PATH:-examples/configs/recipes/vlm/vlm_grpo-nemotron-super-omni-120ba12b-16n8g-megatron-tp8ep16cp2-async-gym.v1.yaml}"
ENTRYPOINT="${ENTRYPOINT:-examples/nemo_gym/run_grpo_nemo_gym.py}"
SLURM_TIME_LIMIT="${SLURM_TIME_LIMIT:-4:0:0}"
SBATCH_NUM_NODES="${SBATCH_NUM_NODES:-$(awk '/^cluster:/{f=1} f && /num_nodes:/{print $2; exit}' "${CONFIG_PATH}")}"
EXTRA_MOUNTS="${EXTRA_MOUNTS:-/scratch:/scratch,/lustre:/lustre}"
EXTRA_HYDRA_ARGS="${EXTRA_HYDRA_ARGS:-}"
CLUSTER_VENV="${CLUSTER_VENV:-}"
NRL_FORCE_REBUILD_VENVS="${NRL_FORCE_REBUILD_VENVS:-${CLUSTER_VENV:+true}}"
NRL_FORCE_REBUILD_VENVS="${NRL_FORCE_REBUILD_VENVS:-false}"
GYM_VENV_DIR="${GYM_VENV_DIR:-/opt/gym_venvs}"
GYM_SKIP_VENV_IF_PRESENT="${GYM_SKIP_VENV_IF_PRESENT:-false}"
WANDB_PROJ="${WANDB_PROJ:-grpo-super-omni}"
WANDB_ENTITY="${WANDB_ENTITY:-}"
TEACHER_MODEL_PATH="${TEACHER_MODEL_PATH:-}"
DRY_RUN="${DRY_RUN:-false}"

while [[ -n "${TEACHER_MODEL_PATH}" && "${TEACHER_MODEL_PATH}" == */ && "${TEACHER_MODEL_PATH}" != "/" ]]; do
    TEACHER_MODEL_PATH="${TEACHER_MODEL_PATH%/}"
done

CHECKPOINT_DIR="results/${EXP_NAME}"
LOG_DIR="logs/${EXP_NAME}"
VLLM_CACHE_DIR="${PERSISTENT_CACHE}/vllm_compile_cache"
FLASHINFER_CUBIN_CACHE="${PERSISTENT_CACHE}/flashinfer_cubins"
FLASHINFER_WS_BASE="${PERSISTENT_CACHE}/flashinfer_workspace"
MEGATRON_CONFIG_LOCK_DIR="${PERSISTENT_CACHE}/hf_config_locks"
NRL_MEGATRON_CHECKPOINT_DIR="${NRL_MEGATRON_CHECKPOINT_DIR:-${PERSISTENT_CACHE}/megatron_ckpt_cache}"
HF_MODULES_CACHE_DIR="${HF_MODULES_CACHE:-${PERSISTENT_CACHE}/hf_modules/${EXP_NAME}}"
CHAT_TEMPLATE="${CHAT_TEMPLATE:-${MODEL_PATH}/chat_template.jinja}"

if [[ -z "${SBATCH_NUM_NODES}" ]]; then
    echo "Error: could not read cluster.num_nodes from ${CONFIG_PATH}" >&2
    exit 1
fi

# Fail on the placeholders above rather than deep inside mkdir/sbatch.
unset_placeholders=()
for var in MODEL_PATH TRAIN_PATH VAL_PATH CONTAINER PERSISTENT_CACHE SLURM_ACCOUNT; do
    case "${!var}" in
        /path/to/*|your_slurm_account) unset_placeholders+=("${var}") ;;
    esac
done
if (( ${#unset_placeholders[@]} )); then
    echo "Error: these still hold placeholder values; export real ones:" >&2
    for var in "${unset_placeholders[@]}"; do
        printf '  %-20s = %s\n' "${var}" "${!var}" >&2
    done
    exit 1
fi

if [[ -n "${CLUSTER_VENV}" ]]; then
    for tool in python python3 ray; do
        if [[ ! -x "${CLUSTER_VENV}/bin/${tool}" ]]; then
            echo "Error: CLUSTER_VENV is missing executable bin/${tool}: ${CLUSTER_VENV}" >&2
            exit 1
        fi
    done

    # Actor environments must use the same Python patch version as the Ray
    # head/driver. A relocatable venv exposes its underlying interpreter here.
    export UV_PYTHON="${UV_PYTHON:-$("${CLUSTER_VENV}/bin/python" -c 'import sys; print(sys._base_executable)')}"

    read -r -d '' cluster_venv_setup <<SETUPEOF || true
set -euo pipefail
CLUSTER_VENV=${CLUSTER_VENV}
for tool in python python3 ray; do
    src="\${CLUSTER_VENV}/bin/\${tool}"
    dst="/opt/nemo_rl_venv/bin/\${tool}"
    test -x "\${src}"
    rm -f "\${dst}"
    printf '#!/bin/sh\nexec "%s" "\$@"\n' "\${src}" > "\${dst}"
    chmod 0755 "\${dst}"
done
hash -r
python -c 'import platform, ray; print("[cluster_venv] Python=" + platform.python_version() + " Ray=" + ray.__version__)'
SETUPEOF
    if [[ -n "${SETUP_COMMAND:-}" ]]; then
        SETUP_COMMAND="${SETUP_COMMAND}"$'\n'"${cluster_venv_setup}"
    else
        SETUP_COMMAND="${cluster_venv_setup}"
    fi
    export SETUP_COMMAND
fi

if [[ -n "${TEACHER_MODEL_PATH}" ]]; then
    EXTRA_HYDRA_ARGS+=" on_policy_distillation.teacher_model_by_agent_name.circle_count_simple_agent=${TEACHER_MODEL_PATH}"
fi
if [[ -n "${WANDB_ENTITY}" ]]; then
    EXTRA_HYDRA_ARGS+=" ++logger.wandb.entity=${WANDB_ENTITY}"
fi

# The driver builds the W&B run before any worker starts, so a missing
# credential kills the job minutes into a full allocation. ~/.netrc is not
# enough when /home is unmounted inside the container; only the environment
# carries the key through sbatch --export=ALL.
WANDB_MODE="${WANDB_MODE:-online}"
if [[ ! "${EXTRA_HYDRA_ARGS}" =~ logger\.wandb_enabled=([Ff]alse|0) ]] \
    && [[ "${WANDB_MODE}" == "online" && -z "${WANDB_API_KEY:-}" ]]; then
    echo "Error: W&B logging is on but WANDB_API_KEY is unset." >&2
    echo "  export WANDB_API_KEY=<key>, or WANDB_MODE=offline, or add" >&2
    echo "  logger.wandb_enabled=false to EXTRA_HYDRA_ARGS." >&2
    exit 1
fi
export WANDB_MODE

if [[ "${DRY_RUN}" != true ]]; then
    mkdir -p "${VLLM_CACHE_DIR}" "${FLASHINFER_CUBIN_CACHE}" "${FLASHINFER_WS_BASE}" \
             "${MEGATRON_CONFIG_LOCK_DIR}" "${HF_MODULES_CACHE_DIR}"
fi
export OMP_NUM_THREADS=16

if [[ "${DRY_RUN}" != true && "${ALLOW_DIRTY_SNAPSHOT:-false}" != true ]]; then
    if ! git diff --quiet || ! git diff --cached --quiet; then
        echo "Error: tracked root changes would make the snapshot disagree with its commit." >&2
        echo "Commit/stash them, or set ALLOW_DIRTY_SNAPSHOT=true for development only." >&2
        exit 1
    fi
    if ! git submodule foreach --quiet --recursive 'git diff --quiet && git diff --cached --quiet'; then
        echo "Error: a submodule has tracked changes; refusing a non-reproducible snapshot." >&2
        exit 1
    fi
fi

SOURCE_SHA="$(git rev-parse --verify HEAD)"
SNAPSHOT_NAME="${EXP_NAME}-${SOURCE_SHA:0:12}"
if [[ "${DRY_RUN}" == true ]]; then
    SNAPSHOT_DIR="$(realpath "${CODE_DIR}")"
else
    SNAPSHOT_ROOT="${CODE_DIR}/${CODE_SNAPSHOT_DIRNAME:-code_snapshots}"
    mkdir -p "${SNAPSHOT_ROOT}"
    SNAPSHOT_LOCK="${SNAPSHOT_ROOT}/.${SNAPSHOT_NAME}.lock"
    exec {SNAPSHOT_LOCK_FD}>"${SNAPSHOT_LOCK}"
    flock "${SNAPSHOT_LOCK_FD}"

    SNAPSHOT_DIR=$(realpath "$(bash "${CODE_DIR}/tools/code_snapshot.sh" "${SNAPSHOT_NAME}")")
    if [[ ! -f "${SNAPSHOT_DIR}/.nrl_snapshot_complete" ]]; then
        echo "Finalizing tracked files in code snapshot: ${SNAPSHOT_DIR}"
        (
            cd "${CODE_DIR}"
            rsync -a --files-from=<(git ls-files --recurse-submodules --cached --full-name) ./ "${SNAPSHOT_DIR}/"
            printf '%s\n' "${SOURCE_SHA}" > "${SNAPSHOT_DIR}/.nrl_source_commit"
            git submodule status --recursive > "${SNAPSHOT_DIR}/.nrl_submodules" || true
            touch "${SNAPSHOT_DIR}/.nrl_snapshot_complete"
        )
    fi
    {
        echo "source_commit=${SOURCE_SHA}"
        echo "config=${CONFIG_PATH}"
        echo "container=${CONTAINER}"
        echo "model=${MODEL_PATH}"
        echo "teacher_model=${TEACHER_MODEL_PATH:-${MODEL_PATH}}"
        echo "cluster_venv=${CLUSTER_VENV:-container-default}"
        if [[ -n "${CLUSTER_VENV}" ]]; then
            "${CLUSTER_VENV}/bin/python" -c 'import platform, ray; print("python=" + platform.python_version()); print("ray=" + ray.__version__)'
        fi
    } > "${SNAPSHOT_DIR}/.nrl_run_manifest"
fi
cd "${SNAPSHOT_DIR}"

# Megatron is imported from the checkout rather than the container's
# site-packages. Ray starts its interpreters before COMMAND runs, so the module
# cache must be exported from the submitting shell for isolated actors to
# import trust_remote_code classes while deserializing their arguments.
MEGATRON_BRIDGE_SRC="${SNAPSHOT_DIR}/3rdparty/Megatron-Bridge-workspace/Megatron-Bridge/src"
MEGATRON_LM_SRC="${SNAPSHOT_DIR}/3rdparty/Megatron-Bridge-workspace/Megatron-Bridge/3rdparty/Megatron-LM"
export HF_MODULES_CACHE="${HF_MODULES_CACHE_DIR}"
export PYTHONPATH="${HF_MODULES_CACHE_DIR}:${SNAPSHOT_DIR}:${MEGATRON_BRIDGE_SRC}:${MEGATRON_LM_SRC}:${PYTHONPATH:-}"

export RAY_DEDUP_LOGS=1
export LISTEN_PORT=6000
export NGINX_PORT=6000
export NEMO_SKILLS_SANDBOX_PORT=6000
export SANDBOX_COMMAND="/start-with-nginx.sh"
export SANDBOX_ENV_VARS="NEMO_SKILLS_SANDBOX_PORT=${NEMO_SKILLS_SANDBOX_PORT}"

export COMMAND="export HF_MODULES_CACHE=${HF_MODULES_CACHE_DIR} ; \
    export PYTHONPATH=${HF_MODULES_CACHE_DIR}:${SNAPSHOT_DIR}:${MEGATRON_BRIDGE_SRC}:${MEGATRON_LM_SRC}:\${PYTHONPATH:-} ; \
    python -c \"from transformers import AutoConfig, AutoProcessor, AutoTokenizer; p='${MODEL_PATH}'; AutoConfig.from_pretrained(p, trust_remote_code=True); AutoProcessor.from_pretrained(p, trust_remote_code=True, use_fast=True); AutoTokenizer.from_pretrained(p, trust_remote_code=True, use_fast=True); print('Prewarmed HF dynamic modules cache')\" ; \
    date ; \
    NRL_WG_USE_RAY_REF=1 \
    NRL_MEGATRON_CHECKPOINT_DIR=${NRL_MEGATRON_CHECKPOINT_DIR} \
    MEGATRON_CONFIG_LOCK_DIR=${MEGATRON_CONFIG_LOCK_DIR} \
    HF_MODULES_CACHE=${HF_MODULES_CACHE_DIR} \
    VLLM_CACHE_ROOT=${VLLM_CACHE_DIR} \
    DG_JIT_CACHE_DIR=${VLLM_CACHE_DIR}/deep_gemm \
    VLLM_DEEP_GEMM_WARMUP=skip \
    FLASHINFER_CUBIN_DIR=${FLASHINFER_CUBIN_CACHE} \
    FLASHINFER_WORKSPACE_BASE=${FLASHINFER_WS_BASE} \
    NEMO_GYM_VENV_DIR=${GYM_VENV_DIR} \
    NRL_VLLM_USE_V1=1 \
    WANDB_MODE=${WANDB_MODE} \
    VLLM_ATTENTION_BACKEND=FLASH_ATTN \
    NRL_FORCE_REBUILD_VENVS=${NRL_FORCE_REBUILD_VENVS} \
    RAY_ENABLE_UV_RUN_RUNTIME_ENV=0 \
    PYTHONPATH=${SNAPSHOT_DIR}:\${PYTHONPATH:-} \
    python ./${ENTRYPOINT} \
    --config ${CONFIG_PATH} \
    ++env.nemo_gym.uv_venv_dir=${GYM_VENV_DIR} \
    env.nemo_gym.skip_venv_if_present=${GYM_SKIP_VENV_IF_PRESENT} \
    policy.model_name=${MODEL_PATH} \
    policy.tokenizer.chat_template=${CHAT_TEMPLATE} \
    policy.generation.vllm_cfg.http_server_serving_chat_kwargs.chat_template=${CHAT_TEMPLATE} \
    checkpointing.checkpoint_dir=${CHECKPOINT_DIR} \
    logger.log_dir=${LOG_DIR} \
    logger.wandb_enabled=True \
    logger.wandb.name=${EXP_NAME} \
    logger.wandb.project=${WANDB_PROJ} \
    data.train.data_path=${TRAIN_PATH} \
    data.validation.data_path=${VAL_PATH} \
    ${EXTRA_HYDRA_ARGS}"

export CONTAINER
export SANDBOX_CONTAINER
BASE_MOUNTS="${SNAPSHOT_DIR}:${SNAPSHOT_DIR}"
BASE_MOUNTS+=",${SNAPSHOT_DIR}/3rdparty/Gym-workspace/Gym:/opt/nemo-rl/3rdparty/Gym-workspace/Gym"
export MOUNTS="${EXTRA_MOUNTS:+${EXTRA_MOUNTS},}${BASE_MOUNTS}"

echo "========================================"
echo " Experiment : ${EXP_NAME}"
echo " Config     : ${CONFIG_PATH}"
echo " Entrypoint : ${ENTRYPOINT}"
echo " Nodes      : ${SBATCH_NUM_NODES}"
echo " Model      : ${MODEL_PATH}"
echo " Container  : ${CONTAINER}"
echo "========================================"

SBATCH_ARGS=(
    sbatch
    --nodes="${SBATCH_NUM_NODES}"
    --account="${SLURM_ACCOUNT}"
    --job-name="${EXP_NAME}"
    --partition="${SLURM_PARTITION}"
    --time="${SLURM_TIME_LIMIT}"
    --gres=gpu:8
    --exclusive
    --dependency=singleton
    ray.sub
)

if [[ "${DRY_RUN}" == true ]]; then
    echo "[dry-run] COMMAND:"
    echo "${COMMAND}"
    echo "[dry-run] sbatch invocation:"
    echo "${SBATCH_ARGS[@]}"
else
    existing_job="$(squeue -h -u "${USER}" -n "${EXP_NAME}" -o '%i' 2>/dev/null | tr -d ' ' || true)"
    if [[ -n "${existing_job}" ]]; then
        echo "Job ${existing_job} already exists for experiment ${EXP_NAME}; not submitting a duplicate."
        exit 0
    fi
    echo "Submitting job: ${EXP_NAME}"
    "${SBATCH_ARGS[@]}"
fi
