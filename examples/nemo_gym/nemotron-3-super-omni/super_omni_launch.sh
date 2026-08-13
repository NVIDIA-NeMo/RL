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
export SANDBOX_CONTAINER="${SANDBOX_CONTAINER:-/path/to/nemo-skills-sandbox.sqsh}"
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
# Read the cluster block from the *resolved* recipe. Scraping the raw file
# with awk only sees keys written literally in it, which forced every recipe
# to repeat cluster: even when it inherits one -- and silently produced an
# empty value for a recipe that does not. tools/config_cli.py resolves
# `defaults:` the same way the driver does. Fall back to the raw scrape if the
# resolver is unavailable (no venv on the submit host), so submission keeps
# working in that case for recipes that do carry a literal cluster block.
_expand_config() {
    # Prefer the project venv: it already has omegaconf and needs no resolution.
    # `uv run` is the documented entry point but fails when the environment is
    # not materialized on this host, and the raw scrape below cannot resolve
    # `defaults:`, so try both before giving up.
    if [[ -x .venv/bin/python ]]; then
        .venv/bin/python tools/config_cli.py expand "${CONFIG_PATH}" 2>/dev/null && return 0
    fi
    uv run --no-sync python tools/config_cli.py expand "${CONFIG_PATH}" 2>/dev/null && return 0
    return 1
}
_expanded_cfg="$(_expand_config || true)"
_read_cluster_key() {
    if [[ -n "${_expanded_cfg}" ]]; then
        awk -v k="$1" '/^cluster:/{f=1} f && $1==k":"{print $2; exit}' <<<"${_expanded_cfg}"
    else
        awk -v k="$1" '/^cluster:/{f=1} f && $1==k":"{print $2; exit}' "${CONFIG_PATH}"
    fi
}
SBATCH_NUM_NODES="${SBATCH_NUM_NODES:-$(_read_cluster_key num_nodes)}"
SBATCH_GPUS_PER_NODE="${SBATCH_GPUS_PER_NODE:-$(_read_cluster_key gpus_per_node)}"
SBATCH_GPUS_PER_NODE="${SBATCH_GPUS_PER_NODE:-8}"
EXTRA_MOUNTS="${EXTRA_MOUNTS:-}"
EXTRA_HYDRA_ARGS="${EXTRA_HYDRA_ARGS:-}"
NRL_FORCE_REBUILD_VENVS="${NRL_FORCE_REBUILD_VENVS:-false}"
GYM_VENV_DIR="${GYM_VENV_DIR:-/opt/gym_venvs}"
WANDB_PROJ="${WANDB_PROJ:-grpo-super-omni}"
DRY_RUN="${DRY_RUN:-false}"

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
for var in MODEL_PATH TRAIN_PATH VAL_PATH CONTAINER SANDBOX_CONTAINER \
           PERSISTENT_CACHE SLURM_ACCOUNT; do
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

mkdir -p "${VLLM_CACHE_DIR}" "${FLASHINFER_CUBIN_CACHE}" "${FLASHINFER_WS_BASE}" \
         "${MEGATRON_CONFIG_LOCK_DIR}" "${HF_MODULES_CACHE_DIR}"
export OMP_NUM_THREADS=16

SNAPSHOT_DIR=$(realpath "$(bash "${CODE_DIR}/tools/code_snapshot.sh" "${EXP_NAME}")")
echo "Refreshing tracked files in code snapshot: ${SNAPSHOT_DIR}"
(
    cd "${CODE_DIR}"
    rsync -a --files-from=<(git ls-files --recurse-submodules --cached --full-name) ./ "${SNAPSHOT_DIR}/"
)
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
    NRL_IGNORE_VERSION_MISMATCH=1 \
    WANDB_MODE=${WANDB_MODE} \
    VLLM_ATTENTION_BACKEND=FLASH_ATTN \
    NRL_FORCE_REBUILD_VENVS=${NRL_FORCE_REBUILD_VENVS} \
    RAY_ENABLE_UV_RUN_RUNTIME_ENV=0 \
    PYTHONPATH=${SNAPSHOT_DIR}:\${PYTHONPATH:-} \
    uv run --no-sync ./${ENTRYPOINT} \
    --config ${CONFIG_PATH} \
    ++env.nemo_gym.uv_venv_dir=${GYM_VENV_DIR} \
    env.nemo_gym.skip_venv_if_present=true \
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
BASE_MOUNTS+=",${CODE_DIR}/3rdparty/Megatron-Bridge-workspace/Megatron-Bridge/3rdparty/Megatron-LM:${SNAPSHOT_DIR}/3rdparty/Megatron-Bridge-workspace/Megatron-Bridge/3rdparty/Megatron-LM"
BASE_MOUNTS+=",${CODE_DIR}/3rdparty/Gym-workspace/Gym:/opt/nemo-rl/3rdparty/Gym-workspace/Gym"
export MOUNTS="${EXTRA_MOUNTS:+${EXTRA_MOUNTS},}${BASE_MOUNTS}"

# The checkpoint, dataset and caches live outside the code snapshot, so their
# filesystems have to be bind-mounted or they simply do not exist inside the
# container. Transformers then treats an absent local path as a Hub repo id and
# fails with "Repo id must be in the form 'repo_name' or 'namespace/repo_name'",
# which says nothing about mounts. Check here instead.
mount_checked_vars=(MODEL_PATH TRAIN_PATH VAL_PATH PERSISTENT_CACHE)
# GYM_VENV_DIR defaults to a container-internal path, which needs no mount. It
# only has to be checked when pointed at host storage -- worth doing, because
# putting it on a shared filesystem is how Gym's skip_venv_if_present actually
# takes effect: the container-local default is discarded with the container, so
# every run rebuilds ~175 packages per resource server.
[[ "${GYM_VENV_DIR}" != /opt/* ]] && mount_checked_vars+=(GYM_VENV_DIR)
for path_var in "${mount_checked_vars[@]}"; do
    path_value="${!path_var}"
    mounted=false
    IFS=',' read -ra mount_specs <<< "${MOUNTS}"
    for spec in "${mount_specs[@]}"; do
        host_path="${spec%%:*}"
        [[ -n "${host_path}" && "${path_value}" == "${host_path}"* ]] && mounted=true && break
    done
    if [[ "${mounted}" != true ]]; then
        echo "Error: ${path_var}=${path_value}" >&2
        echo "  is not under any path in MOUNTS, so it will not exist in the container." >&2
        echo "  Add its filesystem to EXTRA_MOUNTS, e.g." >&2
        echo "    EXTRA_MOUNTS=/host/path:/host/path,/other/path:/other/path" >&2
        exit 1
    fi
done

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
    --gres=gpu:"${SBATCH_GPUS_PER_NODE}"
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
    echo "Submitting job: ${EXP_NAME}"
    "${SBATCH_ARGS[@]}"
fi
