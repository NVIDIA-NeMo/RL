#!/bin/bash

set -euo pipefail

model=${1:-}
dispatcher=${2:-}
mode=${3:-submit}
case "${mode}" in
  submit) submit_mode=(--parsable) ;;
  test-only) submit_mode=(--test-only) ;;
  *) printf 'Usage: %s MODEL {baseline|hybridep} [submit|test-only]\n' "$0" >&2; exit 2 ;;
esac

script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source "${script_dir}/performance_case.sh"

work_root=/lustre/fsw/portfolios/coreai/projects/coreai_chef_posttrain/users/sna
repo=${VALIDATION_REPO_OVERRIDE:-${work_root}/experiments/pr2964-20step-20260807/RL}
experiment_root=${work_root}/experiments/pr2964-20step-20260807
run_name=${RUN_NAME_OVERRIDE:-${model}-sync-${dispatcher}-pr2964-dmabuf-cudava-20step}
run_root=${experiment_root}/runs/${run_name}
container=${CONTAINER_OVERRIDE:-${work_root}/containers/nemo-rl-nightly-cw-fallback-20260808/nemo_rl_nightly_20260805_15171871.sqsh}
hf_home=${work_root}/.cache/huggingface
wheel=${HYBRID_EP_WHEEL_OVERRIDE:-${experiment_root}/deepep-wheels/17cfb817bccec3a9c247013360cc550c2bac441e-dmabuf-506072/deep_ep-1.2.1+17cfb81-cp313-cp313-linux_x86_64.whl}
wheel_sha256=${HYBRID_EP_WHEEL_SHA256_OVERRIDE:-f181085dcbfdcb88bc2a33f9df52d4acfd99d1f5e3a73a03ce3dfa38947f559d}
overlay=/tmp/nemo-rl-hybridep-17cf
job_dependency=${JOB_DEPENDENCY:-}
slurm_exclude=${SLURM_EXCLUDE:-}
validation_head=${VALIDATION_HEAD_OVERRIDE:-a028b33bcde0ef8aeb9fcc626a2e0c57fb568d2f}
mcore_source=${MCORE_SOURCE_OVERRIDE:-${repo}/3rdparty/Megatron-Bridge-workspace/Megatron-Bridge/3rdparty/Megatron-LM}
mcore_commit=${MCORE_EXPECTED_COMMIT_OVERRIDE:-$(git -C "${mcore_source}" rev-parse HEAD)}
job_reaper_comment='{"OccupiedIdleGPUsJobReaper":{"exemptIdleTimeMins":"90","reason":"other","description":"NeMo-RL performance recipe model initialization and colocated vLLM startup"}}'

render_case "${model}" "${dispatcher}" "${run_root}"

case "${model}" in
  qwen3-30ba3b) time_limit=03:00:00 ;;
  qwen3-235b) time_limit=06:00:00 ;;
  nemotron3-super) time_limit=08:00:00 ;;
esac

test "$(git -C "${repo}" rev-parse HEAD)" = "${validation_head}"
test "$(git -C "${mcore_source}" rev-parse HEAD)" = "${mcore_commit}"
git -C "${repo}" merge-base --is-ancestor 60a10b4f54c2754d44150771a06260fe9e8b186f HEAD
git -C "${repo}" merge-base --is-ancestor a9aaa395c37963a9fd8a7320d61a516c7b714e57 HEAD
test -z "$(git -C "${repo}" status --porcelain --untracked-files=no --ignore-submodules=untracked)"
if git -C "${repo}" submodule status --recursive | grep -qE '^[+-U]'; then
  printf 'Submodules do not match the pinned gitlinks.\n' >&2
  exit 2
fi
test -r "${container}"
printf '%s  %s\n' "${wheel_sha256}" "${wheel}" | sha256sum --check --status
mkdir -p "${run_root}/ray" "${run_root}/training"

printf -v COMMAND '%q ' "${driver_args[@]}"
read -r -d '' SETUP_COMMAND <<EOF || true
set -euo pipefail
overlay=${overlay}
wheel=${wheel}
expected_sha256=${wheel_sha256}
[[ "\${overlay}" == /tmp/nemo-rl-hybridep-* ]]
test "\$(sha256sum "\${wheel}" | cut -d' ' -f1)" = "\${expected_sha256}"
rm -rf -- "\${overlay}"
mkdir -p "\${overlay}"
unset UV_CONFIG_FILE
UV_NO_CONFIG=1 uv pip install --python /opt/nemo_rl_venv/bin/python --target "\${overlay}" --reinstall --no-deps --no-index "\${wheel}"
PYTHONPATH="\${overlay}" /opt/nemo_rl_venv/bin/python -c 'import deep_ep, deep_ep_cpp, hybrid_ep_cpp; print(deep_ep.__file__); print(deep_ep_cpp.__file__); print(hybrid_ep_cpp.__file__)'
PYTHONPATH=${mcore_source} /opt/nemo_rl_venv/bin/python -c 'import megatron; print(megatron.__file__)'
EOF

export COMMAND SETUP_COMMAND
export CONTAINER="${container}"
export MOUNTS=/lustre:/lustre
export BASE_LOG_DIR="${run_root}/ray"
export GPUS_PER_NODE=8
export HF_HOME="${hf_home}"
export HF_DATASETS_CACHE="${hf_home}/datasets"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export WANDB_MODE=offline
export NCCL_NVLS_ENABLE=0
export NEMO_RL_PY_EXECUTABLES_SYSTEM=0
export NEMO_RL_VENV_DIR=/opt/ray_venvs
export NRL_FORCE_REBUILD_VENVS=false
export NRL_IGNORE_VERSION_MISMATCH=1
export PYTHONPATH="${overlay}:${repo}:${repo}/3rdparty/Megatron-Bridge-workspace/Megatron-Bridge/src:${mcore_source}"
if [[ "${dispatcher}" == "hybridep" ]]; then
  export NEMO_RL_HYBRIDEP_LOG_PACKING=1
  export NEMO_RL_HYBRIDEP_LOG_PACKING_MAX_CALLS=32
  export NEMO_RL_HYBRIDEP_LOG_PACKING_RANKS=0
  export NEMO_RL_HYBRIDEP_LOG_PACKING_REDUCE=1
fi

slurm_args=(
  "${submit_mode[@]}"
  --export=ALL
  "--nodes=${num_nodes}"
  --gpus-per-node=8
  --exclusive
  --account=coreai_chef_posttrain
  --partition=batch_long
  "--time=${time_limit}"
  "--job-name=coreai_chef_posttrain.${run_name}"
  "--output=${run_root}/slurm-%j.out"
  "--comment=${job_reaper_comment}"
)
if [[ -n "${segment_size}" ]]; then
  slurm_args+=("--segment=${segment_size}")
fi
if [[ -n "${slurm_exclude}" ]]; then
  slurm_args+=("--exclude=${slurm_exclude}")
fi
if [[ -n "${job_dependency}" ]]; then
  slurm_args+=("--dependency=${job_dependency}")
fi

cd "${repo}"
job_output=$(sbatch "${slurm_args[@]}" ray.sub)
printf '%s\n' "${job_output}"

if [[ "${mode}" == submit ]]; then
  printf -v rendered_command '%q ' "${driver_args[@]}"
  printf 'job_id=%s\nrun_name=%s\nmodel=%s\ndispatcher=%s\nrecipe=%s\nnum_nodes=%s\ngpus_per_node=8\nvalidation_head=%s\npr2964_head=%s\npr3436_head=%s\nmcore_source=%s\nmcore_commit=%s\ndeepep_commit=%s\ndeepep_wheel=%s\ndeepep_wheel_sha256=%s\ncontainer=%s\njob_dependency=%s\nslurm_exclude=%s\njob_reaper_comment=%s\ncommand=%s\n' \
    "${job_output}" "${run_name}" "${model}" "${dispatcher}" "${driver_args[3]}" \
    "${num_nodes}" "$(git rev-parse HEAD)" \
    60a10b4f54c2754d44150771a06260fe9e8b186f \
    a9aaa395c37963a9fd8a7320d61a516c7b714e57 \
    "${mcore_source}" "${mcore_commit}" \
    17cfb817bccec3a9c247013360cc550c2bac441e "${wheel}" "${wheel_sha256}" \
    "${container}" "${job_dependency}" "${slurm_exclude}" "${job_reaper_comment}" "${rendered_command}" \
    > "${run_root}/submission.env"
fi
