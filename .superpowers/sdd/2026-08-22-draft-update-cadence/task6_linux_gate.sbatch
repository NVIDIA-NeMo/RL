#!/usr/bin/env bash
#SBATCH --job-name=cadence-task6-split-gate
#SBATCH --partition=batch
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --exclusive
#SBATCH --gres=gpu:4
#SBATCH --time=01:30:00
#SBATCH --output=/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/sna/cadence-task6-bbea05d3e/slurm-%j.out
#SBATCH --error=/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/sna/cadence-task6-bbea05d3e/slurm-%j.err

set -euo pipefail

expected_head=${EXPECTED_HEAD:?EXPECTED_HEAD must name the pushed harness commit}
expected_mcore=14346b65a2d0790e451919858f7771078105c5f0
source_dir=/home/sna/nemo-rl-task6-bbea05d3
result_root=/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/sna/cadence-task6-bbea05d3e
container_path=/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/sna/containers/nemo_rl_nightly_20260818_20260818_6296116.sqsh
container_sha256=6940409542de6669f77e91c7ce7aac0ef7e91bd56839772e1ae7efc371718d44
scratch_root=/raid/scratch/sna/cadence-task6-bbea05d3e-${SLURM_JOB_ID}
mcore_venv=${scratch_root}/mcore-venv
vllm_venv=${scratch_root}/vllm-venv
result_file=${result_root}/result-${SLURM_JOB_ID}.txt
live_log=${result_root}/live-${SLURM_JOB_ID}.log
mcore_phase_result=${result_root}/phase-mcore-${SLURM_JOB_ID}.txt
vllm_phase_result=${result_root}/phase-vllm-${SLURM_JOB_ID}.txt

mkdir -p "${result_root}" "${scratch_root}"
exec > >(tee "${live_log}") 2>&1

cleanup() {
  rc=$?
  printf 'job_id=%s\nexit_code=%s\nsource_sha=%s\nmcore_sha=%s\ncontainer=%s\nscratch=%s\n' \
    "${SLURM_JOB_ID}" \
    "${rc}" \
    "$(git -C "${source_dir}" rev-parse HEAD 2>/dev/null || true)" \
    "$(git -C "${source_dir}/3rdparty/Megatron-Bridge-workspace/Megatron-Bridge/3rdparty/Megatron-LM" rev-parse HEAD 2>/dev/null || true)" \
    "${container_path}" \
    "${scratch_root}" \
    > "${result_file}"
  exit "${rc}"
}
trap cleanup EXIT

test "$(git -C "${source_dir}" rev-parse HEAD)" = "${expected_head}"
test -z "$(git -C "${source_dir}" status --porcelain --untracked-files=all --ignore-submodules=none)"
submodule_status="$(git -C "${source_dir}" submodule status --recursive)"
if printf '%s\n' "${submodule_status}" | grep -Eq '^[-+U]'; then
  printf '%s\n' "${submodule_status}" >&2
  exit 11
fi
test "$(git -C "${source_dir}/3rdparty/Megatron-Bridge-workspace/Megatron-Bridge/3rdparty/Megatron-LM" rev-parse HEAD)" = "${expected_mcore}"
test "$(sha256sum "${container_path}" | awk '{print $1}')" = "${container_sha256}"

printf 'source_sha=%s\n' "${expected_head}"
printf 'mcore_sha=%s\n' "${expected_mcore}"
printf 'container=%s\n' "${container_path}"
printf 'container_sha256=%s\n' "${container_sha256}"
printf '%s\n' "${submodule_status}"

run_mcore_phase() {
  srun \
    --nodes=1 \
    --ntasks=1 \
    --container-image="${container_path}" \
    --container-mounts="${source_dir}:${source_dir},${scratch_root}:${scratch_root},${result_root}:${result_root}" \
    --container-remap-root \
    --mpi=pmix \
    bash -s -- "${source_dir}" "${scratch_root}" "${mcore_venv}" "${expected_head}" "${expected_mcore}" <<'MCORE_INNER'
set -euo pipefail

source_dir=$1
scratch_root=$2
venv_dir=$3
expected_head=$4
expected_mcore=$5

cd "${source_dir}"
export UV_PROJECT_ENVIRONMENT="${venv_dir}"
export UV_CACHE_DIR="${scratch_root}/uv-cache"
export XDG_CACHE_HOME="${scratch_root}/mcore-xdg-cache"
export TRITON_CACHE_DIR="${scratch_root}/mcore-triton-cache"
export TORCH_EXTENSIONS_DIR="${scratch_root}/mcore-torch-extensions"
export TMPDIR="${scratch_root}/mcore-tmp"
export RAY_TMPDIR="/tmp/nr${SLURM_JOB_ID}-m"
unset RAY_ADDRESS
mkdir -p "${UV_CACHE_DIR}" "${XDG_CACHE_HOME}" "${TRITON_CACHE_DIR}" "${TORCH_EXTENSIONS_DIR}" "${TMPDIR}" "${RAY_TMPDIR}"
test ! -e "${UV_PROJECT_ENVIRONMENT}"
uv sync --locked --extra mcore --group test --group dev

export PATH="${UV_PROJECT_ENVIRONMENT}/bin:${PATH}"
export PYTHONPATH="${source_dir}:${source_dir}/3rdparty/Megatron-Bridge-workspace/Megatron-Bridge/src:${source_dir}/3rdparty/Megatron-Bridge-workspace/Megatron-Bridge/3rdparty/Megatron-LM:${PYTHONPATH:-}"
cudnn_lib="${UV_PROJECT_ENVIRONMENT}/lib/python3.13/site-packages/nvidia/cudnn/lib"
test -e "${cudnn_lib}/libcudnn.so.9"
export CUDNN_HOME="${cudnn_lib}"
export CUDNN_PATH="${cudnn_lib}"
export LD_LIBRARY_PATH="${cudnn_lib}:${LD_LIBRARY_PATH:-}"

test "$(git rev-parse HEAD)" = "${expected_head}"
test "$(git -C 3rdparty/Megatron-Bridge-workspace/Megatron-Bridge/3rdparty/Megatron-LM rev-parse HEAD)" = "${expected_mcore}"
python -c 'import ray; import torch; import transformer_engine; import megatron.core; print(f"ray={ray.__version__}"); print(f"torch={torch.__version__}"); print(f"mcore={megatron.core.__file__}")'

python -m pytest -q -rA \
  --confcutdir=tests/unit/models/policy \
  tests/unit/models/policy/test_lm_policy_collective.py

python -m pytest -q -rA \
  --confcutdir=tests/unit/algorithms \
  tests/unit/algorithms/test_grpo.py::test_setup_rejects_colocated_checkpoint_engine_before_logger_setup

python -m pytest -q -rA \
  --confcutdir=tests/unit/single_controller \
  tests/unit/single_controller/test_single_controller_setup.py \
  -k 'fixed_cadence_rejects'

mapfile -t python_files < <(git diff --name-only 5495e172058d08ecf1cf027e4c356d1d44610471 "${expected_head}" -- '*.py')
test "${#python_files[@]}" -gt 0
printf 'changed_python_files=%s\n' "${#python_files[@]}"
ruff check "${python_files[@]}"
ruff format --check "${python_files[@]}"
python -m py_compile "${python_files[@]}"
git diff --check 5495e172058d08ecf1cf027e4c356d1d44610471 "${expected_head}"
printf '%s\n' TASK6_MCORE_PHASE_PASS
MCORE_INNER
}

set +e
run_mcore_phase
mcore_rc=$?
set -e
printf 'phase=mcore\nphase_exit_code=%s\nsource_sha=%s\nvenv=%s\n' \
  "${mcore_rc}" "${expected_head}" "${mcore_venv}" > "${mcore_phase_result}"
if ((mcore_rc != 0)); then
  exit "${mcore_rc}"
fi

run_vllm_phase() {
  srun \
    --nodes=1 \
    --ntasks=1 \
    --container-image="${container_path}" \
    --container-mounts="${source_dir}:${source_dir},${scratch_root}:${scratch_root},${result_root}:${result_root}" \
    --container-remap-root \
    --mpi=pmix \
    bash -s -- "${source_dir}" "${scratch_root}" "${vllm_venv}" "${expected_head}" <<'VLLM_INNER'
set -euo pipefail

source_dir=$1
scratch_root=$2
venv_dir=$3
expected_head=$4

cd "${source_dir}"
export UV_PROJECT_ENVIRONMENT="${venv_dir}"
export UV_CACHE_DIR="${scratch_root}/uv-cache"
export XDG_CACHE_HOME="${scratch_root}/vllm-xdg-cache"
export TRITON_CACHE_DIR="${scratch_root}/vllm-triton-cache"
export TORCH_EXTENSIONS_DIR="${scratch_root}/vllm-torch-extensions"
export TMPDIR="${scratch_root}/vllm-tmp"
export RAY_TMPDIR="/tmp/nr${SLURM_JOB_ID}-v"
unset RAY_ADDRESS
mkdir -p "${UV_CACHE_DIR}" "${XDG_CACHE_HOME}" "${TRITON_CACHE_DIR}" "${TORCH_EXTENSIONS_DIR}" "${TMPDIR}" "${RAY_TMPDIR}"
test ! -e "${UV_PROJECT_ENVIRONMENT}"
uv sync --locked --extra vllm --group test --group dev

export PATH="${UV_PROJECT_ENVIRONMENT}/bin:${PATH}"
export PYTHONPATH="${source_dir}:${PYTHONPATH:-}"

test "$(git rev-parse HEAD)" = "${expected_head}"
python -c 'import ray; import torch; import vllm; print(f"ray={ray.__version__}"); print(f"torch={torch.__version__}"); print(f"vllm={vllm.__version__}")'

python -m pytest -q -rA \
  --confcutdir=tests/unit/weight_sync \
  tests/unit/weight_sync/test_weight_synchronizer.py \
  -k 'selection or component_selection or target_only or default_sync_preserves_legacy_endpoint_call_shape or cadence_fails_before_transport_construction'

python -m pytest -q -rA \
  --confcutdir=tests/unit/models/generation \
  tests/unit/models/generation/test_generation_interfaces_import.py \
  tests/unit/models/generation/test_vllm_speculator_runtime.py \
  tests/unit/models/generation/test_vllm_backend.py \
  -k 'import_in_fresh_process or target_only or mtp_selection or nccl_reshard_full_sync or selection_is_reusable'

python -m pytest -q -rA \
  --confcutdir=tests/unit/weight_sync \
  tests/unit/weight_sync/test_vllm_remote_sparse_weight_synchronizer.py \
  -k 'target_only or cadence'

printf '%s\n' TASK6_VLLM_PHASE_PASS
VLLM_INNER
}

set +e
run_vllm_phase
vllm_rc=$?
set -e
printf 'phase=vllm\nphase_exit_code=%s\nsource_sha=%s\nvenv=%s\n' \
  "${vllm_rc}" "${expected_head}" "${vllm_venv}" > "${vllm_phase_result}"
if ((vllm_rc != 0)); then
  exit "${vllm_rc}"
fi

printf '%s\n' TASK6_LINUX_GATE_PASS
