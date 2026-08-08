#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --account=coreai_chef_posttrain
#SBATCH --partition=batch
#SBATCH --time=01:30:00
#SBATCH --job-name=coreai_chef_posttrain.pr2964-build-hybridep17cf-dmabuf
#SBATCH --output=/lustre/fsw/portfolios/coreai/projects/coreai_chef_posttrain/users/sna/experiments/pr2964-20step-20260807/logs/build-hybridep-17cf-dmabuf-%j.out

set -euo pipefail

container=/lustre/fsw/portfolios/coreai/projects/coreai_chef_posttrain/users/sna/containers/nemo-rl-nightly-cw-fallback-20260808/nemo_rl_nightly_20260805_15171871.sqsh
source_root=${HYBRID_EP_SOURCE_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}
experiment_root=${HYBRID_EP_EXPERIMENT_ROOT:-/lustre/fsw/portfolios/coreai/projects/coreai_chef_posttrain/users/sna/experiments/pr2964-20step-20260807}
script_path=${source_root}/build_hybridep_17cf_dmabuf_b200.sbatch
patch_path=${source_root}/patches/deepep_17cf_dmabuf_fallback.patch
output_root=${experiment_root}/deepep-wheels
deepep_base_commit=17cfb817bccec3a9c247013360cc550c2bac441e
deepep_fix_commit=6d11d654e965b1c967f72adfcec95dbc82043643

if [[ "${HYBRID_EP_BUILD_STAGE:-host}" != container ]]; then
  mkdir -p "${output_root}"
  export HYBRID_EP_BUILD_STAGE=container
  export HYBRID_EP_OUTPUT_ROOT="${output_root}"
  export HYBRID_EP_BASE_COMMIT="${deepep_base_commit}"
  export HYBRID_EP_FIX_COMMIT="${deepep_fix_commit}"
  exec srun \
    --nodes=1 \
    --ntasks-per-node=1 \
    --gpus-per-node=1 \
    --container-image="${container}" \
    --container-mounts=/lustre:/lustre \
    --no-container-mount-home \
    bash "${script_path}"
fi

python_bin=/opt/nemo_rl_venv/bin/python
build_root=$(mktemp -d "/tmp/pr2964-deepep-dmabuf-${SLURM_JOB_ID}.XXXXXX")
cleanup() {
  case "${build_root}" in
    /tmp/pr2964-deepep-dmabuf-"${SLURM_JOB_ID}".*) rm -rf -- "${build_root}" ;;
    *) printf 'Refusing to clean unexpected path: %s\n' "${build_root}" >&2 ;;
  esac
}
trap cleanup EXIT

source_dir=${build_root}/DeepEP
wheel_stage=${build_root}/wheel
import_overlay=${build_root}/import-overlay
artifact_dir=${HYBRID_EP_OUTPUT_ROOT}/${HYBRID_EP_BASE_COMMIT}-dmabuf-${SLURM_JOB_ID}
patch_sha256=$(sha256sum "${patch_path}" | cut -d' ' -f1)

git clone --filter=blob:none --recurse-submodules --branch hybrid-ep \
  https://github.com/deepseek-ai/DeepEP.git "${source_dir}"
git -C "${source_dir}" checkout --detach "${HYBRID_EP_BASE_COMMIT}"
git -C "${source_dir}" submodule update --init --recursive
test "$(git -C "${source_dir}" rev-parse HEAD)" = "${HYBRID_EP_BASE_COMMIT}"
git -C "${source_dir}" apply --check "${patch_path}"
git -C "${source_dir}" apply "${patch_path}"
git -C "${source_dir}" diff --check

mkdir -p "${wheel_stage}" "${import_overlay}"
export HYBRID_EP_MULTINODE=1
export TORCH_CUDA_ARCH_LIST=10.0
export RDMA_CORE_HOME=/usr
unset UV_CONFIG_FILE
UV_NO_CONFIG=1 uv build --wheel --no-build-isolation \
  --python "${python_bin}" \
  --out-dir "${wheel_stage}" \
  "${source_dir}"

mapfile -t wheels < <(find "${wheel_stage}" -maxdepth 1 -type f -name '*.whl')
test "${#wheels[@]}" -eq 1
UV_NO_CONFIG=1 uv pip install \
  --python "${python_bin}" \
  --target "${import_overlay}" \
  --reinstall --no-deps --no-index "${wheels[0]}"
PYTHONPATH="${import_overlay}" "${python_bin}" -c \
  'import deep_ep, deep_ep_cpp, hybrid_ep_cpp; print(deep_ep.__file__); print(deep_ep_cpp.__file__); print(hybrid_ep_cpp.__file__)'

test ! -e "${artifact_dir}"
mkdir "${artifact_dir}"
mv "${wheels[0]}" "${artifact_dir}/"
wheel_path=${artifact_dir}/$(basename "${wheels[0]}")
sha256sum "${wheel_path}" > "${wheel_path}.sha256"
printf 'deepep_base_commit=%s\ndeepep_fix_commit=%s\npatch_sha256=%s\ngpu_arch=10.0\nslurm_job_id=%s\nwheel=%s\n' \
  "${HYBRID_EP_BASE_COMMIT}" \
  "${HYBRID_EP_FIX_COMMIT}" \
  "${patch_sha256}" \
  "${SLURM_JOB_ID}" \
  "${wheel_path}" \
  > "${artifact_dir}/metadata.env"
printf 'HYBRID_EP_WHEEL=%s\n' "${wheel_path}"
