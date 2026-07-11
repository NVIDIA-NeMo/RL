#!/bin/bash
# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

#SBATCH --job-name=cutedsl-qwen3-30ba3b-1n4g
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=72
#SBATCH --mem=0
#SBATCH --exclusive
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null

set -euo pipefail

readonly RECIPE="examples/configs/recipes/llm/performance/grpo-qwen3-30ba3b-1n4g-megatron-mxfp8-cutedsl.yaml"
readonly CONTAINER_REPO_ROOT="/workspace/nemo-rl"
readonly CONTAINER_RESULT_DIR="/results"
readonly CONTAINER_RUNTIME_DIR="/runtime"
readonly PARENT_BASE_SHA="c1868b139819e6364d65f862a843eb39489072a1"
export CONTAINER_REPO_ROOT CONTAINER_RESULT_DIR CONTAINER_RUNTIME_DIR

readonly SLURM_SUBMIT_DIR="${SLURM_SUBMIT_DIR:?submit this wrapper with sbatch from the repository checkout}"
REPO_ROOT=$(git -C "${SLURM_SUBMIT_DIR}" rev-parse --show-toplevel)
readonly REPO_ROOT
readonly EXPERIMENT_DIR="${REPO_ROOT}/experiments/cutedsl_qwen3_30ba3b_oci_1n4g"
source "${EXPERIMENT_DIR}/lib/cluster_profile.sh"
source "${EXPERIMENT_DIR}/lib/events.sh"
load_cutedsl_cluster_profile
readonly IMAGE="${CUTEDSL_IMAGE}"
readonly RUN_ID="${SLURM_JOB_ID:?submit this wrapper with sbatch}${SLURM_RESTART_COUNT:+-r${SLURM_RESTART_COUNT}}"
readonly RESULT_DIR="${EXPERIMENT_DIR}/results/${RUN_ID}"
readonly HOST_RUNTIME_DIR="/tmp/${USER}/nemo-2606-cutedsl/${RUN_ID}"

mkdir -p "${RESULT_DIR}" "${HOST_RUNTIME_DIR}"
export CUTEDSL_EVENT_CLUSTER="${CUTEDSL_PROFILE_NAME}"
export CUTEDSL_EVENT_JOB_ID="${SLURM_JOB_ID}"
cutedsl_events_init "${RESULT_DIR}"
cat > "${HOST_RUNTIME_DIR}/log_runtime_environment.sh" <<'BASH'
printf '[INFO] Runtime environment: VIRTUAL_ENV=%s UV_PROJECT_ENVIRONMENT=%s RUNTIME_PYTHON=%s\n' \
    "${VIRTUAL_ENV-<unset>}" "${UV_PROJECT_ENVIRONMENT-<unset>}" "${RUNTIME_PYTHON-<unset>}"
runtime_uv_version='<unavailable>'
runtime_python_version='<unavailable>'
runtime_python_realpath='<unavailable>'
runtime_python_prefix='<unavailable>'
if [[ -x "${UV_BIN-}" ]]; then
    runtime_uv_version=$("${UV_BIN}" --version 2>&1)
fi
if [[ -x "${RUNTIME_PYTHON-}" ]]; then
    runtime_python_version=$("${RUNTIME_PYTHON}" --version 2>&1)
    runtime_python_realpath=$("${RUNTIME_PYTHON}" -c "import os, sys; print(os.path.realpath(sys.executable))" 2>&1)
    runtime_python_prefix=$("${RUNTIME_PYTHON}" -c "import sys; print(sys.prefix)" 2>&1)
fi
printf '[INFO] Runtime versions: uv=%s python=%s\n' "${runtime_uv_version}" "${runtime_python_version}"
printf '[INFO] Runtime Python: realpath=%s prefix=%s\n' "${runtime_python_realpath}" "${runtime_python_prefix}"
printf '[INFO] Runtime build environment: NVTE_CUDA_ARCHS=%s\n' \
    "${NVTE_CUDA_ARCHS-<unset>}"
unset runtime_uv_version runtime_python_version runtime_python_realpath runtime_python_prefix
BASH
exec > >(tee "${RESULT_DIR}/slurm.out") 2>&1

on_exit() {
    local exit_code=$?
    if ! cutedsl_begin_finalization; then
        return "${exit_code}"
    fi
    set +e
    cutedsl_finalize_run "${exit_code}" "Functional payload finished" \
        "${EXPERIMENT_DIR}/render_cutedsl_report.py"
    local final_exit_code=$?
    rm -rf --one-file-system -- "${HOST_RUNTIME_DIR}"
    exit "${final_exit_code}"
}
trap on_exit EXIT
cutedsl_install_signal_traps

cutedsl_write_event preflight start '' "Validate immutable source and image" slurm.out
cutedsl_write_event image_hash start '' "Verify immutable container SHA256" image.sha256

if [[ ! -r "${IMAGE}" ]]; then
    echo "[ERROR] Required first-choice image is not readable: ${IMAGE}" >&2
    exit 1
fi

SOURCE_STATUS=$(git -C "${REPO_ROOT}" status --porcelain --untracked-files=normal --ignore-submodules=none)
readonly SOURCE_STATUS
if [[ -n "${SOURCE_STATUS}" ]]; then
    echo "[ERROR] Refusing to run with tracked changes or non-ignored untracked files:" >&2
    printf '%s\n' "${SOURCE_STATUS}" >&2
    exit 1
fi
if ! git -C "${REPO_ROOT}" submodule foreach --quiet --recursive \
    'test -z "$(git status --porcelain --untracked-files=normal --ignore-submodules=none)"'; then
    echo "[ERROR] Refusing to run with changes or non-ignored untracked files in a recursive submodule." >&2
    exit 1
fi

SUBMODULE_STATUS=$(git -C "${REPO_ROOT}" submodule status --recursive)
readonly SUBMODULE_STATUS
while IFS= read -r line; do
    if [[ -n "${line}" && "${line:0:1}" != " " ]]; then
        echo "[ERROR] Recursive submodule is not clean and initialized: ${line}" >&2
        exit 1
    fi
done <<< "${SUBMODULE_STATUS}"

IMAGE_SHA256=$(sha256sum "${IMAGE}" | awk '{print $1}')
readonly IMAGE_SHA256
if [[ "${IMAGE_SHA256}" != "${CUTEDSL_IMAGE_SHA256}" ]]; then
    echo "[ERROR] Image SHA256 ${IMAGE_SHA256} does not match profile SHA256 ${CUTEDSL_IMAGE_SHA256}." >&2
    exit 1
fi
cutedsl_write_event image_hash pass 0 "Container SHA256 matches cluster profile" image.sha256
GIT_SHA=$(git -C "${REPO_ROOT}" rev-parse HEAD)
readonly GIT_SHA
GIT_BRANCH=$(git -C "${REPO_ROOT}" branch --show-current)
readonly GIT_BRANCH
validate_cutedsl_runtime_source "${GIT_BRANCH}" "${GIT_SHA}"
GIT_REMOTE=$(git -C "${REPO_ROOT}" remote get-url origin)
readonly GIT_REMOTE
UPSTREAM_REF=$(git -C "${REPO_ROOT}" rev-parse --abbrev-ref --symbolic-full-name '@{upstream}')
readonly UPSTREAM_REF
UPSTREAM_SHA=$(git -C "${REPO_ROOT}" rev-parse '@{upstream}')
readonly UPSTREAM_SHA
if [[ "${GIT_SHA}" != "${UPSTREAM_SHA}" ]]; then
    echo "[ERROR] HEAD ${GIT_SHA} does not match ${UPSTREAM_REF} at ${UPSTREAM_SHA}." >&2
    exit 1
fi

git -C "${REPO_ROOT}" status --porcelain=v1 --branch --untracked-files=normal --ignore-submodules=none \
    > "${RESULT_DIR}/git_status_before.txt"
printf '%s\n' "${SUBMODULE_STATUS}" > "${RESULT_DIR}/submodule_status.txt"
printf '%s  %s\n' "${IMAGE_SHA256}" "${IMAGE}" > "${RESULT_DIR}/image.sha256"
cutedsl_write_event preflight pass 0 \
    "Source, submodules, upstream, and image are immutable" git_status_before.txt

readonly -a GRPO_OVERRIDES=(
    "grpo.max_num_steps=3"
    "cluster.num_nodes=1"
    "cluster.gpus_per_node=4"
    "logger.log_dir=${CONTAINER_RESULT_DIR}/tensorboard"
    "logger.wandb_enabled=false"
    "logger.tensorboard_enabled=true"
    "logger.monitor_gpus=true"
    "checkpointing.enabled=false"
    "checkpointing.checkpoint_dir=${CONTAINER_RESULT_DIR}/checkpoints"
)
printf '%s\n' "${GRPO_OVERRIDES[@]}" > "${RESULT_DIR}/grpo_overrides.txt"

export CUDA_HOME="/usr/local/cuda"
export NVTE_CUDA_ARCHS="100"
export NEMO_RL_VENV_DIR="${CONTAINER_RUNTIME_DIR}/worker_venvs"
export NRL_FORCE_REBUILD_VENVS="true"
export NRL_IGNORE_VERSION_MISMATCH="1"
export PYTHONPATH="${CONTAINER_REPO_ROOT}:${PYTHONPATH:-}"
export TORCH_EXTENSIONS_DIR="${CONTAINER_RUNTIME_DIR}/torch_extensions"
export TRITON_CACHE_DIR="${CONTAINER_RUNTIME_DIR}/triton_cache"
export UV_PROJECT_ENVIRONMENT="${CONTAINER_RUNTIME_DIR}/venv"
export VIRTUAL_ENV="${UV_PROJECT_ENVIRONMENT}"
export RUNTIME_PYTHON="${UV_PROJECT_ENVIRONMENT}/bin/python"
export UV_CACHE_DIR="${CONTAINER_RUNTIME_DIR}/uv-cache"
export UV_VERSION="0.11.6"
export UV_PYTHON_VERSION="3.13.13"
export UV_NO_MODIFY_PATH=1
export UV_INSTALL_DIR="${CONTAINER_RUNTIME_DIR}/uv-bin"
export UV_PYTHON_INSTALL_DIR="${CONTAINER_RUNTIME_DIR}/uv-python"
export UV_BIN="${UV_INSTALL_DIR}/uv"

export CUTEDSL_GATE_GIT_BRANCH="${GIT_BRANCH}"
export CUTEDSL_GATE_GIT_REMOTE="${GIT_REMOTE}"
export CUTEDSL_GATE_GIT_SHA="${GIT_SHA}"
export CUTEDSL_GATE_IMAGE="${IMAGE}"
export CUTEDSL_GATE_IMAGE_SHA256="${IMAGE_SHA256}"
export CUTEDSL_GATE_RECIPE="${RECIPE}"
export CUTEDSL_GATE_RUN_ID="${RUN_ID}"
export CUTEDSL_GATE_UPSTREAM_REF="${UPSTREAM_REF}"
export CUTEDSL_GATE_UPSTREAM_SHA="${UPSTREAM_SHA}"
export PARENT_BASE_SHA

SRUN=(
    srun
    --nodes=1
    --ntasks=1
    --cpus-per-task=72
    --mpi=pmix
    --container-image="${IMAGE}"
    --container-env=VIRTUAL_ENV,UV_PROJECT_ENVIRONMENT,NVTE_CUDA_ARCHS
    --container-mounts="${REPO_ROOT}:${CONTAINER_REPO_ROOT},${RESULT_DIR}:${CONTAINER_RESULT_DIR},${HOST_RUNTIME_DIR}:${CONTAINER_RUNTIME_DIR}"
    --container-workdir="${CONTAINER_REPO_ROOT}"
)
if [[ -n "${CUTEDSL_GRES}" ]]; then
    SRUN+=("--gres=${CUTEDSL_GRES}")
fi
readonly -a SRUN

echo "[INFO] Run ID: ${RUN_ID}"
echo "[INFO] NeMo-RL: ${GIT_SHA} (${GIT_BRANCH})"
echo "[INFO] Image: ${IMAGE}"
echo "[INFO] Image SHA256: ${IMAGE_SHA256}"
echo "[INFO] Result directory: ${RESULT_DIR}"

echo "[INFO] Creating the run-local Linux environment and recording provenance."
cutedsl_write_event runtime_bootstrap start '' \
    "Create locked run-local Python environment" uv_lock_check.log
"${SRUN[@]}" bash -lc '
set -euo pipefail
cd "${CONTAINER_REPO_ROOT}"
[[ "${NVTE_CUDA_ARCHS}" == "100" ]]
curl -LsSf "https://astral.sh/uv/${UV_VERSION}/install.sh" | sh
[[ "$("${UV_BIN}" --version)" == "uv ${UV_VERSION} "* ]]
"${UV_BIN}" python install "${UV_PYTHON_VERSION}"
"${UV_BIN}" python find "${UV_PYTHON_VERSION}"
"${UV_BIN}" sync --locked --extra mcore --group test --group dev --python "${UV_PYTHON_VERSION}"
source "${CONTAINER_RUNTIME_DIR}/log_runtime_environment.sh"
[[ "${VIRTUAL_ENV}" == "${UV_PROJECT_ENVIRONMENT}" ]]
[[ "$("${RUNTIME_PYTHON}" --version)" == "Python ${UV_PYTHON_VERSION}" ]]
expected_runtime_python=$("${RUNTIME_PYTHON}" -c '"'"'import os, sys; print(os.path.realpath(sys.executable))'"'"')
runtime_python=$("${UV_BIN}" run --active --no-sync python -c '"'"'import os, sys; print(os.path.realpath(sys.executable))'"'"')
runtime_prefix=$("${UV_BIN}" run --active --no-sync python -c '"'"'import sys; print(sys.prefix)'"'"')
[[ "${runtime_python}" == "${expected_runtime_python}" ]]
[[ "${runtime_prefix}" == "${VIRTUAL_ENV}" ]]
"${UV_BIN}" lock --check 2>&1 | tee "${CONTAINER_RESULT_DIR}/uv_lock_check.log"
mapfile -t grpo_overrides < "${CONTAINER_RESULT_DIR}/grpo_overrides.txt"
"${UV_BIN}" run --active --no-sync python - "${grpo_overrides[@]}" <<'"'"'PY'"'"'
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
from pathlib import Path

import cutlass
import torch
import transformer_engine
from nemo_rl.algorithms.grpo import MasterConfig
from nemo_rl.utils.config import load_config, parse_hydra_overrides, register_omegaconf_resolvers
from omegaconf import OmegaConf


def command_output(command: list[str]) -> str:
    return subprocess.run(command, check=True, capture_output=True, text=True).stdout.strip()


def package_version(*names: str) -> str | None:
    for name in names:
        try:
            return importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            continue
    return None


result_dir = Path(os.environ["CONTAINER_RESULT_DIR"])
recipe = os.environ["CUTEDSL_GATE_RECIPE"]
overrides = sys.argv[1:]

register_omegaconf_resolvers()
config = parse_hydra_overrides(load_config(recipe), overrides)
resolved_config = OmegaConf.to_container(config, resolve=True)
effective_config = MasterConfig(**resolved_config).model_dump(mode="json")
OmegaConf.save(OmegaConf.create(effective_config), result_dir / "effective_config.yaml")

submodule_lines = (result_dir / "submodule_status.txt").read_text().splitlines()
submodules = []
for line in submodule_lines:
    if not line:
        continue
    fields = line.lstrip(" +-U").split()
    submodules.append({
        "status": line[0],
        "sha": fields[0],
        "path": fields[1],
        "description": " ".join(fields[2:]),
    })

metadata = {
    "run": {
        "run_id": os.environ["CUTEDSL_GATE_RUN_ID"],
        "cluster_profile": os.environ["CUTEDSL_PROFILE_NAME"],
        "created_at_utc": command_output(["date", "-u", "+%Y-%m-%dT%H:%M:%SZ"]),
        "recipe": recipe,
        "overrides": overrides,
        "effective_config": effective_config,
    },
    "source": {
        "remote": os.environ["CUTEDSL_GATE_GIT_REMOTE"],
        "branch": os.environ["CUTEDSL_GATE_GIT_BRANCH"],
        "sha": os.environ["CUTEDSL_GATE_GIT_SHA"],
        "upstream_ref": os.environ["CUTEDSL_GATE_UPSTREAM_REF"],
        "upstream_sha": os.environ["CUTEDSL_GATE_UPSTREAM_SHA"],
        "status_before": (result_dir / "git_status_before.txt").read_text().splitlines(),
        "submodules": submodules,
    },
    "image": {
        "path": os.environ["CUTEDSL_GATE_IMAGE"],
        "sha256": os.environ["CUTEDSL_GATE_IMAGE_SHA256"],
    },
    "runtime": {
        "nvte_cuda_archs": os.environ["NVTE_CUDA_ARCHS"],
        "platform": platform.platform(),
        "machine": platform.machine(),
        "python": sys.version,
        "pytorch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "cuda_driver": command_output(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"]).splitlines()[0],
        "cudnn": torch.backends.cudnn.version(),
        "cudnn_frontend": package_version("nvidia-cudnn-frontend"),
        "nccl": list(torch.cuda.nccl.version()) if torch.cuda.is_available() else None,
        "transformer_engine": package_version("transformer-engine", "transformer-engine-cu13") or getattr(transformer_engine, "__version__", None),
        "cutlass_dsl": package_version("nvidia-cutlass-dsl") or getattr(cutlass, "__version__", None),
        "cutlass_module": cutlass.__file__,
    },
    "hardware": {
        "cuda_device_count": torch.cuda.device_count(),
        "gpus": [
            {
                "index": index,
                "name": torch.cuda.get_device_name(index),
                "compute_capability": list(torch.cuda.get_device_capability(index)),
                "total_memory_bytes": torch.cuda.get_device_properties(index).total_memory,
            }
            for index in range(torch.cuda.device_count())
        ],
        "nvidia_smi_topology": command_output(["nvidia-smi", "topo", "-m"]).splitlines(),
    },
    "slurm": {
        "job_id": os.environ.get("SLURM_JOB_ID"),
        "account": os.environ.get("SLURM_JOB_ACCOUNT"),
        "partition": os.environ.get("SLURM_JOB_PARTITION"),
        "nodes": os.environ.get("SLURM_JOB_NODELIST"),
        "job_num_nodes": os.environ.get("SLURM_JOB_NUM_NODES"),
        "gpus": os.environ.get("SLURM_GPUS"),
    },
}
(result_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
(result_dir / "topology.txt").write_text("\n".join(metadata["hardware"]["nvidia_smi_topology"]) + "\n")
PY
'
cutedsl_write_event runtime_bootstrap pass 0 \
    "Locked runtime environment is active" uv_lock_check.log
cutedsl_write_event config_validation start '' \
    "Resolve and validate the functional GRPO config" effective_config.yaml
cutedsl_write_event config_validation pass 0 \
    "Functional config and provenance recorded" metadata.json

echo "[INFO] Running focused Linux tests before the GPU smoke and GRPO launch."
cutedsl_write_event focused_tests start '' \
    "Run focused tests, type checks, and pre-commit" focused_tests.log
"${SRUN[@]}" bash -lc '
set -euo pipefail
cd "${CONTAINER_REPO_ROOT}"
source "${CONTAINER_RUNTIME_DIR}/log_runtime_environment.sh"
[[ "${NVTE_CUDA_ARCHS}" == "100" ]]
[[ "${VIRTUAL_ENV}" == "${UV_PROJECT_ENVIRONMENT}" ]]
[[ "$("${RUNTIME_PYTHON}" --version)" == "Python ${UV_PYTHON_VERSION}" ]]
expected_runtime_python=$("${RUNTIME_PYTHON}" -c '"'"'import os, sys; print(os.path.realpath(sys.executable))'"'"')
runtime_python=$("${UV_BIN}" run --active --no-sync python -c '"'"'import os, sys; print(os.path.realpath(sys.executable))'"'"')
runtime_prefix=$("${UV_BIN}" run --active --no-sync python -c '"'"'import sys; print(sys.prefix)'"'"')
[[ "${runtime_python}" == "${expected_runtime_python}" ]]
[[ "${runtime_prefix}" == "${VIRTUAL_ENV}" ]]
bridge_root="${CONTAINER_REPO_ROOT}/3rdparty/Megatron-Bridge-workspace/Megatron-Bridge"
(
    cd "${bridge_root}"
    {
        printf '%s\n' \
            "[INFO] Bridge MSC exclusions: exactly 2 tests are deselected because multi-storage-client~=0.50 provides only CPython 3.12 wheels, while the locked NeMo runtime is Python ${UV_PYTHON_VERSION}." \
            "[INFO] Expected Bridge pytest summary: 203 passed, 2 deselected."
        "${RUNTIME_PYTHON}" -m pytest \
            tests/unit_tests/models/test_param_mapping.py \
            tests/unit_tests/training/test_checkpointing.py \
            --deselect=tests/unit_tests/training/test_checkpointing.py::TestCheckpointUtilities::test_ensure_directory_exists_with_msc_url \
            --deselect=tests/unit_tests/training/test_checkpointing.py::TestLoadCheckpointFromPathDirectIterDir::test_fsdp_dtensor_skips_tracker_resolution_with_msc \
            -q
    } 2>&1 | tee "${CONTAINER_RESULT_DIR}/focused_tests.log"
)
(
    cd "${CONTAINER_REPO_ROOT}"
    "${UV_BIN}" run --active --no-sync pytest \
        tests/unit/models/megatron/test_megatron_setup.py \
        tests/unit/models/megatron/test_community_import.py \
        tests/test_cutedsl_policy_recipe.py \
        -q
) 2>&1 | tee -a "${CONTAINER_RESULT_DIR}/focused_tests.log"

"${UV_BIN}" run --active --no-sync pyrefly check 2>&1 | tee "${CONTAINER_RESULT_DIR}/pyrefly.log"

git diff --name-only --diff-filter=ACMR "${PARENT_BASE_SHA}..HEAD" -- \
    . ":(exclude)3rdparty/Megatron-Bridge-workspace/Megatron-Bridge" \
    > "${CONTAINER_RESULT_DIR}/parent_changed_files.txt"
mapfile -t parent_changed_files < "${CONTAINER_RESULT_DIR}/parent_changed_files.txt"
if [[ ${#parent_changed_files[@]} -eq 0 ]]; then
    echo "[ERROR] No changed parent files found for pre-commit validation." >&2
    exit 1
fi
"${UV_BIN}" run --active --no-sync pre-commit run --files "${parent_changed_files[@]}" \
    2>&1 | tee "${CONTAINER_RESULT_DIR}/parent_precommit.log"

git -C "${bridge_root}" diff --name-only --diff-filter=ACMR HEAD^..HEAD \
    > "${CONTAINER_RESULT_DIR}/bridge_changed_files.txt"
mapfile -t bridge_changed_files < "${CONTAINER_RESULT_DIR}/bridge_changed_files.txt"
if [[ ${#bridge_changed_files[@]} -eq 0 ]]; then
    echo "[ERROR] No changed Bridge files found for pre-commit validation." >&2
    exit 1
fi
(
    cd "${bridge_root}"
    "${UV_BIN}" run --active --no-sync pre-commit run --files "${bridge_changed_files[@]}"
) 2>&1 | tee "${CONTAINER_RESULT_DIR}/bridge_precommit.log"
'
cutedsl_write_event focused_tests pass 0 \
    "Focused validation gate passed" focused_tests.log

echo "[INFO] Running the required Cutlass DSL import smoke."
cutedsl_write_event gpu_smoke start '' \
    "Run Cutlass import and four-GPU Transformer Engine smoke" gpu_smoke.log
"${SRUN[@]}" bash -lc '
set -euo pipefail
cd "${CONTAINER_REPO_ROOT}"
source "${CONTAINER_RUNTIME_DIR}/log_runtime_environment.sh"
[[ "${NVTE_CUDA_ARCHS}" == "100" ]]
[[ "${VIRTUAL_ENV}" == "${UV_PROJECT_ENVIRONMENT}" ]]
[[ "$("${RUNTIME_PYTHON}" --version)" == "Python ${UV_PYTHON_VERSION}" ]]
expected_runtime_python=$("${RUNTIME_PYTHON}" -c '"'"'import os, sys; print(os.path.realpath(sys.executable))'"'"')
runtime_python=$("${UV_BIN}" run --active --no-sync python -c '"'"'import os, sys; print(os.path.realpath(sys.executable))'"'"')
runtime_prefix=$("${UV_BIN}" run --active --no-sync python -c '"'"'import sys; print(sys.prefix)'"'"')
[[ "${runtime_python}" == "${expected_runtime_python}" ]]
[[ "${runtime_prefix}" == "${VIRTUAL_ENV}" ]]
"${UV_BIN}" run --active --no-sync python -c '"'"'import cutlass; from cutlass import cute; print(cutlass.__file__)'"'"' \
    2>&1 | tee "${CONTAINER_RESULT_DIR}/cutlass_import.log"
'

echo "[INFO] Running the four-GPU PyTorch and Transformer Engine device smoke."
"${SRUN[@]}" bash -lc '
set -euo pipefail
cd "${CONTAINER_REPO_ROOT}"
source "${CONTAINER_RUNTIME_DIR}/log_runtime_environment.sh"
[[ "${NVTE_CUDA_ARCHS}" == "100" ]]
[[ "${VIRTUAL_ENV}" == "${UV_PROJECT_ENVIRONMENT}" ]]
[[ "$("${RUNTIME_PYTHON}" --version)" == "Python ${UV_PYTHON_VERSION}" ]]
expected_runtime_python=$("${RUNTIME_PYTHON}" -c '"'"'import os, sys; print(os.path.realpath(sys.executable))'"'"')
runtime_python=$("${UV_BIN}" run --active --no-sync python -c '"'"'import os, sys; print(os.path.realpath(sys.executable))'"'"')
runtime_prefix=$("${UV_BIN}" run --active --no-sync python -c '"'"'import sys; print(sys.prefix)'"'"')
[[ "${runtime_python}" == "${expected_runtime_python}" ]]
[[ "${runtime_prefix}" == "${VIRTUAL_ENV}" ]]
"${UV_BIN}" run --active --no-sync python - <<'"'"'PY'"'"' 2>&1 | tee "${CONTAINER_RESULT_DIR}/gpu_smoke.log"
import torch
import transformer_engine.pytorch as te

expected_devices = 4
actual_devices = torch.cuda.device_count()
assert actual_devices == expected_devices, f"expected {expected_devices} CUDA devices, found {actual_devices}"
print(f"PyTorch {torch.__version__}; CUDA {torch.version.cuda}; TE module {te.__file__}")
for device_index in range(actual_devices):
    with torch.cuda.device(device_index):
        model = te.Linear(256, 512, bias=True).cuda(device_index)
        model_input = torch.randn(
            (32, 256), device=f"cuda:{device_index}", requires_grad=True
        )
        output = model(model_input)
        loss = output.square().mean()
        loss.backward()
        torch.cuda.synchronize()
        assert torch.isfinite(output).all(), (
            f"non-finite TE output on CUDA device {device_index}"
        )
        assert model_input.grad is not None and torch.isfinite(model_input.grad).all(), (
            f"non-finite TE input gradient on CUDA device {device_index}"
        )
        for name, parameter in model.named_parameters():
            assert parameter.grad is not None and torch.isfinite(parameter.grad).all(), (
                f"missing or non-finite TE gradient {name} on CUDA device {device_index}"
            )
        print(
            f"cuda:{device_index} {torch.cuda.get_device_name(device_index)} "
            f"capability={torch.cuda.get_device_capability(device_index)} TE forward/backward=PASS"
        )
PY
'
cutedsl_write_event gpu_smoke pass 0 \
    "Cutlass import and four-GPU Transformer Engine smoke passed" gpu_smoke.log

echo "[INFO] Launching the three-update GRPO gate with a step-2 policy-worker Nsight capture."
cutedsl_write_event functional_grpo start '' \
    "Run three-update functional GRPO gate" grpo.log
"${SRUN[@]}" bash -lc '
set -euo pipefail
cd "${CONTAINER_REPO_ROOT}"
source "${CONTAINER_RUNTIME_DIR}/log_runtime_environment.sh"
[[ "${NVTE_CUDA_ARCHS}" == "100" ]]
[[ "${VIRTUAL_ENV}" == "${UV_PROJECT_ENVIRONMENT}" ]]
[[ "$("${RUNTIME_PYTHON}" --version)" == "Python ${UV_PYTHON_VERSION}" ]]
expected_runtime_python=$("${RUNTIME_PYTHON}" -c '"'"'import os, sys; print(os.path.realpath(sys.executable))'"'"')
runtime_python=$("${UV_BIN}" run --active --no-sync python -c '"'"'import os, sys; print(os.path.realpath(sys.executable))'"'"')
runtime_prefix=$("${UV_BIN}" run --active --no-sync python -c '"'"'import sys; print(sys.prefix)'"'"')
[[ "${runtime_python}" == "${expected_runtime_python}" ]]
[[ "${runtime_prefix}" == "${VIRTUAL_ENV}" ]]
export NRL_NSYS_WORKER_PATTERNS="megatron_policy_worker"
export NRL_NSYS_PROFILE_STEP_RANGE="2:3"
export NRL_NSYS_EXTRA_OPTIONS='"'"'{"cuda-memory-usage":"true","cpuctxsw":"none"}'"'"'
export RAY_TMPDIR="${CONTAINER_RUNTIME_DIR}/ray_tmp"
export TMPDIR="${CONTAINER_RUNTIME_DIR}/tmp"
readonly RAY_LOG_DIR="${CONTAINER_RESULT_DIR}/ray_logs"
mkdir -p "${RAY_TMPDIR}" "${TMPDIR}" "${RAY_LOG_DIR}"
mapfile -t grpo_overrides < "${CONTAINER_RESULT_DIR}/grpo_overrides.txt"
set +e
"${UV_BIN}" run --active --no-sync examples/run_grpo.py \
    --config "${CUTEDSL_GATE_RECIPE}" \
    "${grpo_overrides[@]}" \
    2>&1 | tee "${CONTAINER_RESULT_DIR}/grpo.log"
grpo_exit_code=${PIPESTATUS[0]}

mkdir -p "${CONTAINER_RESULT_DIR}/nsight"
: > "${CONTAINER_RESULT_DIR}/kernel_evidence.txt"
while IFS= read -r -d "" report; do
    cp -a "${report}" "${CONTAINER_RESULT_DIR}/nsight/"
    printf "\n===== %s =====\n" "${report}" >> "${CONTAINER_RESULT_DIR}/kernel_evidence.txt"
    nsys stats --report cuda_gpu_kern_sum "${report}" \
        >> "${CONTAINER_RESULT_DIR}/kernel_evidence.txt" 2>&1 || true
done < <(find "${RAY_TMPDIR}" -type f -path '"'"'*/logs/nsight/*.nsys-rep'"'"' -print0 2>/dev/null)
while IFS= read -r -d "" artifact; do
    relative_path=${artifact#"${RAY_TMPDIR}/"}
    destination="${RAY_LOG_DIR}/${relative_path}"
    mkdir -p "$(dirname "${destination}")"
    cp -a "${artifact}" "${destination}"
done < <(
    find "${RAY_TMPDIR}" -type f \
        \( -name "*.log" -o -name "*.out" -o -name "*.err" -o -name "*.json" \) \
        -size -50M -print0 2>/dev/null
)
find "${RAY_LOG_DIR}" -type f -print > "${CONTAINER_RESULT_DIR}/ray_artifacts.txt" 2>/dev/null
set -e
if [[ ${grpo_exit_code} -ne 0 ]]; then
    echo "[ERROR] GRPO exited with status ${grpo_exit_code}; preserved Ray/Nsight artifacts." >&2
    exit "${grpo_exit_code}"
fi

"${UV_BIN}" run --active --no-sync tests/json_dump_tb_logs.py \
    "${CONTAINER_RESULT_DIR}/tensorboard" \
    --output_path "${CONTAINER_RESULT_DIR}/metrics.json"

"${UV_BIN}" run --active --no-sync python - <<'"'"'PY'"'"'
import json
import math
import statistics
from pathlib import Path

result_dir = Path("/results")
metrics = json.loads((result_dir / "metrics.json").read_text())


def values(name: str, *, post_warmup: bool = False) -> list[float]:
    series = metrics.get(name, {})
    ordered = sorted(((int(step), float(value)) for step, value in series.items()))
    if post_warmup and len(ordered) > 1:
        ordered = ordered[1:]
    return [value for _, value in ordered]


losses = values("train/loss")
grad_norms = values("train/grad_norm")
optimizer_updates = values("train/optimizer_update_successful")
policy_times = values("timing/train/policy_training", post_warmup=True)
tokens_per_second = values("performance/policy_training_tokens_per_sec_per_gpu", post_warmup=True)
memory_series = {
    name: [float(value) for value in series.values()]
    for name, series in metrics.items()
    if name.endswith(".mem_gb") and isinstance(series, dict)
}
peak_memory_metrics = {
    name: max(series) for name, series in memory_series.items() if series
}
parity_thresholds = {
    "gen_kl_error_max": 0.05,
    "token_mult_prob_error_max": 2.0,
}
max_generation_kl_error = max(values("train/gen_kl_error"), default=None)
max_token_mult_probability_error = max(
    values("train/token_mult_prob_error"), default=None
)

summary = {
    "successful_optimizer_updates": sum(value == 1.0 for value in optimizer_updates),
    "optimizer_update_results": optimizer_updates,
    "losses_finite": bool(losses) and all(math.isfinite(value) for value in losses),
    "gradients_finite": bool(grad_norms) and all(math.isfinite(value) for value in grad_norms),
    "parity_thresholds": parity_thresholds,
    "max_generation_kl_error": max_generation_kl_error,
    "max_token_mult_probability_error": max_token_mult_probability_error,
    "median_post_warmup_policy_training_time_s": statistics.median(policy_times) if policy_times else None,
    "median_post_warmup_policy_training_tokens_per_s_per_gpu": statistics.median(tokens_per_second) if tokens_per_second else None,
    "peak_memory_metrics_gb": peak_memory_metrics,
    "nsight_reports": sorted(path.name for path in (result_dir / "nsight").glob("*.nsys-rep")),
    "kernel_evidence_file": "kernel_evidence.txt",
}
(result_dir / "metrics_summary.json").write_text(json.dumps(summary, indent=2) + "\n")

assert summary["successful_optimizer_updates"] >= 2, summary
assert summary["losses_finite"], summary
assert summary["gradients_finite"], summary
assert max_generation_kl_error is not None, summary
assert max_generation_kl_error < parity_thresholds["gen_kl_error_max"], summary
assert max_token_mult_probability_error is not None, summary
assert max_token_mult_probability_error < parity_thresholds["token_mult_prob_error_max"], summary
assert policy_times, summary
assert tokens_per_second, summary
assert peak_memory_metrics, summary
PY
'

cutedsl_write_event functional_grpo pass 0 \
    "Functional GRPO gate passed" grpo.log
cutedsl_write_event timing skip 0 \
    "Timing matrix is handled by run_cutedsl_matrix.sbatch" ''
cutedsl_write_event profile pass 0 \
    "Functional Nsight evidence collected" kernel_evidence.txt
cutedsl_write_event metrics_export start '' \
    "Validate and summarize functional metrics" metrics_summary.json
cutedsl_write_event metrics_export pass 0 \
    "Functional metrics summary recorded" metrics_summary.json

echo "[PASS] CuTeDSL functional payload completed. Review metrics_summary.json and kernel_evidence.txt before classifying the GPU gate."
