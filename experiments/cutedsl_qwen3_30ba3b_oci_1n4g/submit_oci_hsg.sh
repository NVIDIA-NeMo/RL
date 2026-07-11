#!/bin/bash
#SBATCH --job-name=cutedsl-qwen3-30ba3b-oci-1n4g
#SBATCH --account=nemotron_n3_post
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=72
#SBATCH --mem=0
#SBATCH --time=02:00:00
#SBATCH --exclusive
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null

set -euo pipefail

readonly IMAGE="/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/sna/containers/nemo_rl_nightly_20260707.sqsh"
readonly RECIPE="examples/configs/recipes/llm/performance/grpo-qwen3-30ba3b-1n4g-megatron-mxfp8-cutedsl.yaml"
readonly CONTAINER_REPO_ROOT="/workspace/nemo-rl"
readonly CONTAINER_RESULT_DIR="/results"
export CONTAINER_REPO_ROOT CONTAINER_RESULT_DIR

readonly SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
readonly EXPERIMENT_DIR="${SCRIPT_DIR}"
readonly REPO_ROOT="$(git -C "${SCRIPT_DIR}" rev-parse --show-toplevel)"
readonly RUN_ID="${SLURM_JOB_ID:?submit this wrapper with sbatch}${SLURM_RESTART_COUNT:+-r${SLURM_RESTART_COUNT}}"
readonly RESULT_DIR="${EXPERIMENT_DIR}/results/${RUN_ID}"

mkdir -p "${RESULT_DIR}"
exec > >(tee "${RESULT_DIR}/slurm.out") 2>&1

on_exit() {
    local exit_code=$?
    set +e
    printf '{\n  "run_id": "%s",\n  "job_id": "%s",\n  "exit_code": %d,\n  "finished_at_utc": "%s"\n}\n' \
        "${RUN_ID}" "${SLURM_JOB_ID}" "${exit_code}" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
        > "${RESULT_DIR}/status.json"
    return "${exit_code}"
}
trap on_exit EXIT

if [[ ! -r "${IMAGE}" ]]; then
    echo "[ERROR] Required first-choice image is not readable: ${IMAGE}" >&2
    exit 1
fi

if ! git -C "${REPO_ROOT}" diff --quiet || ! git -C "${REPO_ROOT}" diff --cached --quiet; then
    echo "[ERROR] Refusing to run with tracked changes in the NeMo-RL checkout." >&2
    exit 1
fi
if ! git -C "${REPO_ROOT}" submodule foreach --quiet --recursive \
    'git diff --quiet && git diff --cached --quiet'; then
    echo "[ERROR] Refusing to run with tracked changes in a recursive submodule." >&2
    exit 1
fi

readonly IMAGE_SHA256="$(sha256sum "${IMAGE}" | awk '{print $1}')"
readonly GIT_SHA="$(git -C "${REPO_ROOT}" rev-parse HEAD)"
readonly GIT_BRANCH="$(git -C "${REPO_ROOT}" branch --show-current)"
readonly GIT_REMOTE="$(git -C "${REPO_ROOT}" remote get-url origin)"

git -C "${REPO_ROOT}" status --short --branch > "${RESULT_DIR}/git_status_before.txt"
git -C "${REPO_ROOT}" submodule status --recursive > "${RESULT_DIR}/submodule_status.txt"
printf '%s  %s\n' "${IMAGE_SHA256}" "${IMAGE}" > "${RESULT_DIR}/image.sha256"

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
export NEMO_RL_VENV_DIR="${CONTAINER_RESULT_DIR}/worker_venvs"
export NRL_FORCE_REBUILD_VENVS="true"
export NRL_IGNORE_VERSION_MISMATCH="1"
export PYTHONPATH="${CONTAINER_REPO_ROOT}:${PYTHONPATH:-}"
export TORCH_EXTENSIONS_DIR="${CONTAINER_RESULT_DIR}/torch_extensions"
export TRITON_CACHE_DIR="${CONTAINER_RESULT_DIR}/triton_cache"
export UV_PROJECT_ENVIRONMENT="${CONTAINER_RESULT_DIR}/venv"
export UV_CACHE_DIR="${CONTAINER_RESULT_DIR}/uv_cache"

export OCI_GATE_GIT_BRANCH="${GIT_BRANCH}"
export OCI_GATE_GIT_REMOTE="${GIT_REMOTE}"
export OCI_GATE_GIT_SHA="${GIT_SHA}"
export OCI_GATE_IMAGE="${IMAGE}"
export OCI_GATE_IMAGE_SHA256="${IMAGE_SHA256}"
export OCI_GATE_RECIPE="${RECIPE}"
export OCI_GATE_RUN_ID="${RUN_ID}"

readonly -a SRUN=(
    srun
    --nodes=1
    --ntasks=1
    --gres=gpu:4
    --cpus-per-task=72
    --mpi=pmix
    --container-image="${IMAGE}"
    --container-mounts="${REPO_ROOT}:${CONTAINER_REPO_ROOT},${RESULT_DIR}:${CONTAINER_RESULT_DIR}"
    --container-workdir="${CONTAINER_REPO_ROOT}"
)

echo "[INFO] Run ID: ${RUN_ID}"
echo "[INFO] NeMo-RL: ${GIT_SHA} (${GIT_BRANCH})"
echo "[INFO] Image: ${IMAGE}"
echo "[INFO] Image SHA256: ${IMAGE_SHA256}"
echo "[INFO] Result directory: ${RESULT_DIR}"

echo "[INFO] Creating the run-local Linux environment and recording provenance."
"${SRUN[@]}" bash -lc '
set -euo pipefail
cd "${CONTAINER_REPO_ROOT}"
uv sync --frozen --extra mcore --group test
mapfile -t grpo_overrides < "${CONTAINER_RESULT_DIR}/grpo_overrides.txt"
uv run --no-sync python - "${grpo_overrides[@]}" <<'"'"'PY'"'"'
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
recipe = os.environ["OCI_GATE_RECIPE"]
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
        "run_id": os.environ["OCI_GATE_RUN_ID"],
        "created_at_utc": command_output(["date", "-u", "+%Y-%m-%dT%H:%M:%SZ"]),
        "recipe": recipe,
        "overrides": overrides,
        "effective_config": effective_config,
    },
    "source": {
        "remote": os.environ["OCI_GATE_GIT_REMOTE"],
        "branch": os.environ["OCI_GATE_GIT_BRANCH"],
        "sha": os.environ["OCI_GATE_GIT_SHA"],
        "status_before": (result_dir / "git_status_before.txt").read_text().splitlines(),
        "submodules": submodules,
    },
    "image": {
        "path": os.environ["OCI_GATE_IMAGE"],
        "sha256": os.environ["OCI_GATE_IMAGE_SHA256"],
    },
    "runtime": {
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

echo "[INFO] Running focused Linux tests before the GPU smoke and GRPO launch."
"${SRUN[@]}" bash -lc '
set -euo pipefail
cd "${CONTAINER_REPO_ROOT}"
uv run --no-sync pytest \
    tests/unit/models/megatron/test_megatron_setup.py \
    tests/unit/models/megatron/test_community_import.py \
    tests/test_cutedsl_policy_recipe.py \
    -q 2>&1 | tee "${CONTAINER_RESULT_DIR}/focused_tests.log"
'

echo "[INFO] Running the required Cutlass DSL import smoke."
"${SRUN[@]}" bash -lc '
set -euo pipefail
cd "${CONTAINER_REPO_ROOT}"
uv run --no-sync python -c '"'"'import cutlass; from cutlass import cute; print(cutlass.__file__)'"'"' \
    2>&1 | tee "${CONTAINER_RESULT_DIR}/cutlass_import.log"
'

echo "[INFO] Running the four-GPU PyTorch and Transformer Engine device smoke."
"${SRUN[@]}" bash -lc '
set -euo pipefail
cd "${CONTAINER_REPO_ROOT}"
uv run --no-sync python - <<'"'"'PY'"'"' 2>&1 | tee "${CONTAINER_RESULT_DIR}/gpu_smoke.log"
import torch
import transformer_engine.pytorch as te

expected_devices = 4
actual_devices = torch.cuda.device_count()
assert actual_devices == expected_devices, f"expected {expected_devices} CUDA devices, found {actual_devices}"
print(f"PyTorch {torch.__version__}; CUDA {torch.version.cuda}; TE module {te.__file__}")
for device_index in range(actual_devices):
    with torch.cuda.device(device_index):
        lhs = torch.randn((256, 256), device="cuda", dtype=torch.bfloat16)
        rhs = torch.randn((256, 256), device="cuda", dtype=torch.bfloat16)
        output = lhs @ rhs
        torch.cuda.synchronize()
        assert torch.isfinite(output).all(), f"non-finite device-smoke output on CUDA device {device_index}"
        print(
            f"cuda:{device_index} {torch.cuda.get_device_name(device_index)} "
            f"capability={torch.cuda.get_device_capability(device_index)} smoke=PASS"
        )
PY
'

echo "[INFO] Launching the three-update GRPO gate with a step-2 policy-worker Nsight capture."
"${SRUN[@]}" bash -lc '
set -euo pipefail
cd "${CONTAINER_REPO_ROOT}"
export NRL_NSYS_WORKER_PATTERNS="megatron_policy_worker"
export NRL_NSYS_PROFILE_STEP_RANGE="2:3"
export NRL_NSYS_EXTRA_OPTIONS='"'"'{"cuda-memory-usage":"true","cpuctxsw":"none"}'"'"'
mapfile -t grpo_overrides < "${CONTAINER_RESULT_DIR}/grpo_overrides.txt"
uv run --no-sync examples/run_grpo.py \
    --config "${OCI_GATE_RECIPE}" \
    "${grpo_overrides[@]}" \
    2>&1 | tee "${CONTAINER_RESULT_DIR}/grpo.log"

mkdir -p "${CONTAINER_RESULT_DIR}/nsight"
: > "${CONTAINER_RESULT_DIR}/kernel_evidence.txt"
while IFS= read -r -d "" report; do
    cp -a "${report}" "${CONTAINER_RESULT_DIR}/nsight/"
    printf "\n===== %s =====\n" "${report}" >> "${CONTAINER_RESULT_DIR}/kernel_evidence.txt"
    nsys stats --report cuda_gpu_kern_sum "${report}" \
        >> "${CONTAINER_RESULT_DIR}/kernel_evidence.txt" 2>&1 || true
done < <(find /tmp/ray -type f -path '"'"'*/logs/nsight/*.nsys-rep'"'"' -print0 2>/dev/null)

uv run --no-sync tests/json_dump_tb_logs.py \
    "${CONTAINER_RESULT_DIR}/tensorboard" \
    --output_path "${CONTAINER_RESULT_DIR}/metrics.json"

uv run --no-sync python - <<'"'"'PY'"'"'
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
policy_times = values("timing/train/policy_training", post_warmup=True)
tokens_per_second = values("performance/policy_training_tokens_per_sec_per_gpu", post_warmup=True)
memory_series = {
    name: [float(value) for value in series.values()]
    for name, series in metrics.items()
    if "memory" in name.lower() and isinstance(series, dict)
}

summary = {
    "completed_policy_updates": len(losses),
    "losses_finite": bool(losses) and all(math.isfinite(value) for value in losses),
    "gradients_finite": bool(grad_norms) and all(math.isfinite(value) for value in grad_norms),
    "max_generation_kl_error": max(values("train/gen_kl_error"), default=None),
    "max_token_mult_probability_error": max(values("train/token_mult_prob_error"), default=None),
    "median_post_warmup_policy_training_time_s": statistics.median(policy_times) if policy_times else None,
    "median_post_warmup_policy_training_tokens_per_s_per_gpu": statistics.median(tokens_per_second) if tokens_per_second else None,
    "peak_memory_metrics": {
        name: max(series) for name, series in memory_series.items() if series
    },
    "nsight_reports": sorted(path.name for path in (result_dir / "nsight").glob("*.nsys-rep")),
    "kernel_evidence_file": "kernel_evidence.txt",
}
(result_dir / "metrics_summary.json").write_text(json.dumps(summary, indent=2) + "\n")

assert summary["completed_policy_updates"] >= 2, summary
assert summary["losses_finite"], summary
assert summary["gradients_finite"], summary
PY
'

echo "[PASS] OCI-HSG wrapper completed. Review metrics_summary.json and kernel_evidence.txt before classifying the GPU gate."
