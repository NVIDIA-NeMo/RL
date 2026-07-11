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

#SBATCH --job-name=cutedsl-ab-qwen3-30ba3b-1n4g
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
readonly CUTEDSL_SELECTOR_PATH="policy.megatron_cfg.env_vars.NVTE_CUTEDSL_FUSED_GROUPED_MLP"
readonly EXPECTED_ON_SELECTOR="1"
readonly OFF_OVERRIDE="${CUTEDSL_SELECTOR_PATH}=\"0\""
export CONTAINER_REPO_ROOT CONTAINER_RESULT_DIR CONTAINER_RUNTIME_DIR
export RECIPE CUTEDSL_SELECTOR_PATH EXPECTED_ON_SELECTOR OFF_OVERRIDE

readonly SLURM_SUBMIT_DIR="${SLURM_SUBMIT_DIR:?submit this benchmark with sbatch from the repository checkout}"
REPO_ROOT=$(git -C "${SLURM_SUBMIT_DIR}" rev-parse --show-toplevel)
readonly REPO_ROOT
readonly EXPERIMENT_DIR="${REPO_ROOT}/experiments/cutedsl_qwen3_30ba3b_oci_1n4g"
source "${EXPERIMENT_DIR}/lib/cluster_profile.sh"
load_cutedsl_cluster_profile
readonly IMAGE="${CUTEDSL_IMAGE}"
readonly RUN_ID="${SLURM_JOB_ID:?submit this benchmark with sbatch}${SLURM_RESTART_COUNT:+-r${SLURM_RESTART_COUNT}}"
readonly RESULT_DIR="${EXPERIMENT_DIR}/benchmark_results/${RUN_ID}"
readonly HOST_RUNTIME_DIR="/tmp/${USER}/nemo-2606-cutedsl-benchmark/${RUN_ID}"

WARMUP_UPDATES="${CUTEDSL_BENCHMARK_WARMUP_UPDATES:-3}"
MEASURED_UPDATES="${CUTEDSL_BENCHMARK_MEASURED_UPDATES:-20}"
TIMING_ORDER="${CUTEDSL_BENCHMARK_ORDER:-on,off}"
PROFILE_ENABLED="${CUTEDSL_BENCHMARK_PROFILE:-1}"
REPLICATE_INDEX="${CUTEDSL_BENCHMARK_REPLICATE:-0}"
SUBMISSION_GROUP="${CUTEDSL_BENCHMARK_SUBMISSION_GROUP:-direct}"
readonly WARMUP_UPDATES MEASURED_UPDATES TIMING_ORDER PROFILE_ENABLED
readonly REPLICATE_INDEX SUBMISSION_GROUP

if [[ ! "${WARMUP_UPDATES}" =~ ^[0-9]+$ ]] || ((WARMUP_UPDATES < 3)); then
    echo "[ERROR] CUTEDSL_BENCHMARK_WARMUP_UPDATES must be an integer >= 3." >&2
    exit 1
fi
if [[ ! "${MEASURED_UPDATES}" =~ ^[0-9]+$ ]] || ((MEASURED_UPDATES < 10)); then
    echo "[ERROR] CUTEDSL_BENCHMARK_MEASURED_UPDATES must be an integer >= 10." >&2
    exit 1
fi
if [[ "${PROFILE_ENABLED}" != "0" && "${PROFILE_ENABLED}" != "1" ]]; then
    echo "[ERROR] CUTEDSL_BENCHMARK_PROFILE must be 0 or 1." >&2
    exit 1
fi
if [[ ! "${REPLICATE_INDEX}" =~ ^[0-9]+$ ]]; then
    echo "[ERROR] CUTEDSL_BENCHMARK_REPLICATE must be a non-negative integer." >&2
    exit 1
fi

IFS=',' read -r -a timing_arms <<< "${TIMING_ORDER}"
if [[ ${#timing_arms[@]} -ne 2 ]]; then
    echo "[ERROR] CUTEDSL_BENCHMARK_ORDER must contain exactly two comma-separated arms." >&2
    exit 1
fi
on_count=0
off_count=0
for arm in "${timing_arms[@]}"; do
    case "${arm}" in
        on) on_count=$((on_count + 1)) ;;
        off) off_count=$((off_count + 1)) ;;
        *)
            echo "[ERROR] Unknown benchmark arm in order: ${arm}" >&2
            exit 1
            ;;
    esac
done
if [[ ${on_count} -ne 1 || ${off_count} -ne 1 ]]; then
    echo "[ERROR] CUTEDSL_BENCHMARK_ORDER must contain on and off exactly once." >&2
    exit 1
fi

TOTAL_UPDATES=$((WARMUP_UPDATES + MEASURED_UPDATES))
readonly TOTAL_UPDATES
export RUN_ID WARMUP_UPDATES MEASURED_UPDATES TIMING_ORDER TOTAL_UPDATES PROFILE_ENABLED
export REPLICATE_INDEX SUBMISSION_GROUP

mkdir -p "${RESULT_DIR}" "${HOST_RUNTIME_DIR}"
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
unset runtime_uv_version runtime_python_version runtime_python_realpath runtime_python_prefix
BASH
exec > >(tee "${RESULT_DIR}/slurm.out") 2>&1

on_exit() {
    local exit_code=$?
    set +e
    printf '{\n  "run_id": "%s",\n  "job_id": "%s",\n  "exit_code": %d,\n  "finished_at_utc": "%s"\n}\n' \
        "${RUN_ID}" "${SLURM_JOB_ID}" "${exit_code}" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
        > "${RESULT_DIR}/status.json"
    rm -rf --one-file-system -- "${HOST_RUNTIME_DIR}"
    return "${exit_code}"
}
trap on_exit EXIT

if [[ ! -r "${IMAGE}" ]]; then
    echo "[ERROR] Required immutable image is not readable: ${IMAGE}" >&2
    exit 1
fi

SOURCE_STATUS=$(git -C "${REPO_ROOT}" status --porcelain --untracked-files=normal --ignore-submodules=none)
readonly SOURCE_STATUS
if [[ -n "${SOURCE_STATUS}" ]]; then
    echo "[ERROR] Refusing to benchmark a dirty parent checkout:" >&2
    printf '%s\n' "${SOURCE_STATUS}" >&2
    exit 1
fi
if ! git -C "${REPO_ROOT}" submodule foreach --quiet --recursive \
    'test -z "$(git status --porcelain --untracked-files=normal --ignore-submodules=none)"'; then
    echo "[ERROR] Refusing to benchmark dirty recursive submodules." >&2
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

GIT_SHA=$(git -C "${REPO_ROOT}" rev-parse HEAD)
UPSTREAM_REF=$(git -C "${REPO_ROOT}" rev-parse --abbrev-ref --symbolic-full-name '@{upstream}')
UPSTREAM_SHA=$(git -C "${REPO_ROOT}" rev-parse '@{upstream}')
IMAGE_SHA256=$(sha256sum "${IMAGE}" | awk '{print $1}')
readonly GIT_SHA UPSTREAM_REF UPSTREAM_SHA IMAGE_SHA256
if [[ "${IMAGE_SHA256}" != "${CUTEDSL_IMAGE_SHA256}" ]]; then
    echo "[ERROR] Image SHA256 ${IMAGE_SHA256} does not match profile SHA256 ${CUTEDSL_IMAGE_SHA256}." >&2
    exit 1
fi
if [[ "${GIT_SHA}" != "${UPSTREAM_SHA}" ]]; then
    echo "[ERROR] HEAD ${GIT_SHA} does not match ${UPSTREAM_REF} at ${UPSTREAM_SHA}." >&2
    exit 1
fi
export GIT_SHA UPSTREAM_REF UPSTREAM_SHA IMAGE IMAGE_SHA256

printf '%s\n' "${SUBMODULE_STATUS}" > "${RESULT_DIR}/submodule_status.txt"
printf '%s  %s\n' "${IMAGE_SHA256}" "${IMAGE}" > "${RESULT_DIR}/image.sha256"

readonly -a COMMON_OVERRIDES=(
    "grpo.max_num_steps=${TOTAL_UPDATES}"
    "cluster.num_nodes=1"
    "cluster.gpus_per_node=4"
    "logger.log_dir=${CONTAINER_RUNTIME_DIR}/benchmark_logs"
    "logger.wandb_enabled=false"
    "logger.tensorboard_enabled=true"
    "logger.monitor_gpus=true"
    "checkpointing.enabled=false"
    "checkpointing.checkpoint_dir=${CONTAINER_RUNTIME_DIR}/benchmark_checkpoints"
)
printf '%s\n' "${COMMON_OVERRIDES[@]}" > "${RESULT_DIR}/common_overrides.txt"
printf '%s\n' "${timing_arms[@]}" > "${RESULT_DIR}/timing_order.txt"

export CUDA_HOME="/usr/local/cuda"
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

SRUN=(
    srun
    --nodes=1
    --ntasks=1
    --cpus-per-task=72
    --mpi=pmix
    --container-image="${IMAGE}"
    --container-env=VIRTUAL_ENV,UV_PROJECT_ENVIRONMENT
    --container-mounts="${REPO_ROOT}:${CONTAINER_REPO_ROOT},${RESULT_DIR}:${CONTAINER_RESULT_DIR},${HOST_RUNTIME_DIR}:${CONTAINER_RUNTIME_DIR}"
    --container-workdir="${CONTAINER_REPO_ROOT}"
)
if [[ -n "${CUTEDSL_GRES}" ]]; then
    SRUN+=("--gres=${CUTEDSL_GRES}")
fi
readonly -a SRUN

echo "[INFO] Benchmark run ID: ${RUN_ID}"
echo "[INFO] Timing order: ${TIMING_ORDER}"
echo "[INFO] Updates per timing arm: ${WARMUP_UPDATES} warmup + ${MEASURED_UPDATES} measured"
echo "[INFO] Source: ${GIT_SHA}; image SHA256: ${IMAGE_SHA256}"

echo "[INFO] Bootstrapping the locked node-local environment and validating matched configs."
"${SRUN[@]}" bash -s <<'BASH'
set -euo pipefail
cd "${CONTAINER_REPO_ROOT}"
curl -LsSf "https://astral.sh/uv/${UV_VERSION}/install.sh" | sh
[[ "$("${UV_BIN}" --version)" == "uv ${UV_VERSION} "* ]]
"${UV_BIN}" python install "${UV_PYTHON_VERSION}"
"${UV_BIN}" python find "${UV_PYTHON_VERSION}"
"${UV_BIN}" sync --locked --extra mcore --group test --group dev --python "${UV_PYTHON_VERSION}"
source "${CONTAINER_RUNTIME_DIR}/log_runtime_environment.sh"
[[ "${VIRTUAL_ENV}" == "${UV_PROJECT_ENVIRONMENT}" ]]
[[ "$("${RUNTIME_PYTHON}" --version)" == "Python ${UV_PYTHON_VERSION}" ]]
expected_runtime_python=$("${RUNTIME_PYTHON}" -c 'import os, sys; print(os.path.realpath(sys.executable))')
runtime_python=$("${UV_BIN}" run --active --no-sync python -c 'import os, sys; print(os.path.realpath(sys.executable))')
runtime_prefix=$("${UV_BIN}" run --active --no-sync python -c 'import sys; print(sys.prefix)')
[[ "${runtime_python}" == "${expected_runtime_python}" ]]
[[ "${runtime_prefix}" == "${VIRTUAL_ENV}" ]]
"${UV_BIN}" lock --check 2>&1 | tee "${CONTAINER_RESULT_DIR}/uv_lock_check.log"

"${UV_BIN}" run --active --no-sync python - <<'PY'
import json
import os
from pathlib import Path
from typing import Any

from nemo_rl.algorithms.grpo import MasterConfig
from nemo_rl.utils.config import (
    load_config,
    parse_hydra_overrides,
    register_omegaconf_resolvers,
)
from omegaconf import OmegaConf


def collect_differences(on_value: Any, off_value: Any, path: str = "") -> dict[str, Any]:
    if isinstance(on_value, dict) and isinstance(off_value, dict):
        differences: dict[str, Any] = {}
        for key in sorted(set(on_value) | set(off_value)):
            child_path = f"{path}.{key}" if path else key
            if key not in on_value or key not in off_value:
                differences[child_path] = {
                    "on": on_value.get(key),
                    "off": off_value.get(key),
                }
            else:
                differences.update(
                    collect_differences(on_value[key], off_value[key], child_path)
                )
        return differences
    if on_value != off_value:
        return {path: {"on": on_value, "off": off_value}}
    return {}


def value_at(config: dict[str, Any], dotted_path: str) -> Any:
    value: Any = config
    for key in dotted_path.split("."):
        value = value[key]
    return value


result_dir = Path(os.environ["CONTAINER_RESULT_DIR"])
recipe = os.environ["RECIPE"]
common_overrides = (result_dir / "common_overrides.txt").read_text().splitlines()
selector_path = os.environ["CUTEDSL_SELECTOR_PATH"]
off_override = os.environ["OFF_OVERRIDE"]

register_omegaconf_resolvers()


def resolve(overrides: list[str]) -> dict[str, Any]:
    config = parse_hydra_overrides(load_config(recipe), overrides)
    resolved = OmegaConf.to_container(config, resolve=True)
    return MasterConfig(**resolved).model_dump(mode="json")


on_config = resolve(common_overrides)
off_config = resolve([*common_overrides, off_override])
on_selector = value_at(on_config, selector_path)
off_selector = value_at(off_config, selector_path)
assert on_selector == "1", on_selector
assert off_selector == "0", off_selector
differences = collect_differences(on_config, off_config)
assert differences == {selector_path: {"on": "1", "off": "0"}}, differences

OmegaConf.save(OmegaConf.create(on_config), result_dir / "effective_config_on.yaml")
OmegaConf.save(OmegaConf.create(off_config), result_dir / "effective_config_off.yaml")
(result_dir / "matched_config_diff.json").write_text(
    json.dumps(differences, indent=2) + "\n"
)

timing_order = os.environ["TIMING_ORDER"].split(",")
manifest = {
    "run_id": os.environ["RUN_ID"],
    "cluster_profile": os.environ["CUTEDSL_PROFILE_NAME"],
    "replicate_index": int(os.environ["REPLICATE_INDEX"]),
    "submission_group": os.environ["SUBMISSION_GROUP"],
    "timing_order": timing_order,
    "warmup_updates": int(os.environ["WARMUP_UPDATES"]),
    "measured_updates": int(os.environ["MEASURED_UPDATES"]),
    "total_updates": int(os.environ["TOTAL_UPDATES"]),
    "profile_enabled": os.environ["PROFILE_ENABLED"] == "1",
    "source_sha": os.environ["GIT_SHA"],
    "upstream_ref": os.environ["UPSTREAM_REF"],
    "upstream_sha": os.environ["UPSTREAM_SHA"],
    "image": os.environ["IMAGE"],
    "image_sha256": os.environ["IMAGE_SHA256"],
    "recipe": recipe,
    "arms": [
        {"arm": arm, "order_index": order_index}
        for order_index, arm in enumerate(timing_order)
    ],
}
(result_dir / "benchmark_manifest.json").write_text(
    json.dumps(manifest, indent=2) + "\n"
)
PY
BASH

echo "[INFO] Running timing-only arms without profiling instrumentation."
"${SRUN[@]}" bash -s <<'BASH'
set -euo pipefail
cd "${CONTAINER_REPO_ROOT}"
source "${CONTAINER_RUNTIME_DIR}/log_runtime_environment.sh"
[[ "${VIRTUAL_ENV}" == "${UV_PROJECT_ENVIRONMENT}" ]]
[[ "$("${RUNTIME_PYTHON}" --version)" == "Python ${UV_PYTHON_VERSION}" ]]
expected_runtime_python=$("${RUNTIME_PYTHON}" -c 'import os, sys; print(os.path.realpath(sys.executable))')
runtime_python=$("${UV_BIN}" run --active --no-sync python -c 'import os, sys; print(os.path.realpath(sys.executable))')
runtime_prefix=$("${UV_BIN}" run --active --no-sync python -c 'import sys; print(sys.prefix)')
[[ "${runtime_python}" == "${expected_runtime_python}" ]]
[[ "${runtime_prefix}" == "${VIRTUAL_ENV}" ]]
mapfile -t COMMON_OVERRIDES < "${CONTAINER_RESULT_DIR}/common_overrides.txt"
IFS=',' read -r -a timing_arms <<< "${TIMING_ORDER}"

run_timing_arm() {
    local arm=$1
    local order_index=$2
    local arm_dir="${CONTAINER_RESULT_DIR}/timing/${order_index}-${arm}"
    local -a arm_overrides=()
    if [[ "${arm}" == "off" ]]; then
        arm_overrides=("${OFF_OVERRIDE}")
    fi

    rm -rf -- "${CONTAINER_RUNTIME_DIR}/benchmark_logs" \
        "${CONTAINER_RUNTIME_DIR}/benchmark_checkpoints" \
        "${CONTAINER_RUNTIME_DIR}/ray_timing"
    mkdir -p "${arm_dir}" "${CONTAINER_RUNTIME_DIR}/ray_timing" \
        "${CONTAINER_RUNTIME_DIR}/tmp"
    printf '%s\n' "${COMMON_OVERRIDES[@]}" "${arm_overrides[@]}" \
        > "${arm_dir}/overrides.txt"

    export BENCHMARK_ARM="${arm}"
    export BENCHMARK_ORDER_INDEX="${order_index}"
    export RAY_TMPDIR="${CONTAINER_RUNTIME_DIR}/ray_timing"
    export TMPDIR="${CONTAINER_RUNTIME_DIR}/tmp"
    unset NRL_NSYS_WORKER_PATTERNS
    unset NRL_NSYS_PROFILE_STEP_RANGE
    unset NRL_NSYS_EXTRA_OPTIONS

    set +e
    "${UV_BIN}" run --active --no-sync examples/run_grpo.py \
        --config "${RECIPE}" \
        "${COMMON_OVERRIDES[@]}" "${arm_overrides[@]}" \
        2>&1 | tee "${arm_dir}/grpo.log"
    local grpo_exit_code=${PIPESTATUS[0]}
    set -e
    if [[ ${grpo_exit_code} -ne 0 ]]; then
        find "${RAY_TMPDIR}" -type f -print > "${arm_dir}/ray_artifacts.txt" 2>/dev/null || true
        return "${grpo_exit_code}"
    fi

    "${UV_BIN}" run --active --no-sync tests/json_dump_tb_logs.py \
        "${CONTAINER_RUNTIME_DIR}/benchmark_logs" \
        --output_path "${arm_dir}/metrics.json"

    "${UV_BIN}" run --active --no-sync python - "${arm_dir}" <<'PY'
import csv
import json
import os
import statistics
import sys
from pathlib import Path

arm_dir = Path(sys.argv[1])
metrics = json.loads((arm_dir / "metrics.json").read_text())
WARMUP_UPDATES = int(os.environ["WARMUP_UPDATES"])
MEASURED_UPDATES = int(os.environ["MEASURED_UPDATES"])
POLICY_TIME_METRIC = "timing/train/policy_training"
TOKEN_COUNT_METRIC = "train/total_num_tokens"
VALID_TOKEN_COUNT_METRIC = "train/global_valid_toks"
NORMALIZED_THROUGHPUT_METRIC = "performance/policy_training_tokens_per_sec_per_gpu"


def ordered_values(name: str) -> list[tuple[int, float]]:
    series = metrics.get(name, {})
    return sorted((int(step), float(value)) for step, value in series.items())


def measured_points(name: str) -> list[tuple[int, float]]:
    points = ordered_values(name)[WARMUP_UPDATES:]
    assert len(points) == MEASURED_UPDATES, {
        "metric": name,
        "warmup_updates": WARMUP_UPDATES,
        "measured_updates_expected": MEASURED_UPDATES,
        "measured_values_found": len(points),
    }
    return points


policy_points = measured_points(POLICY_TIME_METRIC)
token_points = measured_points(TOKEN_COUNT_METRIC)
valid_token_points = measured_points(VALID_TOKEN_COUNT_METRIC)
throughput_points = measured_points(NORMALIZED_THROUGHPUT_METRIC)
measured_steps = [step for step, _ in policy_points]
assert [step for step, _ in token_points] == measured_steps
assert [step for step, _ in valid_token_points] == measured_steps
assert [step for step, _ in throughput_points] == measured_steps
measured_values = [value for _, value in policy_points]

measured_step_workload = [
    {
        "step": step,
        "policy_training_seconds": policy_points[index][1],
        "total_num_tokens": token_points[index][1],
        "global_valid_toks": valid_token_points[index][1],
        "normalized_policy_training_tokens_per_sec_per_gpu": throughput_points[index][1],
    }
    for index, step in enumerate(measured_steps)
]

timing_series = {
    name: [
        {"step": step, "seconds": value}
        for step, value in ordered_values(name)[WARMUP_UPDATES:]
    ]
    for name in sorted(metrics)
    if name.startswith("timing/")
}
raw = {
    "run_id": os.environ["RUN_ID"],
    "arm": os.environ["BENCHMARK_ARM"],
    "order_index": int(os.environ["BENCHMARK_ORDER_INDEX"]),
    "warmup_updates": WARMUP_UPDATES,
    "measured_updates": MEASURED_UPDATES,
    "policy_training_seconds": measured_values,
    "policy_training_median_seconds": statistics.median(measured_values),
    "token_count_metric": TOKEN_COUNT_METRIC,
    "valid_token_count_metric": VALID_TOKEN_COUNT_METRIC,
    "normalized_throughput_metric": NORMALIZED_THROUGHPUT_METRIC,
    "measured_step_workload": measured_step_workload,
    "timing_series": timing_series,
}
(arm_dir / "raw_timing.json").write_text(json.dumps(raw, indent=2) + "\n")

with (arm_dir / "raw_timing.csv").open("w", newline="") as csv_file:
    writer = csv.writer(csv_file)
    writer.writerow(
        [
            "run_id",
            "arm",
            "order_index",
            "step",
            "policy_training_seconds",
            "total_num_tokens",
            "global_valid_toks",
            "normalized_policy_training_tokens_per_sec_per_gpu",
        ]
    )
    for item in measured_step_workload:
        writer.writerow(
            [
                raw["run_id"],
                raw["arm"],
                raw["order_index"],
                item["step"],
                item["policy_training_seconds"],
                item["total_num_tokens"],
                item["global_valid_toks"],
                item["normalized_policy_training_tokens_per_sec_per_gpu"],
            ]
        )
PY
}

for order_index in "${!timing_arms[@]}"; do
    run_timing_arm "${timing_arms[order_index]}" "${order_index}"
done

"${UV_BIN}" run --active --no-sync python - <<'PY'
import json
import os
import statistics
from pathlib import Path

result_dir = Path(os.environ["CONTAINER_RESULT_DIR"])
POLICY_TIME_METRIC = "timing/train/policy_training"
TOKEN_COUNT_METRIC = "train/total_num_tokens"
NORMALIZED_THROUGHPUT_METRIC = "performance/policy_training_tokens_per_sec_per_gpu"
arm_results = [
    json.loads(path.read_text())
    for path in sorted((result_dir / "timing").glob("*-*/raw_timing.json"))
]
assert {result["arm"] for result in arm_results} == {"on", "off"}, arm_results
by_arm = {result["arm"]: result for result in arm_results}
on_values = by_arm["on"]["policy_training_seconds"]
off_values = by_arm["off"]["policy_training_seconds"]
on_workload = [
    item["total_num_tokens"] for item in by_arm["on"]["measured_step_workload"]
]
off_workload = [
    item["total_num_tokens"] for item in by_arm["off"]["measured_step_workload"]
]
on_throughput = [
    item["normalized_policy_training_tokens_per_sec_per_gpu"]
    for item in by_arm["on"]["measured_step_workload"]
]
off_throughput = [
    item["normalized_policy_training_tokens_per_sec_per_gpu"]
    for item in by_arm["off"]["measured_step_workload"]
]
workload_equality_observed = on_workload == off_workload
summary = {
    "run_id": arm_results[0]["run_id"],
    "timing_order": [
        result["arm"] for result in sorted(arm_results, key=lambda item: item["order_index"])
    ],
    "raw_timing_files": [
        str(path.relative_to(result_dir))
        for path in sorted((result_dir / "timing").glob("*-*/raw_timing.json"))
    ],
    "median_policy_training_seconds": {
        "on": statistics.median(on_values),
        "off": statistics.median(off_values),
    },
    "workload_metric": TOKEN_COUNT_METRIC,
    "workload_equality_required": False,
    "workload_equality_observed": workload_equality_observed,
    "measured_total_num_tokens": {"on": on_workload, "off": off_workload},
    "primary_metric": NORMALIZED_THROUGHPUT_METRIC,
    "primary_metric_normalized_by": "train/total_num_tokens and training GPU count",
    "median_normalized_throughput": {
        "on": statistics.median(on_throughput),
        "off": statistics.median(off_throughput),
    },
    "primary_on_over_off_speedup": statistics.median(on_throughput)
    / statistics.median(off_throughput),
    "secondary_metric": POLICY_TIME_METRIC,
    "secondary_metric_confounded_by_live_workload": True,
    "secondary_policy_time_on_over_off_speedup": statistics.median(off_values)
    / statistics.median(on_values),
}
(result_dir / "timing_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
PY
BASH

if [[ "${PROFILE_ENABLED}" == "1" ]]; then
    echo "[INFO] Timing is complete; running separate profiling-only arms."
    "${SRUN[@]}" bash -s <<'BASH'
set -euo pipefail
cd "${CONTAINER_REPO_ROOT}"
source "${CONTAINER_RUNTIME_DIR}/log_runtime_environment.sh"
[[ "${VIRTUAL_ENV}" == "${UV_PROJECT_ENVIRONMENT}" ]]
[[ "$("${RUNTIME_PYTHON}" --version)" == "Python ${UV_PYTHON_VERSION}" ]]
expected_runtime_python=$("${RUNTIME_PYTHON}" -c 'import os, sys; print(os.path.realpath(sys.executable))')
runtime_python=$("${UV_BIN}" run --active --no-sync python -c 'import os, sys; print(os.path.realpath(sys.executable))')
runtime_prefix=$("${UV_BIN}" run --active --no-sync python -c 'import sys; print(sys.prefix)')
[[ "${runtime_python}" == "${expected_runtime_python}" ]]
[[ "${runtime_prefix}" == "${VIRTUAL_ENV}" ]]
mapfile -t common_timing_overrides < "${CONTAINER_RESULT_DIR}/common_overrides.txt"
profile_overrides=()
for override in "${common_timing_overrides[@]}"; do
    if [[ "${override}" != grpo.max_num_steps=* ]]; then
        profile_overrides+=("${override}")
    fi
done
profile_overrides+=("grpo.max_num_steps=2")
IFS=',' read -r -a timing_arms <<< "${TIMING_ORDER}"

run_profile_arm() {
    local arm=$1
    local order_index=$2
    local arm_dir="${CONTAINER_RESULT_DIR}/profiles/${order_index}-${arm}"
    local -a arm_overrides=()
    if [[ "${arm}" == "off" ]]; then
        arm_overrides=("${OFF_OVERRIDE}")
    fi

    rm -rf -- "${CONTAINER_RUNTIME_DIR}/benchmark_logs" \
        "${CONTAINER_RUNTIME_DIR}/benchmark_checkpoints" \
        "${CONTAINER_RUNTIME_DIR}/ray_profile"
    mkdir -p "${arm_dir}/nsight" "${CONTAINER_RUNTIME_DIR}/ray_profile" \
        "${CONTAINER_RUNTIME_DIR}/tmp"
    export RAY_TMPDIR="${CONTAINER_RUNTIME_DIR}/ray_profile"
    export TMPDIR="${CONTAINER_RUNTIME_DIR}/tmp"
    export NRL_NSYS_WORKER_PATTERNS="megatron_policy_worker"
    export NRL_NSYS_PROFILE_STEP_RANGE="1:2"
    export NRL_NSYS_EXTRA_OPTIONS='{"cuda-memory-usage":"true","cpuctxsw":"none"}'

    set +e
    "${UV_BIN}" run --active --no-sync examples/run_grpo.py \
        --config "${RECIPE}" \
        "${profile_overrides[@]}" "${arm_overrides[@]}" \
        2>&1 | tee "${arm_dir}/grpo_profile.log"
    local grpo_exit_code=${PIPESTATUS[0]}

    : > "${arm_dir}/kernel_evidence.txt"
    while IFS= read -r -d "" report; do
        cp -a "${report}" "${arm_dir}/nsight/"
        printf '\n===== %s =====\n' "${report}" >> "${arm_dir}/kernel_evidence.txt"
        nsys stats --report cuda_gpu_kern_sum "${report}" \
            >> "${arm_dir}/kernel_evidence.txt" 2>&1 || true
    done < <(find "${RAY_TMPDIR}" -type f -path '*/logs/nsight/*.nsys-rep' -print0 2>/dev/null)
    find "${RAY_TMPDIR}" -type f -print > "${arm_dir}/ray_artifacts.txt" 2>/dev/null || true
    local report_count
    local profile_artifact_exit_code=0
    report_count=$(find "${arm_dir}/nsight" -type f -name "*.nsys-rep" | wc -l | tr -d " ")
    if ((report_count < 1)); then
        echo "[ERROR] Profiling arm ${arm} produced no .nsys-rep artifact." >&2
        profile_artifact_exit_code=1
    fi
    if [[ ! -s "${arm_dir}/kernel_evidence.txt" ]]; then
        echo "[ERROR] Profiling arm ${arm} produced empty kernel evidence." >&2
        profile_artifact_exit_code=1
    fi
    printf '{\n  "arm": "%s",\n  "order_index": %d,\n  "nsight_report_count": %d,\n  "kernel_evidence": "%s"\n}\n' \
        "${arm}" "${order_index}" "${report_count}" "kernel_evidence.txt" \
        > "${arm_dir}/profile_summary.json"
    set -e
    if [[ ${grpo_exit_code} -ne 0 ]]; then
        return "${grpo_exit_code}"
    fi
    return "${profile_artifact_exit_code}"
}

for order_index in "${!timing_arms[@]}"; do
    run_profile_arm "${timing_arms[order_index]}" "${order_index}"
done
BASH
fi

echo "[PASS] Matched CuTeDSL ON/OFF benchmark completed: ${RESULT_DIR}"
