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

import re
from pathlib import Path

EXPERIMENT_DIR = Path(__file__).parents[1] / "experiments/cutedsl_qwen3_30ba3b_oci_1n4g"
SCRIPT = (EXPERIMENT_DIR / "submit_oci_hsg.sh").read_text()
README = (EXPERIMENT_DIR / "README.md").read_text()
SMOKE_SCRIPT = (
    Path(__file__).parents[1] / "scripts/smoke_nemo_rl_container.sbatch"
).read_text()


def test_wrapper_runs_complete_locked_linux_validation_gate() -> None:
    required_fragments = (
        "Copyright (c) 2026, NVIDIA CORPORATION.",
        '"${UV_BIN}" sync --locked --extra mcore --group test --group dev',
        '"${UV_BIN}" lock --check',
        "3rdparty/Megatron-Bridge-workspace/Megatron-Bridge/tests/unit_tests/models/test_param_mapping.py",
        "3rdparty/Megatron-Bridge-workspace/Megatron-Bridge/tests/unit_tests/training/test_checkpointing.py",
        "tests/unit/models/megatron/test_megatron_setup.py",
        "tests/unit/models/megatron/test_community_import.py",
        "tests/test_cutedsl_policy_recipe.py",
        "pyrefly check",
        "parent_precommit.log",
        "bridge_precommit.log",
    )
    for fragment in required_fragments:
        assert fragment in SCRIPT, fragment
    assert SCRIPT.index('"${UV_BIN}" lock --check') < SCRIPT.index(
        "Running the four-GPU"
    )


def test_wrapper_runs_transformer_engine_forward_backward_on_each_gpu() -> None:
    assert "te.Linear" in SCRIPT
    assert ".backward()" in SCRIPT
    assert "parameter.grad" in SCRIPT
    assert "torch.isfinite(output)" in SCRIPT


def test_wrapper_enforces_clean_pushed_source_and_safe_assignments() -> None:
    assert "status --porcelain" in SCRIPT
    assert "--untracked-files=normal" in SCRIPT
    assert "@{upstream}" in SCRIPT
    assert '"upstream_ref"' in SCRIPT
    assert '"upstream_sha"' in SCRIPT
    assert "submodule status --recursive" in SCRIPT
    assert "line:0:1" in SCRIPT or "${line:0:1}" in SCRIPT
    assert not re.search(r'^\s*readonly\s+\w+="\$\(', SCRIPT, re.MULTILINE)


def test_wrapper_discovers_repository_from_slurm_submit_directory() -> None:
    assert "${SLURM_SUBMIT_DIR:?" in SCRIPT
    assert 'git -C "${SLURM_SUBMIT_DIR}" rev-parse --show-toplevel' in SCRIPT
    assert (
        'EXPERIMENT_DIR="${REPO_ROOT}/experiments/cutedsl_qwen3_30ba3b_oci_1n4g"'
    ) in SCRIPT
    assert "BASH_SOURCE" not in SCRIPT


def test_wrapper_bootstraps_pinned_run_local_uv_and_python() -> None:
    required_fragments = (
        'export UV_VERSION="0.11.6"',
        'export UV_PYTHON_VERSION="3.13.13"',
        "export UV_NO_MODIFY_PATH=1",
        'export UV_INSTALL_DIR="${CONTAINER_RUNTIME_DIR}/uv-bin"',
        'export UV_PYTHON_INSTALL_DIR="${CONTAINER_RUNTIME_DIR}/uv-python"',
        'export UV_BIN="${UV_INSTALL_DIR}/uv"',
        'curl -LsSf "https://astral.sh/uv/${UV_VERSION}/install.sh" | sh',
        '[[ "$("${UV_BIN}" --version)" == "uv ${UV_VERSION} "* ]]',
        '"${UV_BIN}" python install "${UV_PYTHON_VERSION}"',
        '"${UV_BIN}" sync --locked --extra mcore --group test --group dev',
        '"${UV_BIN}" lock --check',
        '"${UV_BIN}" run --no-sync pytest',
        '"${UV_BIN}" run --no-sync pyrefly check',
        '"${UV_BIN}" run --no-sync pre-commit run --files',
        '"${UV_BIN}" run --no-sync examples/run_grpo.py',
        '"${UV_BIN}" run --no-sync tests/json_dump_tb_logs.py',
    )
    for fragment in required_fragments:
        assert fragment in SCRIPT, fragment
    assert not re.search(r"^\s*uv\s", SCRIPT, re.MULTILINE)
    assert "${UV_PROJECT_ENVIRONMENT}/bin/pre-commit" not in SCRIPT
    assert ".bashrc" not in SCRIPT
    assert SCRIPT.index("export UV_NO_MODIFY_PATH=1") < SCRIPT.index(
        'curl -LsSf "https://astral.sh/uv/${UV_VERSION}/install.sh" | sh'
    )


def test_wrapper_keeps_high_churn_runtime_off_lustre_across_srun_steps() -> None:
    required_fragments = (
        'readonly HOST_RUNTIME_DIR="/tmp/${USER}/nemo-2606-cutedsl/${RUN_ID}"',
        'readonly CONTAINER_RUNTIME_DIR="/runtime"',
        '${HOST_RUNTIME_DIR}:${CONTAINER_RUNTIME_DIR}',
        'export UV_INSTALL_DIR="${CONTAINER_RUNTIME_DIR}/uv-bin"',
        'export UV_PYTHON_INSTALL_DIR="${CONTAINER_RUNTIME_DIR}/uv-python"',
        'export UV_PROJECT_ENVIRONMENT="${CONTAINER_RUNTIME_DIR}/venv"',
        'export UV_CACHE_DIR="${CONTAINER_RUNTIME_DIR}/uv-cache"',
        'export NEMO_RL_VENV_DIR="${CONTAINER_RUNTIME_DIR}/worker_venvs"',
        'export TORCH_EXTENSIONS_DIR="${CONTAINER_RUNTIME_DIR}/torch_extensions"',
        'export TRITON_CACHE_DIR="${CONTAINER_RUNTIME_DIR}/triton_cache"',
        'export RAY_TMPDIR="${CONTAINER_RUNTIME_DIR}/ray_tmp"',
        'export TMPDIR="${CONTAINER_RUNTIME_DIR}/tmp"',
        'readonly RAY_LOG_DIR="${CONTAINER_RESULT_DIR}/ray_logs"',
        'rm -rf --one-file-system -- "${HOST_RUNTIME_DIR}"',
    )
    for fragment in required_fragments:
        assert fragment in SCRIPT, fragment
    assert 'export UV_PROJECT_ENVIRONMENT="${CONTAINER_RESULT_DIR}/venv"' not in SCRIPT
    assert 'export UV_CACHE_DIR="${CONTAINER_RESULT_DIR}/uv_cache"' not in SCRIPT
    assert 'export RAY_TMPDIR="${CONTAINER_RESULT_DIR}/ray_tmp"' not in SCRIPT


def test_smoke_uses_result_mount_for_logs_and_image_prebuilt_environment() -> None:
    required_fragments = (
        "${CONTAINER_SMOKE_DIR:?",
        '[[ "${CONTAINER_SMOKE_DIR}" != /* ]]',
        "${CONTAINER_SMOKE_DIR}:/results",
        'readonly python_bin="/opt/nemo_rl_venv/bin/python"',
        '[[ "$("${python_bin}" --version)" == "Python 3.13.13" ]]',
        '"${python_bin}" - <<\'PY\'',
        "import cutlass",
        "from cutlass import cute",
        "import nemo_rl",
        "import transformer_engine.pytorch as te",
        "for device_index in range(actual_devices):",
        "torch.isfinite(outputs).all()",
        "torch.isfinite(inputs.grad).all()",
    )
    for fragment in required_fragments:
        assert fragment in SMOKE_SCRIPT, fragment
    assert ".bashrc" not in SMOKE_SCRIPT


def test_smoke_has_bounded_runtime_and_proves_gb200_allocation() -> None:
    required_fragments = (
        "#SBATCH --time=00:10:00",
        'device_names = [torch.cuda.get_device_name(index) for index in range(actual_devices)]',
        'assert all("GB200" in device_name for device_name in device_names)',
        'tee "${CONTAINER_SMOKE_DIR}/smoke.log"',
    )
    for fragment in required_fragments:
        assert fragment in SMOKE_SCRIPT, fragment


def test_smoke_does_not_create_a_second_uv_environment() -> None:
    assert "export UV_PROJECT_ENVIRONMENT=/results/venv" not in SMOKE_SCRIPT
    assert "export UV_CACHE_DIR=/results/uv-cache" not in SMOKE_SCRIPT
    assert "/tmp/nemo-rl-smoke-" not in SMOKE_SCRIPT


def test_smoke_uses_image_prebuilt_environment_without_sync() -> None:
    required_fragments = (
        'readonly python_bin="/opt/nemo_rl_venv/bin/python"',
        '[[ "$("${python_bin}" --version)" == "Python 3.13.13" ]]',
        '"${python_bin}" - <<\'PY\'',
    )
    for fragment in required_fragments:
        assert fragment in SMOKE_SCRIPT, fragment
    assert "uv sync" not in SMOKE_SCRIPT
    assert "astral.sh/uv" not in SMOKE_SCRIPT


def test_wrapper_preserves_failure_artifacts_and_enforces_metrics() -> None:
    required_fragments = (
        "RAY_TMPDIR",
        "set +e",
        "PIPESTATUS[0]",
        "optimizer_update_successful",
        '"gen_kl_error_max": 0.05',
        '"token_mult_prob_error_max": 2.0',
        '.endswith(".mem_gb")',
        "assert policy_times",
        "assert tokens_per_second",
        "assert peak_memory_metrics",
    )
    for fragment in required_fragments:
        assert fragment in SCRIPT, fragment
    capture_index = SCRIPT.index("PIPESTATUS[0]")
    artifact_index = SCRIPT.index('find "${RAY_TMPDIR}"')
    exit_index = SCRIPT.index('exit "${grpo_exit_code}"')
    assert capture_index < artifact_index < exit_index


def test_readme_matches_enforced_gate_and_requeue_paths() -> None:
    required_fragments = (
        "$JOB_ID-r$SLURM_RESTART_COUNT",
        "train/optimizer_update_successful",
        "train/gen_kl_error",
        "< 0.05",
        "train/token_mult_prob_error",
        "< 2.0",
        "tests/functional/grpo_vllm_mxfp8_rollout_gb200.sh",
        "post-warmup",
        ".mem_gb",
    )
    for fragment in required_fragments:
        assert fragment in README, fragment
    assert "direct tensor equality" not in README.lower()
