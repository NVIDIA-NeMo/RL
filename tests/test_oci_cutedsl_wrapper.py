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

import json
import os
import re
import subprocess
import sys
from pathlib import Path

EXPERIMENT_DIR = Path(__file__).parents[1] / "experiments/cutedsl_qwen3_30ba3b_oci_1n4g"
SCRIPT = (EXPERIMENT_DIR / "run_cutedsl_functional.sbatch").read_text()
README = (EXPERIMENT_DIR / "README.md").read_text()
RECIPE = (
    Path(__file__).parents[1]
    / "examples/configs/recipes/llm/performance/grpo-qwen3-30ba3b-1n4g-megatron-mxfp8-cutedsl.yaml"
).read_text()
BENCHMARK_PATH = EXPERIMENT_DIR / "run_cutedsl_matrix.sbatch"
BENCHMARK_SCRIPT = BENCHMARK_PATH.read_text() if BENCHMARK_PATH.exists() else ""
BENCHMARK_SUBMIT_PATH = EXPERIMENT_DIR / "submit_cutedsl_ab_replicates.sh"
BENCHMARK_SUBMIT_SCRIPT = (
    BENCHMARK_SUBMIT_PATH.read_text() if BENCHMARK_SUBMIT_PATH.exists() else ""
)
SMOKE_SCRIPT = (
    Path(__file__).parents[1] / "scripts/smoke_nemo_rl_container.sbatch"
).read_text()


def _focused_srun_invocation_source() -> str:
    phase_start = SCRIPT.index("cutedsl_write_event focused_tests start")
    invocation_start = SCRIPT.index('"${SRUN[@]}" bash', phase_start)
    invocation_end = SCRIPT.index(
        "\ncutedsl_write_event focused_tests pass", invocation_start
    )
    return SCRIPT[invocation_start:invocation_end]


def test_focused_srun_payload_is_single_parseable_stdin_script(tmp_path: Path) -> None:
    """The exact focused invocation passes one intact, newline-preserving script."""
    capture_script = tmp_path / "capture.py"
    capture_path = tmp_path / "capture.json"
    capture_script.write_text(
        "import json, os, pathlib, sys\n"
        "pathlib.Path(os.environ['CAPTURE_PATH']).write_text(\n"
        "    json.dumps({'argv': sys.argv[1:], 'stdin': sys.stdin.read()})\n"
        ")\n"
    )
    command = (
        "set -eo pipefail\n"
        'SRUN=("$1" "$2")\n'
        "export CAPTURE_PATH=$3\n" + _focused_srun_invocation_source() + "\n"
    )
    captured = subprocess.run(
        [
            "bash",
            "-c",
            command,
            "focused-srun-capture",
            sys.executable,
            str(capture_script),
            str(capture_path),
        ],
        capture_output=True,
        text=True,
    )

    assert captured.returncode == 0, captured.stderr
    invocation = json.loads(capture_path.read_text())
    assert invocation["argv"] == ["bash", "-s"], invocation["argv"]
    payload = invocation["stdin"]
    assert payload.startswith('export TMPDIR="${CONTAINER_RUNTIME_DIR}/tmp"\n')
    parsed = subprocess.run(
        ["bash", "-n"], input=payload, capture_output=True, text=True
    )
    assert parsed.returncode == 0, parsed.stderr

    bridge_printf_start = payload.index("        printf '%s\\n' \\\n")
    bridge_printf_end = payload.index(
        '        "${RUNTIME_PYTHON}" -m pytest', bridge_printf_start
    )
    bridge_diagnostics = subprocess.run(
        ["bash", "-c", payload[bridge_printf_start:bridge_printf_end]],
        env={**os.environ, "UV_PYTHON_VERSION": "3.13.13"},
        capture_output=True,
        text=True,
    )
    assert bridge_diagnostics.returncode == 0, bridge_diagnostics.stderr
    assert bridge_diagnostics.stdout.splitlines() == [
        "[INFO] Bridge MSC exclusions: exactly 2 tests are deselected because "
        "multi-storage-client~=0.50 provides only CPython 3.12 wheels, while the "
        "locked NeMo runtime is Python 3.13.13.",
        "[INFO] Expected Bridge pytest summary: 203 passed, 2 deselected.",
    ]


def test_wrapper_runs_complete_locked_linux_validation_gate() -> None:
    required_fragments = (
        "Copyright (c) 2026, NVIDIA CORPORATION.",
        '"${UV_BIN}" sync --locked --extra mcore --group test --group dev',
        '"${UV_BIN}" lock --check',
        "tests/unit_tests/models/test_param_mapping.py",
        "tests/unit_tests/training/test_checkpointing.py",
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


def test_wrapper_isolates_bridge_and_nemo_rl_pytest_roots() -> None:
    focused_gate = SCRIPT[
        SCRIPT.index("cutedsl_write_event focused_tests start") : SCRIPT.index(
            '"${UV_BIN}" run --active --no-sync pyrefly check'
        )
    ]
    bridge_subshell = """bridge_root="${CONTAINER_REPO_ROOT}/3rdparty/Megatron-Bridge-workspace/Megatron-Bridge"
(
    cd "${bridge_root}"
    {"""  # noqa: E501
    nemo_rl_pytest = """(
    cd "${CONTAINER_REPO_ROOT}"
    trap preserve_nemo_unit_results EXIT
    "${UV_BIN}" run --active --no-sync pytest \\
        tests/unit/models/megatron/test_megatron_setup.py \\
        tests/unit/models/megatron/test_community_import.py \\
        tests/test_cutedsl_policy_recipe.py \\
        -q"""
    lifecycle_pytest = """"${UV_BIN}" run --active --no-sync pytest \\
        tests/unit/algorithms/test_grpo.py \\
        tests/unit/models/generation/test_vllm_generation.py \\
        tests/unit/models/policy/test_megatron_worker.py \\
        -k "${lifecycle_test_filter}" \\
        -q
) 2>&1 | tee -a "${CONTAINER_RESULT_DIR}/focused_tests.log"""

    assert "set -euo pipefail" in focused_gate
    assert bridge_subshell in focused_gate
    assert nemo_rl_pytest in focused_gate
    assert lifecycle_pytest in focused_gate
    assert focused_gate.index(bridge_subshell) < focused_gate.index(nemo_rl_pytest)
    assert focused_gate.index(nemo_rl_pytest) < focused_gate.index(lifecycle_pytest)
    assert "test_megatron_offload_emits_host_memory_at_oom_boundaries" in focused_gate
    assert "test_megatron_offload_memory_diagnostics_are_best_effort" in focused_gate
    assert "test_sync_sleep_memory_diagnostics_are_best_effort" in focused_gate
    assert focused_gate.count('| tee "${CONTAINER_RESULT_DIR}/focused_tests.log"') == 1
    assert (
        focused_gate.count('| tee -a "${CONTAINER_RESULT_DIR}/focused_tests.log"') == 1
    )


def test_wrapper_bounds_bridge_msc_deselections_to_locked_runtime() -> None:
    focused_gate = SCRIPT[
        SCRIPT.index("cutedsl_write_event focused_tests start") : SCRIPT.index(
            '"${UV_BIN}" run --active --no-sync pyrefly check'
        )
    ]
    bridge_gate = focused_gate[
        focused_gate.index('bridge_root="${CONTAINER_REPO_ROOT}') : focused_gate.index(
            '(\n    cd "${CONTAINER_REPO_ROOT}"'
        )
    ]
    deselected_nodes = (
        "tests/unit_tests/training/test_checkpointing.py::TestCheckpointUtilities::test_ensure_directory_exists_with_msc_url",
        "tests/unit_tests/training/test_checkpointing.py::TestLoadCheckpointFromPathDirectIterDir::test_fsdp_dtensor_skips_tracker_resolution_with_msc",
    )

    assert '"${RUNTIME_PYTHON}" -m pytest' in bridge_gate
    assert '"${UV_BIN}" run --active --no-sync pytest' not in bridge_gate
    assert bridge_gate.count("--deselect=") == len(deselected_nodes)
    for node_id in deselected_nodes:
        assert f"--deselect={node_id}" in bridge_gate
    assert "Bridge MSC exclusions: exactly 2 tests" in bridge_gate
    assert "multi-storage-client~=0.50 provides only CPython 3.12 wheels" in bridge_gate
    assert "Expected Bridge pytest summary: 203 passed, 2 deselected" in bridge_gate
    for broad_filter in (" -k ", "--ignore", "--ignore-glob", "pytest.skip"):
        assert broad_filter not in bridge_gate


def _bridge_summary_validator_source() -> str:
    start = "# CUTEDSL_BRIDGE_SUMMARY_VALIDATOR_START\n"
    end = "# CUTEDSL_BRIDGE_SUMMARY_VALIDATOR_END\n"
    assert start in SCRIPT
    assert end in SCRIPT
    return SCRIPT.split(start, 1)[1].split(end, 1)[0]


def _run_bridge_summary_validator(
    tmp_path: Path,
    name: str,
    log_text: str,
) -> subprocess.CompletedProcess[str]:
    log_path = tmp_path / f"{name}.log"
    log_path.write_text(log_text)
    command = (
        "set -euo pipefail\n"
        f"{_bridge_summary_validator_source()}\n"
        'validate_bridge_pytest_summary "$1"\n'
    )
    return subprocess.run(
        ["bash", "-c", command, "bridge-summary-validator", str(log_path)],
        capture_output=True,
        text=True,
    )


def test_bridge_summary_validator_requires_actual_terminal_counts(
    tmp_path: Path,
) -> None:
    diagnostic = "[INFO] Expected Bridge pytest summary: 203 passed, 2 deselected.\n"
    cases = (
        (
            "correct-with-warnings",
            diagnostic
            + "================ 203 passed, 2 deselected, 7 warnings in 12.34s ================\n",
            True,
        ),
        (
            "correct-without-warnings",
            diagnostic
            + "==================== 203 passed, 2 deselected in 9.8s ====================\n",
            True,
        ),
        (
            "wrong-passed",
            diagnostic
            + "================ 202 passed, 2 deselected, 1 warning in 12.34s ================\n",
            False,
        ),
        (
            "wrong-deselected",
            diagnostic
            + "==================== 203 passed, 1 deselected in 12s ====================\n",
            False,
        ),
        ("diagnostic-only-spoof", diagnostic, False),
    )

    for name, log_text, succeeds in cases:
        result = _run_bridge_summary_validator(tmp_path, name, log_text)
        assert (result.returncode == 0) is succeeds, (name, result.stderr)
        if not succeeds:
            assert "Bridge pytest terminal summary mismatch" in result.stderr
            assert len(result.stderr.splitlines()) == 1


def test_wrapper_validates_bridge_summary_before_nemo_rl_tests() -> None:
    bridge_log = '} 2>&1 | tee "${CONTAINER_RESULT_DIR}/focused_tests.log"'
    validator_call = (
        'validate_bridge_pytest_summary "${CONTAINER_RESULT_DIR}/focused_tests.log"'
    )
    nemo_rl_tests = '(\n    cd "${CONTAINER_REPO_ROOT}"'

    assert SCRIPT.index(bridge_log) < SCRIPT.index(validator_call)
    assert SCRIPT.index(validator_call) < SCRIPT.index(nemo_rl_tests)


def _parent_toml_validator_source() -> str:
    start = "# CUTEDSL_PARENT_TOML_VALIDATOR_START\n"
    end = "# CUTEDSL_PARENT_TOML_VALIDATOR_END\n"
    assert start in SCRIPT
    assert end in SCRIPT
    return SCRIPT.split(start, 1)[1].split(end, 1)[0]


def test_parent_toml_validator_fails_on_invalid_changed_file(tmp_path: Path) -> None:
    valid_paths = (tmp_path / "pyproject.toml", tmp_path / "uv.lock")
    for path in valid_paths:
        path.write_text('[project]\nname = "valid"\n')
    valid_result = subprocess.run(
        [sys.executable, "-c", _parent_toml_validator_source(), *valid_paths],
        capture_output=True,
        text=True,
    )
    assert valid_result.returncode == 0, valid_result.stderr
    for path in valid_paths:
        assert f"Valid changed parent TOML: {path}" in valid_result.stdout

    invalid_path = tmp_path / "broken.toml"
    invalid_path.write_text('[project\nname = "broken"\n')
    invalid_result = subprocess.run(
        [sys.executable, "-c", _parent_toml_validator_source(), invalid_path],
        capture_output=True,
        text=True,
    )
    assert invalid_result.returncode != 0
    assert f"Invalid changed parent TOML {invalid_path}" in invalid_result.stderr


def test_wrapper_scopes_taplo_skip_to_parent_precommit() -> None:
    parent_gate = SCRIPT[
        SCRIPT.index("mapfile -t parent_changed_files") : SCRIPT.index(
            'git -C "${bridge_root}" diff'
        )
    ]
    bridge_gate = SCRIPT[
        SCRIPT.index('git -C "${bridge_root}" diff') : SCRIPT.index(
            "cutedsl_write_event focused_tests pass"
        )
    ]
    rationale = (
        "[INFO] Parent pre-commit skips only taplo-format: mirrors-taplo v0.9.3 "
        "sdist is missing crates/taplo-lsp/Cargo.toml for CPython 3.13/aarch64; "
        "validating every changed parent TOML with locked runtime tomllib."
    )

    assert 'for changed_file in "${parent_changed_files[@]}"' in parent_gate
    assert '[[ "${changed_file}" == *.toml ]]' in parent_gate
    assert 'parent_toml_files+=("${changed_file}")' in parent_gate
    assert '"${RUNTIME_PYTHON}" - "${parent_toml_files[@]}"' in parent_gate
    assert rationale in parent_gate
    assert SCRIPT.index('"${UV_BIN}" lock --check') < SCRIPT.index(rationale)
    assert 'parent_precommit_skip="${SKIP:-}"' in parent_gate
    assert 'parent_precommit_skip+=",taplo-format"' in parent_gate
    assert 'parent_precommit_skip="taplo-format"' in parent_gate
    parent_precommit = """(
    export SKIP="${parent_precommit_skip}"
    "${UV_BIN}" run --active --no-sync pre-commit run --files \\
        "${parent_changed_files[@]}"
) 2>&1 | tee -a "${CONTAINER_RESULT_DIR}/parent_precommit.log"""
    assert parent_precommit in parent_gate
    assert parent_gate.count("taplo-format") == 3
    assert '| tee "${CONTAINER_RESULT_DIR}/parent_precommit.log"' in parent_gate
    assert "SKIP=" not in bridge_gate
    for broad_skip in ("SKIP=all", "SKIP=ruff", "SKIP=pyrefly", "SKIP=configs"):
        assert broad_skip not in SCRIPT


def _nemo_unit_results_preserver_source() -> str:
    start = "# CUTEDSL_NEMO_UNIT_RESULTS_PRESERVER_START\n"
    end = "# CUTEDSL_NEMO_UNIT_RESULTS_PRESERVER_END\n"
    assert start in SCRIPT
    assert end in SCRIPT
    return SCRIPT.split(start, 1)[1].split(end, 1)[0]


def _run_nemo_unit_results_preserver(
    tmp_path: Path,
    name: str,
    pytest_exit: int,
    *,
    fail_moves: bool = False,
) -> tuple[subprocess.CompletedProcess[str], Path, Path]:
    repo_root = tmp_path / name / "repo"
    result_dir = tmp_path / name / "result"
    (repo_root / "tests/unit/unit_results").mkdir(parents=True)
    result_dir.mkdir(parents=True)
    (repo_root / "tests/unit/unit_results.json").write_text('{"latest": true}\n')
    (repo_root / "tests/unit/unit_results/20260711_120000.json").write_text(
        '{"dated": true}\n'
    )
    command = f"""
set -uo pipefail
CONTAINER_REPO_ROOT=$1
CONTAINER_RESULT_DIR=$2
{_nemo_unit_results_preserver_source()}
if [[ $4 == 1 ]]; then
    mv() {{ return 9; }}
fi
(
    trap preserve_nemo_unit_results EXIT
    exit "$3"
)
"""
    result = subprocess.run(
        [
            "bash",
            "-c",
            command,
            "nemo-unit-results-preserver",
            str(repo_root),
            str(result_dir),
            str(pytest_exit),
            "1" if fail_moves else "0",
        ],
        capture_output=True,
        text=True,
    )
    return result, repo_root, result_dir


def test_nemo_unit_results_preserver_cleans_success_and_failure(
    tmp_path: Path,
) -> None:
    for name, pytest_exit in (("success", 0), ("failure", 17)):
        result, repo_root, result_dir = _run_nemo_unit_results_preserver(
            tmp_path, name, pytest_exit
        )
        assert result.returncode == pytest_exit, result.stderr
        assert not (repo_root / "tests/unit/unit_results.json").exists()
        assert not (repo_root / "tests/unit/unit_results").exists()
        assert (result_dir / "nemo_unit_results.json").read_text() == (
            '{"latest": true}\n'
        )
        assert (
            result_dir / "nemo_unit_results/20260711_120000.json"
        ).read_text() == '{"dated": true}\n'


def test_nemo_unit_results_preserver_keeps_pytest_error_precedence(
    tmp_path: Path,
) -> None:
    successful_pytest, _, _ = _run_nemo_unit_results_preserver(
        tmp_path, "cleanup-failure", 0, fail_moves=True
    )
    failing_pytest, _, _ = _run_nemo_unit_results_preserver(
        tmp_path, "pytest-and-cleanup-failure", 17, fail_moves=True
    )

    assert successful_pytest.returncode == 9
    assert failing_pytest.returncode == 17


def test_wrapper_installs_nemo_unit_results_exit_trap_before_pytest() -> None:
    focused_gate = SCRIPT[
        SCRIPT.index("validate_bridge_pytest_summary") : SCRIPT.index(
            '"${UV_BIN}" run --active --no-sync pyrefly check'
        )
    ]
    nemo_gate = focused_gate[focused_gate.index('(\n    cd "${CONTAINER_REPO_ROOT}"') :]
    preserver_source = _nemo_unit_results_preserver_source()

    assert "nemo_unit_results.json" in preserver_source
    assert "nemo_unit_results" in preserver_source
    assert "mv --" in preserver_source
    assert nemo_gate.index("trap preserve_nemo_unit_results EXIT") < nemo_gate.index(
        '"${UV_BIN}" run --active --no-sync pytest'
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


def test_payloads_require_and_record_effective_cluster_profile() -> None:
    for script in (SCRIPT, BENCHMARK_SCRIPT):
        assert "CUTEDSL_PROFILE_NAME" in script
        assert "CUTEDSL_IMAGE_SHA256" in script
        assert '"cluster_profile"' in script


def test_submitters_capture_source_and_payloads_reject_checkout_drift() -> None:
    required_branch = "sna/nemo-2606-cutedsl-20260710"
    for submitter in (
        (EXPERIMENT_DIR / "submit_cutedsl_functional.sh").read_text(),
        BENCHMARK_SUBMIT_SCRIPT,
    ):
        assert 'capture_cutedsl_submission_source "${REPO_ROOT}"' in submitter
    for payload in (SCRIPT, BENCHMARK_SCRIPT):
        assert 'source "${EXPERIMENT_DIR}/lib/cluster_profile.sh"' in payload
        assert "validate_cutedsl_runtime_source" in payload
    assert required_branch in (EXPERIMENT_DIR / "lib/cluster_profile.sh").read_text()


def test_payloads_have_no_static_cluster_or_image_directives() -> None:
    for script in (SCRIPT, BENCHMARK_SCRIPT):
        forbidden_fragments = (
            "#SBATCH --account=",
            "#SBATCH --partition=",
            "#SBATCH --gres=",
            "#SBATCH --time=",
            "/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_nemorl",
        )
        for fragment in forbidden_fragments:
            assert fragment not in script
        assert not re.search(r'^readonly IMAGE="/', script, re.MULTILINE)


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
        '"${UV_BIN}" run --active --no-sync pytest',
        '"${UV_BIN}" run --active --no-sync pyrefly check',
        '"${UV_BIN}" run --active --no-sync pre-commit run --files',
        '"${UV_BIN}" run --active --no-sync examples/run_grpo.py',
        '"${UV_BIN}" run --active --no-sync tests/json_dump_tb_logs.py',
    )
    for fragment in required_fragments:
        assert fragment in SCRIPT, fragment
    assert not re.search(r"^\s*uv\s", SCRIPT, re.MULTILINE)
    assert "${UV_PROJECT_ENVIRONMENT}/bin/pre-commit" not in SCRIPT
    assert ".bashrc" not in SCRIPT
    assert SCRIPT.index("export UV_NO_MODIFY_PATH=1") < SCRIPT.index(
        'curl -LsSf "https://astral.sh/uv/${UV_VERSION}/install.sh" | sh'
    )


def test_functional_and_benchmark_force_runtime_interpreter_across_srun() -> None:
    for script in (SCRIPT, BENCHMARK_SCRIPT):
        required_fragments = (
            'export VIRTUAL_ENV="${UV_PROJECT_ENVIRONMENT}"',
            'export RUNTIME_PYTHON="${UV_PROJECT_ENVIRONMENT}/bin/python"',
            '[[ "$("${RUNTIME_PYTHON}" --version)" == "Python ${UV_PYTHON_VERSION}" ]]',
            'expected_runtime_python=$("${RUNTIME_PYTHON}" -c',
            'runtime_python=$("${UV_BIN}" run --active --no-sync python -c',
            'runtime_prefix=$("${UV_BIN}" run --active --no-sync python -c',
            '[[ "${runtime_python}" == "${expected_runtime_python}" ]]',
            '[[ "${runtime_prefix}" == "${VIRTUAL_ENV}" ]]',
        )
        for fragment in required_fragments:
            assert fragment in script, fragment

        uv_run_lines = [
            line.strip() for line in script.splitlines() if '"${UV_BIN}" run ' in line
        ]
        assert uv_run_lines
        assert all(
            line.startswith('"${UV_BIN}" run --active --no-sync ')
            or '$("${UV_BIN}" run --active --no-sync ' in line
            for line in uv_run_lines
        ), uv_run_lines
        assert '"${UV_BIN}" run --no-sync ' not in script
        assert "/opt/nemo_rl_venv" not in script

        srun_count = script.count('"${SRUN[@]}"')
        version_assertion_count = script.count(
            '[[ "$("${RUNTIME_PYTHON}" --version)" == "Python ${UV_PYTHON_VERSION}" ]]'
        )
        assert version_assertion_count == srun_count


def test_pyxis_explicitly_overrides_image_runtime_environment() -> None:
    image_env = {
        "VIRTUAL_ENV": "/opt/nemo_rl_venv",
        "UV_PROJECT_ENVIRONMENT": "/opt/nemo_rl_venv",
        "NVTE_CUDA_ARCHS": "75;80;89;90;100;103;120",
        "TMPDIR": "/lustre/fsw/portfolios/stale-login/tmp",
    }
    host_env = {
        "VIRTUAL_ENV": "/runtime/venv",
        "UV_PROJECT_ENVIRONMENT": "/runtime/venv",
        "NVTE_CUDA_ARCHS": "100",
    }

    def pyxis_container_env(script: str) -> dict[str, str]:
        resolved = image_env.copy()
        match = re.search(r"^\s*--container-env=([^\s]+)$", script, re.MULTILINE)
        if match is not None:
            for name in match.group(1).split(","):
                resolved[name] = host_env[name]
        return resolved

    for script in (SCRIPT, BENCHMARK_SCRIPT):
        expected_option = (
            "--container-env=VIRTUAL_ENV,UV_PROJECT_ENVIRONMENT,NVTE_CUDA_ARCHS"
        )
        assert script.count(expected_option) == 1
        assert not re.search(r"^\s*--container-env=.*TMPDIR", script, re.MULTILINE)
        assert not re.search(
            r"^\s*--container-env=VIRTUAL_ENV\s*$", script, re.MULTILINE
        )
        resolved = pyxis_container_env(script)
        assert resolved["VIRTUAL_ENV"] == "/runtime/venv"
        assert resolved["UV_PROJECT_ENVIRONMENT"] == "/runtime/venv"
        assert resolved["NVTE_CUDA_ARCHS"] == "100"
        assert resolved["TMPDIR"] == image_env["TMPDIR"]


def _runtime_tmpdir_init_source(script: str) -> str:
    start = "# CUTEDSL_RUNTIME_TMPDIR_START\n"
    end = "# CUTEDSL_RUNTIME_TMPDIR_END\n"
    assert script.count(start) == 1
    assert script.count(end) == 1
    return script.split(start, 1)[1].split(end, 1)[0]


def _container_tmpdir_preambles(script: str) -> list[list[str]]:
    lines = script.splitlines()
    return [
        lines[index + 1 : index + 7]
        for index, line in enumerate(lines)
        if '"${SRUN[@]}" bash' in line
    ]


def test_payloads_override_stale_tmpdir_for_every_profile(tmp_path: Path) -> None:
    stale_tmpdir = "/lustre/fsw/portfolios/coreai/projects/stale-login/tmp"
    tmpdir_assertion = '[[ "${TMPDIR}" == "${CONTAINER_RUNTIME_DIR}/tmp" ]]'
    writable_assertion = '[[ -d "${TMPDIR}" && -w "${TMPDIR}" ]]'
    container_diagnostic = "[INFO] Container temporary directory:"

    for payload_name, script in (("functional", SCRIPT), ("matrix", BENCHMARK_SCRIPT)):
        for profile_name in ("pre_tyche", "aws_dfw", "lyris"):
            case_dir = tmp_path / f"{payload_name}-{profile_name}"
            env = os.environ.copy()
            env.update(
                {
                    "CONTAINER_RUNTIME_DIR": "/runtime",
                    "CUTEDSL_PROFILE_NAME": profile_name,
                    "HOST_RUNTIME_DIR": str(case_dir / "host-runtime"),
                    "RESULT_DIR": str(case_dir / "result"),
                    "TMPDIR": stale_tmpdir,
                }
            )
            result = subprocess.run(
                [
                    "bash",
                    "-c",
                    "set -euo pipefail\n"
                    + _runtime_tmpdir_init_source(script)
                    + '\nprintf "%s\\n" "${TMPDIR}"\n',
                ],
                env=env,
                capture_output=True,
                text=True,
            )
            assert result.returncode == 0, (payload_name, profile_name, result.stderr)
            expected_host_tmpdir = case_dir / "host-runtime/tmp"
            assert result.stdout.strip() == str(expected_host_tmpdir)
            assert expected_host_tmpdir.is_dir()

        preambles = _container_tmpdir_preambles(script)
        assert len(preambles) == script.count('"${SRUN[@]}"')
        for index, preamble in enumerate(preambles):
            assert preamble[0] == 'export TMPDIR="${CONTAINER_RUNTIME_DIR}/tmp"'
            container_runtime_dir = tmp_path / payload_name / str(index) / "runtime"
            (container_runtime_dir / "tmp").mkdir(parents=True)
            env = os.environ.copy()
            env.update(
                {
                    "CONTAINER_RUNTIME_DIR": str(container_runtime_dir),
                    "TMPDIR": stale_tmpdir,
                }
            )
            result = subprocess.run(
                [
                    "bash",
                    "-c",
                    "\n".join((preamble[0], preamble[1], preamble[4], preamble[5]))
                    + '\nprintf "%s\\n" "${TMPDIR}"\n',
                ],
                env=env,
                capture_output=True,
                text=True,
            )
            assert result.returncode == 0, (payload_name, index, result.stderr)
            assert result.stdout.splitlines()[-1] == str(container_runtime_dir / "tmp")

        srun_count = script.count('"${SRUN[@]}"')
        assert script.count(tmpdir_assertion) == srun_count
        assert script.count(writable_assertion) == srun_count
        assert script.count(container_diagnostic) == srun_count
        assert (
            "--container-env=VIRTUAL_ENV,UV_PROJECT_ENVIRONMENT,NVTE_CUDA_ARCHS\n"
            in script
        )
        bootstrap_start = script.index("cutedsl_write_event runtime_bootstrap start")
        assert script.index(tmpdir_assertion, bootstrap_start) < script.index(
            'curl -LsSf "https://astral.sh/uv/${UV_VERSION}/install.sh"',
            bootstrap_start,
        )
        srun_positions = [
            match.start() for match in re.finditer(re.escape('"${SRUN[@]}"'), script)
        ]
        for index, srun_position in enumerate(srun_positions):
            block_end = (
                srun_positions[index + 1]
                if index + 1 < len(srun_positions)
                else len(script)
            )
            srun_block = script[srun_position:block_end]
            first_runtime_tool = min(
                srun_block.index(marker)
                for marker in (
                    'curl -LsSf "https://astral.sh/uv/${UV_VERSION}/install.sh"',
                    'source "${CONTAINER_RUNTIME_DIR}/log_runtime_environment.sh"',
                )
                if marker in srun_block
            )
            assert srun_block.index(tmpdir_assertion) < first_runtime_tool
            assert srun_block.index(writable_assertion) < first_runtime_tool

    assert '"tmpdir": os.environ["TMPDIR"]' in SCRIPT
    assert '"TMPDIR": os.environ["TMPDIR"]' in BENCHMARK_SCRIPT


def test_runtime_diagnostics_precede_each_srun_environment_assertion() -> None:
    for script in (SCRIPT, BENCHMARK_SCRIPT):
        srun_count = script.count('"${SRUN[@]}"')
        required_diagnostics = (
            "[INFO] Runtime environment:",
            "[INFO] Runtime versions:",
            "[INFO] Runtime Python:",
            '"${VIRTUAL_ENV-<unset>}"',
            '"${UV_PROJECT_ENVIRONMENT-<unset>}"',
            '"${RUNTIME_PYTHON-<unset>}"',
            '"${UV_BIN}" --version',
            "os.path.realpath(sys.executable)",
            "print(sys.prefix)",
        )
        for fragment in required_diagnostics:
            assert fragment in script, fragment

        diagnostic_call_positions = [
            match.start()
            for match in re.finditer(
                re.escape(
                    'source "${CONTAINER_RUNTIME_DIR}/log_runtime_environment.sh"'
                ),
                script,
            )
        ]
        assertion_positions = [
            match.start()
            for match in re.finditer(
                re.escape('[[ "${VIRTUAL_ENV}" == "${UV_PROJECT_ENVIRONMENT}" ]]'),
                script,
            )
        ]

        assert len(diagnostic_call_positions) == srun_count
        assert len(assertion_positions) == srun_count
        assert all(
            diagnostic < assertion
            for diagnostic, assertion in zip(
                diagnostic_call_positions,
                assertion_positions,
                strict=True,
            )
        )


def test_payloads_pin_sm100_transformer_engine_build_architecture() -> None:
    for script in (SCRIPT, BENCHMARK_SCRIPT):
        export = 'export NVTE_CUDA_ARCHS="100"'
        assertion = '[[ "${NVTE_CUDA_ARCHS}" == "100" ]]'
        assert script.count(export) == 1
        assert script.index(export) < script.index(
            '"${UV_BIN}" sync --locked --extra mcore --group test --group dev'
        )
        assert (
            "--container-env=VIRTUAL_ENV,UV_PROJECT_ENVIRONMENT,NVTE_CUDA_ARCHS"
            in script
        )
        assert script.count(assertion) == script.count('"${SRUN[@]}"')
        assert "Runtime build environment: NVTE_CUDA_ARCHS=%s" in script
        assert "TORCH_CUDA_ARCH_LIST" not in script
        assert "CMAKE_CUDA_ARCHITECTURES" not in script

    assert '"nvte_cuda_archs": os.environ["NVTE_CUDA_ARCHS"]' in SCRIPT
    assert '"NVTE_CUDA_ARCHS": os.environ["NVTE_CUDA_ARCHS"]' in BENCHMARK_SCRIPT


def test_wrapper_keeps_high_churn_runtime_off_lustre_across_srun_steps() -> None:
    required_fragments = (
        'readonly HOST_RUNTIME_DIR="/tmp/${USER}/nemo-2606-cutedsl/${RUN_ID}"',
        'readonly CONTAINER_RUNTIME_DIR="/runtime"',
        "${HOST_RUNTIME_DIR}:${CONTAINER_RUNTIME_DIR}",
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
        "\"${python_bin}\" - <<'PY'",
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
        "device_names = [torch.cuda.get_device_name(index) for index in range(actual_devices)]",
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
        "\"${python_bin}\" - <<'PY'",
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


def test_benchmark_uses_recipe_default_on_and_one_off_override() -> None:
    selector = "policy.megatron_cfg.env_vars.NVTE_CUTEDSL_FUSED_GROUPED_MLP"
    assert 'NVTE_CUTEDSL_FUSED_GROUPED_MLP: "1"' in RECIPE
    required_fragments = (
        f'readonly CUTEDSL_SELECTOR_PATH="{selector}"',
        'readonly EXPECTED_ON_SELECTOR="1"',
        'readonly OFF_OVERRIDE="${CUTEDSL_SELECTOR_PATH}=\\"0\\""',
        "arm_overrides=()",
        'arm_overrides=("${OFF_OVERRIDE}")',
        '"${COMMON_OVERRIDES[@]}" "${arm_overrides[@]}"',
    )
    for fragment in required_fragments:
        assert fragment in BENCHMARK_SCRIPT, fragment
    assert f'{selector}="1"' not in BENCHMARK_SCRIPT


def test_benchmark_verifies_exact_matched_config_diff() -> None:
    required_fragments = (
        'on_selector == "1"',
        'off_selector == "0"',
        'differences == {selector_path: {"on": "1", "off": "0"}}',
        '"matched_config_diff.json"',
        '"effective_config_on.yaml"',
        '"effective_config_off.yaml"',
    )
    for fragment in required_fragments:
        assert fragment in BENCHMARK_SCRIPT, fragment


def test_benchmark_has_recorded_order_and_sufficient_timing_samples() -> None:
    required_fragments = (
        'WARMUP_UPDATES="${CUTEDSL_BENCHMARK_WARMUP_UPDATES:-5}"',
        'MEASURED_UPDATES="${CUTEDSL_BENCHMARK_MEASURED_UPDATES:-20}"',
        "WARMUP_UPDATES < 5",
        "MEASURED_UPDATES < 10",
        "TOTAL_UPDATES=$((WARMUP_UPDATES + MEASURED_UPDATES))",
        'TIMING_ORDER="${CUTEDSL_BENCHMARK_ORDER:-on,off}"',
        '"run_id"',
        '"timing_order"',
        '"warmup_updates"',
        '"measured_updates"',
        '"arm"',
        '"order_index"',
    )
    for fragment in required_fragments:
        assert fragment in BENCHMARK_SCRIPT, fragment


def test_benchmark_separates_profiling_and_collects_raw_timing() -> None:
    required_fragments = (
        "run_timing_arm()",
        "run_profile_arm()",
        "unset NRL_NSYS_WORKER_PATTERNS",
        "unset NRL_NSYS_PROFILE_STEP_RANGE",
        "unset NRL_NSYS_EXTRA_OPTIONS",
        'profile_overrides+=("grpo.max_num_steps=2")',
        'export NRL_NSYS_PROFILE_STEP_RANGE="1:2"',
        '"raw_timing.json"',
        '"raw_timing.csv"',
        '"timing/train/policy_training"',
        '"resolved_metric_names"',
        '"measured_component_series"',
        "measured_step_set",
        "missing measured steps",
    )
    for fragment in required_fragments:
        assert fragment in BENCHMARK_SCRIPT, fragment
    assert 'manifest["resolved_metric_names"]' in BENCHMARK_SCRIPT
    assert 'by_arm[arm]["resolved_metric_names"]' in BENCHMARK_SCRIPT
    assert BENCHMARK_SCRIPT.index("run_timing_arm()") < BENCHMARK_SCRIPT.index(
        "run_profile_arm()"
    )


def test_functional_profiles_first_update_and_keeps_post_update_refit_gate() -> None:
    """Profiling ends before the mature optimizer-state offload on step two."""
    assert (
        "Launching the three-update GRPO gate with a step-1 policy-worker "
        "Nsight capture."
    ) in SCRIPT
    assert "step-2 policy-worker Nsight capture" not in SCRIPT
    assert '"grpo.max_num_steps=3"' in SCRIPT
    assert 'export NRL_NSYS_PROFILE_STEP_RANGE="1:2"' in SCRIPT
    assert 'export NRL_NSYS_PROFILE_STEP_RANGE="2:3"' not in SCRIPT


def _functional_profile_validator_source() -> str:
    start = "# CUTEDSL_FUNCTIONAL_PROFILE_VALIDATOR_START\n"
    end = "# CUTEDSL_FUNCTIONAL_PROFILE_VALIDATOR_END\n"
    assert start in SCRIPT
    assert end in SCRIPT
    return SCRIPT.split(start, 1)[1].split(end, 1)[0]


def _run_functional_profile_validator(
    tmp_path: Path,
    *,
    report_count: int,
    evidence: str,
) -> subprocess.CompletedProcess[str]:
    result_dir = tmp_path / "functional-result"
    (result_dir / "nsight").mkdir(parents=True)
    for report_index in range(report_count):
        (result_dir / "nsight" / f"worker-{report_index}.nsys-rep").touch()
    (result_dir / "kernel_evidence.txt").write_text(evidence)
    return subprocess.run(
        [
            sys.executable,
            "-c",
            _functional_profile_validator_source(),
            str(result_dir),
        ],
        capture_output=True,
        text=True,
    )


def test_functional_profile_validator_fails_closed_and_requires_fused_signatures(
    tmp_path: Path,
) -> None:
    valid_evidence = (
        "ForwardGroupedMLP_CuTeGEMMSwiGLU_MXFP8\n"
        "BlockScaledMoEGroupedGemmDgluDbiasKernel\n"
    )
    cases = (
        ("missing-report", 0, valid_evidence, "no .nsys-rep"),
        ("empty-evidence", 1, "", "kernel evidence is empty"),
        (
            "missing-dgrad",
            1,
            "ForwardGroupedMLP_CuTeGEMMSwiGLU_MXFP8\n",
            "fused CuTeDSL dgrad",
        ),
    )
    for name, report_count, evidence, expected_error in cases:
        case_dir = tmp_path / name
        case_dir.mkdir()
        result = _run_functional_profile_validator(
            case_dir,
            report_count=report_count,
            evidence=evidence,
        )
        assert result.returncode != 0, name
        assert expected_error in result.stderr, (name, result.stderr)

    passing_dir = tmp_path / "passing"
    passing_dir.mkdir()
    passing = _run_functional_profile_validator(
        passing_dir,
        report_count=1,
        evidence=valid_evidence,
    )
    assert passing.returncode == 0, passing.stderr
    attribution = json.loads(
        (
            passing_dir / "functional-result/functional_profile_attribution.json"
        ).read_text()
    )
    assert attribution["passed"] is True
    assert attribution["nsight_report_count"] == 1
    assert attribution["fused_forward_match_count"] > 0
    assert attribution["fused_dgrad_match_count"] > 0


def test_functional_kernel_evidence_preserves_full_kernel_names() -> None:
    """CSV stats avoid the ellipsis emitted by the human-readable table."""
    assert 'nsys stats --report cuda_gpu_kern_sum --format csv "${report}"' in SCRIPT
    assert 'nsys stats --report cuda_gpu_kern_sum "${report}"' not in SCRIPT


def test_functional_profile_validator_matches_cudnn_csv_kernel_names(
    tmp_path: Path,
) -> None:
    """cuDNN CSV embeds class names inside generated CUTLASS identifiers."""
    evidence = (
        "0.1,6551808,144,kernel_cutlass_kernel_cudnngrouped_gemm"
        "grouped_gemm_glumoe_blockscaled_grouped_gemm_glu_bias"
        "BlockScaledMoEGroupedGemmGluBiasKernel_object_at__TiledMMA\n"
        "0.0,2669024,48,kernel_cutlass_kernel_cudnngrouped_gemm"
        "grouped_gemm_dglumoe_blockscaled_grouped_gemm_dglu_dbias"
        "BlockScaledMoEGroupedGemmDgluDbiasKernel_object_at__TiledMMA\n"
    )
    result = _run_functional_profile_validator(
        tmp_path,
        report_count=1,
        evidence=evidence,
    )

    assert result.returncode == 0, result.stderr


def test_functional_profile_pass_occurs_only_after_attribution_validation() -> None:
    assert SCRIPT.index("# CUTEDSL_FUNCTIONAL_PROFILE_VALIDATOR_START") < SCRIPT.index(
        "cutedsl_write_event profile pass"
    )
    assert "functional_profile_attribution.json" in SCRIPT


def test_benchmark_reuses_pinned_image_and_node_local_bootstrap() -> None:
    required_fragments = (
        'IMAGE="${CUTEDSL_IMAGE}"',
        'readonly CONTAINER_RUNTIME_DIR="/runtime"',
        'readonly HOST_RUNTIME_DIR="/tmp/${USER}/nemo-2606-cutedsl-benchmark/${RUN_ID}"',
        "${HOST_RUNTIME_DIR}:${CONTAINER_RUNTIME_DIR}",
        'export UV_VERSION="0.11.6"',
        'export UV_PYTHON_VERSION="3.13.13"',
        "export UV_NO_MODIFY_PATH=1",
        '"${UV_BIN}" sync --locked --extra mcore --group test --group dev',
        '"${UV_BIN}" lock --check',
    )
    for fragment in required_fragments:
        assert fragment in BENCHMARK_SCRIPT, fragment


def test_benchmark_requires_profile_artifacts_for_each_arm() -> None:
    required_fragments = (
        'report_count=$(find "${arm_dir}/nsight" -type f -name "*.nsys-rep"',
        "((report_count < 1))",
        '[[ ! -s "${arm_dir}/kernel_evidence.txt" ]]',
        '"nsight_report_count"',
        '"kernel_evidence"',
        '"kernel_attribution.json"',
        "FUSED_GLU_SIGNATURES",
        "FUSED_DGLU_SIGNATURES",
        "GROUPED_GEMM_SIGNATURES",
        'manifest["kernel_attribution"]',
    )
    for fragment in required_fragments:
        assert fragment in BENCHMARK_SCRIPT, fragment
    assert BENCHMARK_SCRIPT.index("# CUTEDSL_KERNEL_ATTRIBUTION_START") < (
        BENCHMARK_SCRIPT.index("cutedsl_write_event profile pass")
    )


def test_benchmark_records_workload_and_normalized_throughput_as_primary() -> None:
    required_fragments = (
        'TOKEN_COUNT_METRIC = "train/total_num_tokens"',
        'VALID_TOKEN_COUNT_METRIC = "train/global_valid_toks"',
        'NORMALIZED_THROUGHPUT_METRIC = "performance/policy_training_tokens_per_sec_per_gpu"',
        '"measured_step_workload"',
        '"total_num_tokens"',
        '"global_valid_toks"',
        '"policy_training_tokens_per_sec_per_gpu"',
        '"workload_equality_required": True',
        '"workload_equality_observed"',
        '"primary_metric": NORMALIZED_THROUGHPUT_METRIC',
        '"secondary_metric": POLICY_TIME_METRIC',
        '"secondary_metric_confounded_by_live_workload": True',
    )
    for fragment in required_fragments:
        assert fragment in BENCHMARK_SCRIPT, fragment


def _metric_extractor_source() -> str:
    start = "# CUTEDSL_METRIC_EXTRACTOR_START\n"
    end = "# CUTEDSL_METRIC_EXTRACTOR_END\n"
    assert start in BENCHMARK_SCRIPT
    assert end in BENCHMARK_SCRIPT
    return BENCHMARK_SCRIPT.split(start, 1)[1].split(end, 1)[0]


def _required_fake_metrics(*, omit: str | None = None) -> dict[str, dict[str, float]]:
    source_names = (
        "timing/train/total_step_time",
        "timing/train/generation",
        "timing/train/generation_finalize",
        "timing/train/policy_and_reference_logprobs",
        "timing/train/policy_training",
        "timing/train/prepare_for_generation/transfer_and_update_weights",
        "performance/tokens_per_sec_per_gpu",
        "performance/generation_tokens_per_sec_per_gpu",
        "performance/policy_and_reference_logprobs_tokens_per_sec_per_gpu",
        "performance/policy_training_tokens_per_sec_per_gpu",
        "train/total_num_tokens",
        "train/global_valid_toks",
    )
    metrics = {
        name: {str(step): float(step) for step in range(1, 26)}
        for name in source_names
        if name != omit
    }
    refit_name = "timing/train/prepare_for_generation/transfer_and_update_weights"
    if refit_name in metrics:
        metrics[refit_name].pop("1")
    return metrics


def _run_metric_extractor(
    tmp_path: Path,
    metrics: dict[str, dict[str, float]],
) -> subprocess.CompletedProcess[str]:
    arm_dir = tmp_path / "arm"
    arm_dir.mkdir()
    (arm_dir / "metrics.json").write_text(json.dumps(metrics))
    env = os.environ.copy()
    env.update(
        {
            "WARMUP_UPDATES": "5",
            "MEASURED_UPDATES": "20",
            "RUN_ID": "fake-run",
            "BENCHMARK_ARM": "on",
            "BENCHMARK_ORDER_INDEX": "0",
        }
    )
    return subprocess.run(
        [sys.executable, "-c", _metric_extractor_source(), str(arm_dir)],
        env=env,
        capture_output=True,
        text=True,
    )


def test_benchmark_extracts_all_required_measured_component_series(
    tmp_path: Path,
) -> None:
    result = _run_metric_extractor(tmp_path, _required_fake_metrics())
    assert result.returncode == 0, result.stderr
    raw = json.loads((tmp_path / "arm/raw_timing.json").read_text())
    canonical_names = {
        "timing/train/total_step_time",
        "timing/train/generation",
        "timing/train/generation_finalize",
        "timing/train/get_logprobs",
        "timing/train/policy_training",
        "timing/train/prepare_for_generation/transfer_and_update_weights",
        "performance/tokens_per_sec_per_gpu",
        "performance/generation_tokens_per_sec_per_gpu",
        "performance/policy_and_reference_logprobs_tokens_per_sec_per_gpu",
        "performance/policy_training_tokens_per_sec_per_gpu",
        "train/total_num_tokens",
        "train/global_valid_toks",
    }
    assert set(raw["resolved_metric_names"]) == canonical_names
    assert (
        raw["resolved_metric_names"]["timing/train/get_logprobs"]
        == "timing/train/policy_and_reference_logprobs"
    )
    assert set(raw["measured_component_series"]) == canonical_names
    for series in raw["measured_component_series"].values():
        assert [point["step"] for point in series] == list(range(6, 26))
    assert [
        item["refit_effective_tokens_per_sec_per_gpu"]
        for item in raw["measured_step_workload"]
    ] == [0.25] * 20
    csv_header = (tmp_path / "arm/raw_timing.csv").read_text().splitlines()[0]
    for column in (
        "generation_finalize_seconds",
        "e2e_tokens_per_sec_per_gpu",
        "generation_tokens_per_sec_per_gpu",
        "policy_and_reference_logprobs_tokens_per_sec_per_gpu",
        "policy_training_tokens_per_sec_per_gpu",
        "refit_effective_tokens_per_sec_per_gpu",
    ):
        assert column in csv_header


def test_benchmark_metric_extractor_fails_loudly_when_required_metric_is_missing(
    tmp_path: Path,
) -> None:
    missing = "timing/train/generation"
    result = _run_metric_extractor(
        tmp_path,
        _required_fake_metrics(omit=missing),
    )
    assert result.returncode != 0
    assert f"required metric {missing!r}" in result.stderr
    assert not (tmp_path / "arm/raw_timing.json").exists()


def test_benchmark_metric_extractor_rejects_ambiguous_aliases(tmp_path: Path) -> None:
    metrics = _required_fake_metrics()
    metrics["timing/train/get_logprobs"] = metrics[
        "timing/train/policy_and_reference_logprobs"
    ].copy()
    result = _run_metric_extractor(tmp_path, metrics)
    assert result.returncode != 0
    assert "timing/train/get_logprobs" in result.stderr
    assert "ambiguous" in result.stderr
    assert not (tmp_path / "arm/raw_timing.json").exists()


def test_benchmark_metric_extractor_rejects_missing_measured_step(
    tmp_path: Path,
) -> None:
    metrics = _required_fake_metrics()
    metrics["timing/train/generation"].pop("12")
    result = _run_metric_extractor(tmp_path, metrics)
    assert result.returncode != 0
    assert "timing/train/generation" in result.stderr
    assert "missing measured steps: [12]" in result.stderr
    assert not (tmp_path / "arm/raw_timing.json").exists()


def _timing_summarizer_source() -> str:
    start = "# CUTEDSL_TIMING_SUMMARIZER_START\n"
    end = "# CUTEDSL_TIMING_SUMMARIZER_END\n"
    assert start in BENCHMARK_SCRIPT
    assert end in BENCHMARK_SCRIPT
    return BENCHMARK_SCRIPT.split(start, 1)[1].split(end, 1)[0]


def _run_timing_summarizer(
    tmp_path: Path,
    *,
    on_workload: list[float],
    off_workload: list[float],
) -> subprocess.CompletedProcess[str]:
    result_dir = tmp_path / "summary-result"
    for order_index, (arm, workload) in enumerate(
        (("on", on_workload), ("off", off_workload))
    ):
        arm_dir = result_dir / "timing" / f"{order_index}-{arm}"
        arm_dir.mkdir(parents=True)
        raw = {
            "run_id": "fake-run",
            "arm": arm,
            "order_index": order_index,
            "policy_training_seconds": [2.0, 3.0],
            "resolved_metric_names": {},
            "measured_step_workload": [
                {
                    "total_num_tokens": tokens,
                    "policy_training_tokens_per_sec_per_gpu": 10.0,
                }
                for tokens in workload
            ],
        }
        (arm_dir / "raw_timing.json").write_text(json.dumps(raw))
    (result_dir / "benchmark_manifest.json").write_text("{}")
    env = {**os.environ, "CONTAINER_RESULT_DIR": str(result_dir)}
    return subprocess.run(
        [sys.executable, "-c", _timing_summarizer_source()],
        env=env,
        capture_output=True,
        text=True,
    )


def test_benchmark_timing_summary_enforces_identical_measured_workloads(
    tmp_path: Path,
) -> None:
    passing = _run_timing_summarizer(
        tmp_path / "matching",
        on_workload=[100.0, 200.0],
        off_workload=[100.0, 200.0],
    )
    assert passing.returncode == 0, passing.stderr
    passing_summary = json.loads(
        (tmp_path / "matching/summary-result/timing_summary.json").read_text()
    )
    assert passing_summary["workload_equality_required"] is True
    assert passing_summary["workload_equality_observed"] is True

    failing = _run_timing_summarizer(
        tmp_path / "mismatched",
        on_workload=[100.0, 200.0],
        off_workload=[100.0, 201.0],
    )
    assert failing.returncode != 0
    assert "measured workload equality failed" in failing.stderr
    failing_summary = json.loads(
        (tmp_path / "mismatched/summary-result/timing_summary.json").read_text()
    )
    assert failing_summary["workload_equality_required"] is True
    assert failing_summary["workload_equality_observed"] is False


def _kernel_attribution_source() -> str:
    start = "# CUTEDSL_KERNEL_ATTRIBUTION_START\n"
    end = "# CUTEDSL_KERNEL_ATTRIBUTION_END\n"
    assert start in BENCHMARK_SCRIPT
    assert end in BENCHMARK_SCRIPT
    return BENCHMARK_SCRIPT.split(start, 1)[1].split(end, 1)[0]


def _run_kernel_attribution(
    tmp_path: Path,
    *,
    on_evidence: str,
    off_evidence: str,
    grouped_gemm: bool = True,
    op_fuser: bool = True,
) -> subprocess.CompletedProcess[str]:
    result_dir = tmp_path / "result"
    (result_dir / "profiles/0-on").mkdir(parents=True)
    (result_dir / "profiles/1-off").mkdir(parents=True)
    (result_dir / "profiles/0-on/kernel_evidence.txt").write_text(on_evidence)
    (result_dir / "profiles/1-off/kernel_evidence.txt").write_text(off_evidence)
    config_evidence = {
        arm: {
            "policy.megatron_cfg.moe_grouped_gemm": grouped_gemm,
            "policy.megatron_cfg.use_transformer_engine_op_fuser": op_fuser,
        }
        for arm in ("on", "off")
    }
    (result_dir / "benchmark_manifest.json").write_text(
        json.dumps({"fixed_config_evidence": config_evidence})
    )
    return subprocess.run(
        [sys.executable, "-c", _kernel_attribution_source(), str(result_dir)],
        capture_output=True,
        text=True,
    )


def test_benchmark_kernel_attribution_requires_fused_on_and_grouped_both(
    tmp_path: Path,
) -> None:
    result = _run_kernel_attribution(
        tmp_path,
        on_evidence=(
            "ForwardGroupedMLP_CuTeGEMMSwiGLU_MXFP8\n"
            "BlockScaledMoEGroupedGemmDgluDbiasKernel\n"
            "BlockScaledMoEGroupedGemmQuantKernel\n"
        ),
        off_evidence="cutlass grouped_gemm universal kernel\n",
    )
    assert result.returncode == 0, result.stderr
    attribution = json.loads((tmp_path / "result/kernel_attribution.json").read_text())
    assert attribution["passed"] is True
    assert attribution["arms"]["on"]["fused_glu_match_count"] > 0
    assert attribution["arms"]["on"]["fused_dglu_match_count"] > 0
    assert attribution["arms"]["off"]["fused_glu_match_count"] == 0
    assert attribution["arms"]["off"]["fused_dglu_match_count"] == 0
    assert attribution["arms"]["on"]["grouped_gemm_match_count"] > 0
    assert attribution["arms"]["off"]["grouped_gemm_match_count"] > 0
    assert "signature_regexes" in attribution
    manifest = json.loads((tmp_path / "result/benchmark_manifest.json").read_text())
    assert manifest["kernel_attribution"]["passed"] is True
    assert (
        manifest["kernel_attribution"]["signature_regexes"]
        == attribution["signature_regexes"]
    )


def test_benchmark_kernel_attribution_writes_diagnostics_before_failure(
    tmp_path: Path,
) -> None:
    result = _run_kernel_attribution(
        tmp_path,
        on_evidence="ForwardGroupedMLP_CuTeGEMMSwiGLU_MXFP8\n",
        off_evidence="BlockScaledMoEGroupedGemmGluBiasKernel\n",
        op_fuser=False,
    )
    assert result.returncode != 0
    assert "kernel attribution failed" in result.stderr
    attribution = json.loads((tmp_path / "result/kernel_attribution.json").read_text())
    assert attribution["passed"] is False
    assert any("ON fused dGLU" in failure for failure in attribution["failures"])
    assert any("OFF fused GLU" in failure for failure in attribution["failures"])
    assert any("op fuser" in failure for failure in attribution["failures"])


def test_benchmark_submit_driver_requires_three_alternating_replicates() -> None:
    wrapper_fragments = (
        'REPLICATE_INDEX="${CUTEDSL_BENCHMARK_REPLICATE:-0}"',
        '"replicate_index"',
    )
    for fragment in wrapper_fragments:
        assert fragment in BENCHMARK_SCRIPT, fragment

    driver_fragments = (
        'REPLICATES="${CUTEDSL_BENCHMARK_REPLICATES:-3}"',
        'WARMUP_UPDATES="${CUTEDSL_BENCHMARK_WARMUP_UPDATES:-5}"',
        'PROFILE_REPLICATE="${CUTEDSL_BENCHMARK_PROFILE_REPLICATE:-0}"',
        "REPLICATES < 3",
        "PROFILE_REPLICATE >= REPLICATES",
        "replicate_index % 2 == 0",
        'timing_order="on,off"',
        'timing_order="off,on"',
        "CUTEDSL_BENCHMARK_REPLICATE=${replicate_index}",
        "CUTEDSL_BENCHMARK_ORDER=${timing_order}",
        "env -0",
        "-u CUTEDSL_BENCHMARK_ORDER",
        '--export-file="${EXPORT_PAYLOAD}"',
        '"replicate_index"',
        '"timing_order"',
        '"job_id"',
    )
    for fragment in driver_fragments:
        assert fragment in BENCHMARK_SUBMIT_SCRIPT, fragment
    assert "--export=" not in BENCHMARK_SUBMIT_SCRIPT


def test_benchmark_submit_driver_uses_exact_nul_export_payload(
    tmp_path: Path,
) -> None:
    driver = tmp_path / "submit_cutedsl_ab_replicates.sh"
    driver_text = BENCHMARK_SUBMIT_SCRIPT.replace(
        'readonly SUBMISSION_DIR="${EXPERIMENT_DIR}/results/benchmark/submissions"',
        f'readonly SUBMISSION_DIR="{tmp_path}/records"',
    )
    assert driver_text != BENCHMARK_SUBMIT_SCRIPT
    driver.write_text(driver_text)

    mock_bin = tmp_path / "bin"
    mock_bin.mkdir()
    calls_path = tmp_path / "sbatch_calls.jsonl"
    mock_sbatch = mock_bin / "sbatch"
    mock_sbatch.write_text(
        """#!/usr/bin/env python3
import json
import os
import sys
from pathlib import Path

exported = None
mode = None
for argument in sys.argv[1:]:
    if argument.startswith("--export="):
        mode = "comma"
        tokens = argument.removeprefix("--export=").split(",")
        exported = dict(os.environ) if "ALL" in tokens else {}
        for token in tokens:
            if token == "ALL":
                continue
            if "=" not in token:
                if token not in os.environ:
                    print(f"invalid --export token: {token}", file=sys.stderr)
                    raise SystemExit(64)
                exported[token] = os.environ[token]
                continue
            name, value = token.split("=", 1)
            if name not in exported:
                exported[name] = value
    elif argument.startswith("--export-file="):
        mode = "nul_file"
        payload_path = Path(argument.removeprefix("--export-file="))
        exported = {}
        for entry in payload_path.read_bytes().split(b"\\0"):
            if not entry:
                continue
            name, value = entry.decode().split("=", 1)
            exported[name] = value

if exported is None:
    print("missing export option", file=sys.stderr)
    raise SystemExit(64)

required = {
    "CUTEDSL_BENCHMARK_REPLICATE",
    "CUTEDSL_BENCHMARK_ORDER",
    "CUTEDSL_BENCHMARK_SUBMISSION_GROUP",
    "CUTEDSL_BENCHMARK_PROFILE",
    "CUTEDSL_PROFILE_NAME",
    "CUTEDSL_IMAGE",
    "CUTEDSL_IMAGE_SHA256",
    "CUTEDSL_SUBMISSION_GIT_BRANCH",
    "CUTEDSL_SUBMISSION_GIT_SHA",
}
if not required <= exported.keys():
    print("missing benchmark export", file=sys.stderr)
    raise SystemExit(64)

record = {
    "argv": [
        "--export-file=<payload>" if argument.startswith("--export-file=") else argument
        for argument in sys.argv[1:]
    ],
    "mode": mode,
    "replicate": exported["CUTEDSL_BENCHMARK_REPLICATE"],
    "order": exported["CUTEDSL_BENCHMARK_ORDER"],
    "profile_enabled": exported["CUTEDSL_BENCHMARK_PROFILE"],
    "submission_group": exported["CUTEDSL_BENCHMARK_SUBMISSION_GROUP"],
    "submission_branch": exported["CUTEDSL_SUBMISSION_GIT_BRANCH"],
    "submission_sha": exported["CUTEDSL_SUBMISSION_GIT_SHA"],
}
with Path(os.environ["MOCK_SBATCH_CALLS"]).open("a") as output:
    output.write(json.dumps(record) + "\\n")
print(f"mock-{record['replicate']}")
"""
    )
    mock_sbatch.chmod(0o755)

    env = os.environ.copy()
    env.update(
        {
            "PATH": f"{mock_bin}:{env['PATH']}",
            "MOCK_SBATCH_CALLS": str(calls_path),
            "CUTEDSL_CLUSTER_PROFILE": "pre_tyche",
            "CUTEDSL_BENCHMARK_REPLICATES": "3",
            "CUTEDSL_BENCHMARK_PROFILE_REPLICATE": "0",
            "CUTEDSL_BENCHMARK_REPLICATE": "999",
            "CUTEDSL_BENCHMARK_ORDER": "stale,order",
            "CUTEDSL_BENCHMARK_SUBMISSION_GROUP": "stale-group",
        }
    )
    result = subprocess.run(
        ["bash", str(driver)],
        cwd=Path(__file__).parents[1],
        env=env,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr

    calls = [json.loads(line) for line in calls_path.read_text().splitlines()]
    assert [call["mode"] for call in calls] == ["nul_file"] * 3
    expected_argv = [
        "--parsable",
        "--account=coreai_dlalgo_llm",
        "--partition=batch",
        "--comment=metrics",
        "--segment=1",
        "--job-name=coreai_dlalgo_llm-cutedsl.ab",
        "--time=05:00:00",
        "--export-file=<payload>",
        str(BENCHMARK_PATH),
    ]
    assert [call["argv"] for call in calls] == [expected_argv] * 3
    assert [call["replicate"] for call in calls] == ["0", "1", "2"]
    assert [call["order"] for call in calls] == [
        "on,off",
        "off,on",
        "on,off",
    ]
    assert [call["profile_enabled"] for call in calls] == ["1", "0", "0"]
    assert len({call["submission_group"] for call in calls}) == 1
    assert calls[0]["submission_group"] != "stale-group"
    assert [call["submission_branch"] for call in calls] == [
        "sna/nemo-2606-cutedsl-20260710"
    ] * 3
    expected_sha = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=Path(__file__).parents[1],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    assert [call["submission_sha"] for call in calls] == [expected_sha] * 3

    calls_path.write_text("")
    env["CUTEDSL_BENCHMARK_PROFILE"] = "0"
    disabled_result = subprocess.run(
        ["bash", str(driver)],
        cwd=Path(__file__).parents[1],
        env=env,
        capture_output=True,
        text=True,
    )
    assert disabled_result.returncode == 0, disabled_result.stderr
    disabled_calls = [json.loads(line) for line in calls_path.read_text().splitlines()]
    assert [call["profile_enabled"] for call in disabled_calls] == ["0", "0", "0"]


def test_benchmark_submit_driver_rejects_invalid_profile_replicate(
    tmp_path: Path,
) -> None:
    mock_bin = tmp_path / "bin"
    mock_bin.mkdir()
    calls_path = tmp_path / "calls.jsonl"
    mock_sbatch = mock_bin / "sbatch"
    mock_sbatch.write_text("#!/bin/bash\nexit 99\n")
    mock_sbatch.chmod(0o755)
    env = os.environ.copy()
    env.update(
        {
            "PATH": f"{mock_bin}:{env['PATH']}",
            "CUTEDSL_CLUSTER_PROFILE": "pre_tyche",
            "CUTEDSL_BENCHMARK_REPLICATES": "3",
            "CUTEDSL_BENCHMARK_PROFILE_REPLICATE": "3",
            "MOCK_SBATCH_CALLS": str(calls_path),
        }
    )
    result = subprocess.run(
        ["bash", str(BENCHMARK_SUBMIT_PATH), "--test-only"],
        cwd=Path(__file__).parents[1],
        env=env,
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    assert "CUTEDSL_BENCHMARK_PROFILE_REPLICATE" in result.stderr
    assert not calls_path.exists()
