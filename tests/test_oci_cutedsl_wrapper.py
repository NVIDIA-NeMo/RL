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
    }
    host_env = {
        "VIRTUAL_ENV": "/runtime/venv",
        "UV_PROJECT_ENVIRONMENT": "/runtime/venv",
    }

    def pyxis_container_env(script: str) -> dict[str, str]:
        resolved = image_env.copy()
        match = re.search(r"^\s*--container-env=([^\s]+)$", script, re.MULTILINE)
        if match is not None:
            for name in match.group(1).split(","):
                resolved[name] = host_env[name]
        return resolved

    for script in (SCRIPT, BENCHMARK_SCRIPT):
        expected_option = "--container-env=VIRTUAL_ENV,UV_PROJECT_ENVIRONMENT"
        assert script.count(expected_option) == 1
        assert not re.search(
            r"^\s*--container-env=VIRTUAL_ENV\s*$", script, re.MULTILINE
        )
        resolved = pyxis_container_env(script)
        assert resolved["VIRTUAL_ENV"] == "/runtime/venv"
        assert resolved["UV_PROJECT_ENVIRONMENT"] == "/runtime/venv"


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
            'os.path.realpath(sys.executable)',
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
        'WARMUP_UPDATES="${CUTEDSL_BENCHMARK_WARMUP_UPDATES:-3}"',
        'MEASURED_UPDATES="${CUTEDSL_BENCHMARK_MEASURED_UPDATES:-20}"',
        "WARMUP_UPDATES < 3",
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
        'export NRL_NSYS_PROFILE_STEP_RANGE="1:2"',
        '"raw_timing.json"',
        '"raw_timing.csv"',
        '"timing/train/policy_training"',
        "ordered_values(name)[WARMUP_UPDATES:]",
        "len(points) == MEASURED_UPDATES",
    )
    for fragment in required_fragments:
        assert fragment in BENCHMARK_SCRIPT, fragment
    assert BENCHMARK_SCRIPT.index("run_timing_arm()") < BENCHMARK_SCRIPT.index(
        "run_profile_arm()"
    )


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
    )
    for fragment in required_fragments:
        assert fragment in BENCHMARK_SCRIPT, fragment


def test_benchmark_records_workload_and_normalized_throughput_as_primary() -> None:
    required_fragments = (
        'TOKEN_COUNT_METRIC = "train/total_num_tokens"',
        'VALID_TOKEN_COUNT_METRIC = "train/global_valid_toks"',
        'NORMALIZED_THROUGHPUT_METRIC = "performance/policy_training_tokens_per_sec_per_gpu"',
        '"measured_step_workload"',
        '"total_num_tokens"',
        '"global_valid_toks"',
        '"normalized_policy_training_tokens_per_sec_per_gpu"',
        '"workload_equality_required": False',
        '"workload_equality_observed"',
        '"primary_metric": NORMALIZED_THROUGHPUT_METRIC',
        '"secondary_metric": POLICY_TIME_METRIC',
        '"secondary_metric_confounded_by_live_workload": True',
    )
    for fragment in required_fragments:
        assert fragment in BENCHMARK_SCRIPT, fragment


def test_benchmark_submit_driver_requires_three_alternating_replicates() -> None:
    wrapper_fragments = (
        'REPLICATE_INDEX="${CUTEDSL_BENCHMARK_REPLICATE:-0}"',
        '"replicate_index"',
    )
    for fragment in wrapper_fragments:
        assert fragment in BENCHMARK_SCRIPT, fragment

    driver_fragments = (
        'REPLICATES="${CUTEDSL_BENCHMARK_REPLICATES:-3}"',
        "REPLICATES < 3",
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
        'readonly SUBMISSION_DIR="${EXPERIMENT_DIR}/benchmark_submissions"',
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
    "mode": mode,
    "replicate": exported["CUTEDSL_BENCHMARK_REPLICATE"],
    "order": exported["CUTEDSL_BENCHMARK_ORDER"],
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
    assert [call["replicate"] for call in calls] == ["0", "1", "2"]
    assert [call["order"] for call in calls] == [
        "on,off",
        "off,on",
        "on,off",
    ]
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
