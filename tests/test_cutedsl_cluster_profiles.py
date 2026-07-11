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
import subprocess
from pathlib import Path

import pytest

EXPERIMENT_DIR = Path(__file__).parents[1] / "experiments/cutedsl_qwen3_30ba3b_oci_1n4g"
LOADER = EXPERIMENT_DIR / "lib/cluster_profile.sh"

EXPECTED_PROFILES = {
    "pre_tyche": {
        "account": "coreai_dlalgo_llm",
        "partition": "batch",
        "gres": "",
        "segment": "1",
        "comment": "metrics",
        "image": (
            "/lustre/fsw/coreai_dlalgo_llm/users/sna/"
            "nemo-2606-cutedsl-pre-tyche-20260711/containers/"
            "nemo_rl_nightly_20260711_2361407.sqsh"
        ),
        "image_sha256": (
            "dd32f77a0a6fb09710e31f87402f0433413b9c71120fe893297e2f46e32ce8be"
        ),
        "functional_time": "02:00:00",
        "benchmark_time": "05:00:00",
    },
    "aws_dfw": {
        "account": "nemotron_sw_post",
        "partition": "batch_long",
        "gres": "gpu:4",
        "segment": "",
        "comment": "metrics",
        "image": (
            "/lustre/fsw/portfolios/nemotron/projects/nemotron_sw_post/users/sna/"
            "containers/nemo_rl_nightly_20260711_1873004.sqsh"
        ),
        "image_sha256": (
            "a393e1b8f12e5edafa49a84c0b78b172aa163ad29be04fca6e42855a5f16304a"
        ),
        "functional_time": "02:00:00",
        "benchmark_time": "08:00:00",
    },
    "lyris": {
        "account": "coreai_dlalgo_llm",
        "partition": "gb200",
        "gres": "",
        "segment": "1",
        "comment": "metrics",
        "image": (
            "/lustre/fsw/coreai_dlalgo_llm/users/sna/containers/"
            "nemo_rl_nightly_20260711_2346595.sqsh"
        ),
        "image_sha256": (
            "bb5beff9ade16a1eeb6badde7601731bb003a95b4cccf85b3bd9b11c84803a2a"
        ),
        "functional_time": "02:00:00",
        "benchmark_time": "05:00:00",
    },
}


def load_profile(profile_name: str) -> dict[str, object]:
    script = f"""
set -euo pipefail
source {LOADER!s}
CUTEDSL_CLUSTER_PROFILE={profile_name}
load_cutedsl_cluster_profile
python3 - <<'PY'
import json
import os

keys = (
    "CUTEDSL_PROFILE_NAME",
    "CUTEDSL_ACCOUNT",
    "CUTEDSL_PARTITION",
    "CUTEDSL_GRES",
    "CUTEDSL_SEGMENT",
    "CUTEDSL_COMMENT",
    "CUTEDSL_IMAGE",
    "CUTEDSL_IMAGE_SHA256",
    "CUTEDSL_FUNCTIONAL_TIME",
    "CUTEDSL_BENCHMARK_TIME",
)
profile = {{key.removeprefix("CUTEDSL_").lower(): os.environ[key] for key in keys}}
profile["sbatch_args"] = os.environ["CUTEDSL_SBATCH_ARGS"].splitlines()
print(json.dumps(profile))
PY
"""
    result = subprocess.run(
        ["bash", "-c", script],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    return json.loads(result.stdout)


@pytest.mark.parametrize("profile_name", EXPECTED_PROFILES)
def test_profiles_match_immutable_cluster_contracts(profile_name: str) -> None:
    profile = load_profile(profile_name)
    profile_name = str(profile["profile_name"])
    assert profile == {
        "profile_name": profile_name,
        **EXPECTED_PROFILES[profile_name],
        "sbatch_args": profile["sbatch_args"],
    }


def test_pre_tyche_profile_uses_segment_without_gres() -> None:
    profile = load_profile("pre_tyche")
    assert profile["account"] == "coreai_dlalgo_llm"
    assert profile["partition"] == "batch"
    assert profile["segment"] == "1"
    assert profile["gres"] == ""


def test_aws_profile_uses_gres_without_segment() -> None:
    profile = load_profile("aws_dfw")
    assert profile["account"] == "nemotron_sw_post"
    assert profile["partition"] == "batch_long"
    assert profile["segment"] == ""
    assert profile["gres"] == "gpu:4"


def test_lyris_profile_uses_segment_without_gres() -> None:
    profile = load_profile("lyris")
    assert profile["partition"] == "gb200"
    assert profile["segment"] == "1"
    assert profile["gres"] == ""


@pytest.mark.parametrize(
    ("profile_name", "specific_args"),
    [
        ("pre_tyche", ["--segment=1"]),
        ("aws_dfw", ["--gres=gpu:4"]),
        ("lyris", ["--segment=1"]),
    ],
)
def test_loader_builds_exact_cluster_sbatch_args(
    profile_name: str,
    specific_args: list[str],
) -> None:
    profile = load_profile(profile_name)
    expected = EXPECTED_PROFILES[profile_name]
    assert profile["sbatch_args"] == [
        f"--account={expected['account']}",
        f"--partition={expected['partition']}",
        f"--comment={expected['comment']}",
        *specific_args,
    ]


@pytest.mark.parametrize(
    ("profile_name", "override", "error_fragment"),
    [
        ("unknown", "", "Unknown CUTEDSL_CLUSTER_PROFILE"),
        ("pre_tyche", "CUTEDSL_IMAGE=relative.sqsh", "absolute"),
        ("pre_tyche", "CUTEDSL_IMAGE_SHA256=bad", "SHA256"),
        ("pre_tyche", "CUTEDSL_SEGMENT=2", "segment"),
        ("pre_tyche", "CUTEDSL_GRES=gpu:4", "both GRES and segment"),
    ],
)
def test_loader_rejects_invalid_profiles(
    profile_name: str,
    override: str,
    error_fragment: str,
    tmp_path: Path,
) -> None:
    test_loader = LOADER
    if override:
        profile_path = EXPERIMENT_DIR / f"cluster_profiles/{profile_name}.sh"
        mutated_profile = tmp_path / f"{profile_name}.sh"
        mutated_profile.write_text(profile_path.read_text() + f"\nexport {override}\n")
        test_loader = tmp_path / "cluster_profile.sh"
        test_loader.write_text(
            LOADER.read_text().replace(
                'readonly CUTEDSL_CLUSTER_PROFILE_DIR="${CUTEDSL_EXPERIMENT_DIR}/cluster_profiles"',
                f'readonly CUTEDSL_CLUSTER_PROFILE_DIR="{tmp_path}"',
            )
        )

    result = subprocess.run(
        [
            "bash",
            "-c",
            f"source {test_loader!s}; CUTEDSL_CLUSTER_PROFILE={profile_name}; "
            "load_cutedsl_cluster_profile",
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    assert error_fragment in result.stderr


def write_mock_sbatch(mock_bin: Path, calls_path: Path) -> None:
    mock_sbatch = mock_bin / "sbatch"
    mock_sbatch.write_text(
        """#!/usr/bin/env python3
import json
import os
import sys
from pathlib import Path

with Path(os.environ["MOCK_SBATCH_CALLS"]).open("a") as output:
    output.write(json.dumps(sys.argv[1:]) + "\\n")
print("mock-job")
"""
    )
    mock_sbatch.chmod(0o755)
    calls_path.touch()


@pytest.mark.parametrize("profile_name", EXPECTED_PROFILES)
@pytest.mark.parametrize("test_only", [False, True])
def test_functional_submitter_passes_exact_profile_argv(
    profile_name: str,
    test_only: bool,
    tmp_path: Path,
) -> None:
    mock_bin = tmp_path / "bin"
    mock_bin.mkdir()
    calls_path = tmp_path / "calls.jsonl"
    write_mock_sbatch(mock_bin, calls_path)
    env = os.environ.copy()
    env.update(
        {
            "PATH": f"{mock_bin}:{env['PATH']}",
            "MOCK_SBATCH_CALLS": str(calls_path),
            "CUTEDSL_CLUSTER_PROFILE": profile_name,
        }
    )
    command = ["bash", str(EXPERIMENT_DIR / "submit_cutedsl_functional.sh")]
    if test_only:
        command.append("--test-only")
    result = subprocess.run(
        command,
        cwd=Path(__file__).parents[1],
        env=env,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr

    expected = EXPECTED_PROFILES[profile_name]
    cluster_specific = (
        ["--gres=gpu:4"] if expected["gres"] else ["--segment=1"]
    )
    expected_args = [
        f"--account={expected['account']}",
        f"--partition={expected['partition']}",
        f"--comment={expected['comment']}",
        *cluster_specific,
        f"--time={expected['functional_time']}",
        "--export=ALL",
    ]
    if test_only:
        expected_args.append("--test-only")
    expected_args.append(str(EXPERIMENT_DIR / "run_cutedsl_functional.sbatch"))
    calls = [json.loads(line) for line in calls_path.read_text().splitlines()]
    assert calls == [expected_args]


@pytest.mark.parametrize("profile_name", EXPECTED_PROFILES)
@pytest.mark.parametrize("test_only", [False, True])
def test_benchmark_submitter_passes_exact_profile_argv(
    profile_name: str,
    test_only: bool,
    tmp_path: Path,
) -> None:
    mock_bin = tmp_path / "bin"
    mock_bin.mkdir()
    calls_path = tmp_path / "calls.jsonl"
    write_mock_sbatch(mock_bin, calls_path)
    env = os.environ.copy()
    env.update(
        {
            "PATH": f"{mock_bin}:{env['PATH']}",
            "MOCK_SBATCH_CALLS": str(calls_path),
            "CUTEDSL_CLUSTER_PROFILE": profile_name,
            "CUTEDSL_BENCHMARK_REPLICATES": "3",
        }
    )
    driver = tmp_path / "submit_cutedsl_ab_replicates.sh"
    driver.write_text(
        (EXPERIMENT_DIR / "submit_cutedsl_ab_replicates.sh")
        .read_text()
        .replace(
            'readonly SUBMISSION_DIR="${EXPERIMENT_DIR}/benchmark_submissions"',
            f'readonly SUBMISSION_DIR="{tmp_path}/records"',
        )
    )
    command = ["bash", str(driver)]
    if test_only:
        command.append("--test-only")
    result = subprocess.run(
        command,
        cwd=Path(__file__).parents[1],
        env=env,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr

    expected = EXPECTED_PROFILES[profile_name]
    cluster_specific = (
        ["--gres=gpu:4"] if expected["gres"] else ["--segment=1"]
    )
    common_args = [
        "--parsable",
        f"--account={expected['account']}",
        f"--partition={expected['partition']}",
        f"--comment={expected['comment']}",
        *cluster_specific,
        f"--time={expected['benchmark_time']}",
    ]
    if test_only:
        common_args.append("--test-only")

    calls = [json.loads(line) for line in calls_path.read_text().splitlines()]
    assert len(calls) == 3
    for call in calls:
        export_args = [argument for argument in call if argument.startswith("--export-file=")]
        assert len(export_args) == 1
        assert call == [
            *common_args,
            export_args[0],
            str(EXPERIMENT_DIR / "run_cutedsl_matrix.sbatch"),
        ]
