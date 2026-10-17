# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
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

"""Exercise PR platform selection through the actual CI preflight scripts."""

import json
import os
import subprocess
from pathlib import Path

import pytest
import yaml


def _run_preflight(
    tmp_path: Path, labels: list[str | dict[str, str]]
) -> tuple[subprocess.CompletedProcess[str], dict[str, str]]:
    root = Path(__file__).resolve().parents[2]
    workflow = yaml.safe_load((root / ".github/workflows/cicd-main.yml").read_text())
    steps = workflow["jobs"]["pre-flight"]["steps"]
    script = next(step["run"] for step in steps if step.get("id") == "base-head-ref")
    script = script.replace("${{ github.ref_name }}", "pull-request/123")
    script = script.replace("${{ github.sha }}", "2" * 40)
    output = tmp_path / "outputs"
    result = subprocess.run(
        ["bash", "-euo", "pipefail", "-c", script],
        cwd=root,
        env={
            **os.environ,
            "GITHUB_OUTPUT": str(output),
            "IS_PULL_REQUEST_REF": "true",
            "PR_INFO_JSON": json.dumps(
                {
                    "base": {"ref": "r0.8.0", "sha": "1" * 40},
                    "head": {"ref": "feature", "sha": "2" * 40},
                    "merge_commit_sha": "3" * 40,
                    "labels": labels,
                }
            ),
        },
        capture_output=True,
        text=True,
        check=False,
    )
    outputs = (
        dict(line.split("=", 1) for line in output.read_text().splitlines())
        if output.exists()
        else {}
    )
    return result, outputs


@pytest.mark.parametrize("platform", [None, "all", "h100", "gb200"])
@pytest.mark.parametrize("object_labels", [False, True])
def test_pr_platform_preserves_tier_and_release_scope(
    tmp_path: Path, platform: str | None, object_labels: bool
) -> None:
    labels = ["CI:L1"]
    if platform is not None:
        labels.append(f"CI-platform:{platform}")
    result, outputs = _run_preflight(
        tmp_path, [{"name": label} for label in labels] if object_labels else labels
    )
    assert result.returncode == 0, result.stderr
    assert outputs["test_platform"] == (platform or "all")
    assert outputs["ci_label"] == "CI:L1"
    assert outputs["cache_scope"] == "r0.8.0"
    assert outputs["test_sha"] == "3" * 40
    assert outputs["has_skip_cicd"] == "false"


@pytest.mark.parametrize(
    "platform_labels",
    [
        ["CI-platform:h200"],
        ["CI-platform:"],
        ["CI-platform:h100", "CI-platform:gb200"],
        ["CI-platform:all", "CI-platform:h100"],
    ],
)
def test_invalid_platform_selection_stops_preflight(
    tmp_path: Path, platform_labels: list[str]
) -> None:
    result, outputs = _run_preflight(tmp_path, ["CI:L1", *platform_labels])
    assert result.returncode != 0
    assert "Use at most one CI-platform:" in result.stderr
    assert "test_platform" not in outputs
