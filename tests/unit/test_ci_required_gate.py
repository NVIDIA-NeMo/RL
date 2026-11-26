# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""Test the required CI check's event, coverage and exact-head skip policies."""

import json
import os
import subprocess
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[2]
WORKFLOW = yaml.load(
    (ROOT / ".github/workflows/cicd-main.yml").read_text(), Loader=yaml.BaseLoader
)
GATE = WORKFLOW["jobs"]["CI_QA_Gate"]
VERIFY = next(step for step in GATE["steps"] if step.get("id") == "verify-skip")
EVALUATE = next(
    step
    for step in WORKFLOW["jobs"]["pre-flight"]["steps"]
    if step.get("id") == "evaluate"
)
HEAD = "a" * 40


def test_required_check_only_names_unfiltered_mirror_and_merge_group_runs() -> None:
    """Manual and subset runs cannot publish the status required by branch protection."""
    assert WORKFLOW["on"]["merge_group"] == {"types": ["checks_requested"]}
    assert "pull-request/[0-9]+" in WORKFLOW["on"]["push"]["branches"]
    assert " ".join(GATE["name"].split()) == (
        "${{ (github.event_name == 'merge_group' || "
        "(github.event_name == 'push' && startsWith(github.ref, 'refs/heads/pull-request/'))) && "
        "needs.pre-flight.outputs.test_script_pattern == '' && 'CI quality check' || 'CI diagnostics' }}"
    )
    assert "needs.pre-flight.result == 'success'" in VERIFY["if"]
    assert "needs.pre-flight.outputs.head_sha == github.sha" in VERIFY["if"]
    assert GATE["permissions"] == {"pull-requests": "read"}


@pytest.mark.parametrize(
    ("passed", "verified", "success"),
    [
        ("true", "", True),
        ("false", "", False),
        ("false", "true", True),
        ("", "", False),
    ],
)
def test_final_gate_requires_evidence(
    passed: str, verified: str, success: bool
) -> None:
    """Uncertain or absent override evidence leaves the regular CI verdict in force."""
    final = GATE["steps"][-1]
    assert final["env"] == {
        "CI_PASSED": "${{ steps.results.outputs.passed }}",
        "SKIP_VERIFIED": "${{ steps.verify-skip.outputs.verified }}",
    }
    result = subprocess.run(
        ["bash", "-e", "-c", final["run"]],
        env={"CI_PASSED": passed, "SKIP_VERIFIED": verified},
        capture_output=True,
    )
    assert (result.returncode == 0) is success


@pytest.mark.parametrize(
    ("event", "minimum", "label", "level", "pattern"),
    [
        ("push", "Lfast", "CI:docs", "Lfast", ""),
        ("push", "L1", "CI:Lfast", "L1", ""),
        ("push", "Lfast", "CI:L0", "L1", ""),
        ("push", "docs", "", "docs", ""),
        ("push", "docs", "CI:L2", "L2", ""),
        ("push", "L1", "CI:megatron", "L1", "megatron"),
        ("merge_group", "", "", "L1", ""),
        ("workflow_dispatch", "L1", "", "docs", ""),
    ],
)
def test_selected_tier(
    tmp_path: Path, event: str, minimum: str, label: str, level: str, pattern: str
) -> None:
    """Run the workflow's tier selection with representative changed-file floors."""
    output = tmp_path / "outputs"
    subprocess.run(
        ["bash", "-e", "-c", EVALUATE["run"]],
        cwd=ROOT,
        env={
            **os.environ,
            "GITHUB_OUTPUT": str(output),
            "MINIMUM_LEVEL": minimum,
            "EVENT_NAME": event,
            "IS_PULLREQUEST": "false" if event == "merge_group" else "true",
            "LABEL": label,
            "TEST_PLATFORM": "all",
            "DISPATCH_TEST_LEVEL": "docs",
            "DISPATCH_IMAGE_TAG": "",
            "TEST_SCRIPT_PATTERN": "",
            "MEGATRON_LM_COMMIT": "",
            "CACHE_SCOPE": "",
        },
        check=True,
        capture_output=True,
        text=True,
    )
    outputs = dict(line.split("=", 1) for line in output.read_text().splitlines())
    assert (outputs["test_level"], outputs["test_script_pattern"]) == (level, pattern)


@pytest.mark.parametrize(
    ("head", "labeled", "review_change", "role", "accepted"),
    [
        (HEAD, True, {}, "maintain", True),
        (HEAD, True, {"state": "APPROVED"}, "admin", True),
        ("b" * 40, True, {}, "admin", False),
        (HEAD, False, {}, "admin", False),
        (HEAD, True, {"commit_id": "b" * 40}, "admin", False),
        (HEAD, True, {"state": "DISMISSED"}, "admin", False),
        (HEAD, True, {"body": "Please check Skip CICD"}, "admin", False),
        (HEAD, True, {}, "write", False),
        (HEAD, True, {}, "", False),
    ],
)
def test_only_current_maintainer_reviews_authorize_skip(
    tmp_path: Path,
    head: str,
    labeled: bool,
    review_change: dict,
    role: str,
    accepted: bool,
) -> None:
    """Feed recorded-style REST responses through the actual skip gate without network."""
    gh = tmp_path / "gh"
    gh.write_text(
        '#!/bin/bash\nset -eu\ncase "$*" in\n'
        '  "api repos/NVIDIA-NeMo/RL/pulls/42") echo "$PR" ;;\n'
        '  "api --paginate repos/NVIDIA-NeMo/RL/pulls/42/reviews?per_page=100") '
        'echo "[]"; echo "$REVIEWS" ;;\n'
        '  "api repos/NVIDIA-NeMo/RL/collaborators/maintainer%5Bbot%5D/permission --jq .role_name") '
        '[[ -n "$ROLE" ]] && echo "$ROLE" ;;\n'
        "  *) exit 2 ;;\nesac\n"
    )
    gh.chmod(0o755)
    output = tmp_path / "outputs"
    output.touch()
    review = {
        "commit_id": HEAD,
        "state": "COMMENTED",
        "body": "Skip CICD",
        "user": {"login": "maintainer[bot]"},
        **review_change,
    }
    subprocess.run(
        ["bash", "-e", "-c", VERIFY["run"]],
        env={
            **os.environ,
            "PATH": f"{tmp_path}:{os.environ['PATH']}",
            "PR": json.dumps(
                {
                    "state": "open",
                    "head": {"sha": head},
                    "labels": [{"name": "Skip CICD"}] if labeled else [],
                }
            ),
            "REVIEWS": json.dumps([review]),
            "ROLE": role,
            "PR_NUMBER": "42",
            "GITHUB_REPOSITORY": "NVIDIA-NeMo/RL",
            "GITHUB_SHA": HEAD,
            "GITHUB_OUTPUT": str(output),
            "GITHUB_STEP_SUMMARY": str(tmp_path / "summary"),
        },
        check=True,
    )
    assert ("verified=true" in output.read_text().splitlines()) is accepted
