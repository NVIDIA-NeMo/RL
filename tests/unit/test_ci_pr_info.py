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

import json
import os
import subprocess
from collections.abc import Callable
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[2]
ACTION = ROOT / ".github/actions/pr-info/action.yml"
WORKFLOW = ROOT / ".github/workflows/cicd-main.yml"
PREFLIGHT = ROOT / ".github/workflows/_cicd_preflight.yml"
ActionRunner = Callable[..., tuple[subprocess.CompletedProcess[str], Path, Path]]


@pytest.fixture
def action_runner(tmp_path: Path) -> ActionRunner:
    gh = tmp_path / "gh"
    gh.write_text(
        "#!/bin/bash\n"
        'printf "%s\\n" "$@" > "$API_CALL"\n'
        'if [[ "$API_STATUS" != 0 ]]; then exit "$API_STATUS"; fi\n'
        'cat "$API_RESPONSE"\n'
    )
    gh.chmod(0o755)
    script = yaml.safe_load(ACTION.read_text())["runs"]["steps"][0]["run"]

    def run(
        *,
        ref: str = "refs/heads/pull-request/5358",
        event: str = "workflow_dispatch",
        response: str = '{"number":5358,"user":{"login":"pr-author"}}',
        status: int = 0,
    ) -> tuple[subprocess.CompletedProcess[str], Path, Path]:
        output = tmp_path / "output"
        call = tmp_path / "call"
        payload = tmp_path / "response"
        payload.write_text(response)
        env = os.environ | {
            "PATH": f"{tmp_path}:{os.environ['PATH']}",
            "GITHUB_REF": ref,
            "GITHUB_EVENT_NAME": event,
            "GITHUB_REPOSITORY": "NVIDIA-NeMo/RL",
            "GITHUB_OUTPUT": str(output),
            "API_CALL": str(call),
            "API_RESPONSE": str(payload),
            "API_STATUS": str(status),
        }
        result = subprocess.run(
            ["bash", "-c", script], capture_output=True, text=True, env=env, check=False
        )
        return result, output, call

    return run


@pytest.mark.parametrize("event", ["push", "workflow_dispatch"])
def test_pr_mirror_lookup_preserves_live_metadata(
    action_runner: ActionRunner, event: str
) -> None:
    payload = {
        "number": 5358,
        "user": {"login": "pr-author"},
        "base": {"ref": "r0.8.0", "sha": "a" * 40},
        "head": {"ref": "fix/example", "sha": "b" * 40},
        "labels": [{"name": "CI:L1"}],
        "body": "multiline\npr-info=not-an-output",
    }
    result, output, call = action_runner(
        event=event, response=json.dumps(payload, indent=2)
    )
    assert result.returncode == 0, result.stderr
    assert call.read_text().splitlines() == [
        "api",
        "--method",
        "GET",
        "repos/NVIDIA-NeMo/RL/pulls/5358",
    ]
    (line,) = output.read_text().splitlines()
    assert line.startswith("pr-info=")
    assert json.loads(line.removeprefix("pr-info=")) == payload


@pytest.mark.parametrize(
    "ref",
    [
        "refs/heads/main",
        "refs/heads/pull-request/0",
        "refs/heads/pull-request/05358",
        "refs/heads/pull-request/5358/suffix",
        "refs/heads/pull-request/5358\n",
        "refs/heads/pull-request/$(touch bad)",
        "refs/pull/5358/merge",
        "",
    ],
)
def test_invalid_mirror_ref_fails_before_api(
    action_runner: ActionRunner, ref: str
) -> None:
    result, output, call = action_runner(ref=ref)
    assert result.returncode != 0
    assert "Expected a vetted pull-request/<number> branch" in result.stderr
    assert not call.exists()
    assert not output.exists()


def test_api_failure_does_not_publish_metadata(action_runner: ActionRunner) -> None:
    result, output, call = action_runner(status=1)
    assert result.returncode != 0
    assert call.exists()
    assert not output.exists()


@pytest.mark.parametrize(
    "response", ["not JSON", "null", "[]", "{}", '{"number":42}', '{"number":"5358"}']
)
def test_invalid_api_response_does_not_publish_metadata(
    action_runner: ActionRunner, response: str
) -> None:
    result, output, _ = action_runner(response=response)
    assert result.returncode != 0
    assert not output.exists()


def test_all_pr_resolvers_have_a_checkout_and_unchanged_consumers() -> None:
    workflow = yaml.safe_load(WORKFLOW.read_text())
    preflight = yaml.safe_load(PREFLIGHT.read_text())
    for job in [
        workflow["jobs"]["pre-flight"],
        workflow["jobs"]["Coverage"],
        preflight["jobs"]["pre-flight"],
    ]:
        steps = job["steps"]
        index = next(
            i for i, step in enumerate(steps) if step.get("id") == "dispatch-pr-info"
        )
        resolver = steps[index]
        assert resolver["uses"] == "./.github/actions/pr-info"
        assert resolver["if"] == (
            "github.event_name == 'workflow_dispatch' && "
            "startsWith(github.ref, 'refs/heads/pull-request/')"
        )
        legacy = next(step for step in steps if step.get("id") == "get-pr-info")
        assert legacy["uses"].startswith("nv-gha-runners/get-pr-info@")
        assert "github.event_name != 'workflow_dispatch'" in legacy["if"]
        assert any(
            step.get("uses", "").startswith("actions/checkout@")
            for step in steps[:index]
        )
    assert (
        workflow["jobs"]["org-member-pre-flight"]["uses"]
        == "./.github/workflows/_cicd_preflight.yml"
    )
    steps = {step.get("id"): step for step in preflight["jobs"]["pre-flight"]["steps"]}
    assert (
        steps["check-sso"]["uses"] == "./ci-templates/.github/actions/check-nvidia-sso"
    )
    assert steps["check-sso"]["if"] == "${{ inputs.default_runner_prefix != '' }}"
    assert steps["check-sso"]["with"]["username"] == (
        "${{ fromJSON((steps.dispatch-pr-info.outputs.pr-info || steps.get-pr-info.outputs.pr-info) || '{}').user.login || github.actor }}"
    )
    assert (
        steps["check-sso"]["with"]["github_token"]
        == "${{ secrets.NVIDIA_MANAGEMENT_ORG_PAT }}"
    )
    assert (
        "steps.check-sso.outputs.is_member"
        in preflight["jobs"]["pre-flight"]["outputs"]["is_member"]
    )
    assert (
        steps["select_runner"]["env"]["IS_MEMBER"]
        == "${{ steps.check-sso.outputs.is_member }}"
    )
