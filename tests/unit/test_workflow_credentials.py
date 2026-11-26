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

"""Verify credential boundaries at RL workflow call sites."""

import re
from collections.abc import Iterator
from pathlib import Path

import yaml

GITHUB = Path(__file__).resolve().parents[2] / ".github"
TEMPLATES = "NVIDIA-NeMo/FW-CI-templates/.github/workflows"
ROSTER = "NVIDIA_MANAGEMENT_ORG_PAT"
DISPATCH = "NVSKILLS_CI_DISPATCH_TOKEN"
EXCEPTIONS = {
    "cicd-main.yml": (
        "org-member-pre-flight",
        f"{TEMPLATES}/_cicd_preflight.yml",
        {ROSTER: ROSTER},
    ),
    "cicd-approve-test-queue.yml": (
        "approve-test-queue",
        f"{TEMPLATES}/_test_approval_queue.yml",
        {ROSTER: ROSTER, "PAT": "PAT"},
    ),
    "validate-codeowners-membership.yml": (
        "validate-codeowners-membership",
        f"{TEMPLATES}/_validate_codeowners_membership.yml",
        {"ROSTER_TOKEN": ROSTER},
    ),
    "request-nvskills-ci.yml": (
        "request",
        "NVIDIA/skills/.github/workflows/team-request.yml",
        {DISPATCH: DISPATCH},
    ),
}
SECRET_ACCESS = re.compile(
    r"""\bsecrets\b(?:\s*\.\s*(\w+)|\s*\[\s*['"](\w+)['"]\s*\])?""", re.IGNORECASE
)


def _load(path: Path) -> dict:
    """Read workflow keys and GitHub expression strings without YAML boolean coercion."""
    return yaml.load(path.read_text(), Loader=yaml.BaseLoader)


def _restricted_credentials(node: dict | list | str) -> Iterator[str]:
    """Find bulk secret access and credentials reserved for an explicit call site."""
    if isinstance(node, dict):
        if node.get("secrets") == "inherit":
            yield "secrets: inherit"
        for value in node.values():
            yield from _restricted_credentials(value)
    elif isinstance(node, list):
        for value in node:
            yield from _restricted_credentials(value)
    elif isinstance(node, str):
        for expression in re.findall(r"\$\{\{(.*?)\}\}", node, flags=re.DOTALL):
            for match in SECRET_ACCESS.finditer(expression):
                name = (match[1] or match[2] or "").upper()
                if not name:
                    yield "unqualified or dynamic secrets reference"
                elif (
                    "PAT" in name.split("_")
                    or "PERSONAL_ACCESS_TOKEN" in name
                    or name == DISPATCH
                ):
                    yield name


def test_external_credentials_are_scoped_to_their_callers() -> None:
    """Limit roster access, deployment approvals and NVSkills dispatch to their designated jobs."""
    workflows = GITHUB / "workflows"
    definitions = {path: _load(path) for path in workflows.glob("*.y*ml")}
    for filename, (job_id, callee, bindings) in EXCEPTIONS.items():
        job = definitions[workflows / filename]["jobs"][job_id]
        assert job["uses"].split("@")[0] == callee
        for parameter, secret in bindings.items():
            assert job["secrets"].pop(parameter) == "${{ secrets." + secret + " }}"
    definitions.update(
        {path: _load(path) for path in (GITHUB / "actions").rglob("action.y*ml")}
    )
    violations = [
        f"{path.name}: {error}"
        for path, doc in definitions.items()
        for error in _restricted_credentials(doc)
    ]
    assert not violations, "\n".join(violations)


def test_guardwords_uses_only_the_deploy_key_after_mirror_verification() -> None:
    """Require the scanner to use the head commit verified against the PR metadata."""
    workflow = _load(GITHUB / "workflows/guardwords-check.yml")
    assert workflow["on"] == {"push": {"branches": ["pull-request/[0-9]+"]}}
    source = workflow["jobs"]["pr_info"]
    assert source["outputs"]["head_sha"] == "${{ github.sha }}"
    (verify,) = [
        step for step in source["steps"] if "PR_HEAD_SHA" in step.get("env", {})
    ]
    assert (
        verify["env"]["PR_HEAD_SHA"]
        == "${{ fromJSON(steps.pr.outputs.pr-info).head.sha }}"
    )
    assert verify["run"].strip() == 'test "$PR_HEAD_SHA" = "$GITHUB_SHA"'
    assert "if" not in verify and "continue-on-error" not in verify
    scan = workflow["jobs"]["guardwords-check"]
    assert scan["needs"] == "pr_info"
    assert "if" not in scan
    assert scan["uses"].split("@")[0] == f"{TEMPLATES}/_guardwords_check.yml"
    assert scan["with"]["head-sha"] == "${{ needs.pr_info.outputs.head_sha }}"
    assert scan["secrets"] == {
        "GUARDWORDS_DEPLOY_KEY": "${{ secrets.GUARDWORDS_DEPLOY_KEY }}"
    }


def test_reviewer_operations_use_the_pr_app() -> None:
    """Bind reviewer requests and membership lookups to RL's PR App."""
    requests = _load(GITHUB / "workflows/suggest-reviewers.yml")["jobs"][
        "suggest-reviewers"
    ]
    membership = _load(GITHUB / "workflows/validate-codeowners-membership.yml")["jobs"][
        "validate-codeowners-membership"
    ]
    assert (
        requests["with"]["app-id"]
        == membership["with"]["membership_app_id"]
        == "${{ vars.PR_APP_ID }}"
    )
    assert (
        requests["secrets"]["BOT_KEY"]
        == membership["secrets"]["MEMBERSHIP_APP_PRIVATE_KEY"]
    )
    assert requests["secrets"]["BOT_KEY"] == "${{ secrets.PR_APP_PRIVATE_KEY }}"


def test_release_jobs_get_the_release_key_from_the_protected_environment() -> None:
    """Keep the release App key out of caller bindings and all publishing jobs in rl-release."""
    release = _load(GITHUB / "workflows/release.yaml")["jobs"]
    freeze = _load(GITHUB / "workflows/release-freeze.yml")["jobs"]["code-freeze"]
    for job in (freeze, release["bump"], release["finalize"]):
        assert job["with"]["release-environment"] == "rl-release"
        assert "RELEASE_APP_PRIVATE_KEY" not in job.get("secrets", {})
    assert freeze["with"]["freeze-source-branch"] == "main"
    docs = _load(GITHUB / "workflows/release-docs.yml")["jobs"]["publish-docs"]
    assert docs["environment"]["name"] == "rl-release"


def test_docs_publishers_forward_only_named_production_credentials() -> None:
    """Keep both documentation callers in sync with the credentials used by the publisher."""
    workflows = GITHUB / "workflows"
    docs = _load(workflows / "release-docs.yml")
    publish = docs["jobs"]["publish-docs"]
    assert publish["if"] == "${{ !inputs.dry-run }}"
    declared = set(docs["on"]["workflow_call"]["secrets"])
    action_inputs = publish["steps"][0]["with"].values()
    consumed = {
        match[1]
        for value in action_inputs
        if (match := re.fullmatch(r"\$\{\{ secrets\.(\w+) \}\}", value))
    }
    assert len(consumed) == 8
    assert declared == consumed
    nightly = _load(workflows / "release-nightly-docs.yml")["jobs"]["call-release-docs"]
    release = _load(workflows / "release.yaml")["jobs"]["docs"]
    assert (
        nightly["if"]
        == "github.repository == 'NVIDIA-NeMo/RL' && github.ref == 'refs/heads/main'"
    )
    assert nightly["secrets"] == {
        name: "${{ secrets." + name + " }}" for name in declared
    }
    assert release["secrets"] == {
        name: "${{ github.event_name == 'workflow_dispatch' && !inputs.dry-run && secrets."
        + name
        + " || '' }}"
        for name in declared
    }
