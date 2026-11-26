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

"""Exercise CI's tested revision selection against immutable GitHub snapshots."""

import json
import os
import subprocess
from pathlib import Path

import pytest
import yaml

WORKFLOW = Path(__file__).resolve().parents[2] / ".github/workflows/cicd-main.yml"
BASE = "a" * 40
HEAD = "b" * 40
MERGE = "c" * 40
PR = {
    "number": 42,
    "base": {"ref": "r0.8.0", "sha": BASE},
    "head": {"ref": "fix", "sha": HEAD, "label": "contributor:fix"},
    "merge_commit_sha": MERGE,
    "labels": [{"name": "CI:L0"}],
}


def resolve_revision(
    tmp_path: Path,
    *,
    ref: str,
    pr: dict | None,
    parents: tuple[str, ...] = (BASE, HEAD),
) -> tuple[int, dict[str, str]]:
    """Run the workflow step with a local GitHub response and a deterministic non-mirror parent."""
    jobs = yaml.safe_load(WORKFLOW.read_text())["jobs"]
    step = next(
        item
        for item in jobs["pre-flight"]["steps"]
        if item.get("id") == "base-head-ref"
    )
    gh = tmp_path / "gh"
    gh.write_text(
        '#!/bin/sh\nset -eu\n[ "$1" = api ] && [ "$2" = "repos/NVIDIA-NeMo/RL/git/commits/$MERGE_SHA" ]'
        ' && [ "$3" = --jq ] || exit 1\nprintf "%s\\n" "$COMMIT_JSON" | jq -r "$4"\n'
    )
    gh.chmod(0o755)
    git = tmp_path / "git"
    git.write_text(
        '#!/bin/sh\n[ "$*" = "rev-parse HEAD~1" ] || exit 1\necho "$BASE_SHA"\n'
    )
    git.chmod(0o755)
    output = tmp_path / "output"
    output.touch()
    result = subprocess.run(
        ["bash", "-e", "-c", step["run"]],
        env={
            **os.environ,
            "PATH": f"{tmp_path}:{os.environ['PATH']}",
            "PR_INFO_JSON": json.dumps(pr) if pr else "",
            "GITHUB_REF": f"refs/heads/{ref}",
            "GITHUB_REF_NAME": ref,
            "GITHUB_SHA": HEAD,
            "GITHUB_REPOSITORY": "NVIDIA-NeMo/RL",
            "GITHUB_OUTPUT": str(output),
            "BASE_SHA": BASE,
            "MERGE_SHA": MERGE,
            "COMMIT_JSON": json.dumps(
                {"parents": [{"sha": value} for value in parents]}
            ),
        },
        check=False,
    )
    return result.returncode, dict(
        line.split("=", 1) for line in output.read_text().splitlines()
    )


def test_mirror_uses_the_verified_merge_and_release_scope(tmp_path: Path) -> None:
    """Return the merge for testing and the matching head and base for PR checks."""
    status, outputs = resolve_revision(tmp_path, ref="pull-request/42", pr=PR)
    assert status == 0
    assert (
        outputs["pr_number"],
        outputs["head_sha"],
        outputs["base_sha"],
        outputs["test_sha"],
    ) == (
        "42",
        HEAD,
        BASE,
        MERGE,
    )
    assert outputs["cache_scope"] == "r0.8.0"


@pytest.mark.parametrize(
    ("pr", "parents"),
    [
        (None, (BASE, HEAD)),
        ({**PR, "number": 43}, (BASE, HEAD)),
        ({**PR, "head": {**PR["head"], "sha": MERGE}}, (BASE, HEAD)),
        ({**PR, "merge_commit_sha": None}, (BASE, HEAD)),
        ({**PR, "base": {**PR["base"], "sha": ""}}, (BASE, HEAD)),
        (PR, (BASE,)),
        (PR, (BASE, HEAD, MERGE)),
        (PR, (MERGE, HEAD)),
        (PR, (BASE, MERGE)),
    ],
)
def test_stale_or_missing_merge_produces_no_outputs(
    tmp_path: Path, pr: dict | None, parents: tuple[str, ...]
) -> None:
    """Fail before exporting revision, cache routing or CI label outputs."""
    status, outputs = resolve_revision(
        tmp_path, ref="pull-request/42", pr=pr, parents=parents
    )
    assert status != 0
    assert outputs == {}


@pytest.mark.parametrize(
    ("ref", "scope"),
    [("main", ""), ("r0.8.0", "r0.8.0"), ("gh-readonly-queue/main/pr-42", "")],
)
def test_nonmirror_uses_event_commit(tmp_path: Path, ref: str, scope: str) -> None:
    """Keep branch and merge queue revisions on the event commit."""
    status, outputs = resolve_revision(tmp_path, ref=ref, pr=None)
    assert status == 0
    assert (
        outputs["pr_number"],
        outputs["head_sha"],
        outputs["test_sha"],
        outputs["cache_scope"],
    ) == (
        "",
        HEAD,
        HEAD,
        scope,
    )


def git(repo: Path, *args: str) -> str:
    """Run Git against the temporary fixture repository."""
    return subprocess.check_output(
        ["git", "-c", "user.name=CI", "-c", "user.email=ci@example.test", *args],
        cwd=repo,
        text=True,
    ).strip()


def snapshot(repo: Path, parent: str = "") -> str:
    """Create a detached Git commit for the current fixture tree."""
    git(repo, "add", "--all")
    tree = git(repo, "write-tree")
    parents = ["-p", parent] if parent else []
    return git(repo, "commit-tree", tree, *parents, "-m", "fixture")


def classify(repo: Path, base: str, head: str) -> tuple[int, str]:
    """Run the workflow classifier against local commits and return its output."""
    steps = yaml.safe_load(WORKFLOW.read_text())["jobs"]["pre-flight"]["steps"]
    step = next(item for item in steps if item.get("id") == "ci-tier")
    output = repo.parent / "output"
    output.touch()
    result = subprocess.run(
        ["bash", "-e", "-c", step["run"]],
        cwd=repo,
        env={
            **os.environ,
            "BASE_SHA": base,
            "HEAD_SHA": head,
            "GITHUB_OUTPUT": str(output),
        },
        check=False,
    )
    return result.returncode, output.read_text().strip()


@pytest.mark.parametrize(
    ("old", "new", "expected"),
    [
        ("nemo_rl/module.py", None, "Lfast"),
        ("nemo_rl/module.py", "docs/module.md", "Lfast"),
        ("docs/module.md", "nemo_rl/module.py", "Lfast"),
        ("pyproject.toml", "docs/dependency.md", "L1"),
        ("docs/dependency.md", "uv.lock", "L1"),
        ("docs/old.md", "docs/new\nname", "docs"),
    ],
)
def test_tier_classifies_both_rename_paths_and_deletions(
    tmp_path: Path, old: str, new: str | None, expected: str
) -> None:
    """Compare the PR head with its merge base, retaining both sides of renames."""
    repo = tmp_path / "repo"
    repo.mkdir()
    git(repo, "init", "--quiet")
    policy = repo / ".github/scripts/ci_label.py"
    policy.parent.mkdir(parents=True)
    policy.write_bytes((WORKFLOW.parents[1] / "scripts/ci_label.py").read_bytes())
    source = repo / old
    source.parent.mkdir(parents=True, exist_ok=True)
    source.write_text("fixture\n")
    fork = snapshot(repo)
    base_only = repo / "Dockerfile.base"
    base_only.write_text("base branch change\n")
    base = snapshot(repo, fork)
    base_only.unlink()
    if new:
        destination = repo / new
        destination.parent.mkdir(parents=True, exist_ok=True)
        source.rename(destination)
    else:
        source.unlink()
    head = snapshot(repo, fork)
    assert classify(repo, base, head) == (0, f"minimum_level={expected}")


@pytest.mark.parametrize(
    ("base_policy", "status", "output"),
    [
        ("project", 0, "minimum_level=Lfast"),
        ("absent", 0, "minimum_level=L1"),
        ("broken", 1, ""),
        ("missing-commit", 128, ""),
    ],
)
def test_tier_reads_only_verified_base_policy(
    tmp_path: Path, base_policy: str, status: int, output: str
) -> None:
    """Historical trees use full CI, while invalid Git or policy data fails closed."""
    repo = tmp_path / "repo"
    repo.mkdir()
    git(repo, "init", "--quiet")
    policy = repo / ".github/scripts/ci_label.py"
    policy.parent.mkdir(parents=True)
    if base_policy == "project":
        policy.write_bytes((WORKFLOW.parents[1] / "scripts/ci_label.py").read_bytes())
    elif base_policy == "broken":
        policy.write_text("raise RuntimeError('invalid base classifier')\n")
    base = snapshot(repo)
    policy.write_text("raise AssertionError('must never execute head classifier')\n")
    (repo / "fnmatch.py").write_text(
        "raise AssertionError('must never import head modules')\n"
    )
    head = snapshot(repo, base)
    assert classify(repo, BASE if base_policy == "missing-commit" else base, head) == (
        status,
        output,
    )
