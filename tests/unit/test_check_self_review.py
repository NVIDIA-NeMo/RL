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

import importlib.util
import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
_spec = importlib.util.spec_from_file_location(
    "check_self_review", REPO_ROOT / ".github/scripts/check_self_review.py"
)
check = importlib.util.module_from_spec(_spec)
# Dataclasses with postponed annotations look the module up in sys.modules.
sys.modules["check_self_review"] = check
_spec.loader.exec_module(check)

POLICY = json.loads(
    (REPO_ROOT / "skills/review-pr-team/self-review-policy.json").read_text()
)


def marker(
    kind: str = "self",
    harness: str = "claude-code/2.1.330",
    model: str = "claude-opus-5-5",
    effort: str = "medium",
    rev: str = "2",
) -> str:
    return (
        f"Review body.\n<!-- nemo-rl-review: kind={kind} harness={harness} model={model} "
        f"effort={effort} skill_rev={rev} head=abc123 -->"
    )


def run(reviews, threads, main_rev: int = 2) -> tuple[str, str]:
    return check.evaluate(
        pr_number=7,
        author="alice",
        reviews=reviews,
        threads=threads,
        policy=POLICY,
        main_rev=main_rev,
    )


SELF = check.Review(1, "alice", marker(), "2026-11-21T00:00:00Z")


def test_parse_marker() -> None:
    got = check.parse_marker(marker())
    assert got["kind"] == "self"
    assert got["harness"] == "claude-code/2.1.330"
    assert got["skill_rev"] == "2"
    assert check.parse_marker("no marker") is None


@pytest.mark.parametrize(
    "reviews",
    [
        [],
        # Written by someone else.
        [check.Review(1, "bob", marker(), "t")],
        # Author's review without a marker (e.g. a plain comment review).
        [check.Review(1, "alice", "LGTM", "t")],
        # An expert review by the author is not a self-review.
        [check.Review(1, "alice", marker(kind="expert"), "t")],
    ],
)
def test_no_self_review(reviews) -> None:
    state, desc = run(reviews, [])
    assert state == "failure"
    assert desc.startswith("No self-review")
    assert "/review-pr-team 7" in desc
    assert len(desc) <= 140


@pytest.mark.parametrize(
    ("kwargs", "main_rev", "reason"),
    [
        ({"model": "claude-sonnet-5-5"}, 2, "model claude-sonnet-5-5 is not allowed"),
        ({"harness": "pi/1.0"}, 2, "harness pi is not allowed"),
        ({"effort": "low"}, 2, "effort low is below medium"),
        ({"rev": "x"}, 2, "no skill revision"),
        ({"rev": "2"}, 6, "rev 2 is too old"),
        ({"harness": "codex/0.167.2", "model": "gpt-5.6-sol"}, 2, "model gpt-5.6-sol"),
    ],
)
def test_not_accepted(kwargs: dict, main_rev: int, reason: str) -> None:
    review = check.Review(1, "alice", marker(**kwargs), "t")
    state, desc = run([review], [], main_rev=main_rev)
    assert state == "failure"
    assert desc.startswith("Not accepted:")
    assert reason in desc
    assert len(desc) <= 140


def test_revision_within_lag_is_accepted() -> None:
    review = check.Review(1, "alice", marker(rev="2"), "t")
    assert run([review], [], main_rev=5) == ("success", "Resolved (0 threads)")


def test_latest_self_review_decides() -> None:
    old = check.Review(1, "alice", marker(model="claude-sonnet-5-5"), "2026-01-01")
    new = check.Review(2, "alice", marker(), "2026-02-01")
    assert run([old, new], [])[0] == "success"
    assert run([new, old], [])[0] == "success"


@pytest.mark.parametrize(
    ("threads", "expected"),
    [
        # Resolved but never answered stays open.
        (
            [check.Thread(True, 1, 1)],
            ("failure", "Not resolved: 1 of 1 threads open or unanswered"),
        ),
        # Answered but not resolved stays open.
        (
            [check.Thread(False, 2, 1)],
            ("failure", "Not resolved: 1 of 1 threads open or unanswered"),
        ),
        (
            [check.Thread(True, 2, 1), check.Thread(True, 3, 1)],
            ("success", "Resolved (2 threads)"),
        ),
        # Threads from other reviews don't count.
        (
            [check.Thread(False, 1, 99), check.Thread(True, 2, 1)],
            ("success", "Resolved (1 threads)"),
        ),
    ],
)
def test_threads(threads, expected) -> None:
    assert run([SELF], threads) == expected


@pytest.mark.parametrize(
    ("files", "main_rev", "head_rev", "ok"),
    [
        (["nemo_rl/a.py"], 1, 1, True),
        (["skills/review-pr-team/SKILL.md"], 1, 2, True),
        (["skills/review-pr-team/SKILL.md"], 1, 1, False),
        (["skills/review-pr-team/SKILL.md"], 1, 3, False),
        # The PR that adds REVISION: main has none (0), head has 1.
        (["skills/review-pr-team/REVISION"], 0, 1, True),
    ],
)
def test_revision_bump(
    files: list[str], main_rev: int, head_rev: int, ok: bool
) -> None:
    problem = check.revision_bump_problem(files, main_rev, head_rev)
    assert (problem is None) is ok
    if not ok:
        assert f"set it to {main_rev + 1}" in problem
