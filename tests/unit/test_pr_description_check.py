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
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
TEMPLATE = REPO_ROOT / ".github/PULL_REQUEST_TEMPLATE.md"
_spec = importlib.util.spec_from_file_location(
    "check_pr_description", REPO_ROOT / ".github/scripts/check_pr_description.py"
)
check = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(check)

FILLED = """- [x] I, the PR author, have personally reviewed every line of this PR.

## What and why
Fix the loss scale when `token_level_loss` is off.

```python
# this comment is code, not a heading
x = 1
```

## Convergence / perf / functionality proof
N/A: unit-test-only change.

## Needs upstream changes
None
"""


def test_filled_description_passes() -> None:
    assert check.find_problems(FILLED) == []


def test_template_as_shipped_lists_every_problem() -> None:
    problems = check.find_problems(TEMPLATE.read_text())
    assert problems == [
        f'Review box not ticked: "{check.REVIEW_BOX_TEXT}"',
        'Empty section: "## What and why"',
    ]


def test_template_names_every_required_heading() -> None:
    template = TEMPLATE.read_text()
    assert check.REVIEW_BOX_TEXT in template
    for heading in check.REQUIRED_HEADINGS:
        assert f"## {heading}\n" in template


@pytest.mark.parametrize(
    ("body", "expected"),
    [
        ("", "Review box missing"),
        (FILLED.replace("[x]", "[ ]"), "Review box not ticked"),
        (
            FILLED.replace("## Needs upstream changes\n", ""),
            'Missing heading: "## Needs upstream changes"',
        ),
        (
            FILLED.split("## What and why\n")[0]
            + "## What and why\n<!-- todo -->\n\n## Convergence"
            + FILLED.split("## Convergence")[1],
            'Empty section: "## What and why"',
        ),
        # A heading inside fenced code does not count.
        (
            FILLED.replace(
                "## Needs upstream changes\nNone\n",
                "```\n## Needs upstream changes\n```\n",
            ),
            'Missing heading: "## Needs upstream changes"',
        ),
        # A ticked box hidden in an HTML comment does not count.
        (
            "<!-- " + FILLED.splitlines()[0] + " -->\n" + FILLED.split("\n", 1)[1],
            "Review box missing",
        ),
    ],
)
def test_problem_is_reported(body: str, expected: str) -> None:
    assert any(problem.startswith(expected) for problem in check.find_problems(body))


def test_uppercase_tick_and_star_bullet_pass() -> None:
    assert check.find_problems(FILLED.replace("- [x]", "* [X]")) == []


@pytest.mark.parametrize(
    ("author", "author_type", "expected"),
    [
        ("svcnvidia-nemo-ci", "User", True),
        ("svcnemo-autobot", "User", True),
        ("nemo-automation-bot[bot]", "Bot", True),
        ("terrykong", "User", False),
    ],
)
def test_is_bot(author: str, author_type: str, expected: bool) -> None:
    assert check.is_bot(author=author, author_type=author_type) is expected


def test_main_reports_and_fails(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    summary = tmp_path / "summary.md"
    monkeypatch.setenv("PR_BODY", "")
    monkeypatch.setenv("PR_AUTHOR", "someone")
    monkeypatch.setenv("PR_AUTHOR_TYPE", "User")
    monkeypatch.setenv("GITHUB_STEP_SUMMARY", str(summary))
    assert check.main() == 1
    out = capsys.readouterr().out
    assert "::error title=PR description::Review box missing" in out
    assert "Only a human may tick the review box." in out
    assert check.TEMPLATE_URL in summary.read_text()


def test_main_skips_bots(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PR_BODY", "")
    monkeypatch.setenv("PR_AUTHOR", "svcnvidia-nemo-ci")
    monkeypatch.setenv("PR_AUTHOR_TYPE", "User")
    assert check.main() == 0
