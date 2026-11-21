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
import io
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
Fix the loss scale.

```python
# a comment inside code, not a heading
```

## Convergence / perf / functionality proof
N/A: unit-test-only change.

## Needs upstream changes
None
"""
WHAT_EMPTY = (
    FILLED.split("## What and why\n")[0]
    + "## What and why\n<!-- todo -->\n\n## Convergence"
    + FILLED.split("## Convergence")[1]
)


@pytest.mark.parametrize(
    "body",
    [
        FILLED,
        FILLED.replace("- [x]", "* [X]"),
        # Comment markers inside code must not erase a real heading.
        FILLED.replace("## Convergence", "```html\n<!--\n```\n## Convergence").replace(
            "## Needs upstream", "```html\n-->\n```\n## Needs upstream"
        ),
        FILLED.replace("Fix the loss scale.", "Fix the `<!--` parser.") + "\nEnd.\n",
        # Subsections under What and why are content.
        FILLED.replace(
            "Fix the loss scale.",
            "### Problem\nStale results.\n### Change\nRead the live text.",
        ),
        # A code block alone is content.
        FILLED.replace("Fix the loss scale.\n", ""),
    ],
)
def test_passes(body: str) -> None:
    assert check.find_problems(body) == []


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
            FILLED.replace(
                "## Needs upstream changes\nNone\n",
                "```\n## Needs upstream changes\n```\n",
            ),
            'Missing heading: "## Needs upstream changes"',
        ),
        (WHAT_EMPTY, 'Empty section: "## What and why"'),
        (
            FILLED.split("## What and why\n")[0]
            + "## What and why\n\n## Issues\nNone\n## Convergence"
            + FILLED.split("## Convergence")[1],
            'Empty section: "## What and why"',
        ),
        # A ticked box hidden in an HTML comment does not count.
        (
            "<!-- " + FILLED.splitlines()[0] + " -->\n" + FILLED.split("\n", 1)[1],
            "Review box missing",
        ),
        # Everything inside an unclosed fence is code.
        ("```markdown\n" + FILLED, "Review box missing"),
    ],
)
def test_problem_is_reported(body: str, expected: str) -> None:
    assert any(p.startswith(expected) for p in check.find_problems(body))


def test_template_as_shipped() -> None:
    assert check.find_problems(TEMPLATE.read_text()) == [
        f'Review box not ticked: "{check.REVIEW_BOX_TEXT}"',
        'Empty section: "## What and why"',
    ]


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
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:
    summary = tmp_path / "summary.md"
    monkeypatch.setattr("sys.stdin", io.StringIO(""))
    monkeypatch.setenv("PR_AUTHOR", "someone")
    monkeypatch.setenv("PR_AUTHOR_TYPE", "User")
    monkeypatch.setenv("GITHUB_ACTIONS", "true")
    monkeypatch.setenv("GITHUB_STEP_SUMMARY", str(summary))
    assert check.main() == 1
    out = capsys.readouterr().out
    assert "::error title=PR description::Review box missing" in out
    assert "Only a human may tick the review box." in out
    assert check.TEMPLATE_URL in summary.read_text()


def test_main_skips_bots(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("sys.stdin", io.StringIO(""))
    monkeypatch.setenv("PR_AUTHOR", "svcnvidia-nemo-ci")
    monkeypatch.setenv("PR_AUTHOR_TYPE", "User")
    assert check.main() == 0
