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
"""Check a PR/MR description against .github/PULL_REQUEST_TEMPLATE.md.

Reads the description on stdin and the author from PR_AUTHOR / PR_AUTHOR_TYPE, prints one
line per problem and exits 1 if there are any. Standard library only, so GitHub Actions
and GitLab CI can both run it. Keep the box text and headings in sync with the template
and skills/pr-description.
"""

import os
import re
import sys

TEMPLATE_URL = (
    "https://github.com/NVIDIA-NeMo/RL/blob/main/.github/PULL_REQUEST_TEMPLATE.md"
)
REVIEW_BOX_TEXT = "I, the PR author, have personally reviewed every line of this PR."
REQUIRED_HEADINGS = (
    "What and why",
    "Convergence / perf / functionality proof",
    "Needs upstream changes",
)
# Service accounts that open PRs but are GitHub type "User", so the Bot type check misses them.
BOT_AUTHORS = frozenset({"svcnvidia-nemo-ci", "svcnemo-autobot"})

# Code is replaced before HTML comments are dropped, so neither can tick the box,
# add a heading, or hide a real one. An unclosed fence runs to the end.
_FENCE = re.compile(r"^[ \t]*(`{3,}|~{3,}).*?(?:^[ \t]*\1[^\n]*$|\Z)", re.M | re.S)
_INLINE_CODE = re.compile(r"`[^`\n]*`")
_COMMENT = re.compile(r"<!--.*?-->", re.S)
_BOX = re.compile(
    r"^[ \t]*[-*][ \t]+\[(?P<mark>[ xX])\][ \t]+"
    + re.escape(REVIEW_BOX_TEXT.rstrip("."))
    + r"\.?[ \t]*$",
    re.M,
)
# Subsections belong to "What and why"; it ends at a peer or parent heading.
_WHAT_AND_WHY = re.compile(
    r"^(#{1,6})[ \t]+What and why[ \t]*\n(.*?)(?=^(?!\1#)#{1,6}[ \t]|\Z)",
    re.M | re.S | re.I,
)


def is_bot(*, author: str, author_type: str) -> bool:
    return author_type == "Bot" or author.endswith("[bot]") or author in BOT_AUTHORS


def normalize(body: str) -> str:
    text = _FENCE.sub("[code]", body or "")
    text = _INLINE_CODE.sub("[code]", text)
    return _COMMENT.sub("", text)


def find_problems(body: str) -> list[str]:
    """Return one human-readable line per template problem in the description."""
    text = normalize(body)
    problems = []

    box = _BOX.search(text)
    if box is None:
        problems.append(
            f'Review box missing: add "- [ ] {REVIEW_BOX_TEXT}" at the top.'
        )
    elif box.group("mark") == " ":
        problems.append(f'Review box not ticked: "{REVIEW_BOX_TEXT}"')

    for heading in REQUIRED_HEADINGS:
        if not re.search(
            rf"^#{{1,6}}[ \t]+{re.escape(heading)}[ \t]*$", text, re.M | re.I
        ):
            problems.append(f'Missing heading: "## {heading}"')

    section = _WHAT_AND_WHY.search(text)
    if section is not None and not section.group(2).strip():
        problems.append(f'Empty section: "## {REQUIRED_HEADINGS[0]}"')
    return problems


def main() -> int:
    author = os.environ.get("PR_AUTHOR", "")
    if is_bot(author=author, author_type=os.environ.get("PR_AUTHOR_TYPE", "")):
        print(f"Bot-authored PR ({author}), skipped.")
        return 0

    problems = find_problems(sys.stdin.read())
    if not problems:
        print("PR description follows the template.")
        return 0

    report = "\n".join(
        ["PR description does not follow the template:"]
        + [f"  - {problem}" for problem in problems]
        + [
            f"Template: {TEMPLATE_URL}",
            "Agents: see skills/pr-description. Only a human may tick the review box.",
        ]
    )
    print(report)
    if os.environ.get("GITHUB_ACTIONS") == "true":
        for problem in problems:
            print(f"::error title=PR description::{problem}")
    summary_path = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary_path:
        with open(summary_path, "a", encoding="utf-8") as summary:
            summary.write(f"```\n{report}\n```\n")
    return 1


if __name__ == "__main__":
    sys.exit(main())
