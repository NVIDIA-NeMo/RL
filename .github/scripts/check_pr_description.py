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
"""Check that a PR description follows .github/PULL_REQUEST_TEMPLATE.md.

Reads PR_BODY, PR_AUTHOR and PR_AUTHOR_TYPE from the environment, prints one
error per problem and exits 1 if any are found. Uses only the standard library
so it runs on a bare GitHub-hosted runner.
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

_HTML_COMMENT = re.compile(r"<!--.*?-->", re.DOTALL)
# Fenced code is content, but a "# comment" line inside it must not read as a heading.
_CODE_FENCE = re.compile(r"^\s*(```|~~~).*?^\s*\1[^\n]*$", re.DOTALL | re.MULTILINE)
_REVIEW_BOX = re.compile(
    r"^\s*[-*]\s+\[(?P<mark>[ xX])\]\s+"
    + re.escape(REVIEW_BOX_TEXT.rstrip("."))
    + r"\.?\s*$",
    re.MULTILINE,
)
_HEADING = re.compile(r"^#{1,6}\s+(?P<title>.+?)\s*#*\s*$", re.MULTILINE)


def is_bot(*, author: str, author_type: str) -> bool:
    return author_type == "Bot" or author.endswith("[bot]") or author in BOT_AUTHORS


def _section_bodies(body: str) -> dict[str, str]:
    """Map each heading title (lowercased) to the text under it, up to the next heading."""
    matches = list(_HEADING.finditer(body))
    sections = {}
    for i, match in enumerate(matches):
        end = matches[i + 1].start() if i + 1 < len(matches) else len(body)
        sections[match.group("title").strip().lower()] = body[match.end() : end]
    return sections


def find_problems(body: str) -> list[str]:
    """Return one human-readable line per template problem in the PR description."""
    text = _CODE_FENCE.sub("[code block]", _HTML_COMMENT.sub("", body or ""))
    problems = []

    box = _REVIEW_BOX.search(text)
    if box is None:
        problems.append(
            f'Review box missing: add "- [ ] {REVIEW_BOX_TEXT}" at the top.'
        )
    elif box.group("mark") == " ":
        problems.append(f'Review box not ticked: "{REVIEW_BOX_TEXT}"')

    sections = _section_bodies(text)
    for heading in REQUIRED_HEADINGS:
        if heading.lower() not in sections:
            problems.append(f'Missing heading: "## {heading}"')

    what_and_why = sections.get(REQUIRED_HEADINGS[0].lower())
    if what_and_why is not None and not what_and_why.strip():
        problems.append(f'Empty section: "## {REQUIRED_HEADINGS[0]}"')

    return problems


def main() -> int:
    author = os.environ.get("PR_AUTHOR", "")
    if is_bot(author=author, author_type=os.environ.get("PR_AUTHOR_TYPE", "")):
        print(f"Bot-authored PR ({author}), skipped.")
        return 0

    problems = find_problems(os.environ.get("PR_BODY", ""))
    if not problems:
        print("PR description follows the template.")
        return 0

    lines = ["PR description does not follow the template:"]
    lines += [f"  - {problem}" for problem in problems]
    lines += [
        f"Template: {TEMPLATE_URL}",
        "Agents: see skills/pr-description. Only a human may tick the review box.",
    ]
    report = "\n".join(lines)
    print(report)
    for problem in problems:
        print(f"::error title=PR description::{problem}")
    summary_path = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary_path:
        with open(summary_path, "a", encoding="utf-8") as summary:
            summary.write(f"```\n{report}\n```\n")
    return 1


if __name__ == "__main__":
    sys.exit(main())
