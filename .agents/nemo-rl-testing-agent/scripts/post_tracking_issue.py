#!/usr/bin/env python3
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

# /// script
# requires-python = ">=3.10"
# dependencies = []
# ///

"""Renders and upserts the watchdog's single tracking issue.

One long-lived issue, edited in place, rather than a comment per PR or an issue
per failure. The audience is whoever wants to know "is NeMo-RL currently healthy
against megatron-core main, and who is on it" -- a question that should have one
answer in one place, not one answer per labeled PR.

The delta section is the part people read: `still broken` is background noise,
`broke since the last pass` is someone's afternoon.

Usage:
    uv run --script post_tracking_issue.py --suite l1 --results r.json \
        --meta-env m.env --integration i.json [--previous p.json] \
        --out body.md [--publish]
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

MARKER = "<!-- nrlta-watchdog -->"
TITLE = "NeMo-RL vs megatron-core `main`: current functional-test breakage"
FAILING = {"fail", "fail (pre-existing)", "incomplete", "pass (suspect)"}
ICONS = {
    "pass": "✅",
    "fail": "❌",
    "fail (pre-existing)": "❌",
    "incomplete": "⚠️",
    "pass (suspect)": "🚩",
}


def load_env(path: Path | None) -> dict[str, str]:
    values: dict[str, str] = {}
    if not path or not path.exists():
        return values
    for line in path.read_text().splitlines():
        if "=" in line and not line.startswith("#"):
            key, _, value = line.partition("=")
            values[key.strip()] = value.strip()
    return values


def short(sha: str) -> str:
    return sha[:8] if sha else "unknown"


def render(
    suite: str,
    results: dict[str, Any],
    meta: dict[str, str],
    manifest: dict[str, Any],
    previous: dict[str, Any] | None,
) -> str:
    tests = results.get("tests", [])
    failing = [test for test in tests if test.get("status") in FAILING]

    lines = [
        MARKER,
        f"## {suite.upper()} functional tests against megatron-core `main`",
        "",
        f"Last pass: {datetime.now(timezone.utc):%Y-%m-%d %H:%M} UTC · "
        f"{len(tests) - len(failing)}/{len(tests)} green.",
        "",
        "This issue is maintained automatically and edited in place. It tracks breakage "
        "that is **not** caused by any particular Megatron-LM pull request, so that "
        "labeled PRs do not each have to rediscover it.",
        "",
        "### Stack under test",
        "",
        f"- megatron-core: `main` @ `{short(meta.get('MCORE_SHA', ''))}`",
        f"- Megatron-Bridge: `{short(meta.get('BRIDGE_SHA', ''))}`",
        f"- NeMo-RL: `main` plus the pending fixes below @ `{short(meta.get('NEMO_RL_SHA', ''))}`",
        "",
    ]

    applied = manifest.get("applied", [])
    if applied:
        lines += [
            "### Fixes applied here but not yet merged",
            "",
            "These are in review. Until they land, every labeled PR is tested with them "
            "applied, so their breakage does not get re-reported on each one.",
            "",
        ]
        for item in applied:
            url = item.get("url", "")
            link = f"[#{item['pr']}]({url})" if url else f"#{item['pr']}"
            lines.append(f"- {link} — {item.get('title', '')}")
        lines.append("")
    skipped = [
        item for item in manifest.get("skipped", []) if item.get("reason") == "conflict"
    ]
    if skipped:
        lines += [
            "> **Needs a human:** "
            + ", ".join(f"#{item['pr']}" for item in skipped)
            + " no longer applies cleanly on top of `main` and was left out. Rebase it.",
            "",
        ]

    if previous:
        previous_status = {
            test.get("name"): test.get("status") for test in previous.get("tests", [])
        }
        newly_broken = [
            test["name"]
            for test in tests
            if test.get("status") in FAILING
            and previous_status.get(test["name"]) not in FAILING
            and test["name"] in previous_status
        ]
        newly_fixed = [
            test["name"]
            for test in tests
            if test.get("status") == "pass"
            and previous_status.get(test["name"]) in FAILING
        ]
        if newly_broken or newly_fixed:
            lines += ["### Since the previous pass", ""]
            for name in newly_broken:
                lines.append(f"- 🔴 **newly broken**: `{name}`")
            for name in newly_fixed:
                lines.append(f"- 🟢 recovered: `{name}`")
            lines.append("")

    lines += [
        "### Current status",
        "",
        "| Test | Status | Assessment |",
        "| --- | --- | --- |",
    ]
    # One registry entry routinely claims every test in the suite, so writing its
    # diagnosis into each row turned the table into the same paragraph repeated
    # seven times. Reference it from the row and write it out once below.
    diagnoses: dict[str, str] = {}
    for test in tests:
        status = test.get("status", "unknown")
        icon = ICONS.get(status, "❔")
        issue_id = test.get("known_issue")
        if status == "pass":
            assessment = "—"
        elif test.get("known_issue_stale"):
            assessment = (
                "**Regressed after its fix was already applied. Needs investigation.**"
            )
        elif issue_id:
            diagnosis = (test.get("comment") or "Known issue.").strip()
            if "](http" not in diagnosis:
                diagnosis += " **No fix raised yet.**"
            diagnoses.setdefault(issue_id, diagnosis)
            assessment = f"Known issue `{issue_id}` (see below)."
        else:
            assessment = "**Not yet diagnosed.**"
        if test.get("metric_failure"):
            assessment = f"{test['metric_failure']}. {assessment}"
        lines.append(f"| `{test.get('name', '?')}` | {icon} {status} | {assessment} |")

    if diagnoses:
        lines += ["", "### Diagnoses", ""]
        for issue_id, diagnosis in diagnoses.items():
            affected = [
                t.get("name", "?") for t in tests if t.get("known_issue") == issue_id
            ]
            lines += [
                f"**`{issue_id}`** — {len(affected)} test(s): "
                + ", ".join(f"`{name}`" for name in affected),
                "",
                diagnosis,
                "",
            ]

    if any(test.get("status") == "pass (suspect)" for test in tests):
        lines += [
            "",
            "🚩 means the test exited 0 while swallowing errors, so its green is not "
            "trustworthy.",
        ]
    return "\n".join(lines) + "\n"


def find_issue(repo: str) -> int | None:
    """Finds this issue by its marker, scanning our own open issues.

    Deliberately not `--search TITLE`: the title contains backticks and a colon,
    which GitHub's search tokenizer does not match literally, so the lookup
    silently found nothing and a second run opened a duplicate tracking issue
    instead of editing the first.
    """
    result = subprocess.run(
        [
            "gh",
            "issue",
            "list",
            "--repo",
            repo,
            "--state",
            "open",
            "--author",
            "@me",
            "--limit",
            "100",
            "--json",
            "number,body",
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    matches = [
        int(issue["number"])
        for issue in json.loads(result.stdout or "[]")
        if MARKER in (issue.get("body") or "")
    ]
    if len(matches) > 1:
        # Editing an arbitrary one of several would leave the others stale and
        # contradicting it, which is worse than refusing.
        raise RuntimeError(
            f"{len(matches)} issues in {repo} carry the watchdog marker "
            f"({', '.join(f'#{n}' for n in sorted(matches))}); close all but one."
        )
    return matches[0] if matches else None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", required=True)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--meta-env", type=Path)
    parser.add_argument("--integration", type=Path)
    parser.add_argument("--previous", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--publish",
        action="store_true",
        help="Create or edit the GitHub issue. Without it, only --out is written.",
    )
    parser.add_argument(
        "--repo", default=os.environ.get("NEMO_RL_REPO", "NVIDIA-NeMo/RL")
    )
    args = parser.parse_args()

    results = json.loads(args.results.read_text())
    manifest = (
        json.loads(args.integration.read_text())
        if args.integration and args.integration.exists()
        else {}
    )
    previous = (
        json.loads(args.previous.read_text())
        if args.previous and args.previous.exists()
        else None
    )

    body = render(args.suite, results, load_env(args.meta_env), manifest, previous)
    args.out.write_text(body)
    print(f"tracking issue body written to {args.out}")

    if not args.publish:
        return 0

    number = find_issue(args.repo)
    if number is None:
        result = subprocess.run(
            [
                "gh",
                "issue",
                "create",
                "--repo",
                args.repo,
                "--title",
                TITLE,
                "--body-file",
                str(args.out),
            ],
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            print(result.stderr, file=sys.stderr)
            return 1
        print(f"created {result.stdout.strip()}")
    else:
        result = subprocess.run(
            [
                "gh",
                "issue",
                "edit",
                str(number),
                "--repo",
                args.repo,
                "--body-file",
                str(args.out),
            ],
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            print(result.stderr, file=sys.stderr)
            return 1
        print(f"updated {args.repo}#{number}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
