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

"""Render the results table and upsert it as a sticky comment on a PR.

The comment is identified by an HTML marker, so re-running the agent on the same
PR edits the existing comment instead of adding a new one. Because it is sticky,
the agent claims the comment as soon as testing starts (`--state running`) and
overwrites it as the run progresses; a PR author should never have to wonder
whether the agent picked their PR up. A run that dies before any test executes
still reports (`--state infra`) rather than leaving the PR silent.

Usage:
    uv run --script post_report.py --pr 5700 --state running \
        --meta mcore_sha=abc1234

    uv run --script post_report.py --pr 5700 \
        --results l1.json --results l2.json \
        --meta mcore_sha=abc1234 --meta base_sha=def5678
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

MARKER = "<!-- nemo-rl-testing-agent -->"

# The comment is read by external Megatron-LM contributors who have no access to
# our cluster. A `nrlta-pr5700-l1-a7` run name or a /lustre path means nothing to
# them and actively misleads -- one reviewer reasonably read a run name as a branch
# they were supposed to look at. Internal references belong in the local ledger, so
# rendering refuses to emit them rather than trusting every caller's --note.
INTERNAL_REF_PATTERNS = [
    (re.compile(r"\bnrlta-[\w.-]+"), "an internal cog run name"),
    (re.compile(r"/lustre/\S+"), "a cluster filesystem path"),
    (re.compile(r"\b(?:oci-hsg|oci)\s*:\s*/\S+"), "a cluster ssh destination"),
]

STATUS_ICONS = {
    "pass": "✅",
    "pass (suspect)": "🚩",
    "fail": "❌",
    "fail (pre-existing)": "⚠️",
    "fail (infra)": "⚠️",
    "fixed": "🛠️",
    "not run": "⏭️",
    "incomplete": "⏱️",
}


def load_config(config_path: Path) -> dict[str, str]:
    """Reads the plain KEY=value config.env, expanding $HOME."""
    config: dict[str, str] = {}
    if not config_path.is_file():
        return config
    for line in config_path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        config[key.strip()] = os.path.expandvars(value.strip())
    return config


def find_internal_refs(body: str) -> list[str]:
    """Returns human-readable descriptions of any internal-only references found."""
    found: list[str] = []
    for pattern, description in INTERNAL_REF_PATTERNS:
        for match in dict.fromkeys(pattern.findall(body)):
            found.append(f"{match!r} looks like {description}")
    return found


def cell(text: str) -> str:
    """Makes arbitrary text safe for a single markdown table cell."""
    collapsed = re.sub(r"\s*\n\s*", "<br>", str(text).strip())
    return collapsed.replace("|", "\\|")


def render_pending_fixes(manifest_path: Path | None) -> list[str]:
    """Discloses that the suite ran against NeMo-RL `main` plus unmerged fixes.

    Without saying so, a green table would be misleading: it would look like the
    PR is fine against released NeMo-RL when it is really fine against a NeMo-RL
    that carries patches nobody has merged yet.
    """
    if not manifest_path or not manifest_path.exists():
        return []
    manifest = json.loads(manifest_path.read_text())
    applied = manifest.get("applied", [])
    if not applied:
        return []
    links = ", ".join(
        f"[#{item['pr']}]({item['url']})" if item.get("url") else f"#{item['pr']}"
        for item in applied
    )
    return [
        f"<sub>Run against NeMo-RL `main` plus {len(applied)} fix(es) already in review "
        f"({links}), so breakage that is already being dealt with does not show up here "
        "as if it were yours.</sub>",
        "",
    ]


def render(
    state: str,
    results_files: list[Path],
    meta: dict[str, str],
    note: str,
    integration: Path | None = None,
) -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    meta_bits = [
        f"{key.replace('_', ' ')}: `{value}`" for key, value in meta.items() if value
    ]
    meta_bits.append(f"updated {stamp}")

    lines = [
        MARKER,
        "### NeMo-RL functional tests",
        "",
        " · ".join(meta_bits),
        "",
        *render_pending_fixes(integration),
    ]

    if state == "running":
        lines.append(
            "⏳ Running the NeMo-RL functional suites against this PR. This comment will be updated with the results."
        )
        if note:
            lines += ["", note]
    elif state == "infra":
        lines.append(
            "⚠️ **The run did not reach the tests.** This is an infrastructure problem on the NeMo-RL side, not a verdict on this PR."
        )
        lines += ["", note or "See the run log for details."]
    else:
        rows: list[tuple[str, str, str]] = []
        for path in results_files:
            payload: dict[str, Any] = json.loads(path.read_text())
            for test in payload.get("tests", []):
                status = test.get("status", "unknown")
                icon = STATUS_ICONS.get(status, "")
                comment = test.get("comment") or ""
                if not comment and status not in ("pass", "not run"):
                    comment = test.get("error_signature", "")
                rows.append(
                    (test.get("name", "?"), f"{icon} {status}".strip(), comment)
                )

        lines += ["| Test | Status | Comment |", "| --- | --- | --- |"]
        if rows:
            lines.extend(
                f"| `{cell(name)}` | {cell(status)} | {cell(comment)} |"
                for name, status, comment in rows
            )
        else:
            lines.append(
                "| _no tests ran_ | ⚠️ | The job failed before any test started; see the run log. |"
            )
        if note:
            lines += ["", note]

    lines += [
        "",
        "<sub>Posted by the nemo-rl-testing-agent. Re-runs edit this comment in place.</sub>",
    ]
    return "\n".join(lines) + "\n"


def gh(args: list[str]) -> str:
    result = subprocess.run(["gh", *args], capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"gh {' '.join(args)} failed: {result.stderr.strip()}")
    return result.stdout


def find_sticky_comment_id(repo: str, pr: int) -> str | None:
    out = gh(
        [
            "api",
            f"repos/{repo}/issues/{pr}/comments",
            "--paginate",
            "--jq",
            f'.[] | select(.body | contains("{MARKER}")) | .id',
        ]
    )
    ids = [line.strip() for line in out.splitlines() if line.strip()]
    return ids[0] if ids else None


def upsert(repo: str, pr: int, body: str) -> str:
    with tempfile.NamedTemporaryFile("w", suffix=".md", delete=False) as handle:
        handle.write(body)
        body_path = handle.name
    try:
        comment_id = find_sticky_comment_id(repo, pr)
        if comment_id:
            gh(
                [
                    "api",
                    "-X",
                    "PATCH",
                    f"repos/{repo}/issues/comments/{comment_id}",
                    "-F",
                    f"body=@{body_path}",
                ]
            )
            return f"updated comment {comment_id} on {repo}#{pr}"
        gh(
            [
                "api",
                "-X",
                "POST",
                f"repos/{repo}/issues/{pr}/comments",
                "-F",
                f"body=@{body_path}",
            ]
        )
        return f"posted a new comment on {repo}#{pr}"
    finally:
        os.unlink(body_path)


def main() -> int:
    default_config = Path(__file__).resolve().parent.parent / "config.env"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pr", type=int, required=True, help="Megatron-LM PR number.")
    parser.add_argument(
        "--repo", default="", help="Overrides MEGATRON_REPO from config.env."
    )
    parser.add_argument(
        "--state",
        choices=("running", "results", "infra"),
        default="results",
        help="running: claim the comment before tests start. results: render the table. infra: the run never reached the tests.",
    )
    parser.add_argument(
        "--results",
        type=Path,
        action="append",
        default=[],
        help="Results JSON (repeatable, rendered in order). Required for --state results.",
    )
    parser.add_argument(
        "--note",
        default="",
        help="Extra markdown shown below the body (e.g. the failing prep step).",
    )
    parser.add_argument(
        "--meta",
        action="append",
        default=[],
        help="key=value shown in the comment header (repeatable).",
    )
    parser.add_argument(
        "--integration",
        type=Path,
        help="integration.json, to disclose the unmerged fixes the run carried.",
    )
    parser.add_argument("--config", type=Path, default=default_config)
    parser.add_argument(
        "--out", type=Path, help="Also write the rendered markdown here."
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the comment instead of posting it.",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    repo = args.repo or config.get("MEGATRON_REPO", "")
    if not repo:
        print(
            "post_report: no repo given and MEGATRON_REPO is missing from config.env",
            file=sys.stderr,
        )
        return 1

    if args.state == "results" and not args.results:
        print(
            "post_report: --state results needs at least one --results file",
            file=sys.stderr,
        )
        return 1

    missing = [str(path) for path in args.results if not path.is_file()]
    if missing:
        print(
            f"post_report: missing results file(s): {', '.join(missing)}",
            file=sys.stderr,
        )
        return 1

    meta: dict[str, str] = {}
    for item in args.meta:
        if "=" not in item:
            print(
                f"post_report: --meta expects key=value, got '{item}'", file=sys.stderr
            )
            return 1
        key, value = item.split("=", 1)
        meta[key] = value

    body = render(args.state, args.results, meta, args.note, args.integration)

    internal = find_internal_refs(body)
    if internal:
        print(
            "post_report: refusing to post -- the comment contains references only "
            "meaningful inside NVIDIA:",
            file=sys.stderr,
        )
        for item in internal:
            print(f"  {item}", file=sys.stderr)
        print(
            "Describe the evidence in words instead, and keep run names and cluster "
            "paths in the local ledger.",
            file=sys.stderr,
        )
        return 1

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(body)

    if args.dry_run:
        print(body)
        return 0

    print(upsert(repo, args.pr, body))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
