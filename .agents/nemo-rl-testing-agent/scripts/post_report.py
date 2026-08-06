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


def slug_of(url: str) -> str:
    """owner/name out of a clone URL, in either of the forms we clone with."""
    match = re.search(r"(?:github\.com[:/])([^/]+/[^/]+?)(?:\.git)?/?$", url or "")
    return match.group(1) if match else ""


def commit_link(sha: str, url: str) -> str:
    short = sha[:8] if re.fullmatch(r"[0-9a-f]{40}", sha or "") else sha
    slug = slug_of(url)
    if not short:
        return "?"
    if not slug:
        return f"`{short}`"
    return f"[`{short}`](https://github.com/{slug}/commit/{sha})"


def describe_ref(ref: str, url: str) -> str:
    """What the ref means, not just what it is called.

    A reader who has to decode `refs/pull/5382/head` or tell a 40-character
    Bridge sha apart from a branch name is being asked to reconstruct the run,
    which is the job this table exists to do for them.
    """
    if not ref or ref.startswith("<"):
        return ref or "?"
    pull = re.fullmatch(r"refs/pull/(\d+)/head", ref)
    if pull:
        slug = slug_of(url)
        number = pull.group(1)
        link = (
            f"[#{number}](https://github.com/{slug}/pull/{number})"
            if slug
            else f"#{number}"
        )
        return f"{link} head"
    if re.fullmatch(r"[0-9a-f]{40}", ref):
        return "the sha NeMo-RL pins"
    return f"`{ref.removeprefix('refs/heads/')}`"


def describe_bridge_ref(ref: str) -> str:
    """Whether the Bridge under test is the one NeMo-RL ships, or a substitute.

    Only a sha is the pin. Anything else is an override -- normally a fix branch
    the harness carried because the fix is still in review -- and that changes
    what a green table means: the suite passed against a Bridge that nobody can
    `pip install` yet. Saying only the branch name leaves the reader to infer it.
    """
    if re.fullmatch(r"[0-9a-f]{40}", ref or ""):
        return "the sha NeMo-RL pins"
    if not ref or ref.startswith("<"):
        return "whatever the container image carries"
    return f"`{ref}` — an override, **not** the Bridge NeMo-RL pins"


def render_stack(
    payloads: list[dict[str, Any]], heading: str = "**Exactly what was tested**"
) -> list[str]:
    """Exactly which three revisions produced the table below.

    Every one of them is a choice the harness made rather than an obvious
    default: megatron-core is the PR's head, Megatron-Bridge is either NeMo-RL's
    pin or an unmerged fix branch checked out in its place, and NeMo-RL is
    normally a fork branch of `main` plus fixes still in review. A reader who
    cannot see that cannot tell what a green table is a statement about, and the
    shas were previously retyped by hand into `--meta`, where the refs were lost
    and a typo looked exactly like a result.
    """
    prep: dict[str, str] = {}
    baseline: dict[str, str] = {}
    for payload in payloads:
        prep = prep or (payload.get("prep") or {})
        baseline = baseline or (payload.get("baseline") or {})
    if not prep:
        return []

    rows = [
        (
            "megatron-core",
            describe_ref(prep.get("mcore_fetch_ref", ""), prep.get("mcore_url", "")),
            commit_link(prep.get("mcore_sha", ""), prep.get("mcore_url", "")),
        ),
        (
            "Megatron-Bridge",
            describe_bridge_ref(prep.get("bridge_fetch_ref", "")),
            commit_link(prep.get("bridge_sha", ""), prep.get("bridge_url", "")),
        ),
        (
            "NeMo-RL",
            describe_ref(
                prep.get("nemo_rl_fetch_ref", ""), prep.get("nemo_rl_url", "")
            ),
            commit_link(prep.get("nemo_rl_sha", ""), prep.get("nemo_rl_url", "")),
        ),
    ]
    lines = [
        heading,
        "",
        "| Component | Ref | Commit |",
        "| --- | --- | --- |",
    ]
    lines += [f"| {name} | {ref} | {sha} |" for name, ref, sha in rows]
    if baseline.get("mcore_sha"):
        lines += [
            "",
            "Compared against a baseline of the same suite on megatron-core `main` at "
            + commit_link(baseline["mcore_sha"], prep.get("mcore_url", ""))
            + ".",
        ]
    lines += [""]
    return lines


# Superseded by the stack table, which says the same thing with the ref that
# gives each sha its meaning. Kept out of the header rather than rejected at the
# CLI so that an older invocation still renders, just without the duplication.
STACK_META_KEYS = {
    "mcore_sha",
    "bridge_sha",
    "nemo_rl_sha",
    "base_sha",
    "baseline_main_sha",
}


def render(
    state: str,
    results_files: list[Path],
    meta: dict[str, str],
    note: str,
    integration: Path | None = None,
) -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    payloads = [json.loads(path.read_text()) for path in results_files]
    stack = render_stack(payloads)
    meta_bits = [
        f"{key.replace('_', ' ')}: `{value}`"
        for key, value in meta.items()
        if value and not (stack and key in STACK_META_KEYS)
    ]
    meta_bits.append(f"updated {stamp}")

    lines = [
        MARKER,
        "### NeMo-RL functional tests",
        "",
        " · ".join(meta_bits),
        "",
        *stack,
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
        for payload in payloads:
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
