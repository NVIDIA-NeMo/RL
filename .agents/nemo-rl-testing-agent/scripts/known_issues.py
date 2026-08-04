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

"""Cross-PR memory of failures the agent has already diagnosed.

Each labeled PR is tested on its own, but the breakages they hit are mostly not
their own: a bug on `main` fails the suite for every PR until its fix merges.
Without memory the agent re-derives the same diagnosis every time, and may open a
second fix PR for a bug that already has one. This registry makes "have we seen
this before?" a lookup, keyed by test name plus a normalized error signature.

Two things keep it from becoming a source of wrong answers:

* Entries retire themselves. ``refresh`` asks GitHub whether each fix PR is still
  open, so a merged fix stops excusing failures that are now real regressions.
* A match is not automatically believed. If the fix is already applied to the
  branch under test (per the integration manifest) and the failure still occurs,
  ``annotate`` says so loudly instead of quietly labelling it known -- the entry
  is stale, or this is a different bug wearing the same signature.

Usage:
    uv run --script known_issues.py annotate --results r.json [--integration m.json]
    uv run --script known_issues.py record --id <slug> --test <name> \
        --signature "<error line>" --diagnosis "..." --repo owner/name --fix-pr 3363
    uv run --script known_issues.py refresh
    uv run --script known_issues.py list
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# Everything that makes a signature specific to one run rather than to one bug.
# Two runs of the same failure differ in pids, addresses, line numbers, tensor
# shapes and absolute paths; the exception type and message shape do not.
NORMALIZERS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"0x[0-9a-f]+", re.IGNORECASE), "<addr>"),
    (re.compile(r"\bpid=\d+"), "pid=<n>"),
    (re.compile(r"\brank=?\s*\d+", re.IGNORECASE), "rank<n>"),
    (re.compile(r"(?<![\w/])/[\w./+-]+"), "<path>"),
    (re.compile(r"\bline \d+"), "line <n>"),
    (re.compile(r"\b\d+\.\d+\b"), "<float>"),
    (re.compile(r"\b\d+\b"), "<n>"),
    (re.compile(r"\s+"), " "),
]

STALE_WARNING = (
    "known-issue entry '{issue_id}' claims this is fixed by {repo}#{pr}, and that "
    "fix IS applied to the branch under test, yet the failure persists. Treat this "
    "as a NEW failure and investigate it; the entry is stale or the signature "
    "collides with a different bug."
)


def now_utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def normalize(signature: str) -> str:
    """Reduces an error line to the part that identifies the bug rather than the run."""
    text = signature.strip().lower()
    for pattern, replacement in NORMALIZERS:
        text = pattern.sub(replacement, text)
    return text.strip()


def registry_path(explicit: Path | None) -> Path:
    if explicit:
        return explicit
    from_env = os.environ.get("KNOWN_ISSUES_FILE")
    if from_env:
        return Path(os.path.expandvars(from_env)).expanduser()
    return Path.home() / ".nemo-rl-testing-agent" / "known_issues.json"


def load(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"issues": []}
    return json.loads(path.read_text())


def save(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n")


def find_match(
    issues: list[dict[str, Any]], test: str, signature: str
) -> dict[str, Any] | None:
    """Matches on test name AND signature.

    Both are required. Signature alone over-matches: `AssertionError:
    token_mult_prob_error` is the visible symptom of several unrelated numerics
    bugs. Test name alone over-matches even more badly.
    """
    if not signature:
        return None
    normalized = normalize(signature)
    for issue in issues:
        if issue.get("state") not in (None, "open"):
            continue
        tests = issue.get("tests") or []
        if tests and test not in tests:
            continue
        known = issue.get("signature_normalized") or normalize(
            issue.get("signature", "")
        )
        if not known:
            continue
        if known == normalized or known in normalized or normalized in known:
            return issue
    return None


def applied_prs(manifest: Path | None, results: dict[str, Any]) -> set[int]:
    """Fixes that were applied to the revision these results actually came from.

    The manifest describes the integration branch *now*, which moves every time a
    fix is raised. Judging a run against a later manifest credits it with fixes it
    never carried: a cached baseline was labelled "regressed after its fix was
    already applied" for three tests within minutes of that fix being created,
    which is a confidently wrong answer and worse than no answer. So the manifest
    only counts when its sha is the one the run tested.
    """
    if not manifest or not manifest.exists():
        return set()
    payload = json.loads(manifest.read_text())
    tested_sha = (results.get("prep") or {}).get("nemo_rl_sha", "")
    if not tested_sha or tested_sha != payload.get("integration_sha"):
        return set()
    return {int(entry["pr"]) for entry in payload.get("applied", [])}


def cmd_annotate(args: argparse.Namespace) -> int:
    path = registry_path(args.registry)
    registry = load(path)
    issues = registry.get("issues", [])
    results = json.loads(args.results.read_text())
    already_applied = applied_prs(args.integration, results)

    matched = 0
    stale = 0
    fresh: list[str] = []

    for test in results.get("tests", []):
        # Clear any previous verdict first. Annotation is re-run whenever the
        # registry or the manifest changes, and a label left over from the last
        # pass would outlive the reasoning behind it -- a corrected "stale" verdict
        # stayed on the tracking issue after the mistake that produced it was
        # fixed, because rendering read the file rather than this run's output.
        test.pop("known_issue", None)
        test.pop("known_issue_stale", None)

        # A clean pass needs no attribution; `pass (suspect)` does, since it is a
        # failure the test happened to swallow.
        if test.get("status") == "pass":
            continue
        issue = find_match(
            issues, test.get("name", ""), test.get("error_signature", "")
        )
        if issue is None:
            fresh.append(test.get("name", "?"))
            continue

        fix_pr = issue.get("fix_pr")
        if fix_pr and int(fix_pr) in already_applied:
            test["known_issue_stale"] = issue.get("id")
            message = STALE_WARNING.format(
                issue_id=issue.get("id"), repo=issue.get("repo", ""), pr=fix_pr
            )
            print(f"STALE   {test.get('name')}: {message}")
            stale += 1
            continue

        test["known_issue"] = issue.get("id")
        test["comment"] = build_comment(issue)
        issue["last_seen_utc"] = now_utc()
        matched += 1
        fix = f"{issue.get('repo')}#{fix_pr}" if fix_pr else "no fix raised yet"
        print(f"known   {test.get('name')} -> {issue.get('id')} ({fix})")

    for name in fresh:
        print(f"new     {name}: no registry entry; needs investigation")

    args.results.write_text(json.dumps(results, indent=2) + "\n")
    save(path, registry)
    print(
        f"known_issues: {matched} known, {stale} stale-match, {len(fresh)} needing "
        f"investigation -> {args.results}"
    )
    return 0


def build_comment(issue: dict[str, Any]) -> str:
    parts = [issue.get("diagnosis", "").strip()]
    repo = issue.get("repo", "")
    fix_pr = issue.get("fix_pr")
    if repo and fix_pr:
        # Just the link. Whether it is "this PR's fault" is the caller's framing --
        # apply_baseline.py adds that wording for a PR comment, and the tracking
        # issue has no "this PR" to speak of.
        parts.append(
            f"Fix for review: [{repo}#{fix_pr}](https://github.com/{repo}/pull/{fix_pr})."
        )
    elif issue.get("tracking_url"):
        parts.append(f"Already diagnosed; tracked at {issue['tracking_url']}.")
    return " ".join(part for part in parts if part)


def cmd_record(args: argparse.Namespace) -> int:
    path = registry_path(args.registry)
    registry = load(path)
    issues = registry.setdefault("issues", [])

    existing = next((issue for issue in issues if issue.get("id") == args.id), None)
    entry = existing or {"id": args.id, "first_seen_utc": now_utc()}

    if args.test:
        entry["tests"] = sorted(set(entry.get("tests", [])) | set(args.test))
    if args.signature:
        entry["signature"] = args.signature
        entry["signature_normalized"] = normalize(args.signature)
    for field, value in (
        ("diagnosis", args.diagnosis),
        ("repo", args.repo),
        ("fix_branch", args.fix_branch),
        ("tracking_url", args.tracking_url),
        ("first_seen_megatron_pr", args.first_seen_megatron_pr),
    ):
        if value:
            entry[field] = value
    if args.fix_pr:
        entry["fix_pr"] = args.fix_pr
    entry.setdefault("state", "open")
    entry["updated_utc"] = now_utc()

    if existing is None:
        issues.append(entry)
    save(path, registry)
    print(f"known_issues: recorded '{args.id}' -> {path}")
    return 0


def cmd_refresh(args: argparse.Namespace) -> int:
    """Retires entries whose fix has landed.

    A registry that never forgets is dangerous: once a fix merges, the same
    failure recurring is a real regression, and an entry still saying "known,
    fix in review" would wave it through.
    """
    path = registry_path(args.registry)
    registry = load(path)
    changed = 0

    for issue in registry.get("issues", []):
        repo, fix_pr = issue.get("repo"), issue.get("fix_pr")
        if not repo or not fix_pr:
            continue
        try:
            state = subprocess.run(
                [
                    "gh",
                    "pr",
                    "view",
                    str(fix_pr),
                    "--repo",
                    repo,
                    "--json",
                    "state",
                    "--jq",
                    ".state",
                ],
                capture_output=True,
                text=True,
                check=True,
            ).stdout.strip()
        except (subprocess.CalledProcessError, FileNotFoundError) as error:
            print(f"could not check {repo}#{fix_pr}: {error}", file=sys.stderr)
            continue

        new_state = {"OPEN": "open", "MERGED": "merged", "CLOSED": "closed"}.get(
            state, "open"
        )
        if new_state != issue.get("state"):
            print(f"{issue['id']}: {issue.get('state')} -> {new_state}")
            issue["state"] = new_state
            issue["updated_utc"] = now_utc()
            changed += 1

    save(path, registry)
    print(f"known_issues: {changed} entry state(s) changed -> {path}")
    return 0


def cmd_list(args: argparse.Namespace) -> int:
    path = registry_path(args.registry)
    registry = load(path)
    issues = registry.get("issues", [])
    if not issues:
        print("known_issues: registry is empty")
        return 0
    for issue in issues:
        fix = f"{issue.get('repo', '?')}#{issue.get('fix_pr', '?')}"
        print(f"[{issue.get('state', 'open'):6}] {issue.get('id')}  fix={fix}")
        print(f"          tests: {', '.join(issue.get('tests', [])) or '<any>'}")
        print(f"          sig:   {issue.get('signature', '')[:100]}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", type=Path, help="Override $KNOWN_ISSUES_FILE.")
    sub = parser.add_subparsers(dest="command", required=True)

    annotate = sub.add_parser(
        "annotate", help="Label a results JSON with known issues."
    )
    annotate.add_argument("--results", type=Path, required=True)
    annotate.add_argument(
        "--integration",
        type=Path,
        help="integration.json manifest, so an already-applied fix is not credited twice.",
    )
    annotate.set_defaults(func=cmd_annotate)

    record = sub.add_parser("record", help="Add or update an entry.")
    record.add_argument(
        "--id", required=True, help="Stable slug, e.g. mcore-5918-prompt-tokens."
    )
    record.add_argument(
        "--test", action="append", help="Test this affects; repeatable."
    )
    record.add_argument("--signature", help="A representative error line.")
    record.add_argument(
        "--diagnosis", help="One or two sentences a PR author can act on."
    )
    record.add_argument("--repo", help="Repo the fix lands in, owner/name.")
    record.add_argument("--fix-pr", type=int)
    record.add_argument("--fix-branch")
    record.add_argument("--tracking-url")
    record.add_argument("--first-seen-megatron-pr", type=int)
    record.set_defaults(func=cmd_record)

    refresh = sub.add_parser(
        "refresh", help="Retire entries whose fix merged or closed."
    )
    refresh.set_defaults(func=cmd_refresh)

    listing = sub.add_parser("list", help="Show the registry.")
    listing.set_defaults(func=cmd_list)

    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
