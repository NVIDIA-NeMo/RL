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

"""Queue of corrections the agent owes its own instructions.

``known_issues.py`` remembers *what was broken*. This remembers *what the agent
should do differently next time* -- a flag that has to be passed, a step whose
order matters, a symptom that means something other than what the skill says.
Without it that knowledge lives only in the sweep's context window and dies with
it, so the next sweep repeats the mistake and pays for it in cluster hours.

Two properties make the queue worth having over just editing a skill on the spot:

* It survives an interrupted sweep. A learning is usually found mid-PR, when
  stopping to edit a skill and raise a PR would strand a GPU job; recording takes
  a second and the promotion happens at the end of the sweep.
* It notices when a skill edit did not work. Re-recording an id that was already
  promoted means the instruction was written down and the agent still got it
  wrong -- the wording is too weak, or it sits somewhere nothing reads. That is
  reported loudly rather than silently counted, because the second occurrence is
  the evidence that the fix has to be structural (a script guard, a config key)
  rather than another sentence of prose.

Usage:
    uv run --script learnings.py record --id <slug> \
        --trigger "what happened" --lesson "what to do instead" \
        --target .agents/contributor-skills/megatron-pr-test-run/SKILL.md \
        [--severity blocking] [--context-pr 5700]
    uv run --script learnings.py list [--state pending] [--format markdown]
    uv run --script learnings.py resolve --id <slug> --as promoted --pr 2931
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[3]

# A lesson ends up in a public repo and has to apply to runs other than the one
# that produced it. Both of those rule out the same tokens post_report.py rejects
# in a PR comment: a run name or a cluster path is by construction specific to a
# single run, so a "lesson" containing one is a ledger note that took a wrong
# turn.
INTERNAL_REF_PATTERNS = [
    (re.compile(r"\bnrlta-[\w.-]+"), "an internal cog run name"),
    (re.compile(r"/lustre/\S+"), "a cluster filesystem path"),
    (re.compile(r"/Users/\S+|/home/\S+"), "a local home directory path"),
]

SEVERITIES = ("blocking", "routine")
STATES = ("pending", "promoted", "rejected")

REGRESSION_WARNING = (
    "REGRESSED '{issue_id}': this was promoted on {promoted_utc} into {targets} "
    "and the agent hit it again anyway. Prose did not hold. Prefer a structural "
    "fix this time -- a guard in a script, a config key, or a checklist line in "
    "the agent definition -- over restating the sentence."
)


def now_utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def queue_path(explicit: Path | None) -> Path:
    if explicit:
        return explicit
    from_env = os.environ.get("LEARNINGS_FILE")
    if from_env:
        return Path(os.path.expandvars(from_env)).expanduser()
    return Path.home() / ".nemo-rl-testing-agent" / "learnings.json"


def load(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"learnings": []}
    return json.loads(path.read_text())


def save(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n")


def check_lesson(lesson: str) -> list[str]:
    problems = []
    if len(lesson.split()) < 5:
        problems.append(
            "the lesson is too short to act on; say what a future run should do "
            "differently, not just what broke"
        )
    for pattern, what in INTERNAL_REF_PATTERNS:
        if pattern.search(lesson):
            problems.append(
                f"the lesson names {what}, which is specific to one run; keep that "
                "in ledger.md and state the general rule here"
            )
    return problems


def resolve_target(raw: str) -> tuple[str, str | None]:
    """Maps a target path to the file an edit must actually touch.

    Skills are reachable under two paths: the real file in
    ``.agents/contributor-skills/`` and a symlink in ``.claude/skills/``. Writing
    through the symlink lands in the right file but reading the path back tells
    the next agent to edit the link, so the canonical path is recorded and the
    symlink is reported.
    """
    candidate = Path(raw)
    if not candidate.is_absolute():
        candidate = REPO_ROOT / candidate
    if not candidate.exists():
        return raw, f"target does not exist: {raw}"
    real = candidate.resolve()
    try:
        canonical = str(real.relative_to(REPO_ROOT))
    except ValueError:
        return str(real), None
    note = None
    if real != candidate:
        note = f"{raw} resolves through a symlink; recorded {canonical} instead"
    return canonical, note


def cmd_record(args: argparse.Namespace) -> int:
    problems = check_lesson(args.lesson)
    if problems:
        for problem in problems:
            print(f"learnings: refusing to record -- {problem}", file=sys.stderr)
        return 2

    targets: list[str] = []
    for raw in args.target:
        canonical, note = resolve_target(raw)
        if note and note.startswith("target does not exist"):
            print(f"learnings: refusing to record -- {note}", file=sys.stderr)
            return 2
        if note:
            print(f"learnings: {note}")
        targets.append(canonical)

    path = queue_path(args.queue)
    payload = load(path)
    learnings = payload.setdefault("learnings", [])
    existing = next((item for item in learnings if item.get("id") == args.id), None)

    if existing is None:
        entry: dict[str, Any] = {
            "id": args.id,
            "first_seen_utc": now_utc(),
            "occurrences": 1,
        }
        learnings.append(entry)
    else:
        entry = existing
        entry["occurrences"] = int(entry.get("occurrences", 1)) + 1
        if entry.get("state") == "promoted":
            resolution = entry.get("resolution") or {}
            print(
                REGRESSION_WARNING.format(
                    issue_id=args.id,
                    promoted_utc=resolution.get("utc", "?"),
                    targets=", ".join(entry.get("targets", [])) or "?",
                ),
                file=sys.stderr,
            )
            entry["regressed"] = True

    entry["trigger"] = args.trigger
    entry["lesson"] = args.lesson
    entry["targets"] = sorted(set(entry.get("targets", [])) | set(targets))
    entry["severity"] = args.severity
    if args.context_pr:
        seen_on = set(entry.get("seen_on_prs", [])) | {args.context_pr}
        entry["seen_on_prs"] = sorted(seen_on)
    entry["state"] = "pending"
    entry["updated_utc"] = now_utc()

    save(path, payload)
    marker = "BLOCKING " if args.severity == "blocking" else ""
    print(
        f"learnings: {marker}recorded '{args.id}' "
        f"(seen {entry['occurrences']}x) -> {path}"
    )
    if args.severity == "blocking":
        print(
            "learnings: blocking means the current sweep is wrong without it -- "
            "apply the edit to the working tree now, then keep going."
        )
    return 0


def render_text(entries: list[dict[str, Any]]) -> str:
    lines = []
    for entry in entries:
        flags = []
        if entry.get("severity") == "blocking":
            flags.append("BLOCKING")
        if entry.get("regressed"):
            flags.append("REGRESSED")
        suffix = f"  [{' '.join(flags)}]" if flags else ""
        lines.append(f"[{entry.get('state', 'pending'):8}] {entry['id']}{suffix}")
        lines.append(f"          seen:    {entry.get('occurrences', 1)}x")
        lines.append(f"          trigger: {entry.get('trigger', '')}")
        lines.append(f"          lesson:  {entry.get('lesson', '')}")
        lines.append(f"          target:  {', '.join(entry.get('targets', []))}")
    return "\n".join(lines)


def render_markdown(entries: list[dict[str, Any]]) -> str:
    lines = []
    for entry in entries:
        lines.append(f"- **{entry['id']}** — {entry.get('lesson', '')}")
        lines.append(f"  - Trigger: {entry.get('trigger', '')}")
        lines.append(f"  - Applied to: `{'`, `'.join(entry.get('targets', []))}`")
        if entry.get("occurrences", 1) > 1:
            lines.append(f"  - Hit {entry['occurrences']} times before this change.")
    return "\n".join(lines)


def cmd_list(args: argparse.Namespace) -> int:
    path = queue_path(args.queue)
    learnings = load(path).get("learnings", [])
    if args.state != "all":
        learnings = [
            item for item in learnings if item.get("state", "pending") == args.state
        ]
    # Blocking first, then repeat offenders: both need a decision before the
    # merely-nice-to-write-down ones.
    learnings.sort(
        key=lambda item: (
            item.get("severity") != "blocking",
            -int(item.get("occurrences", 1)),
            item.get("id", ""),
        )
    )

    if args.format == "json":
        print(json.dumps(learnings, indent=2))
        return 0
    if not learnings:
        print(f"learnings: nothing {args.state} in {path}")
        return 0
    if args.format == "markdown":
        print(render_markdown(learnings))
        return 0
    print(render_text(learnings))
    blocking = sum(1 for item in learnings if item.get("severity") == "blocking")
    print(f"\nlearnings: {len(learnings)} {args.state}, {blocking} blocking")
    return 0


def cmd_resolve(args: argparse.Namespace) -> int:
    path = queue_path(args.queue)
    payload = load(path)
    entry = next(
        (item for item in payload.get("learnings", []) if item.get("id") == args.id),
        None,
    )
    if entry is None:
        print(f"learnings: no entry '{args.id}' in {path}", file=sys.stderr)
        return 1
    if args.state == "rejected" and not args.note:
        print(
            "learnings: --note is required when rejecting, so the next sweep does "
            "not re-litigate it",
            file=sys.stderr,
        )
        return 2

    entry["state"] = args.state
    entry["resolution"] = {
        "as": args.state,
        "note": args.note or "",
        "pr": args.pr,
        "utc": now_utc(),
    }
    entry.pop("regressed", None)
    entry["updated_utc"] = now_utc()
    save(path, payload)
    print(f"learnings: '{args.id}' -> {args.state}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queue", type=Path, help="Override $LEARNINGS_FILE.")
    sub = parser.add_subparsers(dest="command", required=True)

    record = sub.add_parser("record", help="Queue a correction to the instructions.")
    record.add_argument(
        "--id", required=True, help="Stable slug, e.g. bridge-pin-needed-with-mcore."
    )
    record.add_argument(
        "--trigger", required=True, help="What happened that taught this."
    )
    record.add_argument(
        "--lesson",
        required=True,
        help="What a future run should do differently, in general terms.",
    )
    record.add_argument(
        "--target",
        action="append",
        required=True,
        help="File the edit belongs in; repeatable.",
    )
    record.add_argument("--severity", choices=SEVERITIES, default="routine")
    record.add_argument(
        "--context-pr", type=int, help="Megatron-LM PR being tested when this surfaced."
    )
    record.set_defaults(func=cmd_record)

    listing = sub.add_parser("list", help="Show the queue.")
    listing.add_argument("--state", choices=(*STATES, "all"), default="pending")
    listing.add_argument(
        "--format", choices=("text", "markdown", "json"), default="text"
    )
    listing.set_defaults(func=cmd_list)

    resolve = sub.add_parser("resolve", help="Close an entry out.")
    resolve.add_argument("--id", required=True)
    resolve.add_argument(
        "--as", dest="state", choices=("promoted", "rejected"), required=True
    )
    resolve.add_argument("--note", help="Required when rejecting.")
    resolve.add_argument("--pr", type=int, help="PR carrying the promoted edit.")
    resolve.set_defaults(func=cmd_resolve)

    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
