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
"""Who is running this review, and may it count as a self-review?

Reads facts, not the model's own claim: harness and version from environment variables,
model and effort from the harness's session transcript, the skill revision from the
REVISION file next to this copy of the skill.

    review_identity.py preflight --pr N   # exit 2: this session wrote the PR's code
                                          # exit 3: self-review with a harness/model/effort not allowed
    review_identity.py footer --pr N      # the review-body footer: visible line + hidden marker

Standard library only, and runs on the older system `python3` some harnesses use (e.g. Codex).
`gh` is used to read the PR.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

SKILL_DIR = Path(__file__).resolve().parents[1]
POLICY = json.loads((SKILL_DIR / "self-review-policy.json").read_text())
REPO = "NVIDIA-NeMo/RL"
MARKER = "nemo-rl-review"

CLAUDE_EDIT_TOOLS = {"Edit", "Write", "MultiEdit", "NotebookEdit"}
# Codex writes patches inside code text, where line breaks may be a literal "\n".
_PATCH_FILE = re.compile(r"\*\*\* (?:Update|Add|Delete) File: ([^\s\\'\"]+)")


@dataclass
class Identity:
    harness: str  # "claude-code", "codex" or "unknown"
    version: str
    model: str
    effort: str
    transcript: Path | None

    def label(self) -> str:
        names = {"claude-code": "Claude Code", "codex": "Codex"}
        return f"{names.get(self.harness, self.harness)} {self.version}"


def skill_revision() -> int:
    return int((SKILL_DIR / "REVISION").read_text().strip())


def _jsonl(path: Path):
    with open(path, encoding="utf-8") as f:
        for line in f:
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def detect(env: dict[str, str] | None = None, home: Path | None = None) -> Identity:
    env = dict(os.environ) if env is None else env
    home = Path.home() if home is None else home
    if env.get("CLAUDECODE") == "1" and env.get("CLAUDE_CODE_SESSION_ID"):
        sid = env["CLAUDE_CODE_SESSION_ID"]
        found = glob.glob(str(home / ".claude/projects/*" / f"{sid}.jsonl"))
        transcript = Path(found[0]) if found else None
        model = "unknown"
        if transcript:
            for row in _jsonl(transcript):
                msg = row.get("message") or {}
                if row.get("type") == "assistant" and msg.get("model"):
                    model = msg["model"]  # Keep the latest.
        version = Path(env.get("CLAUDE_CODE_EXECPATH", "")).name or "unknown"
        return Identity(
            "claude-code",
            version,
            model,
            env.get("CLAUDE_EFFORT", "unknown"),
            transcript,
        )
    if env.get("CODEX_SESSION_ID") or env.get("CODEX_THREAD_ID"):
        sid = env.get("CODEX_SESSION_ID") or env["CODEX_THREAD_ID"]
        found = glob.glob(
            str(home / ".codex/sessions/**" / f"rollout-*-{sid}.jsonl"), recursive=True
        )
        transcript = Path(found[0]) if found else None
        model = effort = "unknown"
        if transcript:
            for row in _jsonl(transcript):
                if row.get("type") == "turn_context":
                    payload = row.get("payload") or {}
                    model = payload.get("model", model)
                    effort = payload.get("effort", effort)
        return Identity(
            "codex", env.get("CODEX_VERSION", "unknown"), model, effort, transcript
        )
    return Identity("unknown", "unknown", "unknown", "unknown", None)


def not_allowed_reason(ident: Identity) -> str | None:
    """Return why this harness/model/effort can't count as a self-review, or None if it can."""
    models = POLICY["allowed_models"].get(ident.harness)
    if models is None:
        return f"harness `{ident.harness}` is not allowed; use Claude Code or Codex"
    if ident.model not in models:
        return f"model `{ident.model}` is not allowed with {ident.harness}; use one of {', '.join(models)}"
    order = POLICY["effort_order"][ident.harness]
    floor = POLICY["min_effort"]
    if ident.effort not in order or order.index(ident.effort) < order.index(floor):
        return f"effort `{ident.effort}` is below `{floor}`"
    return None


def same_session_evidence(
    transcript: Path | None, pr_files: list[str], repo_root: str
) -> list[str]:
    """Lines of evidence that this session edited the PR's files or committed in this repo."""
    if transcript is None:
        return []
    evidence = []

    def touches(path: str) -> str | None:
        return next((f for f in pr_files if path == f or path.endswith("/" + f)), None)

    for row in _jsonl(transcript):
        if row.get("type") == "assistant":  # Claude Code
            cwd = row.get("cwd", "")
            for block in (row.get("message") or {}).get("content") or []:
                if not isinstance(block, dict) or block.get("type") != "tool_use":
                    continue
                args = block.get("input") or {}
                if block.get("name") in CLAUDE_EDIT_TOOLS and (
                    hit := touches(args.get("file_path", ""))
                ):
                    evidence.append(f"edited {hit}")
                if block.get("name") == "Bash" and re.search(
                    r"\bgit (commit|push)\b", args.get("command", "")
                ):
                    if cwd.startswith(repo_root):
                        evidence.append("ran git commit/push in this repo")
        payload = row.get("payload") or {}
        if (
            row.get("type") == "response_item"
            and payload.get("type") == "custom_tool_call"
        ):  # Codex
            text = payload.get("input") or ""
            for path in _PATCH_FILE.findall(text):
                if hit := touches(path):
                    evidence.append(f"edited {hit}")
            if re.search(r"\bgit (commit|push)\b", text) and repo_root in text:
                evidence.append("ran git commit/push in this repo")
    return sorted(set(evidence))


def _gh(*args: str) -> str:
    return subprocess.run(
        ["gh", *args], check=True, capture_output=True, text=True
    ).stdout


def _pr(pr: int) -> dict:
    return json.loads(
        _gh("pr", "view", str(pr), "-R", REPO, "--json", "author,headRefOid,files")
    )


def footer(*, ident: Identity, kind: str, head: str) -> str:
    rev = skill_revision()
    visible = (
        f"_{kind.capitalize()}-review: {ident.label()} · {ident.model} · effort {ident.effort} · "
        f"review-pr-team rev {rev} · reviewed {head[:7]}_"
    )
    marker = (
        f"<!-- {MARKER}: kind={kind} harness={ident.harness}/{ident.version} model={ident.model} "
        f"effort={ident.effort} skill_rev={rev} head={head} -->"
    )
    return f"{visible}\n{marker}"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("command", choices=["preflight", "footer"])
    parser.add_argument("--pr", type=int, required=True)
    args = parser.parse_args(argv)

    pr = _pr(args.pr)
    me = _gh("api", "user", "--jq", ".login").strip()
    kind = "self" if me == pr["author"]["login"] else "expert"
    ident = detect()

    if args.command == "footer":
        print(footer(ident=ident, kind=kind, head=pr["headRefOid"]))
        return 0

    print(
        f"Review kind: {kind} ({me}); {ident.label()}, model {ident.model}, effort {ident.effort}; "
        f"review-pr-team rev {skill_revision()}"
    )
    repo_root = subprocess.run(
        ["git", "rev-parse", "--show-toplevel"], capture_output=True, text=True
    ).stdout.strip()
    evidence = same_session_evidence(
        ident.transcript, [f["path"] for f in pr["files"]], repo_root or "/"
    )
    if evidence:
        print(
            "STOP: this session wrote code in this PR ("
            + "; ".join(evidence[:5])
            + "), so its review would be biased."
        )
        print(
            f"Open a fresh Claude Code or Codex session and run /review-pr-team {args.pr} there."
        )
        return 2
    if kind == "self" and (reason := not_allowed_reason(ident)):
        print(f"STOP: this can't count as a self-review: {reason}.")
        print(
            f"Run /review-pr-team {args.pr} in a fresh Claude Code or Codex session with an allowed model."
        )
        return 3
    print("OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
