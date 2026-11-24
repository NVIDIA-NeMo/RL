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
model from the harness's session transcript, effort from Claude's CLAUDE_EFFORT
environment variable or Codex's transcript, and the skill revision from the
REVISION file next to this copy of the skill.

    review_identity.py preflight --pr N             # exit 2: this session wrote the PR's code
                                                    # exit 3: self-review with a harness/model/effort not allowed
    review_identity.py footer --pr N --head SHA     # the review-body footer: one `nemo-rl-review: …` line

Standard library only, and runs on the older system `python3` some harnesses use (e.g. Codex).
Needs the GitHub CLI `gh`, logged in, to read the PR, even in Codex sessions that use HTTP elsewhere.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
import shlex
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
_PATCH_BODY = re.compile(r"\*\*\* Begin Patch.*?(?:\*\*\* End Patch|$)", re.S)
_JS_STRING = re.compile(r"\"((?:[^\"\\]|\\.)*)\"|'((?:[^'\\]|\\.)*)'|`([^`]*)`")
_JS_WORKDIR = re.compile(r"\b(?:workdir|cwd)\b\s*[:=]\s*[\"'`]([^\"'`]+)")
_HEREDOC = re.compile(r"(?<!<)<<-?\s*['\"]?(\w+)['\"]?")
_SHELL_SEPARATORS = {"&&", "||", ";", "|", "&", "(", ")"}
_REDIRECTS = {">", ">>", ">|", "&>", "&>>"}
_GIT_OPTS_WITH_VALUE = {"-c", "--git-dir", "--work-tree", "--namespace"}


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
        claude_root = Path(env.get("CLAUDE_CONFIG_DIR") or home / ".claude")
        found = glob.glob(str(claude_root / "projects/*" / f"{sid}.jsonl"))
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
        codex_root = Path(env.get("CODEX_HOME") or home / ".codex")
        found = glob.glob(
            str(codex_root / "sessions/**" / f"rollout-*-{sid}.jsonl"), recursive=True
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


def _resolve(base: str | None, target: str) -> str | None:
    """`target` as an absolute normalized path, or None if it is relative and `base` is unknown."""
    target = os.path.expanduser(target)
    if os.path.isabs(target):
        return os.path.normpath(target)
    return os.path.normpath(os.path.join(base, target)) if base else None


def _repo_path(path: str, cwd: str | None, repo_root: str) -> str | None:
    """`path` relative to `repo_root`, or None if it lies outside the repo."""
    full = _resolve(cwd, path)
    if full is not None:
        full = os.path.realpath(full)
    root = os.path.realpath(repo_root)
    if full is None or (full != root and not full.startswith(root.rstrip("/") + "/")):
        return None
    return os.path.relpath(full, root)


def _written_files(prog: str, args: list[str]) -> list[str]:
    """Files a `tee`, `cp`, `mv`, `install`, `sed -i` or `perl -i` command writes."""
    operands, in_place, has_script_opt, skip = [], False, False, False
    for arg in args:
        if skip:
            skip = False
        elif arg in ("-e", "-f", "--expression", "--file"):
            has_script_opt = skip = True
        elif arg.startswith("--"):
            in_place = in_place or arg.startswith("--in-place")
        elif arg.startswith("-") and len(arg) > 1:
            in_place = in_place or "i" in arg[1:]
        else:
            operands.append(arg)
    if prog == "tee":
        return operands
    if prog in ("cp", "mv", "install"):
        return operands[-1:]
    if prog in ("sed", "perl") and in_place:
        return operands if has_script_opt else operands[1:]
    return []


def _drop_heredoc_bodies(command: str) -> list[str]:
    """The command's lines without heredoc bodies, which are data, not shell."""
    lines, end = [], None
    for line in command.split("\n"):
        if end is not None:
            if line.strip() == end:
                end = None
            continue
        lines.append(line)
        if m := _HEREDOC.search(line):
            end = m.group(1)
    return lines


def _lex_lines(lines: list[str]) -> list[str]:
    """Shell tokens, lexed line by line so a `#` comment or a stray quote only hides its own line.

    A quote may span lines (e.g. a multi-line `git commit -m`), so a line that does not lex is
    retried joined with the lines after it; if that never lexes, only that line is skipped.
    """
    tokens: list[str] = []
    i = 0
    while i < len(lines):
        for j in range(i, len(lines)):
            lexer = shlex.shlex(
                "\n".join(lines[i : j + 1]), posix=True, punctuation_chars=True
            )
            lexer.whitespace_split = True
            try:
                tokens += [*lexer, ";"]
                break
            except ValueError:
                continue
        else:
            j = i  # Never balanced: skip this line alone.
        i = j + 1
    return tokens


def _shell_actions(
    command: str, cwd: str | None, repo_root: str
) -> tuple[list[str], bool]:
    """Repo-relative files a shell command writes, and whether it runs git commit/push in the repo."""
    tokens = _lex_lines(_drop_heredoc_bodies(command))
    segments: list[list[str]] = [[]]
    for tok in tokens:
        if tok in _SHELL_SEPARATORS:
            segments.append([])
        else:
            segments[-1].append(tok)
    writes, committed = [], False
    for words in segments:
        args, targets, i = [], [], 0
        while i < len(words):
            if words[i] in _REDIRECTS and i + 1 < len(words):
                targets.append(words[i + 1])
                i += 2
            else:
                args.append(words[i])
                i += 1
        while args and re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*=.*", args[0]):
            args.pop(0)  # VAR=value prefixes
        prog = os.path.basename(args[0]) if args else ""
        targets += _written_files(prog, args[1:])
        writes += [p for p in (_repo_path(t, cwd, repo_root) for t in targets) if p]
        if prog == "cd":
            target = args[1] if len(args) > 1 else "-"
            cwd = None if target == "-" else _resolve(cwd, target)
        elif prog == "git":
            git_dir, j = cwd, 1
            while j < len(args) and args[j].startswith("-"):
                if args[j] == "-C" and j + 1 < len(args):
                    git_dir = _resolve(git_dir, args[j + 1])
                j += 2 if args[j] == "-C" or args[j] in _GIT_OPTS_WITH_VALUE else 1
            if (
                j < len(args)
                and args[j] in ("commit", "push")
                and git_dir is not None
                and _repo_path(git_dir, None, repo_root) is not None
            ):
                committed = True
    return writes, committed


def _js_strings(code: str) -> list[str]:
    """String literals in Codex's code-mode tool input, with common escapes undone."""
    out = []
    for m in _JS_STRING.finditer(code):
        s = next(g for g in m.groups() if g is not None)
        for esc, ch in (("\\n", "\n"), ('\\"', '"'), ("\\'", "'"), ("\\\\", "\\")):
            s = s.replace(esc, ch)
        out.append(s)
    return out


def same_session_evidence(
    transcript: Path | None, pr_files: list[str], repo_root: str
) -> list[str]:
    """Lines of evidence that this session edited the PR's files or committed in this repo."""
    if transcript is None:
        return []
    # Subagents (Agent tool, forks) log their tool calls in <sid>/subagents/, not in <sid>.jsonl.
    subagents = sorted((transcript.with_suffix("") / "subagents").glob("*.jsonl"))
    rows = [row for path in [transcript, *subagents] for row in _jsonl(path)]
    pr_set = set(pr_files)
    evidence = []

    def record(writes: list[str], committed: bool) -> None:
        evidence.extend(f"edited {p}" for p in writes if p in pr_set)
        if committed:
            evidence.append("ran git commit/push in this repo")

    failed = {  # Claude Code tool calls whose result was an error
        block.get("tool_use_id")
        for row in rows
        if row.get("type") == "user"
        for block in (row.get("message") or {}).get("content") or []
        if isinstance(block, dict)
        and block.get("type") == "tool_result"
        and block.get("is_error")
    }
    codex_cwd = None
    for row in rows:
        if row.get("type") == "assistant":  # Claude Code
            cwd = row.get("cwd") or None
            for block in (row.get("message") or {}).get("content") or []:
                if (
                    not isinstance(block, dict)
                    or block.get("type") != "tool_use"
                    or block.get("id") in failed
                ):
                    continue
                args = block.get("input") or {}
                if block.get("name") in CLAUDE_EDIT_TOOLS:
                    path = args.get("file_path") or args.get("notebook_path") or ""
                    record([_repo_path(path, cwd, repo_root) or ""], False)
                if block.get("name") == "Bash":
                    record(*_shell_actions(args.get("command", ""), cwd, repo_root))
        payload = row.get("payload") or {}
        if row.get("type") == "turn_context":  # Codex
            codex_cwd = payload.get("cwd") or codex_cwd
        if row.get("type") != "response_item":
            continue
        if payload.get("type") in ("custom_tool_call", "function_call"):  # Codex
            # Patches (apply_patch, or a patch passed to a command) name their files.
            text = payload.get("input") or payload.get("arguments") or ""
            record(
                [
                    _repo_path(p, codex_cwd, repo_root) or ""
                    for p in _PATCH_FILE.findall(text)
                ],
                False,
            )
        if payload.get("type") == "custom_tool_call":  # Codex code mode
            code = _PATCH_BODY.sub("", payload.get("input") or "")
            workdir = _JS_WORKDIR.search(code)
            shell_dir = _resolve(codex_cwd, workdir.group(1)) if workdir else codex_cwd
            for literal in _js_strings(code):
                record(*_shell_actions(literal, shell_dir, repo_root))
        if payload.get("type") == "function_call":  # Codex exec_command / shell
            command, workdir = _function_call_command(payload.get("arguments") or "")
            shell_dir = _resolve(codex_cwd, workdir) if workdir else codex_cwd
            record(*_shell_actions(_PATCH_BODY.sub("", command), shell_dir, repo_root))
    return sorted(set(evidence))


def _function_call_command(arguments: str) -> tuple[str, str | None]:
    """The shell command and working directory in a Codex function call's JSON arguments."""
    try:
        args = json.loads(arguments)
    except json.JSONDecodeError:
        return "", None
    if not isinstance(args, dict):
        return "", None
    command = args.get("cmd") or args.get("command") or ""
    if isinstance(command, list):
        words = [str(w) for w in command]
        is_shell_c = (
            len(words) >= 3
            and os.path.basename(words[0]) in ("bash", "sh", "zsh")
            and words[1] in ("-c", "-lc")
        )
        command = words[2] if is_shell_c else shlex.join(words)
    workdir = args.get("workdir")
    return str(command), workdir if isinstance(workdir, str) and workdir else None


def _gh(*args: str) -> str:
    try:
        return subprocess.run(
            ["gh", *args], check=True, capture_output=True, text=True
        ).stdout
    except FileNotFoundError:
        sys.exit(
            "The review-pr-team preflight needs the GitHub CLI `gh`. Install it and run `gh auth login`."
        )


def _login() -> str | None:
    """The reviewer's GitHub login, or None for a bot app token.

    App tokens, such as the one Autobot reviews with, get HTTP 403 on `GET /user`: there is no
    user behind them, so the review can never be a self-review. Any other failure (not logged
    in, network, rate limit) is a broken setup and stops the review.
    """
    try:
        return _gh("api", "user", "--jq", ".login").strip()
    except subprocess.CalledProcessError as e:
        if "HTTP 403" in (e.stderr or ""):
            return None
        sys.exit(
            "Could not read your GitHub user with `gh api user`:\n"
            f"{(e.stderr or '').strip()}\nRun `gh auth status` (or `gh auth login`) and retry."
        )


def _pr(pr: int) -> dict:
    info = json.loads(
        _gh("pr", "view", str(pr), "-R", REPO, "--json", "author,headRefOid")
    )
    # `gh pr view --json files` stops at 100 files; the REST list pages through all of them.
    paths = _gh(
        "api", "--paginate", f"repos/{REPO}/pulls/{pr}/files", "--jq", ".[].filename"
    )
    info["files"] = [{"path": p} for p in paths.splitlines() if p]
    return info


def footer(*, ident: Identity, kind: str, head: str) -> str:
    """The one-line identity footer, a code span of space-separated key=value pairs."""
    fields = {
        "kind": kind,
        "harness": f"{ident.harness}/{ident.version}",
        "model": ident.model,
        "effort": ident.effort,
        "review-pr-team": f"v{skill_revision()}",
        "reviewed": head[:7],
    }
    # Values are split on spaces by the reader, so none may contain one.
    pairs = " ".join(k + "=" + re.sub(r"\s+", "-", v) for k, v in fields.items())
    return f"`{MARKER}: {pairs}`"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("command", choices=["preflight", "footer"])
    parser.add_argument("--pr", type=int, required=True)
    parser.add_argument(
        "--head", help="footer only: the full SHA of the commit you reviewed"
    )
    args = parser.parse_args(argv)
    if args.command == "footer" and not re.fullmatch(r"[0-9a-f]{40}", args.head or ""):
        parser.error("footer needs --head with the full SHA of the reviewed commit")

    me = _login()
    pr = _pr(args.pr)
    kind = "self" if me == pr["author"]["login"] else "expert"
    ident = detect()

    if args.command == "footer":
        if args.head != pr["headRefOid"]:
            parser.error(
                f"the PR head is now {pr['headRefOid']}, not the reviewed {args.head}; "
                "review the new head before writing a footer"
            )
        print(footer(ident=ident, kind=kind, head=args.head))
        return 0

    print(
        f"Review kind: {kind} ({me or 'bot-app-token'}); {ident.label()}, model {ident.model}, effort {ident.effort}; "
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
    if me is None:
        print("OK (bot-app-token)")
    elif ident.transcript is None:
        print(
            "WARNING: no session log found, so same-session edits could not be checked"
        )
    else:
        print("OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
