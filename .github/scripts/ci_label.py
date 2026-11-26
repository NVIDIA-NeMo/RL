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
"""Pick a PR's CI:* label from its changed files, and keep only maintainers' overrides.

Rules (first match wins):
  CI:L1    any dependency or container file (including the docker/ .sh and .py
           build helpers), or a 3rdparty/ submodule bump
  CI:docs  every file is docs, skill text, or a .github/ file the main CI run
           does not read
  CI:Lfast everything else

The bot never picks CI:L0 or CI:L2. A CI:* or "Skip CICD" label set by a repo
maintainer (admin or maintain role) is kept until a later push changes the bot's
own pick; if a maintainer set several CI:* labels, the newest one stays. One set
by anyone else is replaced, with one PR comment that is edited in place.

The rules and `decide` are pure, so other CI systems can reuse them; `main`
is the GitHub Actions entry point and uses only the standard library.
"""

import fnmatch
import json
import os
import sys
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field

L1_GLOBS = (
    "pyproject.toml",
    "*/pyproject.toml",
    "uv.lock",
    "*/uv.lock",
    "Dockerfile*",
    "*/Dockerfile*",
    "docker/*.sh",
    "docker/*.py",
    "3rdparty/*",
)
DOCS_GLOBS = (
    "*.md",
    "docs/*",
    "skills/*",
    ".agents/*",
    ".claude/*",
    ".github/*",
)
# The .github/ files the main CI run reads; changing one still needs CI:Lfast.
MAIN_CI_FILES = (
    ".github/workflows/cicd-main.yml",
    ".github/workflows/_build_container.yml",
    ".github/actions/test-template/*",
    ".github/scripts/ci_label.py",
    ".github/scripts/plan-tests.sh",
    ".github/scripts/checkout-megatron-lm-commit.sh",
    ".github/scripts/install-megatron-lock-artifact.sh",
    ".github/scripts/known-flake-patterns.txt",
)
SKIP_LABEL = "Skip CICD"
COMMENT_MARKER = "<!-- ci-label-bot -->"


def _matches(path: str, globs: tuple[str, ...]) -> bool:
    return any(fnmatch.fnmatch(path, glob) for glob in globs)


def pick_label(files: list[str]) -> tuple[str, str]:
    """Return (label, reason) for a PR that changes `files`."""
    l1 = [f for f in files if _matches(f, L1_GLOBS)]
    if l1:
        return "CI:L1", f"dependency, container or submodule change (`{l1[0]}`)"
    if files and all(
        _matches(f, DOCS_GLOBS) and not _matches(f, MAIN_CI_FILES) for f in files
    ):
        return (
            "CI:docs",
            "only docs, skills or .github/ files the main CI run does not read changed",
        )
    return "CI:Lfast", "default for code and main CI changes"


@dataclass
class Plan:
    add: list[str] = field(default_factory=list)
    remove: list[str] = field(default_factory=list)
    reverted: list[tuple[str, str]] = field(default_factory=list)  # (label, who set it)
    kept: str | None = None  # The maintainer's CI:* label that stays, if any.


def decide(
    *,
    pick: str,
    pick_changed: bool,
    labels: dict[str, tuple[str, bool]],
    set_after_push: frozenset[str] = frozenset(),
) -> Plan:
    """Plan label changes.

    `labels` maps each current CI:* / Skip CICD label to (who set it, is that person a maintainer), oldest first.
    `pick_changed` is True when the push this run handles changed the bot's own pick.
    Labels in `set_after_push` were set after that push, so the push does not expire them.
    """

    def kept_override(name: str) -> bool:
        maintainer = labels[name][1]
        return maintainer and (not pick_changed or name in set_after_push)

    plan = Plan()
    ci = [name for name in labels if name.startswith("CI:")]
    overrides = [name for name in ci if kept_override(name)]
    if overrides:
        plan.kept = overrides[-1]  # The newest maintainer choice wins.
    for name in ci:
        if name == plan.kept or (plan.kept is None and name == pick):
            continue
        plan.remove.append(name)
        who, maintainer = labels[name]
        if not maintainer and who != "bot":
            plan.reverted.append((name, who))
    if plan.kept is None and pick not in ci:
        plan.add.append(pick)
    if SKIP_LABEL in labels and not kept_override(SKIP_LABEL):
        plan.remove.append(SKIP_LABEL)
        who, maintainer = labels[SKIP_LABEL]
        if not maintainer:
            plan.reverted.append((SKIP_LABEL, who))
    return plan


def comment_body(
    *, label: str, pick: str, reason: str, reverted: list[tuple[str, str]]
) -> str:
    if label == pick:
        status = f"**CI label:** `{pick}`, set automatically ({reason})."
    else:
        status = f"**CI label:** `{label}`, set by a maintainer. The automatic pick is `{pick}` ({reason})."
    lines = [COMMENT_MARKER, status]
    for name, who in reverted:
        lines.append(
            f"- Removed `{name}` added by `{who}`: only repo maintainers can change the CI label or skip CI."
        )
    lines.append("If this label is wrong, ask a maintainer to set the right one.")
    return "\n".join(lines)


class GitHub:
    def __init__(self, repo: str, token: str):
        self.base = f"https://api.github.com/repos/{repo}"
        self.token = token

    def call(self, method: str, path: str, body: dict | None = None):
        req = urllib.request.Request(
            path if path.startswith("https://") else self.base + path,
            method=method,
            data=json.dumps(body).encode() if body is not None else None,
            headers={
                "Authorization": f"Bearer {self.token}",
                "Accept": "application/vnd.github+json",
            },
        )
        with urllib.request.urlopen(req) as resp:
            data = resp.read()
            return json.loads(data) if data else None

    def pages(self, path: str) -> list:
        out, page = [], 1
        while True:
            sep = "&" if "?" in path else "?"
            batch = self.call("GET", f"{path}{sep}per_page=100&page={page}")
            out += batch
            if len(batch) < 100:
                return out
            page += 1

    def is_maintainer(self, login: str) -> bool:
        if login.endswith("[bot]"):
            return False
        try:
            role = self.call("GET", f"/collaborators/{login}/permission").get(
                "role_name"
            )
        except urllib.error.HTTPError:
            return False
        return role in ("admin", "maintain")


def main() -> int:
    gh = GitHub(os.environ["REPO"], os.environ["GITHUB_TOKEN"])
    event = json.load(open(os.environ["GITHUB_EVENT_PATH"]))
    pr = event["pull_request"]
    number = pr["number"]

    pick, reason = pick_label(
        [f["filename"] for f in gh.pages(f"/pulls/{number}/files")]
    )
    pick_changed = False
    if event["action"] == "synchronize" and event.get("before"):
        try:
            before = gh.call("GET", f"/compare/{pr['base']['sha']}...{event['before']}")
            pick_changed = (
                pick_label([f["filename"] for f in before.get("files", [])])[0] != pick
            )
        except urllib.error.HTTPError:
            pick_changed = (
                False  # Previous head is gone (force-push): keep a maintainer's choice.
            )

    # Read the labels now, not from the event: a queued run must see labels set while it waited.
    current = [
        l["name"]
        for l in gh.pages(f"/issues/{number}/labels")
        if l["name"].startswith("CI:") or l["name"] == SKIP_LABEL
    ]
    # Events come oldest first; re-inserting keeps `setters` ordered by each label's last "labeled" event.
    setters = {}
    for ev in gh.pages(f"/issues/{number}/events"):
        name = (ev.get("label") or {}).get("name")
        if ev.get("event") == "labeled" and name in current:
            setters.pop(name, None)
            setters[name] = (ev["actor"]["login"], ev["created_at"])
    labels = {}
    for name in [n for n in current if n not in setters] + list(setters):
        who = setters[name][0] if name in setters else "unknown"
        labels[name] = (
            "bot" if who == "github-actions[bot]" else who,
            gh.is_maintainer(who),
        )
    # A synchronize event's updated_at is the push time; a label set later was chosen after this push.
    pushed_at = pr["updated_at"]
    set_after_push = frozenset(
        name for name, (_, at) in setters.items() if pick_changed and at > pushed_at
    )

    plan = decide(
        pick=pick,
        pick_changed=pick_changed,
        labels=labels,
        set_after_push=set_after_push,
    )
    print(
        f"PR #{number}: pick={pick} ({reason}); pick_changed={pick_changed}; labels={labels}; plan={plan}"
    )
    for name in plan.remove:
        try:
            gh.call("DELETE", f"/issues/{number}/labels/{urllib.parse.quote(name)}")
        except urllib.error.HTTPError as err:
            if err.code != 404:
                raise
    if plan.add:
        gh.call("POST", f"/issues/{number}/labels", {"labels": plan.add})
    if plan.reverted:
        body = comment_body(
            label=plan.kept or pick, pick=pick, reason=reason, reverted=plan.reverted
        )
        mine = [
            c
            for c in gh.pages(f"/issues/{number}/comments")
            if c["user"]["login"] == "github-actions[bot]"
            and COMMENT_MARKER in (c.get("body") or "")
        ]
        if mine:
            gh.call("PATCH", f"/issues/comments/{mine[-1]['id']}", {"body": body})
        else:
            gh.call("POST", f"/issues/{number}/comments", {"body": body})
    return 0


if __name__ == "__main__":
    sys.exit(main())
