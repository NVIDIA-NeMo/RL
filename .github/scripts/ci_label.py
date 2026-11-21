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
  CI:L1    any dependency or container file, or a 3rdparty/ submodule bump
  CI:docs  every file is docs, skill text, or a .github/ file the main CI run
           does not read
  CI:Lfast everything else

The bot never picks CI:L0 or CI:L2. A CI:* or "Skip CICD" label set by a repo
maintainer (admin or maintain role) is kept until a push changes the bot's own
pick; one set by anyone else is replaced, with one PR comment that is edited
in place.

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


def decide(
    *,
    pick: str,
    pick_changed: bool,
    labels: dict[str, tuple[str, bool]],
) -> Plan:
    """Plan label changes.

    `labels` maps each current CI:* / Skip CICD label to (who set it, is that person a maintainer).
    `pick_changed` is True when this push changed the bot's own pick.
    """
    plan = Plan()
    ci = {name: setter for name, setter in labels.items() if name.startswith("CI:")}
    kept = None
    for name, (who, maintainer) in ci.items():
        if maintainer and not pick_changed and kept is None:
            kept = name
    for name, (who, maintainer) in ci.items():
        if name == kept or (kept is None and name == pick):
            continue
        plan.remove.append(name)
        if not maintainer and who != "bot":
            plan.reverted.append((name, who))
    if kept is None and pick not in ci:
        plan.add.append(pick)
    if SKIP_LABEL in labels:
        who, maintainer = labels[SKIP_LABEL]
        if not maintainer:
            plan.remove.append(SKIP_LABEL)
            plan.reverted.append((SKIP_LABEL, who))
    return plan


def comment_body(*, pick: str, reason: str, reverted: list[tuple[str, str]]) -> str:
    lines = [COMMENT_MARKER, f"**CI label:** `{pick}`, set automatically ({reason})."]
    for label, who in reverted:
        lines.append(
            f"- Removed `{label}` added by `{who}`: only repo maintainers can change the CI label or skip CI."
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

    current = [
        l["name"]
        for l in pr["labels"]
        if l["name"].startswith("CI:") or l["name"] == SKIP_LABEL
    ]
    setters = {}
    for ev in gh.pages(f"/issues/{number}/events"):
        if ev.get("event") == "labeled" and ev["label"]["name"] in current:
            setters[ev["label"]["name"]] = ev["actor"][
                "login"
            ]  # Last "labeled" event wins.
    labels = {}
    for name in current:
        who = setters.get(name, "unknown")
        labels[name] = (
            "bot" if who == "github-actions[bot]" else who,
            gh.is_maintainer(who),
        )

    plan = decide(pick=pick, pick_changed=pick_changed, labels=labels)
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
        body = comment_body(pick=pick, reason=reason, reverted=plan.reverted)
        mine = [
            c
            for c in gh.pages(f"/issues/{number}/comments")
            if COMMENT_MARKER in (c.get("body") or "")
        ]
        if mine:
            gh.call("PATCH", f"/issues/comments/{mine[-1]['id']}", {"body": body})
        else:
            gh.call("POST", f"/issues/{number}/comments", {"body": body})
    return 0


if __name__ == "__main__":
    sys.exit(main())
