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
"""Check the PR author's self-review and post it as the `Self-review` commit status.

States (shown in the status description):
  No self-review           the author has not published a review made by review-pr-team
  Not accepted: <reason>   the review's marker names a harness/model/effort or skill
                           revision that skills/review-pr-team/self-review-policy.json rejects
  Not resolved: N of M     some self-review threads have no reply or are not resolved
  Resolved (M threads)     every self-review thread has a reply and is resolved

Also fails the job when a PR changes skills/review-pr-team/ without bumping its REVISION by one.

The pure functions (parse_marker, not_accepted_reason, evaluate, revision_bump_problem) have no
GitHub dependency so other CI systems can reuse them; `main` is the GitHub Actions entry point.
Standard library only.
"""

from __future__ import annotations

import json
import os
import re
import sys
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path

CONTEXT = "Self-review"
SKILL_DIR = "skills/review-pr-team/"
_MARKER = re.compile(r"<!--\s*nemo-rl-review:\s*(.*?)\s*-->", re.S)


@dataclass
class Review:
    id: int
    author: str
    body: str
    submitted_at: str


@dataclass
class Thread:
    is_resolved: bool
    comment_count: int
    first_review_id: int | None  # The review the thread's first comment belongs to.


def parse_marker(body: str) -> dict[str, str] | None:
    """Return the fields of the last `<!-- nemo-rl-review: k=v ... -->` marker in body."""
    found = _MARKER.findall(body or "")
    if not found:
        return None
    return dict(part.split("=", 1) for part in found[-1].split() if "=" in part)


def not_accepted_reason(
    marker: dict[str, str], policy: dict, main_rev: int
) -> str | None:
    """Why this self-review's marker is rejected, or None if it is accepted."""
    harness = marker.get("harness", "").split("/", 1)[0]
    models = policy["allowed_models"].get(harness)
    if models is None:
        return f"harness {harness or 'unknown'} is not allowed"
    model = marker.get("model", "unknown")
    if model not in models:
        return f"model {model} is not allowed"
    order = policy["effort_order"][harness]
    effort = marker.get("effort", "unknown")
    if effort not in order or order.index(effort) < order.index(policy["min_effort"]):
        return f"effort {effort} is below {policy['min_effort']}"
    try:
        rev = int(marker.get("skill_rev", ""))
    except ValueError:
        return "no skill revision in the marker"
    if main_rev - rev > policy["max_revision_lag"]:
        return f"review-pr-team rev {rev} is too old (main is {main_rev}); update the skill"
    return None


def evaluate(
    *,
    pr_number: int,
    author: str,
    reviews: list[Review],
    threads: list[Thread],
    policy: dict,
    main_rev: int,
) -> tuple[str, str]:
    """Return (commit status state, description) for the author's self-review."""
    own = [
        r
        for r in reviews
        if r.author == author and (parse_marker(r.body) or {}).get("kind") == "self"
    ]
    if not own:
        return (
            "failure",
            f"No self-review — run /review-pr-team {pr_number} in a fresh Claude Code or Codex session",
        )
    latest = max(own, key=lambda r: r.submitted_at)
    reason = not_accepted_reason(parse_marker(latest.body), policy, main_rev)
    if reason:
        return "failure", f"Not accepted: {reason}"[:140]
    ids = {r.id for r in own}
    mine = [t for t in threads if t.first_review_id in ids]
    open_ = [t for t in mine if not t.is_resolved or t.comment_count < 2]
    if open_:
        return (
            "failure",
            f"Not resolved: {len(open_)} of {len(mine)} threads open or unanswered",
        )
    return "success", f"Resolved ({len(mine)} threads)"


def revision_bump_problem(
    changed_files: list[str], main_rev: int, head_rev: int
) -> str | None:
    """A PR that changes the review skill must bump its REVISION by exactly one."""
    if not any(f.startswith(SKILL_DIR) for f in changed_files):
        return None
    if head_rev == main_rev + 1:
        return None
    return (
        f"This PR changes {SKILL_DIR} but {SKILL_DIR}REVISION is {head_rev}; "
        f"set it to {main_rev + 1} (main is {main_rev})."
    )


class GitHub:
    def __init__(self, repo: str, token: str):
        self.repo = repo
        self.token = token

    def call(self, method: str, url: str, body: dict | None = None, raw: bool = False):
        if not url.startswith("https://"):
            url = f"https://api.github.com/repos/{self.repo}{url}"
        req = urllib.request.Request(
            url,
            method=method,
            data=json.dumps(body).encode() if body is not None else None,
            headers={
                "Authorization": f"Bearer {self.token}",
                "Accept": "application/vnd.github.raw"
                if raw
                else "application/vnd.github+json",
            },
        )
        with urllib.request.urlopen(req) as resp:
            data = resp.read()
        if raw:
            return data.decode()
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

    def threads(self, number: int) -> list[Thread]:
        owner, name = self.repo.split("/")
        query = """query($owner:String!,$name:String!,$number:Int!,$cursor:String){
          repository(owner:$owner,name:$name){pullRequest(number:$number){
          reviewThreads(first:100,after:$cursor){pageInfo{hasNextPage endCursor}
          nodes{isResolved comments(first:1){totalCount nodes{pullRequestReview{databaseId}}}}}}}}"""
        out, cursor = [], None
        while True:
            data = self.call(
                "POST",
                "https://api.github.com/graphql",
                {
                    "query": query,
                    "variables": {
                        "owner": owner,
                        "name": name,
                        "number": number,
                        "cursor": cursor,
                    },
                },
            )
            conn = data["data"]["repository"]["pullRequest"]["reviewThreads"]
            for n in conn["nodes"]:
                first = n["comments"]["nodes"]
                review = (first[0].get("pullRequestReview") or {}) if first else {}
                out.append(
                    Thread(
                        n["isResolved"],
                        n["comments"]["totalCount"],
                        review.get("databaseId"),
                    )
                )
            if not conn["pageInfo"]["hasNextPage"]:
                return out
            cursor = conn["pageInfo"]["endCursor"]


def _read_rev(text: str | None) -> int:
    try:
        return int((text or "").strip())
    except ValueError:
        return 0


def main() -> int:
    gh = GitHub(os.environ["REPO"], os.environ["GITHUB_TOKEN"])
    event = json.load(open(os.environ["GITHUB_EVENT_PATH"]))
    pr = event["pull_request"]
    number, head, author = pr["number"], pr["head"]["sha"], pr["user"]["login"]
    policy = json.loads(Path(SKILL_DIR, "self-review-policy.json").read_text())
    main_rev_path = Path(SKILL_DIR, "REVISION")
    main_rev = _read_rev(main_rev_path.read_text() if main_rev_path.exists() else None)

    reviews = [
        Review(
            r["id"],
            (r.get("user") or {}).get("login", ""),
            r.get("body") or "",
            r.get("submitted_at") or "",
        )
        for r in gh.pages(f"/pulls/{number}/reviews")
    ]
    state, description = evaluate(
        pr_number=number,
        author=author,
        reviews=reviews,
        threads=gh.threads(number),
        policy=policy,
        main_rev=main_rev,
    )
    run_url = f"{os.environ.get('GITHUB_SERVER_URL', 'https://github.com')}/{gh.repo}/actions/runs/{os.environ.get('GITHUB_RUN_ID', '')}"
    gh.call(
        "POST",
        f"/statuses/{head}",
        {
            "state": state,
            "description": description,
            "context": CONTEXT,
            "target_url": run_url,
        },
    )
    print(f"{CONTEXT}: {state} — {description}")

    changed = [f["filename"] for f in gh.pages(f"/pulls/{number}/files")]
    head_rev = 0
    if any(f.startswith(SKILL_DIR) for f in changed):
        try:
            head_rev = _read_rev(
                gh.call("GET", f"/contents/{SKILL_DIR}REVISION?ref={head}", raw=True)
            )
        except urllib.error.HTTPError as err:
            if err.code != 404:
                raise
    problem = revision_bump_problem(changed, main_rev, head_rev)
    if problem:
        print(f"::error title=review-pr-team REVISION::{problem}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
