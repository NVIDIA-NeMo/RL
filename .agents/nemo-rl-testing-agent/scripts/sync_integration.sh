#!/usr/bin/env bash
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

# Rebuilds the NeMo-RL integration branch: `main` plus every agent fix that is
# raised but not yet merged.
#
# Why this exists: a fix the agent raises can sit in review for days. Without it
# applied, the same break fails the suite on every labeled PR that arrives in the
# meantime -- each one burning a fresh debugging budget to re-derive a diagnosis
# we already have, and, worse, keeping the suite red so a genuine problem in
# those PRs stays invisible behind it. Testing against `main` + pending fixes
# means each break is paid for once.
#
# The branch is REBUILT from main each time rather than rebased. Rebuilding is
# self-healing: a fix that merged upstream simply stops being listed, a fix that
# no longer applies is reported and skipped, and the branch never accumulates
# state that has to be untangled by hand. It is force-pushed to a fork so a
# rewritten history never touches the shared repo.
#
# Usage:
#   sync_integration.sh [--dry-run] [--exclude <pr>]...
#
# Writes a manifest to $STATE_DIR/integration.json describing exactly what the
# branch contains, which is what the PR comment discloses.

set -euo pipefail
source "$(cd -P "$(dirname "${BASH_SOURCE[0]}")" && pwd)/lib.sh"

dry_run=0
excludes=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dry-run) dry_run=1; shift ;;
    --exclude) excludes+=("$2"); shift 2 ;;
    *) nrlta_die "unknown argument: $1" ;;
  esac
done

nrlta_require NEMO_RL_REPO NEMO_RL_CLONE_URL NEMO_RL_FORK_URL NEMO_RL_BASE_BRANCH \
  NEMO_RL_INTEGRATION_BRANCH NEMO_RL_REPO_PATH STATE_DIR

worktree="${STATE_DIR}/integration-wt"
manifest="${STATE_DIR}/integration.json"

is_excluded() {
  local number="$1" item
  for item in ${excludes[@]+"${excludes[@]}"}; do
    [[ "${item}" == "${number}" ]] && return 0
  done
  return 1
}

# Open fix PRs, oldest first so the branch content is stable run to run rather
# than reshuffling with GitHub's default ordering.
prs_json="$(gh pr list --repo "${NEMO_RL_REPO}" --state open --limit 100 \
  --json number,headRefName,title,url,mergeable \
  --jq '[.[] | select(.headRefName | test("^mcore-.*-fix$"))] | sort_by(.number)')"

pr_numbers=($(printf '%s' "${prs_json}" | python3 -c 'import json,sys; print(" ".join(str(p["number"]) for p in json.load(sys.stdin)))'))

nrlta_log "found ${#pr_numbers[@]} open fix PR(s) in ${NEMO_RL_REPO}"

if [[ "${dry_run}" -eq 1 ]]; then
  printf '%s\n' "${prs_json}" | python3 -c '
import json, sys
for pr in json.load(sys.stdin):
    print("  #{number} {headRefName}: {title}".format(**pr))
'
  exit 0
fi

# A dedicated worktree keeps this off whatever the operator has checked out.
if [[ ! -d "${worktree}/.git" ]] && [[ ! -f "${worktree}/.git" ]]; then
  rm -rf "${worktree}"
  mkdir -p "$(dirname "${worktree}")"
  git -C "${NEMO_RL_REPO_PATH}" worktree add --detach "${worktree}" HEAD >/dev/null
fi

git -C "${worktree}" fetch --force --quiet "${NEMO_RL_CLONE_URL}" \
  "refs/heads/${NEMO_RL_BASE_BRANCH}:refs/nrlta/base"
base_sha="$(git -C "${worktree}" rev-parse refs/nrlta/base)"

# Detach before resetting so the branch ref can be moved freely.
git -C "${worktree}" checkout --quiet --detach refs/nrlta/base
git -C "${worktree}" reset --hard --quiet refs/nrlta/base
git -C "${worktree}" clean -fdq

applied=()
skipped=()

for number in ${pr_numbers[@]+"${pr_numbers[@]}"}; do
  if is_excluded "${number}"; then
    nrlta_log "skipping #${number} (excluded)"
    skipped+=("${number}:excluded")
    continue
  fi

  git -C "${worktree}" fetch --force --quiet "${NEMO_RL_CLONE_URL}" \
    "refs/pull/${number}/head:refs/nrlta/pr-${number}" || {
      nrlta_log "could not fetch #${number}; skipping"
      skipped+=("${number}:unfetchable")
      continue
    }

  head_sha="$(git -C "${worktree}" rev-parse "refs/nrlta/pr-${number}")"
  merge_base="$(git -C "${worktree}" merge-base refs/nrlta/base "refs/nrlta/pr-${number}")"

  if [[ "${merge_base}" == "${head_sha}" ]]; then
    # Already contained in main: the PR merged, or it is empty.
    nrlta_log "#${number} is already in ${NEMO_RL_BASE_BRANCH}; nothing to apply"
    skipped+=("${number}:already-in-base")
    continue
  fi

  # Pick commit by commit with the committer date forced to the author date.
  # Left to itself, cherry-pick stamps "now", so an unchanged set of fixes would
  # produce a different integration sha on every sync -- and the sha is what the
  # reports compare to answer "was this tested against the same stack?".
  pick_failed=0
  for commit in $(git -C "${worktree}" rev-list --reverse "${merge_base}..${head_sha}"); do
    author_date="$(git -C "${worktree}" show -s --format=%aI "${commit}")"
    if ! GIT_COMMITTER_DATE="${author_date}" \
        git -C "${worktree}" cherry-pick --allow-empty "${commit}" >/dev/null 2>&1; then
      pick_failed=1
      break
    fi
  done

  if [[ "${pick_failed}" -eq 0 ]]; then
    nrlta_log "applied #${number} (${head_sha:0:8})"
    applied+=("${number}")
  else
    # Leave the branch usable: drop this fix, keep the rest, and say so loudly.
    # A conflicting fix means main moved under it and a human needs to rebase the
    # PR -- silently shipping a half-applied fix would be far worse.
    git -C "${worktree}" cherry-pick --abort >/dev/null 2>&1 || true
    nrlta_log "CONFLICT applying #${number}; left out of the integration branch"
    skipped+=("${number}:conflict")
  fi
done

git -C "${worktree}" branch --force "${NEMO_RL_INTEGRATION_BRANCH}" HEAD
integration_sha="$(git -C "${worktree}" rev-parse HEAD)"

# Borrow gh's credentials rather than requiring a configured helper or a token in
# the URL, which would end up in the process list.
git -C "${worktree}" -c "credential.helper=!gh auth git-credential" \
  push --force --quiet "${NEMO_RL_FORK_URL}" \
  "${NEMO_RL_INTEGRATION_BRANCH}:refs/heads/${NEMO_RL_INTEGRATION_BRANCH}"

python3 - "${manifest}" "${integration_sha}" "${base_sha}" \
  "${applied[*]-}" "${skipped[*]-}" "${prs_json}" <<'PY'
import json, sys
from datetime import datetime, timezone

manifest, integration_sha, base_sha, applied, skipped, prs_json = sys.argv[1:7]
by_number = {str(pr["number"]): pr for pr in json.loads(prs_json)}

payload = {
    "integration_sha": integration_sha,
    "base_sha": base_sha,
    "synced_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    "applied": [
        {
            "pr": int(n),
            "title": by_number.get(n, {}).get("title", ""),
            "url": by_number.get(n, {}).get("url", ""),
        }
        for n in applied.split()
    ],
    "skipped": [
        {"pr": int(entry.split(":")[0]), "reason": entry.split(":", 1)[1]}
        for entry in skipped.split()
    ],
}
with open(manifest, "w") as handle:
    json.dump(payload, handle, indent=2)
    handle.write("\n")
print(f"integration branch = {base_sha[:8]} + {len(payload['applied'])} fix(es) -> {integration_sha[:8]}")
for item in payload["skipped"]:
    print(f"  NOT applied: #{item['pr']} ({item['reason']})")
PY

nrlta_log "manifest written to ${manifest}"
