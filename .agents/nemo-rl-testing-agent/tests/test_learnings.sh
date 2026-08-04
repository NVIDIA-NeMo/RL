#!/bin/bash
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

# Checks the learnings queue on the three things that decide whether it is worth
# having.
#
# It has to survive being written to mid-sweep, so a re-record must accumulate
# rather than duplicate. It has to notice when a promoted instruction failed to
# stick, since that is the signal to stop writing prose and add a guard. And it
# has to refuse a "lesson" that is really a ledger note -- one naming a run or a
# cluster path cannot help a later run and would leak an internal reference into
# a public repo.
#
#   bash .agents/nemo-rl-testing-agent/tests/test_learnings.sh

set -uo pipefail

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRIPT="${here}/../scripts/learnings.py"
TARGET=".agents/contributor-skills/megatron-pr-test-run/SKILL.md"
SYMLINKED_TARGET=".claude/skills/megatron-pr-test-run/SKILL.md"
work="$(mktemp -d)"
trap 'rm -rf "${work}"' EXIT

QUEUE="${work}/learnings.json"
failures=0

run() { uv run --script "${SCRIPT}" --queue "${QUEUE}" "$@"; }

check() {
  local label="$1" want="$2" got="$3"
  if [[ "${got}" == "${want}" ]]; then
    echo "ok: ${label}"
  else
    echo "FAIL: ${label}"
    echo "    wanted: ${want}"
    echo "    got:    ${got}"
    failures=$((failures + 1))
  fi
}

field_of() {
  python3 - "$1" "$2" "$3" <<'PY'
import json, sys
payload = json.load(open(sys.argv[1]))
for item in payload["learnings"]:
    if item["id"] == sys.argv[2]:
        print(item.get(sys.argv[3], "<none>"))
PY
}

record_one() {
  run record --id "$1" --trigger "Prep swapped mcore but left the image's Bridge." \
    --lesson "$2" --target "${3:-${TARGET}}" "${@:4}"
}

record_one bridge-pin "Always pin Megatron-Bridge in the same prep step that swaps megatron-core." >/dev/null
check "records a first sighting" "1" "$(field_of "${QUEUE}" bridge-pin occurrences)"

# The same gap hit again before anyone promoted it: one entry, counted twice.
record_one bridge-pin "Always pin Megatron-Bridge in the same prep step that swaps megatron-core." >/dev/null
check "accumulates a repeat instead of duplicating it" \
  "2" "$(field_of "${QUEUE}" bridge-pin occurrences)"

# A promoted entry that comes back means the written instruction did not hold.
run resolve --id bridge-pin --as promoted --pr 2931 >/dev/null
check "resolving marks it promoted" "promoted" "$(field_of "${QUEUE}" bridge-pin state)"

out="$(record_one bridge-pin "Always pin Megatron-Bridge in the same prep step that swaps megatron-core." 2>&1)"
if grep -q "REGRESSED" <<<"${out}"; then
  echo "ok: says loudly that a promoted instruction failed to stick"
else
  echo "FAIL: expected a REGRESSED warning, got:"
  printf '%s\n' "${out}" | sed 's/^/    /'
  failures=$((failures + 1))
fi
check "reopens a regressed entry for promotion" \
  "pending" "$(field_of "${QUEUE}" bridge-pin state)"

# A lesson is written into a public repo and must generalise past one run.
record_one leaky "Re-run with the artifacts under /lustre/fsw/portfolios/coreai/users/x/runs." >/dev/null 2>&1
check "refuses a lesson carrying a cluster path" "" "$(field_of "${QUEUE}" leaky state)"

record_one terse "It broke." >/dev/null 2>&1
check "refuses a lesson too vague to act on" "" "$(field_of "${QUEUE}" terse state)"

record_one nowhere "Pass the bridge ref through to the remote runner every time." \
  ".agents/contributor-skills/does-not-exist/SKILL.md" >/dev/null 2>&1
check "refuses a target that does not exist" "" "$(field_of "${QUEUE}" nowhere state)"

# Skills are reachable through .claude/skills symlinks, but a commit has to touch
# the real file, so that is what gets recorded.
if [[ -e "${here}/../../../${SYMLINKED_TARGET}" ]]; then
  record_one via-symlink \
    "Name all three shas in the report so a green table is interpretable." \
    "${SYMLINKED_TARGET}" >/dev/null
  check "records the real path behind a skill symlink" \
    "['${TARGET}']" "$(field_of "${QUEUE}" via-symlink targets)"
else
  echo "skip: ${SYMLINKED_TARGET} is not present"
fi

# Blocking entries sort first: they are the ones that invalidate a running sweep.
record_one urgent "Verify the megatron import guard passed before trusting any result." \
  "${TARGET}" --severity blocking >/dev/null
first="$(run list --format json | python3 -c 'import json,sys; print(json.load(sys.stdin)[0]["id"])')"
check "lists blocking entries first" "urgent" "${first}"

check "leaves promoted entries out of the pending list" \
  "" "$(run list --format json | python3 -c 'import json,sys; print("".join(i["id"] for i in json.load(sys.stdin) if i["state"]!="pending"))')"

# Rejecting without a reason invites the same entry back next month.
run resolve --id urgent --as rejected >/dev/null 2>&1
check "refuses a rejection with no reason" "pending" "$(field_of "${QUEUE}" urgent state)"
run resolve --id urgent --as rejected --note "Already enforced by the prep guard." >/dev/null
check "accepts a rejection that says why" "rejected" "$(field_of "${QUEUE}" urgent state)"

echo
if [[ "${failures}" -eq 0 ]]; then
  echo "learnings: all checks passed"
else
  echo "learnings: ${failures} check(s) failed"
fi
exit $(( failures > 0 ))
