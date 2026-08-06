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

# Checks the provenance table that tells a reader which three revisions produced
# a result.
#
# The thing being defended is that a reader can tell, without asking, whether a
# green table is a statement about the stack NeMo-RL ships or about a stack that
# exists only inside this harness. Those look identical once the refs are
# dropped, which is what the old hand-typed `--meta` shas did. The Bridge row
# carries the most weight: a sha there is NeMo-RL's pin, while a branch is a fix
# nobody has merged.

set -uo pipefail

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRIPT="${here}/../scripts/post_report.py"
work="$(mktemp -d)"
trap 'rm -rf "${work}"' EXIT

failures=0

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

contains() {
  local label="$1" needle="$2" haystack="$3"
  if [[ "${haystack}" == *"${needle}"* ]]; then
    echo "ok: ${label}"
  else
    echo "FAIL: ${label}"
    echo "    expected to find: ${needle}"
    failures=$((failures + 1))
  fi
}

absent() {
  local label="$1" needle="$2" haystack="$3"
  if [[ "${haystack}" != *"${needle}"* ]]; then
    echo "ok: ${label}"
  else
    echo "FAIL: ${label}"
    echo "    expected NOT to find: ${needle}"
    failures=$((failures + 1))
  fi
}

# A run of a labeled PR, carrying an unmerged Bridge fix on a fork branch of
# NeMo-RL: the shape every sweep produces while a Bridge fix is in review.
cat > "${work}/results.json" <<'JSON'
{
  "prep": {
    "mcore_fetch_ref": "refs/pull/5382/head",
    "mcore_sha": "eb01b689c4ddf1034db987a25736aee12552d2ea",
    "mcore_url": "https://github.com/NVIDIA/Megatron-LM.git",
    "bridge_fetch_ref": "mcore-5382-fix",
    "bridge_sha": "2940a635731da68c5960a020f8ea2f678fa1e073",
    "bridge_url": "https://github.com/NVIDIA-NeMo/Megatron-Bridge.git",
    "nemo_rl_fetch_ref": "nrlta/integration",
    "nemo_rl_sha": "86774472e229b457314418fe3f25e8e844775596",
    "nemo_rl_url": "https://github.com/shanmugamr1992/RL.git"
  },
  "baseline": {"mcore_sha": "3aee84c392e3a265c497991d0d07b27bfffb2490"},
  "tests": [{"name": "grpo_megatron_generation", "status": "pass"}]
}
JSON

body="$(uv run --script "${SCRIPT}" --pr 5382 --state results --dry-run \
  --results "${work}/results.json" --meta suite=L1 \
  --meta mcore_sha=eb01b689c4ddf1034db987a25736aee12552d2ea)"

contains "names the PR the mcore revision came from" \
  "[#5382](https://github.com/NVIDIA/Megatron-LM/pull/5382) head" "${body}"
contains "links the mcore commit to Megatron-LM" \
  "[\`eb01b689\`](https://github.com/NVIDIA/Megatron-LM/commit/eb01b689c4ddf1034db987a25736aee12552d2ea)" \
  "${body}"

# The fork matters: NeMo-RL's sha does not exist in NVIDIA-NeMo/RL while the
# integration branch lives on a fork, so a link to the upstream repo 404s.
contains "links the NeMo-RL commit to the fork it was fetched from" \
  "https://github.com/shanmugamr1992/RL/commit/86774472" "${body}"
contains "names the NeMo-RL branch, not just its sha" "\`nrlta/integration\`" "${body}"

contains "says a Bridge branch is an override rather than the pin" \
  "an override, **not** the Bridge NeMo-RL pins" "${body}"
contains "still records the baseline it was compared against" \
  "megatron-core \`main\` at [\`3aee84c3\`]" "${body}"

# The header used to carry the same shas, typed by hand. Two places to read one
# fact is one place to get it wrong.
absent "drops a sha from the header once the table states it" \
  "mcore sha: \`eb01b689" "${body}"
contains "keeps header fields the table does not cover" "suite: \`L1\`" "${body}"

# A Bridge sha is the pin, and must not be described as an override.
python3 - "${work}" <<'PY'
import json, sys
work = sys.argv[1]
payload = json.load(open(f"{work}/results.json"))
payload["prep"]["bridge_fetch_ref"] = "573e088c1234567890abcdef1234567890abcdef"
payload["prep"]["mcore_fetch_ref"] = "refs/heads/main"
json.dump(payload, open(f"{work}/pinned.json", "w"), indent=2)
PY
pinned="$(uv run --script "${SCRIPT}" --pr 5382 --state results --dry-run \
  --results "${work}/pinned.json" --meta suite=L1)"
contains "calls a Bridge sha the pin" "the sha NeMo-RL pins" "${pinned}"
absent "does not call the pin an override" "an override" "${pinned}"
contains "renders a branch ref without its refs/heads prefix" "| megatron-core | \`main\` |" "${pinned}"

# Results from before the URLs were recorded must still render, minus the links.
python3 - "${work}" <<'PY'
import json, sys
work = sys.argv[1]
payload = json.load(open(f"{work}/results.json"))
for key in ("mcore_url", "bridge_url", "nemo_rl_url"):
    payload["prep"].pop(key)
json.dump(payload, open(f"{work}/nourls.json", "w"), indent=2)
PY
nourls="$(uv run --script "${SCRIPT}" --pr 5382 --state results --dry-run \
  --results "${work}/nourls.json" --meta suite=L1)"
contains "still shows the refs when no URL was recorded" "\`nrlta/integration\`" "${nourls}"
contains "falls back to a bare sha rather than a broken link" "| \`86774472\` |" "${nourls}"

# `--state running` is posted before any suite has run, so there is nothing to
# describe and the table must not appear half-filled.
running="$(uv run --script "${SCRIPT}" --pr 5382 --state running --dry-run --meta suite=L1)"
absent "omits the table entirely before the run has results" "Exactly what was tested" "${running}"

# The watchdog issue must render the same table from the same code.
issue_out="${work}/issue.md"
uv run --script "${here}/../scripts/post_tracking_issue.py" --suite l1 \
  --results "${work}/results.json" --out "${issue_out}" >/dev/null 2>&1
issue="$(cat "${issue_out}" 2>/dev/null)"
contains "the watchdog issue carries the same Bridge warning" \
  "an override, **not** the Bridge NeMo-RL pins" "${issue}"
check "the watchdog issue heads the table its own way" \
  "1" "$(grep -c '^### Stack under test$' "${issue_out}" 2>/dev/null || echo 0)"
absent "does not stack two headings on one table" "**Exactly what was tested**" "${issue}"

echo
if [[ "${failures}" -eq 0 ]]; then
  echo "report stack: all checks passed"
else
  echo "report stack: ${failures} check(s) failed"
fi
exit $(( failures > 0 ))
