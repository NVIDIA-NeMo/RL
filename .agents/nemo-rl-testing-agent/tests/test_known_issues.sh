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

# Checks the known-issues registry on the two ways it can do damage.
#
# Under-matching wastes a debugging budget re-deriving a diagnosis we already
# have. Over-matching is worse: it stamps "known, not your fault" on a genuine
# regression and the PR author ships it. So this exercises a signature that
# differs only in the run-specific parts (pids, paths, line numbers), a
# same-signature-different-test case, and the stale-entry guard that fires when a
# fix is applied to the branch under test and the failure happens anyway.

set -uo pipefail

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRIPT="${here}/../scripts/known_issues.py"
work="$(mktemp -d)"
trap 'rm -rf "${work}"' EXIT

REGISTRY="${work}/known_issues.json"
failures=0

run() { uv run --script "${SCRIPT}" --registry "${REGISTRY}" "$@"; }

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

status_of() {
  python3 - "$1" "$2" <<'PY'
import json, sys
payload = json.load(open(sys.argv[1]))
for test in payload["tests"]:
    if test["name"] == sys.argv[2]:
        print(test.get("known_issue") or test.get("known_issue_stale") or "<none>")
PY
}

run record \
  --id mcore-5918-prompt-tokens \
  --test grpo_megatron_generation \
  --test grpo_megatron_generation_async \
  --signature "AttributeError: 'NoneType' object has no attribute 'tolist'" \
  --diagnosis "megatron-core stopped echoing prompt_tokens back." \
  --repo NVIDIA-NeMo/RL --fix-pr 3363 >/dev/null

# The same bug, seen in a later run: different pid, different path, different
# line number. It must still match.
cat > "${work}/results.json" <<'JSON'
{
  "tests": [
    {
      "name": "grpo_megatron_generation",
      "status": "fail",
      "error_signature": "AttributeError: 'NoneType' object has no attribute 'tolist'"
    },
    {
      "name": "grpo_megatron_generation_async",
      "status": "pass (suspect)",
      "error_signature": "ray::MegatronPolicyWorker.generate_async() (pid=99887) AttributeError: 'NoneType' object has no attribute 'tolist'"
    },
    {
      "name": "grpo_megatron_generation_multiturn",
      "status": "fail",
      "error_signature": "AssertionError: median(data[\"train/token_mult_prob_error\"]) < 1.1"
    },
    {
      "name": "grpo_megatron_generation_non_colocated",
      "status": "pass",
      "error_signature": ""
    }
  ]
}
JSON

run annotate --results "${work}/results.json" >/dev/null

check "matches a known failure despite pid/path/line noise" \
  "mcore-5918-prompt-tokens" "$(status_of "${work}/results.json" grpo_megatron_generation)"
check "matches a suspect pass too, since it is a swallowed failure" \
  "mcore-5918-prompt-tokens" "$(status_of "${work}/results.json" grpo_megatron_generation_async)"
check "leaves an unrelated failure for investigation" \
  "<none>" "$(status_of "${work}/results.json" grpo_megatron_generation_multiturn)"
check "ignores a clean pass" \
  "<none>" "$(status_of "${work}/results.json" grpo_megatron_generation_non_colocated)"

# Same signature, a test the entry does not claim: must NOT be excused.
cat > "${work}/other.json" <<'JSON'
{
  "tests": [
    {
      "name": "some_unrelated_test",
      "status": "fail",
      "error_signature": "AttributeError: 'NoneType' object has no attribute 'tolist'"
    }
  ]
}
JSON
run annotate --results "${work}/other.json" >/dev/null
check "does not excuse a test the entry never claimed" \
  "<none>" "$(status_of "${work}/other.json" some_unrelated_test)"

# The stale guard: the fix is on the branch under test, yet the failure recurred.
cat > "${work}/integration.json" <<'JSON'
{"integration_sha": "abc123", "applied": [{"pr": 3363, "title": "fix", "url": ""}], "skipped": []}
JSON
cat > "${work}/stale.json" <<'JSON'
{
  "prep": {"nemo_rl_sha": "abc123"},
  "tests": [
    {
      "name": "grpo_megatron_generation",
      "status": "fail",
      "error_signature": "AttributeError: 'NoneType' object has no attribute 'tolist'"
    }
  ]
}
JSON
out="$(run annotate --results "${work}/stale.json" --integration "${work}/integration.json")"
check "flags a recurrence of an already-applied fix instead of excusing it" \
  "mcore-5918-prompt-tokens" "$(status_of "${work}/stale.json" grpo_megatron_generation)"
if grep -q "STALE" <<<"${out}"; then
  echo "ok: says loudly that the entry is stale"
else
  echo "FAIL: expected a STALE warning, got:"
  printf '%s\n' "${out}" | sed 's/^/    /'
  failures=$((failures + 1))
fi

# Results predating the manifest must NOT be judged against it. The manifest
# moves whenever a fix is raised, so a cached baseline would otherwise be accused
# of regressing seconds after somebody opened a PR it never contained.
cat > "${work}/older.json" <<'JSON'
{
  "prep": {"nemo_rl_sha": "0ldsha0"},
  "tests": [
    {
      "name": "grpo_megatron_generation",
      "status": "fail",
      "error_signature": "AttributeError: 'NoneType' object has no attribute 'tolist'"
    }
  ]
}
JSON
out="$(run annotate --results "${work}/older.json" --integration "${work}/integration.json")"
check "does not accuse a run that predates the fix of regressing" \
  "mcore-5918-prompt-tokens" "$(status_of "${work}/older.json" grpo_megatron_generation)"
if grep -q "STALE" <<<"${out}"; then
  echo "FAIL: called a pre-fix run stale"
  failures=$((failures + 1))
else
  echo "ok: treats a pre-fix run as a plain known issue"
fi

# Re-annotating must replace the previous verdict, not sit on top of it.
run annotate --results "${work}/stale.json" --integration "${work}/integration.json" >/dev/null
run annotate --results "${work}/stale.json" >/dev/null
check "drops a stale verdict when re-annotated without the manifest" \
  "mcore-5918-prompt-tokens" "$(status_of "${work}/stale.json" grpo_megatron_generation)"

# A merged fix must stop excusing anything: the failure is a regression now.
python3 - "${REGISTRY}" <<'PY'
import json, sys
payload = json.load(open(sys.argv[1]))
payload["issues"][0]["state"] = "merged"
json.dump(payload, open(sys.argv[1], "w"), indent=2)
PY
cat > "${work}/after_merge.json" <<'JSON'
{
  "tests": [
    {
      "name": "grpo_megatron_generation",
      "status": "fail",
      "error_signature": "AttributeError: 'NoneType' object has no attribute 'tolist'"
    }
  ]
}
JSON
run annotate --results "${work}/after_merge.json" >/dev/null
check "stops excusing the failure once the fix merged" \
  "<none>" "$(status_of "${work}/after_merge.json" grpo_megatron_generation)"

echo
if [[ "${failures}" -eq 0 ]]; then
  echo "known_issues: all checks passed"
else
  echo "known_issues: ${failures} check(s) failed"
fi
exit $(( failures > 0 ))
