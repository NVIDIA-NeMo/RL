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

# Exercises run_suite.sh's refusal to submit twice under one run name.
#
# The rule was prose in megatron-pr-test-run for a long time and a resumed sweep
# still re-used a name, because the run that spent it happened in an earlier
# session. Two revisions then shared one artifact directory, which nothing
# downstream can undo: ensure_baseline.sh parses a run with `cat *.out`. So the
# rule is enforced in the script, and enforcement gets a test. Run after touching
# the guard:
#
#   bash .agents/nemo-rl-testing-agent/tests/test_run_suite_guards.sh

set -uo pipefail

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRIPT="${here}/../scripts/run_suite.sh"
ROOT=/tmp/nrlta-run-suite-test
CONFIG="${ROOT}/config.env"
GUARD_TEXT="was already used"

failures=0

reset_fixture() {
  rm -rf "${ROOT}"
  mkdir -p "${ROOT}/state"
  # Inherit the real config, then redirect the state dir and point the clone URLs
  # at paths that cannot resolve. The guard runs before any ref lookup, so the
  # unresolvable URLs make the no-guard cases fail fast and offline instead of
  # reaching for the network or, worse, for cog.
  {
    echo "source '${here}/../config.env'"
    echo "STATE_DIR='${ROOT}/state'"
    echo "NEMO_RL_CLONE_URL='${ROOT}/nonexistent.git'"
    echo "NEMO_RL_FORK_URL='${ROOT}/nonexistent.git'"
  } > "${CONFIG}"
}

spend_run_name() { # name
  mkdir -p "${ROOT}/state/runs/$1"
  touch "${ROOT}/state/runs/$1/cog.log"
}

run_suite() { # extra args...
  NRLTA_CONFIG="${CONFIG}" bash "${SCRIPT}" \
    --suite l1 --mcore-ref refs/heads/main "$@" 2>&1
}

check_guard_fires() { # description, args...
  local desc="$1"; shift
  local out rc
  out="$(run_suite "$@")" && rc=0 || rc=$?
  if [ "${rc}" -eq 0 ]; then
    echo "FAIL: ${desc}: expected a non-zero exit"
    printf '%s\n' "${out}" | sed 's/^/    /'
    failures=$((failures + 1))
    return
  fi
  if ! printf '%s' "${out}" | grep -q "${GUARD_TEXT}"; then
    echo "FAIL: ${desc}: output missing '${GUARD_TEXT}'"
    printf '%s\n' "${out}" | sed 's/^/    /'
    failures=$((failures + 1))
    return
  fi
  echo "ok: ${desc}"
}

check_guard_silent() { # description, args...
  local desc="$1"; shift
  local out
  out="$(run_suite "$@")"
  if printf '%s' "${out}" | grep -q "${GUARD_TEXT}"; then
    echo "FAIL: ${desc}: the guard fired when it should not have"
    printf '%s\n' "${out}" | sed 's/^/    /'
    failures=$((failures + 1))
    return
  fi
  echo "ok: ${desc}"
}

reset_fixture
check_guard_silent "an unused run name submits" --run-name nrlta-fresh-l1-a1

reset_fixture
spend_run_name nrlta-spent-l1-a1
check_guard_fires "a run name that already has a cog.log is refused" \
  --run-name nrlta-spent-l1-a1

# The override exists for the rare deliberate overwrite, and has to keep working
# or the guard becomes something to route around by editing the script.
reset_fixture
spend_run_name nrlta-spent-l1-a1
check_guard_silent "--reuse-run-name overrides the refusal" \
  --run-name nrlta-spent-l1-a1 --reuse-run-name

# A dry run submits nothing, so refusing it would only obstruct inspecting what a
# previous attempt did.
reset_fixture
spend_run_name nrlta-spent-l1-a1
check_guard_silent "--dry-run is never refused" \
  --run-name nrlta-spent-l1-a1 --dry-run

# The default name is a UTC stamp, which collides only if two submits land in the
# same second; it must not be refused on a clean state dir.
reset_fixture
check_guard_silent "the default run name is not refused"

rm -rf "${ROOT}"

echo
if [ "${failures}" -eq 0 ]; then
  echo "run_suite guards: all checks passed"
else
  echo "run_suite guards: ${failures} check(s) failed"
fi
exit $(( failures > 0 ))
