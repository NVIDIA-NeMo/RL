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

# Runs INSIDE the container, on the head node only.
#
# Executes every sub-test of a NeMo-RL functional suite as an independent step.
# The suite scripts themselves are `set -e`, so running them directly would stop
# at the first failure and hide the status of every later test; the per-PR
# report needs a verdict for each one.
#
# The sub-test list is read out of the suite file's `run_test` lines, so the two
# stay in sync automatically (including `fast` markers and commented-out tests).
#
# Usage: run_suite_remote.sh <path-to-suite.sh>
# Environment:
#   CONTAINER_ROOT       NeMo-RL project root to run from (default /opt/nemo-rl).
#   NRLTA_ARTIFACT_DIR   Shared-scratch dir for per-test logs and results.tsv.
#   ONLY_TESTS           Space-separated sub-test names to run (default: all).

set -uo pipefail

suite="${1:?usage: run_suite_remote.sh <path-to-suite.sh>}"
[ -f "${suite}" ] || { echo "NRLTA_FAIL: suite not found: ${suite}"; exit 1; }

project_root="${CONTAINER_ROOT:-$(cd "$(dirname "${suite}")/../.." && pwd)}"
artifact_dir="${NRLTA_ARTIFACT_DIR:-${project_root}/tests/functional/nrlta}"
only_tests=" ${ONLY_TESTS:-} "
suite_name="$(basename "${suite}" .sh)"

mkdir -p "${artifact_dir}"
results_tsv="${artifact_dir}/results.tsv"
: > "${results_tsv}"

cd "${project_root}"

# `run_test [fast] <command...>`; commented-out tests never match.
suite_lines=()
while IFS= read -r suite_line; do
  suite_lines+=("${suite_line}")
done < <(grep -E '^[[:space:]]*run_test[[:space:]]' "${suite}")

if [ "${#suite_lines[@]}" -eq 0 ]; then
  echo "NRLTA_FAIL: no run_test lines found in ${suite}"
  exit 1
fi

trim() {
  local s="$1"
  s="${s#"${s%%[![:space:]]*}"}"
  s="${s%"${s##*[![:space:]]}"}"
  printf '%s' "${s}"
}

overall_rc=0
echo "===NRLTA_SUITE_BEGIN suite=${suite_name} tests=${#suite_lines[@]}==="

for line in "${suite_lines[@]}"; do
  cmd="$(trim "${line#*run_test}")"
  if [[ "${cmd}" == fast\ * || "${cmd}" == fast$'\t'* ]]; then
    cmd="$(trim "${cmd#fast}")"
  fi

  if [[ "${cmd}" =~ ([A-Za-z0-9_.-]+)\.sh ]]; then
    name="$(basename "${BASH_REMATCH[1]}")"
  else
    echo "NRLTA_WARN: cannot derive a test name from: ${cmd}"
    continue
  fi

  if [ -n "$(trim "${only_tests}")" ] && [[ "${only_tests}" != *" ${name} "* ]]; then
    echo "===NRLTA_TEST_SKIP name=${name} reason=not-selected==="
    continue
  fi

  test_log="${artifact_dir}/${name}.log"
  echo "===NRLTA_TEST_BEGIN name=${name} suite=${suite_name} cmd=${cmd}==="
  start_secs=${SECONDS}
  eval "${cmd}" 2>&1 | tee "${test_log}"
  rc=${PIPESTATUS[0]}
  elapsed=$((SECONDS - start_secs))
  echo "===NRLTA_TEST_END name=${name} rc=${rc} secs=${elapsed}==="

  printf '%s\t%s\t%s\t%s\n' "${name}" "${rc}" "${elapsed}" "${suite_name}" >> "${results_tsv}"
  if [ "${rc}" -ne 0 ]; then
    overall_rc=1
  fi
done

echo "===NRLTA_SUITE_SUMMARY suite=${suite_name}==="
cat "${results_tsv}"
echo "===NRLTA_SUITE_END suite=${suite_name} overall_rc=${overall_rc}==="

exit "${overall_rc}"
