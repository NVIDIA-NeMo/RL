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

# Makes sure a recent baseline of a suite against megatron-core `main` exists, and
# prints the path to its results JSON.
#
# Why this exists: a test that also fails on `main` is not the labeled PR's fault,
# and telling the two apart after the fact costs a second full suite run at exactly
# the moment someone is waiting on an answer. Baselining once per day and caching
# the result makes "is this pre-existing?" a lookup instead of a cluster job. It
# also catches NeMo-RL/Bridge/mcore integration drift on its own, rather than
# letting it surface as a mystery failure on an unrelated PR.
#
# The baseline is keyed by suite and UTC date, and records the exact megatron-core
# and Megatron-Bridge shas it used, since `main` moves several times a day.
#
# Usage:
#   ensure_baseline.sh --suite l1 [--force] [--max-age-hours N] [--print-only]
#                      [--bridge-ref <ref>]
#
#   --bridge-ref <ref>   Passed through to run_suite.sh. Needed while a raised but
#                        unmerged Megatron-Bridge fix is what stands between the
#                        suite and a usable baseline: a Bridge fix does not ride the
#                        NeMo-RL integration branch, so without this the baseline
#                        keeps recording a break that every PR run has already
#                        worked around, and attribution silently reads off a stack
#                        nobody is testing against.

set -euo pipefail
# shellcheck source=lib.sh
source "$(cd -P "$(dirname "${BASH_SOURCE[0]}")" && pwd)/lib.sh"

suite=""
force=0
print_only=0
max_age_hours=24
bridge_ref=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --suite) suite="$2"; shift 2 ;;
    --force) force=1; shift ;;
    --print-only) print_only=1; shift ;;
    --max-age-hours) max_age_hours="$2"; shift 2 ;;
    --bridge-ref) bridge_ref="$2"; shift 2 ;;
    *) nrlta_die "unknown argument: $1" ;;
  esac
done

[[ -n "${suite}" ]] || nrlta_die "--suite is required (l1 or l2)"
nrlta_require STATE_DIR MEGATRON_REPO MEGATRON_BASE_BRANCH

baseline_dir="${STATE_DIR}/baselines"
mkdir -p "${baseline_dir}"
results_path="${baseline_dir}/${suite}-baseline.json"
meta_path="${baseline_dir}/${suite}-baseline.env"

baseline_is_fresh() {
  [[ -f "${results_path}" && -f "${meta_path}" ]] || return 1
  local age_seconds now mtime
  now="$(date -u +%s)"
  # stat is not portable between macOS and Linux; try both spellings.
  mtime="$(stat -f %m "${results_path}" 2>/dev/null || stat -c %Y "${results_path}" 2>/dev/null || echo 0)"
  age_seconds=$(( now - mtime ))
  [[ "${age_seconds}" -lt $(( max_age_hours * 3600 )) ]]
}

if [[ "${print_only}" -eq 1 ]]; then
  if baseline_is_fresh; then
    echo "${results_path}"
    exit 0
  fi
  nrlta_log "no baseline for ${suite} newer than ${max_age_hours}h"
  exit 1
fi

if [[ "${force}" -eq 0 ]] && baseline_is_fresh; then
  nrlta_log "reusing baseline $(basename "${results_path}") ($(grep -E '^MCORE_SHA=' "${meta_path}" 2>/dev/null || echo 'sha unknown'))"
  echo "${results_path}"
  exit 0
fi

main_sha="$(gh api "repos/${MEGATRON_REPO}/commits/${MEGATRON_BASE_BRANCH}" --jq .sha 2>/dev/null || true)"
[[ -n "${main_sha}" ]] || nrlta_die "could not resolve ${MEGATRON_REPO}@${MEGATRON_BASE_BRANCH}"

run_name="nrlta-baseline-${suite}-$(date -u +%Y%m%d-%H%M)"
nrlta_log "running ${suite} baseline against ${MEGATRON_BASE_BRANCH} (${main_sha:0:8}) as ${run_name}"

# A baseline is a normal suite run whose revision under test happens to be main.
baseline_args=(
  --suite "${suite}"
  --mcore-ref "refs/heads/${MEGATRON_BASE_BRANCH}"
  --mcore-sha "${main_sha}"
  --run-name "${run_name}"
)
[[ -n "${bridge_ref}" ]] && baseline_args+=(--bridge-ref "${bridge_ref}")

"${NRLTA_SCRIPT_DIR}/run_suite.sh" "${baseline_args[@]}" \
  || nrlta_log "baseline run exited non-zero (expected when tests fail)"

slurm_log_dir="${CLUSTER_RUNS_ROOT}/${run_name}/slurm"
combined="${STATE_DIR}/runs/${run_name}/combined.log"
mkdir -p "$(dirname "${combined}")"
if ! ssh "${CLUSTER_SSH_ALIAS}" "cat ${slurm_log_dir}/*.out" > "${combined}" 2>/dev/null; then
  nrlta_die "could not fetch baseline logs from ${slurm_log_dir}"
fi

# Parse to a staging path first. A baseline is cached for a day and silently
# decides every "is this pre-existing?" question in that window, so caching a run
# that never reached the tests is worse than having no baseline at all: every
# later failure comes back "absent from the baseline" and the reason is a day old
# by the time anyone looks. An infra death mid-prep is routine -- one baseline run
# died on `cp: Cannot send after transport endpoint shutdown` -- so check for
# actual test results before promoting.
staged="${results_path}.staged"
uv run --script "${NRLTA_SCRIPT_DIR}/parse_results.py" "${combined}" --out "${staged}"

test_count="$(python3 -c 'import json,sys; print(len(json.load(open(sys.argv[1])).get("tests", [])))' "${staged}" 2>/dev/null || echo 0)"
if [[ "${test_count}" -eq 0 ]]; then
  rm -f "${staged}"
  nrlta_die "baseline run produced no test results (the job died before the suite started); \
not caching it. Check ${slurm_log_dir} and re-run."
fi

mv "${staged}" "${results_path}"

# The baseline decides attribution for a whole day, so it has to record all three
# legs of the stack it used, not just megatron-core. A baseline taken against a
# different NeMo-RL than the PR run is not a baseline for that run.
nemo_rl_sha="$(python3 -c '
import json, sys
print(json.load(open(sys.argv[1])).get("prep", {}).get("nemo_rl_sha", ""))
' "${results_path}" 2>/dev/null || true)"
bridge_sha="$(python3 -c '
import json, sys
print(json.load(open(sys.argv[1])).get("prep", {}).get("bridge_sha", ""))
' "${results_path}" 2>/dev/null || true)"

{
  echo "SUITE=${suite}"
  echo "TEST_COUNT=${test_count}"
  echo "MCORE_SHA=${main_sha}"
  echo "MCORE_REF=refs/heads/${MEGATRON_BASE_BRANCH}"
  echo "NEMO_RL_SHA=${nemo_rl_sha}"
  echo "NEMO_RL_REF=${NEMO_RL_INTEGRATION_BRANCH}"
  echo "BRIDGE_SHA=${bridge_sha}"
  echo "RUN_NAME=${run_name}"
  echo "CREATED_UTC=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
} > "${meta_path}"

nrlta_log "baseline written to ${results_path}"
echo "${results_path}"
