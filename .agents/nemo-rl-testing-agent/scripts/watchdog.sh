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

# The standing NeMo-RL-against-megatron-core-`main` watch.
#
# This exists so that pre-existing breakage is owned in one place instead of by
# whichever labeled PR happens to arrive first. Before it, the cost of a bug on
# `main` fell entirely on an unlucky author: their PR comment filled with
# failures they did not cause, and the agent spent its whole fix budget on
# someone else's bug. Now the watchdog finds those breaks on a schedule, fixes
# them, and parks the fixes on the integration branch; the per-PR runs test
# against that branch and only have to explain what is genuinely new.
#
# One pass:
#   1. rebuild the integration branch (main + fixes raised but not yet merged)
#   2. retire registry entries whose fix has merged
#   3. run the suite: megatron-core `main` x integration NeMo-RL x pinned Bridge
#   4. label the results from the registry
#   5. report what changed since the last pass
#
# Anything still failing at step 5 without a registry entry is new breakage the
# watchdog owns: diagnose it, fix it, raise the fix, record it.
#
# Usage:
#   watchdog.sh [--suite l1] [--skip-run] [--publish]
#
# --skip-run reuses the cached baseline (for re-reporting without a cluster job).
# --publish updates the tracking issue on GitHub; without it the rendered issue
# body is only written locally, so a first run can be inspected before it posts.

set -euo pipefail
source "$(cd -P "$(dirname "${BASH_SOURCE[0]}")" && pwd)/lib.sh"

suite="l1"
skip_run=0
publish=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --suite) suite="$2"; shift 2 ;;
    --skip-run) skip_run=1; shift ;;
    --publish) publish=1; shift ;;
    *) nrlta_die "unknown argument: $1" ;;
  esac
done

nrlta_require STATE_DIR NEMO_RL_REPO KNOWN_ISSUES_FILE

baseline_dir="${STATE_DIR}/baselines"
results_path="${baseline_dir}/${suite}-baseline.json"
meta_path="${baseline_dir}/${suite}-baseline.env"
previous_path="${baseline_dir}/${suite}-baseline.previous.json"
manifest="${STATE_DIR}/integration.json"

nrlta_log "step 1/5: rebuilding the integration branch"
"${NRLTA_SCRIPT_DIR}/sync_integration.sh"

nrlta_log "step 2/5: retiring registry entries whose fix has merged"
uv run --script "${NRLTA_SCRIPT_DIR}/known_issues.py" refresh

if [[ "${skip_run}" -eq 1 ]]; then
  nrlta_log "step 3/5: skipped, reusing the cached ${suite} baseline"
  [[ -f "${results_path}" ]] || nrlta_die "no cached baseline at ${results_path}"
else
  # Keep the prior pass so step 5 can say what changed rather than just what is
  # broken. "Still broken" and "broke today" need very different responses.
  [[ -f "${results_path}" ]] && cp "${results_path}" "${previous_path}"
  nrlta_log "step 3/5: running the ${suite} suite against megatron-core main"
  "${NRLTA_SCRIPT_DIR}/ensure_baseline.sh" --suite "${suite}" --force >/dev/null
fi

nrlta_log "step 4/5: labelling results from the known-issues registry"
uv run --script "${NRLTA_SCRIPT_DIR}/known_issues.py" annotate \
  --results "${results_path}" --integration "${manifest}"

nrlta_log "step 5/5: reporting"
body_path="${STATE_DIR}/watchdog-${suite}-issue.md"
publish_args=()
[[ "${publish}" -eq 1 ]] && publish_args+=(--publish)

uv run --script "${NRLTA_SCRIPT_DIR}/post_tracking_issue.py" \
  --suite "${suite}" \
  --results "${results_path}" \
  --meta-env "${meta_path}" \
  --integration "${manifest}" \
  --previous "${previous_path}" \
  --out "${body_path}" \
  ${publish_args[@]+"${publish_args[@]}"}

if [[ "${publish}" -eq 0 ]]; then
  nrlta_log "not published; review ${body_path} then re-run with --publish"
fi
