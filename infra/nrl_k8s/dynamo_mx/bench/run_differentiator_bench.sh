#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
#
# Unified runner for the seven MX differentiator scenarios.
# Every executed scenario writes one schema-versioned JSON result.

set -uo pipefail

DIR="$(cd "$(dirname "$0")" && pwd)"
OUT="${OUT:-./differentiator_results}"
PYTHON="${PYTHON:-python3}"
FULL_BYTES="${FULL_BYTES:-61064245248}"
FULL_EXPERT_BYTES="${FULL_EXPERT_BYTES:-58000000000}"
EXPERTS="${EXPERTS:-128}"
ROLLOUT_EP="${ROLLOUT_EP:-2}"
TP="${TP:-2}"
mkdir -p "${OUT}"

run() {
  local name="$1"
  shift
  printf '\n=== %s ===\n' "${name}"
  if "$@"; then
    printf '[PASS] %s\n' "${name}"
  else
    printf '[FAIL] %s\n' "${name}"
    return 1
  fi
}

skip() {
  printf '[SKIP] %s\n' "$1"
}

# D1 — full EP filtering. Set EP_ACTUAL_BYTES from the live receiver log;
# otherwise this is the planner/expected-byte assertion.
ep_args=(
  "${PYTHON}" "${DIR}/differentiator_suite.py"
  --out "${OUT}/01_ep_filter.json"
  ep-filter
  --experts "${EXPERTS}"
  --rollout-ep "${ROLLOUT_EP}"
  --full-expert-bytes "${FULL_EXPERT_BYTES}"
)
if [[ -n "${EP_ACTUAL_BYTES:-}" ]]; then
  ep_args+=(--actual-bytes "${EP_ACTUAL_BYTES}")
fi
run "D1 expert filtering" "${ep_args[@]}"

# D2 — TP-local bytes. Set TP_ACTUAL_BYTES after a live sliced-pull run.
tp_args=(
  "${PYTHON}" "${DIR}/differentiator_suite.py"
  --out "${OUT}/02_tp_slice.json"
  tp-slice
  --tp "${TP}"
  --full-bytes "${FULL_BYTES}"
)
if [[ -n "${TP_ACTUAL_BYTES:-}" ]]; then
  tp_args+=(--actual-bytes "${TP_ACTUAL_BYTES}")
fi
run "D2 TP-local slicing" "${tp_args[@]}"

# D3 — partial refit. Manifest entries are {name, bytes}; selectors repeat.
if [[ -n "${PARTIAL_MANIFEST:-}" && -n "${PARTIAL_SELECTOR:-}" ]]; then
  partial_args=(
    "${PYTHON}" "${DIR}/differentiator_suite.py"
    --out "${OUT}/03_partial.json"
    partial
    --manifest "${PARTIAL_MANIFEST}"
  )
  IFS=',' read -ra selectors <<< "${PARTIAL_SELECTOR}"
  for selector in "${selectors[@]}"; do
    partial_args+=(--selector "${selector}")
  done
  run "D3 partial refit" "${partial_args[@]}"
else
  skip "D3 partial refit: set PARTIAL_MANIFEST and PARTIAL_SELECTOR"
fi

# D4/D5 — use result_*.json produced by elastic_bench.py.
if [[ -n "${ELASTIC_RESULTS:-}" ]]; then
  run "D4 elastic join" \
    "${PYTHON}" "${DIR}/differentiator_suite.py" \
    --out "${OUT}/04_elastic.json" elastic --results "${ELASTIC_RESULTS}"
else
  skip "D4 elastic join: set ELASTIC_RESULTS"
fi

if [[ -n "${STRAGGLER_RESULTS:-${ELASTIC_RESULTS:-}}" ]]; then
  run "D5 straggler isolation" \
    "${PYTHON}" "${DIR}/differentiator_suite.py" \
    --out "${OUT}/05_straggler.json" straggler \
    --results "${STRAGGLER_RESULTS:-${ELASTIC_RESULTS}}"
else
  skip "D5 straggler isolation: set STRAGGLER_RESULTS"
fi

# D6 — direct/tree result directories produced by fanout_bench.py.
if [[ -n "${FANOUT_DIRECT:-}" && -n "${FANOUT_TREE:-}" ]]; then
  run "D6 tree fan-out" \
    "${PYTHON}" "${DIR}/differentiator_suite.py" \
    --out "${OUT}/06_fanout.json" fanout \
    --direct "${FANOUT_DIRECT}" --tree "${FANOUT_TREE}"
else
  skip "D6 tree fan-out: set FANOUT_DIRECT and FANOUT_TREE"
fi

# D7 — parse source-ranked RDMA lines from the production rollout log.
if [[ -n "${MX_LOG:-}" ]]; then
  run "D7 trainer egress balance" \
    "${PYTHON}" "${DIR}/differentiator_suite.py" \
    --out "${OUT}/07_egress.json" egress --log "${MX_LOG}"
else
  skip "D7 trainer egress balance: set MX_LOG"
fi

printf '\nResults: %s\n' "${OUT}"
