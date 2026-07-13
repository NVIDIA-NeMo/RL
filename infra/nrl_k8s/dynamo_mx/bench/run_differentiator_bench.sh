#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
#
# Unified runner for the seven MX differentiator scenarios.
# Every executed scenario writes one schema-versioned JSON result.

set -uo pipefail

DIR="$(cd "$(dirname "$0")" && pwd)"
OUT="${OUT:-./differentiator_results}"
PYTHON="${PYTHON:-python3}"
MODEL_ID="Qwen/Qwen3-30B-A3B-Instruct-2507"
FULL_BYTES="61064245248"
EXPERTS="${EXPERTS:-128}"
ROLLOUT_EP="${ROLLOUT_EP:-2}"
TP="${TP:-2}"
FANOUT_N="${FANOUT_N:-13}"
mkdir -p "${OUT}"

if [[ -n "${EP_ACTUAL_BYTES:-}" || -n "${TP_ACTUAL_BYTES:-}" ||
      -n "${FULL_BYTES_OVERRIDE:-}" || -n "${SYNTHETIC:-}" ]]; then
  printf '[FAIL] synthetic/numeric-only inputs are disabled; provide real artifact JSON\n' >&2
  exit 2
fi

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

# D1 — full EP filtering from a real 30B receiver artifact.
if [[ -n "${EP_ARTIFACT:-}" && -n "${FULL_EXPERT_BYTES:-}" ]]; then
  run "D1 expert filtering" \
    "${PYTHON}" "${DIR}/differentiator_suite.py" \
    --out "${OUT}/01_ep_filter.json" ep-filter \
    --experts "${EXPERTS}" --rollout-ep "${ROLLOUT_EP}" \
    --full-expert-bytes "${FULL_EXPERT_BYTES}" --artifact "${EP_ARTIFACT}"
else
  skip "D1 expert filtering: set EP_ARTIFACT and measured FULL_EXPERT_BYTES"
fi

# D2 — TP-local bytes from a real 30B sliced-pull artifact.
if [[ -n "${TP_ARTIFACT:-}" ]]; then
  run "D2 TP-local slicing" \
    "${PYTHON}" "${DIR}/differentiator_suite.py" \
    --out "${OUT}/02_tp_slice.json" tp-slice \
    --tp "${TP}" --full-bytes "${FULL_BYTES}" --artifact "${TP_ARTIFACT}"
else
  skip "D2 TP-local slicing: set TP_ARTIFACT"
fi

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
  fanout_args=(
    "${PYTHON}" "${DIR}/differentiator_suite.py"
    --out "${OUT}/06_fanout.json" fanout
    --direct "${FANOUT_DIRECT}" --tree "${FANOUT_TREE}"
  )
  if [[ "${FANOUT_N}" != "adaptive" ]]; then
    fanout_args+=(--workers "${FANOUT_N}")
  fi
  run "D6 tree fan-out" "${fanout_args[@]}"
else
  skip "D6 tree fan-out: set FANOUT_DIRECT and FANOUT_TREE"
fi

# D7 — parse source-ranked RDMA lines from the production rollout log.
if [[ -n "${MX_LOG:-}" && -n "${MX_ARTIFACT:-}" ]]; then
  run "D7 trainer egress balance" \
    "${PYTHON}" "${DIR}/differentiator_suite.py" \
    --out "${OUT}/07_egress.json" egress --log "${MX_LOG}" \
    --artifact "${MX_ARTIFACT}" --steps "${MX_STEPS:-1}"
else
  skip "D7 trainer egress balance: set MX_LOG and MX_ARTIFACT"
fi

printf '\nModel: %s (%s bytes)\nResults: %s\n' "${MODEL_ID}" "${FULL_BYTES}" "${OUT}"
