#!/usr/bin/env bash
# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# Private integration harness; run inside the allocated container.
set -euo pipefail
set +x
umask 077
phase_name=$1
shift
if [[ ! "$phase_name" =~ ^[a-z0-9][a-z0-9-]*$ ]]; then
  echo 'Invalid phase name' >&2
  exit 2
fi
cd /flint-work
mkdir -p evidence
if [[ -e "evidence/$phase_name.log" ]]; then
  echo 'Phase receipt already exists; choose a fresh phase name.' >&2
  exit 2
fi
export FLINT_METRICS_PATH="/flint-work/evidence/$phase_name.metrics.jsonl"
export FLINT_REFIT_AUDIT_DIR="/flint-work/evidence/$phase_name.refit"
export FLINT_REFIT_AUDIT_HEAD_ROW_START=${FLINT_REFIT_AUDIT_HEAD_ROW_START:-0}
cp /flint-work/RL/examples/configs/recipes/llm/grpo-flint-49layer-1n4g-automodel-sglang-actckpt-quick.yaml "evidence/$phase_name.base-recipe.yaml"
if [[ "${FLINT_SYNTHETIC_SMOKE:-0}" == 1 ]]; then
  cp /flint-work/RL/examples/configs/recipes/llm/grpo-flint-49layer-1n4g-automodel-sglang-actckpt-synthetic-quick.yaml "evidence/$phase_name.selected-recipe.yaml"
fi
printf '%q ' "$@" >"evidence/$phase_name.overrides.txt"
printf '\n' >>"evidence/$phase_name.overrides.txt"
bash /flint-work/tools/run_grpo.sh "$@" >"evidence/$phase_name.log" 2>&1 &
phase_pid=$!
/opt/flint/bin/python-driver /flint-work/tools/monitor_phase.py --pid "$phase_pid" --output "evidence/$phase_name.memory.jsonl" &
monitor_pid=$!
set +e
wait "$phase_pid"
phase_status=$?
set -e
kill "$monitor_pid" 2>/dev/null || true
wait "$monitor_pid" 2>/dev/null || true
printf '%s\n' "$phase_status" >"evidence/$phase_name.exit-code"
printf 'Phase %s exited %s\n' "$phase_name" "$phase_status"
exit "$phase_status"
