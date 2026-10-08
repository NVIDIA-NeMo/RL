#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd -P)"
RUNTIME_ROOT="${OSWORLD_RUNTIME_ROOT:-$(dirname "${ROOT}")/osworld-cc-runtime}"
PRIVATE_ENV="${OSWORLD_PRIVATE_ENV:-${RUNTIME_ROOT}/private/opensandbox.env}"
POLL_SECONDS="${MOLT_RESUME_POLL_SECONDS:-300}"

set -a
source "${PRIVATE_ENV}"
set +a

run_names=(
  osworld361-molt-nemogym-parity8
  osworld361-molt-nemogym-parity32
)
targets=(300 300)
submitters=(
  "${ROOT}/examples/nemo_gym/submit_osworld_molt_nemogym_parity8.sh"
  "${ROOT}/examples/nemo_gym/submit_osworld_molt_nemogym_parity32.sh"
)

latest_step() {
  local run_name="$1"
  python - "${RUNTIME_ROOT}/results/${run_name}/checkpoints" <<'PY'
import json
import sys
from pathlib import Path

steps = []
for info_path in Path(sys.argv[1]).glob("step_*/training_info.json"):
    try:
        steps.append(int(json.loads(info_path.read_text())["current_step"]))
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError):
        continue
print(max(steps, default=0))
PY
}

while true; do
  all_complete=true
  for index in 0 1; do
    run_name="${run_names[index]}"
    step="$(latest_step "${run_name}")"
    if (( step >= targets[index] )); then
      continue
    fi
    all_complete=false

    job_name="molt-${run_name}"
    active="$(squeue -h -u "${USER}" -n "${job_name}" -o '%i|%T')"
    if [[ -n "${active}" ]]; then
      printf '[%s] run=%s step=%s active=%s\n' \
        "$(date -Is)" "${run_name}" "${step}" "${active//$'\n'/; }"
      continue
    fi

    job_id="$("${submitters[index]}")"
    printf '[%s] resumed run=%s from_step=%s target=%s job=%s\n' \
      "$(date -Is)" "${run_name}" "${step}" "${targets[index]}" "${job_id}"
  done

  if [[ "${all_complete}" == true ]]; then
    printf '[%s] both parity experiments reached target steps; stopping\n' "$(date -Is)"
    exit 0
  fi
  sleep "${POLL_SECONDS}"
done
