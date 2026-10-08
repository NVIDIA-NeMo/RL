#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd -P)"
RUNTIME_ROOT="${OSWORLD_RUNTIME_ROOT:-$(dirname "${ROOT}")/osworld-cc-runtime}"
PRIVATE_ENV="${OSWORLD_PRIVATE_ENV:-${RUNTIME_ROOT}/private/opensandbox.env}"
STATE_DIR="${RUNTIME_ROOT}/results/.molt-auto-resume"
STATE_FILE="${STATE_DIR}/next_experiment"
POLL_SECONDS="${MOLT_RESUME_POLL_SECONDS:-300}"

mkdir -p "${STATE_DIR}"
set -a
source "${PRIVATE_ENV}"
set +a

run_names=(
  osworld361-molt-nemogym-parity32
  osworld361-molt-nemogym-parity8
  osworld361-molt-legacy-chunk1-repro
)
targets=(300 300 200)
submitters=(
  "${ROOT}/examples/nemo_gym/submit_osworld_molt_nemogym_parity32.sh"
  "${ROOT}/examples/nemo_gym/submit_osworld_molt_nemogym_parity8.sh"
  "${ROOT}/examples/nemo_gym/submit_osworld_molt_legacy_chunk1_repro.sh"
)

latest_step() {
  local run_name="$1"
  python - "${RUNTIME_ROOT}/results/${run_name}/checkpoints" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
steps = []
for info_path in root.glob("step_*/training_info.json"):
    try:
        steps.append(int(json.loads(info_path.read_text())["current_step"]))
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError):
        continue
print(max(steps, default=0))
PY
}

opensandbox_reachable() {
  python <<'PY'
import os
import socket
from urllib.parse import urlparse

raw = os.environ.get("OPENSANDBOX_BASE_URL")
if not raw:
    raw = "http://" + os.environ["OPENSANDBOX_DOMAIN"]
url = urlparse(raw)
port = url.port or (443 if url.scheme == "https" else 80)
try:
    with socket.create_connection((url.hostname, port), timeout=5):
        pass
except OSError:
    raise SystemExit(1)
PY
}

while true; do
  # Serialize the three large OSWorld experiments. This prevents the prior
  # 3x128 sandbox-create storm while still round-robin resuming every run.
  active="$(
    squeue -h -u "${USER}" \
      -n "molt-${run_names[0]},molt-${run_names[1]},molt-${run_names[2]}" \
      -o '%i|%j|%T'
  )"
  if [[ -n "${active}" ]]; then
    printf '[%s] active Molt job; waiting: %s\n' "$(date -Is)" "${active//$'\n'/; }"
    sleep "${POLL_SECONDS}"
    continue
  fi

  if ! opensandbox_reachable; then
    printf '[%s] OpenSandbox unreachable; deferring all submissions\n' "$(date -Is)"
    sleep "${POLL_SECONDS}"
    continue
  fi

  next_index=0
  if [[ -f "${STATE_FILE}" ]]; then
    read -r next_index < "${STATE_FILE}"
  fi
  [[ "${next_index}" =~ ^[0-2]$ ]] || next_index=0

  submitted=false
  for offset in 0 1 2; do
    index=$(((next_index + offset) % 3))
    step="$(latest_step "${run_names[index]}")"
    if (( step >= targets[index] )); then
      continue
    fi

    job_id="$("${submitters[index]}")"
    printf '%s\n' "$(((index + 1) % 3))" > "${STATE_FILE}"
    printf '[%s] submitted run=%s from_step=%s target=%s job=%s\n' \
      "$(date -Is)" "${run_names[index]}" "${step}" "${targets[index]}" "${job_id}"
    submitted=true
    break
  done

  if [[ "${submitted}" == false ]]; then
    printf '[%s] all Molt experiments reached their target steps; stopping\n' "$(date -Is)"
    exit 0
  fi
  sleep "${POLL_SECONDS}"
done
