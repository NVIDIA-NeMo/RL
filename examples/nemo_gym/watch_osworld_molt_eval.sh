#!/usr/bin/env bash
set -Eeuo pipefail

physical_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd -P)"
if [[ "${physical_root}" == /lustre/fs?/* ]]; then
  ROOT="/lustre/fsw/${physical_root#/*/*/}"
else
  ROOT="${physical_root}"
fi
RUNTIME_ROOT="${OSWORLD_RUNTIME_ROOT:-$(dirname "${ROOT}")/osworld-cc-runtime}"
if [[ -z "${OPENSANDBOX_DOMAIN:-}" && -n "${OPENSANDBOX_BASE_URL:-}" ]]; then
  opensandbox_host="${OPENSANDBOX_BASE_URL#*://}"
  export OPENSANDBOX_DOMAIN="${opensandbox_host%%/*}"
fi

: "${MOLT_RUN_NAME:?Set MOLT_RUN_NAME}"
: "${MOLT_CHECKPOINT_DIR:?Set MOLT_CHECKPOINT_DIR}"
: "${OSWORLD_GRPO_VAL_DATA:?Set OSWORLD_GRPO_VAL_DATA}"
: "${OPENSANDBOX_DOMAIN:?Set OPENSANDBOX_DOMAIN or OPENSANDBOX_BASE_URL}"
: "${OPENSANDBOX_API_KEY:?Set OPENSANDBOX_API_KEY}"

EVAL_EVERY="${EVAL_EVERY:-25}"
EVAL_POLL_SECONDS="${EVAL_POLL_SECONDS:-60}"
EVAL_STATE_DIR="${EVAL_STATE_DIR:-${RUNTIME_ROOT}/results/osworld-cc-eval/.submitted/${MOLT_RUN_NAME}}"
SUBMIT="${ROOT}/examples/nemo_gym/submit_osworld_cc_eval.sh"
mkdir -p "${EVAL_STATE_DIR}"

submit_once() {
  local step="$1"
  local checkpoint_path="${2:-}"
  local marker="${EVAL_STATE_DIR}/step_${step}.submitted"
  [[ -e "${marker}" ]] && return 0

  local eval_name="${MOLT_RUN_NAME}-step${step}"
  local job_id
  if [[ -n "${checkpoint_path}" ]]; then
    job_id="$(
      EVAL_NAME="${eval_name}" \
      EVAL_CHECKPOINT_PATH="${checkpoint_path}" \
      "${SUBMIT}"
    )"
  else
    job_id="$(EVAL_NAME="${eval_name}" "${SUBMIT}")"
  fi
  printf '%s\n' "${job_id}" > "${marker}"
  echo "Submitted OSWorld eval step=${step} job=${job_id}"
}

# Step 0 evaluates the RFC0037 SFT checkpoint directly.
submit_once 0

while true; do
  for step_dir in "${MOLT_CHECKPOINT_DIR}"/step_*; do
    [[ -d "${step_dir}" ]] || continue
    step="${step_dir##*/step_}"
    [[ "${step}" =~ ^[0-9]+$ ]] || continue
    (( step > 0 && step % EVAL_EVERY == 0 )) || continue
    weights="${step_dir}/policy/weights"
    [[ -f "${weights}/latest_checkpointed_iteration.txt" ]] || continue
    submit_once "${step}" "${weights}"
  done
  sleep "${EVAL_POLL_SECONDS}"
done
