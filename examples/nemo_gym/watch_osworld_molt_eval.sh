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
EVAL_AT_START="${EVAL_AT_START:-true}"
EVAL_STATE_DIR="${EVAL_STATE_DIR:-${RUNTIME_ROOT}/results/osworld-cc-eval/.submitted/${MOLT_RUN_NAME}}"
EVAL_SNAPSHOT_DIR="${EVAL_SNAPSHOT_DIR:-${RUNTIME_ROOT}/results/osworld-cc-eval/.checkpoint-snapshots/${MOLT_RUN_NAME}}"
SUBMIT="${ROOT}/examples/nemo_gym/submit_osworld_cc_eval_jianh_parity.sh"
mkdir -p "${EVAL_STATE_DIR}" "${EVAL_SNAPSHOT_DIR}"

snapshot_weights() {
  local step="$1"
  local source_weights="$2"
  local snapshot_step="${EVAL_SNAPSHOT_DIR}/step_${step}"
  local snapshot_weights="${snapshot_step}/weights"
  [[ -f "${snapshot_weights}/latest_checkpointed_iteration.txt" ]] && {
    printf '%s\n' "${snapshot_weights}"
    return 0
  }

  local tmp_step="${snapshot_step}.tmp.$$"
  rm -rf "${tmp_step}"
  mkdir -p "${tmp_step}/weights"
  # Checkpoint files are immutable after finalization. Hard links protect the
  # model weights from keep_top_k pruning without duplicating their data blocks;
  # optimizer state is deliberately excluded from eval snapshots.
  cp -al "${source_weights}/." "${tmp_step}/weights/"
  rm -rf "${snapshot_step}"
  mv "${tmp_step}" "${snapshot_step}"
  printf '%s\n' "${snapshot_weights}"
}

submit_once() {
  local step="$1"
  local checkpoint_path="${2:-}"
  local marker="${EVAL_STATE_DIR}/step_${step}.submitted"
  [[ -e "${marker}" ]] && return 0

  local eval_name="${MOLT_RUN_NAME}-step${step}"
  local job_id
  if [[ -n "${checkpoint_path}" ]]; then
    checkpoint_path="$(snapshot_weights "${step}" "${checkpoint_path}")"
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

# Step 0 evaluates the RFC0037 SFT checkpoint directly. Runs sharing that
# identical base model may reuse one Eval0 instead of submitting duplicates.
if [[ "${EVAL_AT_START}" == "true" ]]; then
  submit_once 0
fi

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
