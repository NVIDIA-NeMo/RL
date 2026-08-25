#!/usr/bin/env bash
# Submit dependent MPO segments that share one checkpoint directory and W&B run.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
NEMORL="${NEMORL:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
CHAIN_SEGMENTS="${CHAIN_SEGMENTS:-2}"
CHAIN_STEP_TARGETS="${CHAIN_STEP_TARGETS:-}"
CHAIN_ID="${CHAIN_ID:-$(date +%Y%m%d-%H%M%S)}"
CHAIN_NAME="${CHAIN_NAME:-nemotron-omni-mpo-chain-${CHAIN_ID}}"
RESULTS_NAME="${RESULTS_NAME:-${CHAIN_NAME}}"
WANDB_NAME="${WANDB_NAME:-${CHAIN_NAME}}"
WANDB_RUN_ID="${WANDB_RUN_ID:-mpo$(od -An -N3 -tx1 /dev/urandom | tr -d ' \n')}"
RESULTS_ROOT="${RESULTS_ROOT:-${NEMORL}/results}"
final_max_steps="${MPO_MAX_NUM_STEPS:-}"

[[ "${CHAIN_SEGMENTS}" =~ ^[1-9][0-9]*$ ]] || {
  echo "CHAIN_SEGMENTS must be a positive integer" >&2
  exit 2
}
if ((CHAIN_SEGMENTS > 1)) && [[ -z "${CHAIN_STEP_TARGETS}" ]]; then
  echo "CHAIN_STEP_TARGETS is required when CHAIN_SEGMENTS is greater than 1" >&2
  exit 2
fi
step_targets=()
if [[ -n "${CHAIN_STEP_TARGETS}" ]]; then
  IFS=',' read -r -a step_targets <<<"${CHAIN_STEP_TARGETS}"
  [[ "${#step_targets[@]}" -eq "${CHAIN_SEGMENTS}" ]] || {
    echo "CHAIN_STEP_TARGETS must contain one comma-separated target per segment" >&2
    exit 2
  }
  previous_target=0
  for target in "${step_targets[@]}"; do
    [[ "${target}" =~ ^[1-9][0-9]*$ ]] && ((target > previous_target)) || {
      echo "CHAIN_STEP_TARGETS must be strictly increasing positive integers" >&2
      exit 2
    }
    previous_target="${target}"
  done

  last_target="${step_targets[${#step_targets[@]} - 1]}"
  if [[ -z "${final_max_steps}" ]]; then
    final_max_steps="${last_target}"
  elif [[ "${final_max_steps}" != "${last_target}" ]]; then
    echo "MPO_MAX_NUM_STEPS must equal the final CHAIN_STEP_TARGETS value" >&2
    exit 2
  fi
fi
[[ ! -e "${RESULTS_ROOT}/${RESULTS_NAME}" ]] || {
  echo "Refusing to start a fresh chain in existing results: ${RESULTS_ROOT}/${RESULTS_NAME}" >&2
  exit 2
}

dependency=""
job_ids=()
for ((segment = 1; segment <= CHAIN_SEGMENTS; segment++)); do
  segment_stop_after_step=""
  if [[ "${#step_targets[@]}" -gt 0 ]]; then
    segment_stop_after_step="${step_targets[segment - 1]}"
  fi
  output="$(
    NEMORL="${NEMORL}" \
    RUN_ID="${CHAIN_ID}-s${segment}" \
    JOB_NAME="${CHAIN_NAME}-s${segment}" \
    RESULTS_ROOT="${RESULTS_ROOT}" \
    RESULTS_NAME="${RESULTS_NAME}" \
    WANDB_NAME="${WANDB_NAME}" \
    WANDB_RUN_ID="${WANDB_RUN_ID}" \
    WANDB_RESUME="allow" \
    MPO_MAX_NUM_STEPS="${final_max_steps}" \
    MPO_STOP_AFTER_STEP="${segment_stop_after_step}" \
    SBATCH_DEPENDENCY="${dependency}" \
    "${SCRIPT_DIR}/vlm_mpo.sh"
  )"
  printf '%s\n' "${output}"
  job_id="$(printf '%s\n' "${output}" | awk '/Submitted batch job/ {id=$NF} END {print id}')"
  [[ "${job_id}" =~ ^[0-9]+$ ]] || {
    echo "Unable to parse Slurm job ID for segment ${segment}" >&2
    exit 1
  }
  job_ids+=("${job_id}")
  dependency="${job_id}"
done

printf 'W&B run ID: %s\n' "${WANDB_RUN_ID}"
printf 'Results: %s\n' "${RESULTS_ROOT}/${RESULTS_NAME}"
printf 'Slurm chain: %s\n' "${job_ids[*]}"
