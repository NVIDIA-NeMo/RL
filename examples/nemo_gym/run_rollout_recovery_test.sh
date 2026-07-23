#!/usr/bin/env bash
# Run a small GRPO math job through SingleController + NeMo-Gym and exercise
# completed-sibling rollout persistence and recovery inside a Linux container.
#
# Full crash-and-recover smoke test in one container:
#
#   RUN_ROOT=/shared/path/rollout-recovery-smoke \
#     ./examples/nemo_gym/run_rollout_recovery_test.sh all
#
# To test recovery across separate container or Slurm allocations, mount the
# same shared RUN_ROOT and run the phases separately:
#
#   RUN_ROOT=/shared/path/rollout-recovery-smoke RESET=1 \
#     ./examples/nemo_gym/run_rollout_recovery_test.sh crash
#   RUN_ROOT=/shared/path/rollout-recovery-smoke \
#     ./examples/nemo_gym/run_rollout_recovery_test.sh recover
#
# Use "persist" to run uninterrupted and only verify that sibling checkpoints
# are written. Additional arguments are passed through as Hydra overrides.

set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
PROJECT_ROOT=$(realpath "${SCRIPT_DIR}/../..")
CONFIG_PATH="${SCRIPT_DIR}/grpo_qwen3_0_6b_rollout_recovery.yaml"
SAMPLE_SOURCE="${PROJECT_ROOT}/3rdparty/Gym-workspace/Gym/resources_servers/math_with_judge/data/example.jsonl"

MODE="${1:-all}"
case "${MODE}" in
  all|crash|recover|persist) ;;
  *)
    echo "Usage: $0 {all|crash|recover|persist} [Hydra overrides ...]" >&2
    exit 2
    ;;
esac
shift || true

RUN_ID=${RUN_ID:-$(date -u +%Y%m%d-%H%M%S)}
# Use a mounted shared-filesystem path here when crash and recover run in
# different containers or Slurm allocations.
RUN_ROOT=${RUN_ROOT:-"${PROJECT_ROOT}/results/grpo-math-qwen3-0.6b-sc/${RUN_ID}"}
DATA_DIR="${RUN_ROOT}/data"
LOG_DIR="${RUN_ROOT}/logs"
MODEL_CHECKPOINT_DIR=${MODEL_CHECKPOINT_DIR:-"${RUN_ROOT}/model-checkpoints"}
ROLLOUT_CHECKPOINT_DIR=${ROLLOUT_CHECKPOINT_DIR:-"${RUN_ROOT}/rollout-checkpoints"}
TRAIN_PATH="${DATA_DIR}/train.jsonl"
PERSIST_LOG="${RUN_ROOT}/persist.log"
CRASH_LOG="${RUN_ROOT}/crash.log"
RECOVERY_LOG="${RUN_ROOT}/recover.log"
PHASE1_CHECKSUMS="${RUN_ROOT}/phase1-rollout-checksums.txt"
PHASE1_READY="${RUN_ROOT}/phase1-ready"
MIN_CHECKPOINTS=${MIN_CHECKPOINTS:-1}
WAIT_TIMEOUT_S=${WAIT_TIMEOUT_S:-1800}

if [[ ! -f "${SAMPLE_SOURCE}" ]]; then
  echo "ERROR: NeMo-Gym sample data not found at ${SAMPLE_SOURCE}" >&2
  echo "Initialize the 3rdparty/Gym-workspace/Gym submodule first." >&2
  exit 1
fi

for required_command in jq setsid sha256sum uv; do
  if ! command -v "${required_command}" >/dev/null 2>&1; then
    echo "ERROR: required command is unavailable: ${required_command}" >&2
    exit 1
  fi
done

if [[ "${RESET:-0}" == "1" ]]; then
  case "${RUN_ROOT}" in
    /|""|"${PROJECT_ROOT}")
      echo "ERROR: refusing to reset unsafe RUN_ROOT=${RUN_ROOT}" >&2
      exit 2
      ;;
  esac
  rm -rf "${RUN_ROOT}"
fi

mkdir -p \
  "${DATA_DIR}" \
  "${LOG_DIR}" \
  "${MODEL_CHECKPOINT_DIR}" \
  "${ROLLOUT_CHECKPOINT_DIR}"

# Keep four checked-in examples and add the agent routing field expected by
# NeMo-Gym. Recreating this file is deterministic across recovery phases.
head -n 4 "${SAMPLE_SOURCE}" | jq -c \
  '. + {agent_ref: {
      type: "responses_api_agents",
      name: "math_with_judge_simple_agent"
  }}' \
  > "${TRAIN_PATH}"

export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"
cd "${PROJECT_ROOT}"

TRAIN_CMD=(
  uv run python -u
  examples/run_grpo_single_controller.py
  --config "${CONFIG_PATH}"
  "data.train.data_path=${TRAIN_PATH}"
  "logger.log_dir=${LOG_DIR}"
  "checkpointing.checkpoint_dir=${MODEL_CHECKPOINT_DIR}"
  "rollout_checkpointing.root_dir=${ROLLOUT_CHECKPOINT_DIR}"
  "$@"
)

TRAIN_PID=""

terminate_training() {
  if [[ -n "${TRAIN_PID}" ]] && kill -0 "${TRAIN_PID}" 2>/dev/null; then
    kill -KILL -- "-${TRAIN_PID}" 2>/dev/null || true
    wait "${TRAIN_PID}" 2>/dev/null || true
  fi
  TRAIN_PID=""
}

trap terminate_training EXIT INT TERM

count_rollout_checkpoints() {
  if [[ ! -d "${ROLLOUT_CHECKPOINT_DIR}/active" ]]; then
    echo 0
    return
  fi
  find "${ROLLOUT_CHECKPOINT_DIR}/active" -type f -name 'g*.pt' \
    | wc -l \
    | tr -d ' '
}

write_checkpoint_checksums() {
  local checkpoint_list="${RUN_ROOT}/phase1-rollout-files.txt"
  find "${ROLLOUT_CHECKPOINT_DIR}/active" -type f -name 'g*.pt' \
    | sort \
    > "${checkpoint_list}"

  : > "${PHASE1_CHECKSUMS}"
  while IFS= read -r checkpoint; do
    sha256sum "${checkpoint}" >> "${PHASE1_CHECKSUMS}"
  done < "${checkpoint_list}"
  test -s "${PHASE1_CHECKSUMS}"
}

require_fresh_rollout_root() {
  if [[ -e "${ROLLOUT_CHECKPOINT_DIR}/recovery_manifest.json" ]]; then
    echo "ERROR: rollout checkpoint directory already contains a recovery lineage." >&2
    echo "Use a new RUN_ROOT or rerun with RESET=1." >&2
    exit 2
  fi
}

run_foreground() {
  local run_log=$1
  set +e
  "${TRAIN_CMD[@]}" 2>&1 | tee "${run_log}"
  local training_status=${PIPESTATUS[0]}
  set -e
  return "${training_status}"
}

run_persistence_test() {
  require_fresh_rollout_root

  echo "Run directory: ${RUN_ROOT}"
  echo "Rollout checkpoints: ${ROLLOUT_CHECKPOINT_DIR}"
  run_foreground "${PERSIST_LOG}"

  local checkpoint_count
  checkpoint_count=$(count_rollout_checkpoints)
  if (( checkpoint_count == 0 )); then
    echo "ERROR: training finished without a completed-sibling checkpoint." >&2
    exit 1
  fi

  echo "Wrote ${checkpoint_count} completed-sibling checkpoint(s)."
  find "${ROLLOUT_CHECKPOINT_DIR}" -type f -name 'g*.pt' -print | sort
}

run_crash_phase() {
  require_fresh_rollout_root
  rm -f "${PHASE1_READY}" "${PHASE1_CHECKSUMS}"

  echo "Run directory: ${RUN_ROOT}"
  echo "Starting crash phase; training will be killed after ${MIN_CHECKPOINTS} durable sibling checkpoint(s)."
  echo "Crash log: ${CRASH_LOG}"

  setsid "${TRAIN_CMD[@]}" > "${CRASH_LOG}" 2>&1 &
  TRAIN_PID=$!
  local start_time=${SECONDS}

  while true; do
    local checkpoint_count
    checkpoint_count=$(count_rollout_checkpoints)
    if (( checkpoint_count >= MIN_CHECKPOINTS )); then
      write_checkpoint_checksums
      touch "${PHASE1_READY}"
      sync

      echo "Observed ${checkpoint_count} durable sibling checkpoint(s)."
      echo "Killing training process group ${TRAIN_PID} to simulate interruption."
      terminate_training

      if find "${MODEL_CHECKPOINT_DIR}" -maxdepth 1 -type d -name 'step_*' \
        | grep -q .; then
        echo "ERROR: a finalized training checkpoint was written before interruption." >&2
        echo "Increase grpo.num_generations_per_prompt or reduce MIN_CHECKPOINTS." >&2
        exit 1
      fi

      echo "Crash phase complete. Durable rollout files were left in place."
      return
    fi

    if ! kill -0 "${TRAIN_PID}" 2>/dev/null; then
      wait "${TRAIN_PID}" 2>/dev/null || true
      TRAIN_PID=""
      echo "ERROR: training exited before a sibling checkpoint was persisted." >&2
      tail -n 100 "${CRASH_LOG}" >&2
      exit 1
    fi

    if (( SECONDS - start_time >= WAIT_TIMEOUT_S )); then
      terminate_training
      echo "ERROR: timed out waiting for a completed-sibling checkpoint." >&2
      tail -n 100 "${CRASH_LOG}" >&2
      exit 1
    fi
    sleep 0.2
  done
}

run_recovery_phase() {
  if [[ ! -f "${PHASE1_READY}" || ! -s "${PHASE1_CHECKSUMS}" ]]; then
    echo "ERROR: crash phase marker or checksum list is missing." >&2
    exit 2
  fi
  if [[ ! -f "${ROLLOUT_CHECKPOINT_DIR}/recovery_manifest.json" ]]; then
    echo "ERROR: durable rollout recovery manifest is missing." >&2
    exit 1
  fi
  if find "${MODEL_CHECKPOINT_DIR}" -maxdepth 1 -type d -name 'step_*' \
    | grep -q .; then
    echo "ERROR: a finalized training checkpoint already exists." >&2
    echo "This would not exercise recovery before the first train step." >&2
    exit 1
  fi

  local before_count
  before_count=$(count_rollout_checkpoints)
  echo "Starting recovery with ${before_count} completed sibling checkpoint(s)."
  echo "Recovery log: ${RECOVERY_LOG}"
  run_foreground "${RECOVERY_LOG}"

  if ! grep -Eq \
    'ROLLOUT_CHECKPOINT_RECOVERY restored=[1-9][0-9]* ' \
    "${RECOVERY_LOG}"; then
    echo "ERROR: recovery completed without restoring a rollout sibling." >&2
    exit 1
  fi

  sha256sum --check "${PHASE1_CHECKSUMS}"

  if ! find "${MODEL_CHECKPOINT_DIR}" -maxdepth 1 -type d -name 'step_*' \
    | grep -q .; then
    echo "ERROR: recovery finished without a finalized training checkpoint." >&2
    exit 1
  fi

  local after_count
  after_count=$(count_rollout_checkpoints)
  echo "Recovery succeeded."
  echo "Completed sibling checkpoints: ${before_count} before, ${after_count} after."
  echo "Finalized model checkpoint root: ${MODEL_CHECKPOINT_DIR}"
}

case "${MODE}" in
  persist)
    run_persistence_test
    ;;
  crash)
    run_crash_phase
    ;;
  recover)
    run_recovery_phase
    ;;
  all)
    run_crash_phase
    run_recovery_phase
    ;;
esac
