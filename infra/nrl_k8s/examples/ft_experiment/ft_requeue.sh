#!/usr/bin/env bash
# FT Experiment — requeue-on-failure driver for the CONTROL setups (#2, #4).
#
# These runs have NO fault recovery. A simulated infrastructure failure kills
# the training driver mid-run; the only way forward is to restart and let
# run_grpo.py auto-resume from the last checkpoint (CheckpointManager.
# get_latest_checkpoint_path on startup — see nemo_rl/algorithms/grpo.py).
# We loop until ${FT_TARGET_STEP} is reached.
#
# Fault timing (the spec: 2-3 faults over 20 steps, never during startup):
# each attempt arms a watcher that greps the attempt's fresh driver log for
# "[STEP-START] step=N". Once N reaches START_STEP + FT_STEPS_PER_ATTEMPT
# (a step strictly past where this attempt resumed), it kills the driver —
# UNLESS that kill step is >= FT_TARGET_STEP, in which case the fault is
# disarmed so the final leg can actually finish. Because the watcher keys off
# real steps (not wall-clock), the fault never fires during the multi-minute
# cold start, and the cadence is deterministic regardless of step time:
#   save_period=1 (#2),  FT_STEPS_PER_ATTEMPT=6 -> faults at steps 6,11,16
#                        (lose ~1 step each; resume from step_5/10/15)
#   save_period=5 (#4),  FT_STEPS_PER_ATTEMPT=8 -> faults at steps 8,13,18
#                        (lose ~3 steps each; resume from step_5/10/15)
#
# Required env (exported by the infra entrypoint):
#   FT_CKPT_DIR FT_CONFIG FT_LOG_DIR FT_TARGET_STEP FT_STEPS_PER_ATTEMPT FT_RUN_TAG
set -u

CKPT_DIR="${FT_CKPT_DIR:?FT_CKPT_DIR required}"
CONFIG="${FT_CONFIG:?FT_CONFIG required}"
LOG_DIR="${FT_LOG_DIR:?FT_LOG_DIR required}"
TARGET_STEP="${FT_TARGET_STEP:-20}"
STEPS_PER_ATTEMPT="${FT_STEPS_PER_ATTEMPT:-6}"
RUN_TAG="${FT_RUN_TAG:-control}"
SETTLE_S="${FT_SETTLE_S:-45}"

RECIPE_BASE="$(basename "${CONFIG}" .yaml)"
mkdir -p "${LOG_DIR}"

# Latest FINALIZED checkpoint step (keep_top_k=1 leaves only one step_N dir;
# tmp_step_N in-progress dirs are excluded by the step_* glob).
latest_step() {
  ls -d "${CKPT_DIR}"/step_* 2>/dev/null | sed 's#.*/step_##' | sort -n | tail -1
}

attempt=0
while true; do
  attempt=$((attempt + 1))
  START_STEP="$(latest_step)"; START_STEP="${START_STEP:-0}"
  if [ "${START_STEP}" -ge "${TARGET_STEP}" ]; then
    echo "[REQUEUE] reached step ${START_STEP} >= ${TARGET_STEP}; training COMPLETE after $((attempt - 1)) attempt(s) unix_ts=$(date -u +%s.%N)"
    break
  fi
  KILL_AT=$((START_STEP + STEPS_PER_ATTEMPT))
  TS="$(date -u +%Y%m%d-%H%M%S-%N)"
  LOG="${LOG_DIR}/ft-experiment-${RUN_TAG}-attempt${attempt}-${TS}.log"
  echo "[REQUEUE] attempt=${attempt} resume_from_step=${START_STEP} kill_at_step=${KILL_AT} target=${TARGET_STEP} log=${LOG} unix_ts=$(date -u +%s.%N)"

  # Arm the fault watcher (skipped on the final leg, KILL_AT >= TARGET).
  FAULT_PID=""
  if [ "${KILL_AT}" -lt "${TARGET_STEP}" ]; then
    (
      # Wait for the fresh log to appear, then for a step >= KILL_AT.
      while [ ! -f "${LOG}" ]; do sleep 2; done
      while true; do
        CUR="$(grep -oE '\[STEP-START\] step=[0-9]+' "${LOG}" 2>/dev/null | grep -oE '[0-9]+$' | sort -n | tail -1)"
        CUR="${CUR:-0}"
        if [ "${CUR}" -ge "${KILL_AT}" ]; then
          echo "[FAULT-SIM] $(date -u +%Y-%m-%dT%H:%M:%S.%NZ) attempt=${attempt} log_step=${CUR} >= kill_at=${KILL_AT}; killing training driver (simulated infra failure) unix_ts=$(date -u +%s.%N)"
          pkill -TERM -f "run_grpo.py.*${RECIPE_BASE}" || true
          sleep 5
          pkill -KILL -f "run_grpo.py.*${RECIPE_BASE}" || true
          break
        fi
        sleep 5
      done
    ) &
    FAULT_PID=$!
  fi

  python -u examples/run_grpo.py --config "${CONFIG}" 2>&1 | tee "${LOG}"
  EXIT_CODE="${PIPESTATUS[0]}"

  # Tear down the watcher if it's still waiting (training finished first).
  if [ -n "${FAULT_PID}" ]; then
    kill "${FAULT_PID}" 2>/dev/null || true
    wait "${FAULT_PID}" 2>/dev/null || true
  fi
  echo "[REQUEUE] attempt=${attempt} run_grpo exited code=${EXIT_CODE} unix_ts=$(date -u +%s.%N)"

  # If the driver was faulted (non-zero exit), let Ray reclaim the dead
  # actors' GPUs / placement groups before the next attempt re-allocates.
  if [ "${EXIT_CODE}" -ne 0 ]; then
    echo "[REQUEUE] settling ${SETTLE_S}s for GPU/PG cleanup before restart"
    sleep "${SETTLE_S}"
  fi
done
