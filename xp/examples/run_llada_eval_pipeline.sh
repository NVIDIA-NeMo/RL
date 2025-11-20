#!/bin/bash
#
# LLaDA server + evaluation pipeline launcher.
# - Launches the batch inference server on SLURM.
# - Waits for the server-info file to report a usable address.
# - Runs one sequential evaluation job.
# - Fires off additional evaluation jobs (optionally in parallel).
#
# Customize the variables in the "Default configuration" section below or
# override them with environment variables before invoking the script.
#
# Usage (from repo root):
#   bash xp/examples/run_llada_eval_pipeline.sh
# ---------------------------------------------------------------------------

set -euo pipefail

# === SLURM configuration ====================================================
ACCOUNT="${ACCOUNT:-your_slurm_account}"

SERVER_JOB_NAME="${SERVER_JOB_NAME:-llada-batch-canonical}"
SERVER_PARTITION="${SERVER_PARTITION:-interactive}"
SERVER_TIME="${SERVER_TIME:-04:00:00}"
SERVER_GPUS="${SERVER_GPUS:-8}"

SEQ_EVAL_JOB_NAME="${SEQ_EVAL_JOB_NAME:-llada-eval-sequential}"
SEQ_EVAL_CPUS="${SEQ_EVAL_CPUS:-48}"
SEQ_EVAL_TIME="${SEQ_EVAL_TIME:-02:00:00}"
SEQ_EVAL_PARTITION="${SEQ_EVAL_PARTITION:-cpu}"
SEQ_EVAL_NO_WAIT_SERVER="${SEQ_EVAL_NO_WAIT_SERVER:-false}"
SEQ_EVAL_USE_SAME_NODE="${SEQ_EVAL_USE_SAME_NODE:-false}"

# === Pipeline / model configuration =========================================
SERVER_INFO_FILE="${SERVER_INFO_FILE:-}"
SERVER_BATCH_SIZE="${SERVER_BATCH_SIZE:-1}"
SERVER_MODEL_PATH="${SERVER_MODEL_PATH:-}"
SERVER_BASE_MODEL="${SERVER_BASE_MODEL:-GSAI-ML/LLaDA-8B-Instruct}"
SERVER_DCP_PATH="${SERVER_DCP_PATH:-}"
SERVER_ENGINE="${SERVER_ENGINE:-}"
SERVER_EXTRA_ARGS="${SERVER_EXTRA_ARGS:-}"

SEQ_EVAL_BENCHMARK="${SEQ_EVAL_BENCHMARK:-}"
SEQ_EVAL_EXPNAME="${SEQ_EVAL_EXPNAME:-}"
SEQ_EVAL_OUTPUT_DIR="${SEQ_EVAL_OUTPUT_DIR:-}"
SEQ_EVAL_GENERATION_ALGORITHM="${SEQ_EVAL_GENERATION_ALGORITHM:-}"
SEQ_EVAL_THRESHOLD="${SEQ_EVAL_THRESHOLD:-}"
SEQ_EVAL_TOKENS_TO_GENERATE="${SEQ_EVAL_TOKENS_TO_GENERATE:-}"
SEQ_EVAL_STEPS="${SEQ_EVAL_STEPS:-}"
SEQ_EVAL_BLOCK_LENGTH="${SEQ_EVAL_BLOCK_LENGTH:-}"
SEQ_EVAL_TEMPERATURE="${SEQ_EVAL_TEMPERATURE:-}"
SEQ_EVAL_EXTRA_ARGS="${SEQ_EVAL_EXTRA_ARGS:-}"

# Additional flags that should be applied to every evaluation launch
GLOBAL_EVAL_FLAGS="${GLOBAL_EVAL_FLAGS:-}"

# Control whether to wait for sequential eval to complete before launching parallel evals
# Set to "false" to launch sequential eval in background and immediately proceed to parallel evals
WAIT_FOR_SEQUENTIAL="${WAIT_FOR_SEQUENTIAL:-true}"

# Parallel evaluation jobs (newline-separated list; edit or override as needed)
read -r -d '' DEFAULT_PARALLEL_JOBS <<'EOF' || true
EOF
PARALLEL_EVAL_JOBS="${PARALLEL_EVAL_JOBS_OVERRIDE:-$DEFAULT_PARALLEL_JOBS}"
# ===========================================================================

export ACCOUNT

# Ensure the server-info directory exists so the server launcher can write to it
mkdir -p "$(dirname "$SERVER_INFO_FILE")"

# Clear any stale server info file from previous runs to prevent race conditions
if [[ -f "$SERVER_INFO_FILE" ]]; then
  echo "[pipeline] removing stale server info file: $SERVER_INFO_FILE"
  rm -f "$SERVER_INFO_FILE"
fi

# Write a placeholder to ensure we don't read stale data
cat > "$SERVER_INFO_FILE" <<EOF
# Placeholder - waiting for server launcher to start
SERVER_STATUS="initializing"
SERVER_INFO_GENERATED_AT="$(date -Iseconds)"
EOF
echo "[pipeline] initialized server info file: $SERVER_INFO_FILE"

# Build server arguments
SERVER_ARGS=(
  "--server-info-file" "$SERVER_INFO_FILE"
  "--job-name" "$SERVER_JOB_NAME"
  "--partition" "$SERVER_PARTITION"
  "--time" "$SERVER_TIME"
  "--gpus" "$SERVER_GPUS"
  "--batch-size" "$SERVER_BATCH_SIZE"
)
if [[ -n "$SERVER_MODEL_PATH" ]]; then
  SERVER_ARGS+=("--model-path" "$SERVER_MODEL_PATH")
fi
if [[ -n "$SERVER_DCP_PATH" ]]; then
  SERVER_ARGS+=("--dcp-path" "$SERVER_DCP_PATH")
fi
if [[ -n "$SERVER_BASE_MODEL" ]]; then
  SERVER_ARGS+=("--base-model" "$SERVER_BASE_MODEL")
fi
if [[ -n "$SERVER_ENGINE" ]]; then
  SERVER_ARGS+=("--engine" "$SERVER_ENGINE")
fi
if [[ -n "$SERVER_EXTRA_ARGS" ]]; then
  read -r -a SERVER_EXTRA_ARRAY <<<"$SERVER_EXTRA_ARGS"
  SERVER_ARGS+=("${SERVER_EXTRA_ARRAY[@]}")
fi

# Build sequential evaluation arguments
SEQ_EVAL_ARGS=(
  "--job-name" "$SEQ_EVAL_JOB_NAME"
  "--cpus" "$SEQ_EVAL_CPUS"
  "--time" "$SEQ_EVAL_TIME"
  "--partition" "$SEQ_EVAL_PARTITION"
  "--server-info-file" "$SERVER_INFO_FILE"
)
if [[ "$SEQ_EVAL_NO_WAIT_SERVER" == "true" ]]; then
  SEQ_EVAL_ARGS+=("--no-wait-for-server")
fi
if [[ "$SEQ_EVAL_USE_SAME_NODE" == "true" ]]; then
  SEQ_EVAL_ARGS+=("--use-same-node")
fi
SEQ_EVAL_ARGS+=("--")
if [[ -n "$SEQ_EVAL_BENCHMARK" ]]; then
  SEQ_EVAL_ARGS+=("--benchmark" "$SEQ_EVAL_BENCHMARK")
fi
if [[ -n "$SEQ_EVAL_EXPNAME" ]]; then
  SEQ_EVAL_ARGS+=("--expname" "$SEQ_EVAL_EXPNAME")
fi
if [[ -n "$SEQ_EVAL_OUTPUT_DIR" ]]; then
  SEQ_EVAL_ARGS+=("--output-dir" "$SEQ_EVAL_OUTPUT_DIR")
fi
if [[ -n "$SEQ_EVAL_GENERATION_ALGORITHM" ]]; then
  SEQ_EVAL_ARGS+=("--generation-algorithm" "$SEQ_EVAL_GENERATION_ALGORITHM")
fi
if [[ -n "$SEQ_EVAL_THRESHOLD" ]]; then
  SEQ_EVAL_ARGS+=("--threshold" "$SEQ_EVAL_THRESHOLD")
fi
if [[ -n "$SEQ_EVAL_TOKENS_TO_GENERATE" ]]; then
  SEQ_EVAL_ARGS+=("--tokens-to-generate" "$SEQ_EVAL_TOKENS_TO_GENERATE")
fi
if [[ -n "$SEQ_EVAL_STEPS" ]]; then
  SEQ_EVAL_ARGS+=("--steps" "$SEQ_EVAL_STEPS")
fi
if [[ -n "$SEQ_EVAL_BLOCK_LENGTH" ]]; then
  SEQ_EVAL_ARGS+=("--block-length" "$SEQ_EVAL_BLOCK_LENGTH")
fi
if [[ -n "$SEQ_EVAL_TEMPERATURE" ]]; then
  SEQ_EVAL_ARGS+=("--temperature" "$SEQ_EVAL_TEMPERATURE")
fi
if [[ -n "$GLOBAL_EVAL_FLAGS" ]]; then
  read -r -a GLOBAL_FLAGS_ARRAY <<<"$GLOBAL_EVAL_FLAGS"
  SEQ_EVAL_ARGS+=("${GLOBAL_FLAGS_ARRAY[@]}")
fi
if [[ -n "$SEQ_EVAL_EXTRA_ARGS" ]]; then
  read -r -a EXTRA_SEQ_ARRAY <<<"$SEQ_EVAL_EXTRA_ARGS"
  SEQ_EVAL_ARGS+=("${EXTRA_SEQ_ARRAY[@]}")
fi

# Build parallel evaluation argument strings
PARALLEL_EVAL_JOBS="${PARALLEL_EVAL_JOBS//\$\{SERVER_INFO_FILE\}/$SERVER_INFO_FILE}"
PAR_EVAL_ARGS_LIST=()
while IFS= read -r line; do
  [[ -z "$line" ]] && continue
  PAR_EVAL_ARGS_LIST+=("$line")
done <<<"$PARALLEL_EVAL_JOBS"
if [[ -n "$GLOBAL_EVAL_FLAGS" ]]; then
  for i in "${!PAR_EVAL_ARGS_LIST[@]}"; do
    PAR_EVAL_ARGS_LIST[$i]+=" $GLOBAL_EVAL_FLAGS"
  done
fi

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SERVER_LAUNCHER="$ROOT_DIR/llada_api/scripts/start_llada_batch_server.sh"
EVAL_LAUNCHER="$ROOT_DIR/llada_api/scripts/start_llada_eval.sh"

if [[ ! -x "$SERVER_LAUNCHER" ]]; then
  echo "[pipeline] ERROR: missing or non-executable server launcher at $SERVER_LAUNCHER" >&2
  exit 1
fi
if [[ ! -x "$EVAL_LAUNCHER" ]]; then
  echo "[pipeline] ERROR: missing or non-executable eval launcher at $EVAL_LAUNCHER" >&2
  exit 1
fi

trap 'echo "[pipeline] terminating..."; [[ -n "${SERVER_PID:-}" ]] && kill "$SERVER_PID" 2>/dev/null || true' EXIT

echo "[pipeline] server launcher: $SERVER_LAUNCHER"
echo "[pipeline] server args: ${SERVER_ARGS[*]}"

"$SERVER_LAUNCHER" "${SERVER_ARGS[@]}" &
SERVER_PID=$!
sleep 2
if ! kill -0 "$SERVER_PID" 2>/dev/null; then
  if wait "$SERVER_PID"; then
    STATUS=0
  else
    STATUS=$?
  fi
  echo "[pipeline] server launcher exited immediately (status $STATUS)." >&2
  exit "$STATUS"
fi


echo "[pipeline] waiting for server info at $SERVER_INFO_FILE ..."
SERVER_SLURM_JOB_ID=""
for attempt in {1..180}; do
  if [[ -f "$SERVER_INFO_FILE" ]]; then
    # shellcheck disable=SC1090
    source "$SERVER_INFO_FILE"
    # Check all fields required by the eval launcher
    if [[ "${SERVER_STATUS:-}" == "running" && -n "${SERVER_ADDRESS:-}" && "${SERVER_ADDRESS:-}" != *"0.0.0.0"* ]]; then
      # For SLURM servers, also verify SERVER_CLIENT_HOST is set
      if [[ "${SERVER_INFO_SOURCE:-}" == "slurm" ]]; then
        if [[ -n "${SERVER_CLIENT_HOST:-}" && "${SERVER_CLIENT_HOST:-}" != "0.0.0.0" ]]; then
          echo "[pipeline] server ready at $SERVER_ADDRESS"
          echo "[pipeline] server info: STATUS=$SERVER_STATUS, ADDRESS=$SERVER_ADDRESS, CLIENT_HOST=$SERVER_CLIENT_HOST, PORT=${SERVER_PORT:-unset}"
          # Capture SLURM job ID for cleanup
          SERVER_SLURM_JOB_ID="${SLURM_JOB_ID:-}"
          if [[ -n "$SERVER_SLURM_JOB_ID" ]]; then
            echo "[pipeline] server SLURM job ID: $SERVER_SLURM_JOB_ID (will be cancelled at end)"
          fi
          break
        fi
      else
        # For local servers, CLIENT_HOST check not required
        echo "[pipeline] server ready at $SERVER_ADDRESS"
        echo "[pipeline] server info: STATUS=$SERVER_STATUS, ADDRESS=$SERVER_ADDRESS, PORT=${SERVER_PORT:-unset}"
        break
      fi
    fi
    if [[ $attempt -eq 1 || $((attempt % 12)) -eq 0 ]]; then
      echo "[pipeline] waiting... (attempt $attempt/180, status=${SERVER_STATUS:-unset}, source=${SERVER_INFO_SOURCE:-unset}, client_host=${SERVER_CLIENT_HOST:-unset})"
    fi
  fi
  sleep 5
  if [[ $attempt -eq 180 ]]; then
    echo "[pipeline] ERROR: server never reached running state" >&2
    echo "[pipeline] Last known status: SERVER_STATUS=${SERVER_STATUS:-unset}, SERVER_ADDRESS=${SERVER_ADDRESS:-unset}, SERVER_CLIENT_HOST=${SERVER_CLIENT_HOST:-unset}" >&2
    exit 1
  fi
done

echo "[pipeline] launching sequential eval job..."
echo "[pipeline] eval launcher: $EVAL_LAUNCHER"
echo "[pipeline] eval args: ${SEQ_EVAL_ARGS[*]}"
echo ""

# Launch the eval job in background to capture PID (output will be visible)
"$EVAL_LAUNCHER" "${SEQ_EVAL_ARGS[@]}" &
SEQ_EVAL_PID=$!
echo ""
echo "[pipeline] sequential eval job launched with PID $SEQ_EVAL_PID"
echo "[pipeline] eval job is now running via SLURM - output should appear above"

if [[ "$WAIT_FOR_SEQUENTIAL" == "true" ]]; then
  echo "[pipeline] waiting for sequential eval job to complete (WAIT_FOR_SEQUENTIAL=true)..."
  echo "[pipeline] TIP: If no output appears, check 'squeue -u $USER' to see job status"
  echo ""
  
  # Show progress while waiting
  WAIT_COUNT=0
  while kill -0 "$SEQ_EVAL_PID" 2>/dev/null; do
    sleep 10
    WAIT_COUNT=$((WAIT_COUNT + 1))
    if [[ $((WAIT_COUNT % 6)) -eq 0 ]]; then
      WAIT_TIME=$((WAIT_COUNT * 10))
      echo "[pipeline] still waiting for eval job... (${WAIT_TIME}s elapsed)"
    fi
  done
  
  if wait "$SEQ_EVAL_PID"; then
    echo "[pipeline] sequential eval job completed successfully"
  else
    SEQ_STATUS=$?
    echo "[pipeline] WARNING: sequential eval job exited with status $SEQ_STATUS" >&2
  fi
else
  echo "[pipeline] not waiting for sequential eval (WAIT_FOR_SEQUENTIAL=false)"
  echo "[pipeline] sequential eval job running in background with PID $SEQ_EVAL_PID"
fi

echo "[pipeline] launching parallel eval jobs..."
PARALLEL_PIDS=()
if [[ ${#PAR_EVAL_ARGS_LIST[@]} -eq 0 ]]; then
  echo "[pipeline] no parallel eval jobs configured (PARALLEL_EVAL_JOBS_OVERRIDE is empty)"
else
  echo "[pipeline] launching ${#PAR_EVAL_ARGS_LIST[@]} parallel eval job(s)..."
  for eval_cmd in "${PAR_EVAL_ARGS_LIST[@]}"; do
    eval "$EVAL_LAUNCHER $eval_cmd &"
    PARALLEL_PIDS+=($!)
  done
fi

# Collect all eval PIDs to wait for
ALL_EVAL_PIDS=()
if [[ "$WAIT_FOR_SEQUENTIAL" != "true" ]] && kill -0 "$SEQ_EVAL_PID" 2>/dev/null; then
  echo "[pipeline] will wait for sequential eval (PID $SEQ_EVAL_PID) to complete"
  ALL_EVAL_PIDS+=("$SEQ_EVAL_PID")
fi
ALL_EVAL_PIDS+=("${PARALLEL_PIDS[@]}")

if [[ ${#ALL_EVAL_PIDS[@]} -gt 0 ]]; then
  echo "[pipeline] waiting for ${#ALL_EVAL_PIDS[@]} eval job(s) to finish..."
  for pid in "${ALL_EVAL_PIDS[@]}"; do
    if wait "$pid"; then
      echo "[pipeline] eval job with PID $pid completed successfully"
    else
      echo "[pipeline] WARNING: eval job with PID $pid failed" >&2
    fi
  done
fi

echo "[pipeline] all evaluation jobs completed."

# Shut down the server SLURM job
if [[ -n "$SERVER_SLURM_JOB_ID" ]]; then
  echo "[pipeline] shutting down server SLURM job $SERVER_SLURM_JOB_ID..."
  if scancel "$SERVER_SLURM_JOB_ID" 2>/dev/null; then
    echo "[pipeline] server job cancelled successfully"
  else
    echo "[pipeline] WARNING: failed to cancel server job $SERVER_SLURM_JOB_ID (may have already terminated)" >&2
  fi
else
  echo "[pipeline] no SLURM job ID found for server (may be running locally)"
  echo "[pipeline] if server is still running, use Ctrl+C or manual scancel"
fi

echo "[pipeline] pipeline completed."

