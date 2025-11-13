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

# === Pipeline / model configuration =========================================
SERVER_INFO_FILE="${SERVER_INFO_FILE:-/lustre/${USER}/tmp/llada_server.env}"
SERVER_BATCH_SIZE="${SERVER_BATCH_SIZE:-1}"
SERVER_MODEL_PATH="${SERVER_MODEL_PATH:-}"
SERVER_BASE_MODEL="${SERVER_BASE_MODEL:-GSAI-ML/LLaDA-8B-Instruct}"
SERVER_DCP_PATH="${SERVER_DCP_PATH:-}"
SERVER_ENGINE="${SERVER_ENGINE:-}"
SERVER_EXTRA_ARGS="${SERVER_EXTRA_ARGS:-}"

SEQ_EVAL_BENCHMARK="${SEQ_EVAL_BENCHMARK:-}"
SEQ_EVAL_EXPNAME="${SEQ_EVAL_EXPNAME:-}"
SEQ_EVAL_GENERATION_ALGORITHM="${SEQ_EVAL_GENERATION_ALGORITHM:-}"
SEQ_EVAL_THRESHOLD="${SEQ_EVAL_THRESHOLD:-}"
SEQ_EVAL_TOKENS_TO_GENERATE="${SEQ_EVAL_TOKENS_TO_GENERATE:-}"
SEQ_EVAL_STEPS="${SEQ_EVAL_STEPS:-}"
SEQ_EVAL_BLOCK_LENGTH="${SEQ_EVAL_BLOCK_LENGTH:-}"
SEQ_EVAL_EXTRA_ARGS="${SEQ_EVAL_EXTRA_ARGS:-}"

# Additional flags that should be applied to every evaluation launch
GLOBAL_EVAL_FLAGS="${GLOBAL_EVAL_FLAGS:-}"

# Parallel evaluation jobs (newline-separated list; edit or override as needed)
read -r -d '' DEFAULT_PARALLEL_JOBS <<'EOF' || true
--job-name llada-eval-par-1 --cpus 64 --time 03:00:00 --server-info-file ${SERVER_INFO_FILE} -- --benchmark gsm8k:1 --generation-algorithm nemotron --model nemotron-4b --threshold 0.9 --tokens-to-generate 512 --steps 512 --block-length 32 --expname llada-gsm8k-par-1
--job-name llada-eval-par-2 --cpus 64 --time 03:00:00 --server-info-file ${SERVER_INFO_FILE} -- --benchmark gsm8k:1 --generation-algorithm nemotron --model nemotron-4b --threshold 0.9 --tokens-to-generate 512 --steps 512 --block-length 32 --expname llada-gsm8k-par-2
EOF
PARALLEL_EVAL_JOBS="${PARALLEL_EVAL_JOBS_OVERRIDE:-$DEFAULT_PARALLEL_JOBS}"
# ===========================================================================

export ACCOUNT

# Ensure the server-info directory exists so the server launcher can write to it
mkdir -p "$(dirname "$SERVER_INFO_FILE")"

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
  "--server-info-file" "$SERVER_INFO_FILE"
  "--"
)
if [[ -n "$SEQ_EVAL_BENCHMARK" ]]; then
  SEQ_EVAL_ARGS+=("--benchmark" "$SEQ_EVAL_BENCHMARK")
fi
if [[ -n "$SEQ_EVAL_EXPNAME" ]]; then
  SEQ_EVAL_ARGS+=("--expname" "$SEQ_EVAL_EXPNAME")
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
for attempt in {1..180}; do
  if [[ -f "$SERVER_INFO_FILE" ]]; then
    # shellcheck disable=SC1090
    source "$SERVER_INFO_FILE"
    if [[ "${SERVER_STATUS:-}" == "running" && "${SERVER_ADDRESS:-}" != "" && "${SERVER_ADDRESS}" != *"0.0.0.0"* ]]; then
      echo "[pipeline] server ready at $SERVER_ADDRESS"
      break
    fi
  fi
  sleep 5
  if [[ $attempt -eq 180 ]]; then
    echo "[pipeline] ERROR: server never reached running state" >&2
    exit 1
  fi
done

echo "[pipeline] launching sequential eval job..."
"$EVAL_LAUNCHER" "${SEQ_EVAL_ARGS[@]}"

echo "[pipeline] launching parallel eval jobs..."
PARALLEL_PIDS=()
for eval_cmd in "${PAR_EVAL_ARGS_LIST[@]}"; do
  eval "$EVAL_LAUNCHER $eval_cmd &"
  PARALLEL_PIDS+=($!)
done

echo "[pipeline] waiting for parallel eval jobs to finish..."
for pid in "${PARALLEL_PIDS[@]}"; do
  wait "$pid"
done

echo "[pipeline] all evaluation jobs completed. Use Ctrl+C or scancel to stop the server when finished."

