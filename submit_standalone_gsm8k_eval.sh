#!/usr/bin/env bash
set -euo pipefail

# Submit the standalone No-Ray math evaluator that reproduces the NeMoRL
# validation prompt/generation/grading path without Ray.
#
# Common usage:
#   ALG=FastDiffuser BS=16 TEMP=0 TAG=my_eval ./submit_standalone_gsm8k_eval.sh
#   ALG=LinearSpec   BS=16 TEMP=0.0 TAG=my_greedy_ls_b16 ./submit_standalone_gsm8k_eval.sh
#   BENCHMARK=aime2024 ALG=FastDiffuser BS=16 TEMP=0 TAG=my_aime24 ./submit_standalone_gsm8k_eval.sh
#   BENCHMARK=aime2025 ALG=FastDiffuser BS=16 TEMP=0 TAG=my_aime25 ./submit_standalone_gsm8k_eval.sh

ACCOUNT=${ACCOUNT:-coreai_dlalgo_llm}
PARTITION=${PARTITION:-batch_short}
TIME_LIMIT=${TIME_LIMIT:-02:00:00}
GPUS_PER_NODE=${GPUS_PER_NODE:-1}
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)

BENCHMARK=${BENCHMARK:-gsm8k}        # gsm8k, aime2024/aime24, or aime2025/aime25
ALG=${ALG:-FastDiffuser}        # FastDiffuser or LinearSpec
BS=${BS:-16}                    # diffusion block size
TEMP=${TEMP:-0}                 # paper-style AIME greedy decoding uses temperature 0
MAX_STEPS=${MAX_STEPS:-8192}
THRESHOLD=${THRESHOLD:-0.9}
SELECTION_POLICY=${SELECTION_POLICY:-confidence}
MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-8192}
CONTEXT_LENGTH=${CONTEXT_LENGTH:-20480}
TOP_P=${TOP_P:-1.0}
TOP_K=${TOP_K:--1}
CONCURRENT=${CONCURRENT:-1}
GENERATION_API=${GENERATION_API:-generate}
MAX_RUNNING_REQUESTS=${MAX_RUNNING_REQUESTS:-}
MAX_TOTAL_TOKENS=${MAX_TOTAL_TOKENS:-20000}
MEM_FRACTION_STATIC=${MEM_FRACTION_STATIC:-0.55}
ATTENTION_BACKEND=${ATTENTION_BACKEND:-flashinfer}
SERVER_RANDOM_SEED=${SERVER_RANDOM_SEED:-0}
NUM_SAMPLES=${NUM_SAMPLES:--1}
SHARD_DP_SIZE=${SHARD_DP_SIZE:-1}
SHARD_RANK=${SHARD_RANK:-0}
VAL_BATCH_SIZE=${VAL_BATCH_SIZE:-128}
PORT=${PORT:-32617}

CKPT=${CKPT:-/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_genai/users/snorouzi/checkpoints/gsm8k_nd3b_run1_step200_dlm_hf_nemoskills_ropefix_materialized_20260605}
TOKENIZER=${TOKENIZER:-/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_genai/users/snorouzi/checkpoints/Nemotron-Labs-Diffusion-3B}
CONTAINER=${CONTAINER:-/lustre/fsw/portfolios/coreai/projects/coreai_dlalgo_llm/users/sfawzy/nemo-rl-nightly.sqsh}
WORKDIR=${WORKDIR:-$SCRIPT_DIR}
ROOT=${ROOT:-/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_genai/users/snorouzi/eval_results}
SGLANG_REPO=${SGLANG_REPO:-/home/snorouzi/code/sglang-nemotron-dllm-a652eb48}
SGLANG_COMMIT=${SGLANG_COMMIT:-$(git -C "$SGLANG_REPO" rev-parse HEAD)}
PROMPT_FILE=${PROMPT_FILE:-$WORKDIR/examples/prompts/cot.txt}

usage() {
  cat <<EOF
Usage: $0 [--concurrency N] [--max-running-requests N]

Options:
  --concurrency, --concurrent N      Number of parallel client generation requests.
  --max-running-requests N          SGLang server max running requests. Defaults to concurrency.
  -h, --help                        Show this help.

All existing environment-variable overrides are still supported.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --concurrency|--concurrent)
      [[ $# -ge 2 ]] || { echo "$1 requires a value" >&2; exit 2; }
      CONCURRENT="$2"
      shift 2
      ;;
    --max-running-requests)
      [[ $# -ge 2 ]] || { echo "$1 requires a value" >&2; exit 2; }
      MAX_RUNNING_REQUESTS="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

MAX_RUNNING_REQUESTS=${MAX_RUNNING_REQUESTS:-$CONCURRENT}

if [[ "$ALG" != "FastDiffuser" && "$ALG" != "LinearSpec" ]]; then
  echo "ALG must be FastDiffuser or LinearSpec, got: $ALG" >&2
  exit 2
fi
if [[ "$GENERATION_API" != "generate" && "$GENERATION_API" != "chat_completions" ]]; then
  echo "GENERATION_API must be generate or chat_completions, got: $GENERATION_API" >&2
  exit 2
fi
case "$BENCHMARK" in
  gsm8k|aime24|aime2024|aime25|aime2025) ;;
  *)
    echo "BENCHMARK must be gsm8k, aime2024/aime24, or aime2025/aime25, got: $BENCHMARK" >&2
    exit 2
    ;;
esac

if [[ -z "${TAG:-}" ]]; then
  alg_tag=$(echo "$ALG" | tr '[:upper:]' '[:lower:]')
  temp_tag=${TEMP//./p}
  bench_tag=$(echo "$BENCHMARK" | tr '[:upper:]' '[:lower:]')
  TAG="standalone_${bench_tag}_${alg_tag}_b${BS}_t${temp_tag}_$(date +%Y%m%d_%H%M%S)"
fi

OUTDIR=${OUTDIR:-$ROOT/$TAG}
mkdir -p "$OUTDIR"

JOB_NAME=${JOB_NAME:-${BENCHMARK}_${ALG}_b${BS}}
JOB_NAME=${JOB_NAME:0:24}

JOBID=$(sbatch \
  -J "$JOB_NAME" \
  -A "$ACCOUNT" \
  -p "$PARTITION" \
  -N 1 \
  --gpus-per-node="$GPUS_PER_NODE" \
  -t "$TIME_LIMIT" \
  -o "$OUTDIR/slurm-%j.out" \
  --export=ALL,OUTDIR="$OUTDIR",CKPT="$CKPT",TOKENIZER="$TOKENIZER",CONTAINER="$CONTAINER",WORKDIR="$WORKDIR",SGLANG_REPO="$SGLANG_REPO",SGLANG_COMMIT="$SGLANG_COMMIT",PROMPT_FILE="$PROMPT_FILE",PORT="$PORT",BENCHMARK="$BENCHMARK",ALG="$ALG",BS="$BS",TEMP="$TEMP",MAX_STEPS="$MAX_STEPS",THRESHOLD="$THRESHOLD",SELECTION_POLICY="$SELECTION_POLICY",MAX_NEW_TOKENS="$MAX_NEW_TOKENS",CONTEXT_LENGTH="$CONTEXT_LENGTH",TOP_P="$TOP_P",TOP_K="$TOP_K",CONCURRENT="$CONCURRENT",GENERATION_API="$GENERATION_API",MAX_RUNNING_REQUESTS="$MAX_RUNNING_REQUESTS",MAX_TOTAL_TOKENS="$MAX_TOTAL_TOKENS",MEM_FRACTION_STATIC="$MEM_FRACTION_STATIC",ATTENTION_BACKEND="$ATTENTION_BACKEND",SERVER_RANDOM_SEED="$SERVER_RANDOM_SEED",NUM_SAMPLES="$NUM_SAMPLES",SHARD_DP_SIZE="$SHARD_DP_SIZE",SHARD_RANK="$SHARD_RANK",VAL_BATCH_SIZE="$VAL_BATCH_SIZE" \
  <<'SBATCH'
#!/bin/bash
set -euo pipefail

echo "GpuFreq=control_disabled"
echo "OUTDIR=$OUTDIR"
echo "CKPT=$CKPT"
echo "TOKENIZER=$TOKENIZER"
echo "PORT=$PORT"
echo "CONTAINER=$CONTAINER"
echo "WORKDIR=$WORKDIR"
echo "SGLANG_REPO=$SGLANG_REPO"
echo "SGLANG_COMMIT=$SGLANG_COMMIT"
echo "PROMPT_FILE=$PROMPT_FILE"
echo "BENCHMARK=$BENCHMARK"
echo "ALG=$ALG"
echo "BS=$BS"
echo "TEMP=$TEMP"
echo "MAX_STEPS=$MAX_STEPS"
echo "MAX_NEW_TOKENS=$MAX_NEW_TOKENS"
echo "GENERATION_API=$GENERATION_API"
echo "CONCURRENT=$CONCURRENT"
echo "MAX_RUNNING_REQUESTS=$MAX_RUNNING_REQUESTS"
echo "NUM_SAMPLES=$NUM_SAMPLES"
echo "SHARD_DP_SIZE=$SHARD_DP_SIZE"
echo "SHARD_RANK=$SHARD_RANK"

srun --container-image="$CONTAINER" \
  --container-mounts=/home/snorouzi:/home/snorouzi,/lustre:/lustre \
  --container-workdir="$WORKDIR" \
  bash -lc '
    set -euo pipefail
    export PYTHONPATH=/opt/nemo_rl_venv/lib/python3.12/site-packages${PYTHONPATH:+:$PYTHONPATH}
    export HF_HOME=/lustre/fsw/portfolios/coreai/users/snorouzi/hf_home
    export HF_HUB_CACHE=/lustre/fsw/portfolios/coreai/users/snorouzi/hf_home/hub
    export TRANSFORMERS_CACHE=/lustre/fsw/portfolios/coreai/users/snorouzi/hf_home/hub
    export HF_DATASETS_CACHE=/lustre/fsw/portfolios/coreai/users/snorouzi/hf_home/datasets
    export XDG_CACHE_HOME=/lustre/fsw/portfolios/coreai/users/snorouzi/xdg_cache
    export CUDA_VISIBLE_DEVICES=0
    /usr/bin/python3 "$WORKDIR/examples/eval_grpo_checkpoint_validation.py" \
      --model-path "$CKPT" \
      --tokenizer-path "$TOKENIZER" \
      --outdir "$OUTDIR" \
      --benchmark "$BENCHMARK" \
      --prompt-file "$PROMPT_FILE" \
      --sglang-repo "$SGLANG_REPO" \
      --sglang-commit "$SGLANG_COMMIT" \
      --num-samples "$NUM_SAMPLES" \
      --port "$PORT" \
      --base-url "http://127.0.0.1:$PORT" \
      --server-random-seed "$SERVER_RANDOM_SEED" \
      --cuda-visible-devices 0 \
      --dllm-algorithm "$ALG" \
      --block-size "$BS" \
      --max-steps "$MAX_STEPS" \
      --temperature "$TEMP" \
      --top-p "$TOP_P" \
      --top-k "$TOP_K" \
      --generation-api "$GENERATION_API" \
      --threshold "$THRESHOLD" \
      --selection-policy "$SELECTION_POLICY" \
      --max-new-tokens "$MAX_NEW_TOKENS" \
      --context-length "$CONTEXT_LENGTH" \
      --concurrent "$CONCURRENT" \
      --max-running-requests "$MAX_RUNNING_REQUESTS" \
      --max-total-tokens "$MAX_TOTAL_TOKENS" \
      --mem-fraction-static "$MEM_FRACTION_STATIC" \
      --attention-backend "$ATTENTION_BACKEND" \
      --val-batch-size "$VAL_BATCH_SIZE" \
      --shard-dp-size "$SHARD_DP_SIZE" \
      --shard-rank "$SHARD_RANK"
  '
SBATCH
)

echo "$JOBID" | tee "$OUTDIR/submit.log"
echo "OUTDIR=$OUTDIR"
echo "BENCHMARK=$BENCHMARK ALG=$ALG BS=$BS TEMP=$TEMP CONCURRENT=$CONCURRENT MAX_RUNNING_REQUESTS=$MAX_RUNNING_REQUESTS SHARD=$SHARD_RANK/$SHARD_DP_SIZE"
