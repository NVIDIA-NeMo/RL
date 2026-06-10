#!/usr/bin/env bash
set -euo pipefail

# Submit the standalone No-Ray GSM8K evaluator that reproduces the NeMoRL
# validation prompt/generation/grading path without Ray.
#
# Common usage:
#   ALG=FastDiffuser BS=32 TEMP=1.0 TAG=my_eval ./submit_standalone_gsm8k_eval.sh
#   ALG=LinearSpec   BS=16 TEMP=0.0 TAG=my_greedy_ls_b16 ./submit_standalone_gsm8k_eval.sh
#   ALG=AR TEMP=0.0 TAG=my_sglang_ar_mode ./submit_standalone_gsm8k_eval.sh
#   ./submit_standalone_gsm8k_eval.sh --concurrency 1
#   BACKEND=ar_native TEMP=0.0 SHARD_DP_SIZE=8 SHARD_RANK=0 TAG=my_ar_shard0 ./submit_standalone_gsm8k_eval.sh

ACCOUNT=${ACCOUNT:-coreai_dlalgo_llm}
PARTITION=${PARTITION:-batch_short}
TIME_LIMIT=${TIME_LIMIT:-00:45:00}
GPUS_PER_NODE=${GPUS_PER_NODE:-1}

BACKEND=${BACKEND:-sglang_dllm}      # sglang_dllm or ar_native
ALG=${ALG:-FastDiffuser}        # FastDiffuser, LinearSpec, or AR (SGLang ar_mode)
BS=${BS:-32}                    # diffusion block size
TEMP=${TEMP:-1.0}               # 1.0 for NeMoRL validation-style sampling, 0.0 for greedy
MAX_STEPS=${MAX_STEPS:-32}
THRESHOLD=${THRESHOLD:-0.9}
SELECTION_POLICY=${SELECTION_POLICY:-confidence}
MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-750}
CONTEXT_LENGTH=${CONTEXT_LENGTH:-1024}
TOP_P=${TOP_P:-1.0}
TOP_K=${TOP_K:--1}
CONCURRENT=${CONCURRENT:-8}
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
WORKDIR=${WORKDIR:-/home/snorouzi/diffusion_RL/RL}
ROOT=${ROOT:-/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_genai/users/snorouzi/eval_results}
PROMPT_FILE=${PROMPT_FILE:-$WORKDIR/examples/prompts/cot.txt}
JSON_MODEL_OVERRIDE_ARGS=${JSON_MODEL_OVERRIDE_ARGS:-}

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
SGLANG_REPO=${SGLANG_REPO:-/home/snorouzi/code/sglang-nemotron-dllm-a652eb48}
SGLANG_COMMIT=${SGLANG_COMMIT:-$(git -C "$SGLANG_REPO" rev-parse HEAD)}

if [[ "$ALG" != "FastDiffuser" && "$ALG" != "LinearSpec" && "$ALG" != "AR" ]]; then
  echo "ALG must be FastDiffuser, LinearSpec, or AR, got: $ALG" >&2
  exit 2
fi
if [[ "$BACKEND" != "sglang_dllm" && "$BACKEND" != "ar_native" ]]; then
  echo "BACKEND must be sglang_dllm or ar_native, got: $BACKEND" >&2
  exit 2
fi

if [[ -z "${TAG:-}" ]]; then
  alg_tag=$(echo "${BACKEND}_${ALG}" | tr '[:upper:]' '[:lower:]')
  temp_tag=${TEMP//./p}
  TAG="standalone_gsm8k_${alg_tag}_b${BS}_t${temp_tag}_$(date +%Y%m%d_%H%M%S)"
fi

OUTDIR=${OUTDIR:-$ROOT/$TAG}
mkdir -p "$OUTDIR"

JOB_NAME=${JOB_NAME:-gsm8k_${ALG}_b${BS}}
JOB_NAME=${JOB_NAME:0:24}

JOBID=$(sbatch \
  -J "$JOB_NAME" \
  -A "$ACCOUNT" \
  -p "$PARTITION" \
  -N 1 \
  --gpus-per-node="$GPUS_PER_NODE" \
  -t "$TIME_LIMIT" \
  -o "$OUTDIR/slurm-%j.out" \
  --export=ALL,OUTDIR="$OUTDIR",CKPT="$CKPT",TOKENIZER="$TOKENIZER",CONTAINER="$CONTAINER",WORKDIR="$WORKDIR",PROMPT_FILE="$PROMPT_FILE",SGLANG_REPO="$SGLANG_REPO",SGLANG_COMMIT="$SGLANG_COMMIT",JSON_MODEL_OVERRIDE_ARGS="$JSON_MODEL_OVERRIDE_ARGS",PORT="$PORT",BACKEND="$BACKEND",ALG="$ALG",BS="$BS",TEMP="$TEMP",MAX_STEPS="$MAX_STEPS",THRESHOLD="$THRESHOLD",SELECTION_POLICY="$SELECTION_POLICY",MAX_NEW_TOKENS="$MAX_NEW_TOKENS",CONTEXT_LENGTH="$CONTEXT_LENGTH",TOP_P="$TOP_P",TOP_K="$TOP_K",CONCURRENT="$CONCURRENT",MAX_RUNNING_REQUESTS="$MAX_RUNNING_REQUESTS",MAX_TOTAL_TOKENS="$MAX_TOTAL_TOKENS",MEM_FRACTION_STATIC="$MEM_FRACTION_STATIC",ATTENTION_BACKEND="$ATTENTION_BACKEND",SERVER_RANDOM_SEED="$SERVER_RANDOM_SEED",NUM_SAMPLES="$NUM_SAMPLES",SHARD_DP_SIZE="$SHARD_DP_SIZE",SHARD_RANK="$SHARD_RANK",VAL_BATCH_SIZE="$VAL_BATCH_SIZE" \
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
echo "PROMPT_FILE=$PROMPT_FILE"
echo "SGLANG_REPO=$SGLANG_REPO"
echo "SGLANG_COMMIT=$SGLANG_COMMIT"
echo "JSON_MODEL_OVERRIDE_ARGS=$JSON_MODEL_OVERRIDE_ARGS"
echo "BACKEND=$BACKEND"
echo "ALG=$ALG"
echo "BS=$BS"
echo "TEMP=$TEMP"
echo "MAX_STEPS=$MAX_STEPS"
echo "MAX_NEW_TOKENS=$MAX_NEW_TOKENS"
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
    /usr/bin/python3 /home/snorouzi/diffusion_RL/RL/examples/eval_grpo_checkpoint_validation.py \
      --model-path "$CKPT" \
      --tokenizer-path "$TOKENIZER" \
      --outdir "$OUTDIR" \
      --prompt-file "$PROMPT_FILE" \
      --sglang-repo "$SGLANG_REPO" \
      --sglang-commit "$SGLANG_COMMIT" \
      --json-model-override-args "$JSON_MODEL_OVERRIDE_ARGS" \
      --backend "$BACKEND" \
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
echo "BACKEND=$BACKEND ALG=$ALG BS=$BS TEMP=$TEMP CONCURRENT=$CONCURRENT MAX_RUNNING_REQUESTS=$MAX_RUNNING_REQUESTS SHARD=$SHARD_RANK/$SHARD_DP_SIZE"
