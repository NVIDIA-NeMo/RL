# Run evaluation (Math or MMLU) for Llama 3.2 1B checkpoints from the root of NeMo RL repo
#
# IMPORTANT: The checkpoint must be in standard HuggingFace format (not FSDP shards).
#   If your checkpoint is in NeMo-RL sharded format (policy/weights/model/shard-*.safetensors),
#   first consolidate it using:
#     python examples/converters/consolidate_checkpoint.py \
#       --input /path/to/step_N/policy/weights \
#       --output /path/to/hf_checkpoint \
#       --model-name meta-llama/Llama-3.2-1B
#
# Usage:
#   bash submit_eval_llama.sh -c /path/to/hf_checkpoint -e math
#   bash submit_eval_llama.sh -c /path/to/hf_checkpoint -e mmlu
#
# Options:
#   -c  Path to the consolidated HuggingFace checkpoint directory (required)
#   -e  Eval benchmark: "math" or "mmlu" (default: math)
#   -t  Tokenizer name or path (default: same as checkpoint path)
NUM_ACTOR_NODES=1

CKPT_PATH=""
EVAL_TYPE="math"
TOKENIZER=""

while getopts "c:e:t:" opt; do
  case $opt in
    c) CKPT_PATH=$OPTARG;;
    e) EVAL_TYPE=$OPTARG;;
    t) TOKENIZER=$OPTARG;;
  esac
done

if [ -z "$CKPT_PATH" ]; then
  echo "Error: checkpoint path is required. Usage: bash submit_eval_llama.sh -c /path/to/hf_checkpoint -e math"
  exit 1
fi

# Default tokenizer to checkpoint path (works when checkpoint has tokenizer files)
if [ -z "$TOKENIZER" ]; then
  TOKENIZER="${CKPT_PATH}"
fi

# Select eval config based on type
case $EVAL_TYPE in
  math)
    EVAL_CONFIG="/lustre/fsw/portfolios/coreai/users/avenkateshha/nemo_rl/RL/examples/configs/evals/llama_math_eval.yaml"
    ;;
  mmlu)
    EVAL_CONFIG="/lustre/fsw/portfolios/coreai/users/avenkateshha/nemo_rl/RL/examples/configs/evals/llama_mmlu_eval.yaml"
    ;;
  *)
    echo "Error: unknown eval type '$EVAL_TYPE'. Use 'math' or 'mmlu'."
    exit 1
    ;;
esac

EXP_NAME=Eval-${EVAL_TYPE}-Llama

read -r -d '' COMMAND <<EOF
export WANDB_API_KEY=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
export HF_TOKEN=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

export HF_HOME=/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_genai/users/avenkateshha/hf
export CUDA_DEVICE_MAX_CONNECTIONS=1
export HF_DATASETS_CACHE=/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_genai/users/avenkateshha/hf_datasets_cache

# Run evaluation
uv run /lustre/fsw/portfolios/coreai/users/avenkateshha/nemo_rl/RL/examples/run_eval.py \
  --config ${EVAL_CONFIG} \
  generation.model_name=${CKPT_PATH} \
  tokenizer.name=${TOKENIZER}
EOF

export COMMAND

# Ray logs go to $BASE_LOG_DIR/$SLURM_JOB_ID-logs (see ray.sub)
export BASE_LOG_DIR="$(pwd)/llama-eval"

MY_CONTAINER="/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_genai/users/avenkateshha/nemo_rl/nemo-rl.sqsh"

echo "Submitting ${EVAL_TYPE} evaluation for checkpoint: ${CKPT_PATH} (tokenizer: ${TOKENIZER})"
export CONTAINER="${MY_CONTAINER}"
# Mount fs1 (HF cache), fsw repo (code), and checkpoint path
export MOUNTS="/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_genai/users/avenkateshha:/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_genai/users/avenkateshha,/lustre/fsw/portfolios/coreai/users/avenkateshha/nemo_rl:/lustre/fsw/portfolios/coreai/users/avenkateshha/nemo_rl,/lustre/fsw/portfolios/llmservice/users/sdiao/data:/lustre/fsw/portfolios/llmservice/users/sdiao/data"
sbatch \
  --nodes=${NUM_ACTOR_NODES} \
  --account=coreai_dlalgo_genai \
  --job-name=nemo-rl.${EXP_NAME} \
  --partition=batch \
  --time=1:0:0 \
  --gres=gpu:8 \
  ray.sub
