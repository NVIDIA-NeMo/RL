#!/bin/bash
# Run SFT on arrow data with inline MATH + MMLU evaluation
#
# Usage:
#   bash submit_sft_arrow_eval.sh
#
# Launches a 4-node job that trains Llama-3.2-1B on arrow text data
# and evaluates on MATH + MMLU every 50 steps.

NUM_NODES=4

read -r -d '' COMMAND <<'EOF'
export WANDB_API_KEY=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
export HF_TOKEN=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

export HF_HOME=/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_genai/users/avenkateshha/hf
export CUDA_DEVICE_MAX_CONNECTIONS=1
export HF_DATASETS_CACHE=/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_genai/users/avenkateshha/hf_datasets_cache

uv run /lustre/fsw/portfolios/coreai/users/avenkateshha/nemo_rl/RL/examples/run_sft_arrow_with_eval.py \
  --config /lustre/fsw/portfolios/coreai/users/avenkateshha/nemo_rl/RL/examples/configs/llama_sft_arrow.yaml
EOF

export COMMAND

export BASE_LOG_DIR="$(pwd)/sft-arrow-eval-logs"

MY_CONTAINER="/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_genai/users/avenkateshha/nemo_rl/nemo-rl.sqsh"

echo "Submitting SFT arrow training with MATH+MMLU eval (${NUM_NODES} nodes)"
export CONTAINER="${MY_CONTAINER}"
export MOUNTS="/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_genai/users/avenkateshha:/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_genai/users/avenkateshha,/lustre/fsw/portfolios/coreai/users/avenkateshha/nemo_rl:/lustre/fsw/portfolios/coreai/users/avenkateshha/nemo_rl,/lustre/fsw/portfolios/llmservice/users/sdiao/data:/lustre/fsw/portfolios/llmservice/users/sdiao/data"
sbatch \
  --nodes=${NUM_NODES} \
  --account=coreai_dlalgo_genai \
  --job-name=nemo-rl.sft-arrow-eval \
  --partition=batch \
  --time=4:0:0 \
  --gres=gpu:8 \
  ray.sub
