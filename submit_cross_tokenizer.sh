# Run Cross-Tokenizer Off-Policy Distillation (Llama 1B student, Qwen 8B teacher)
# Run 5 times: bash submit_cross_tokenizer.sh -n 5
NUM_ACTOR_NODES=2

# default: single submission; use -n to repeat
N_CALLS=1
while getopts "n:" opt; do
  case $opt in
    n) N_CALLS=$OPTARG;;
  esac
done

EXP_NAME=CrossTokenizer-Distillation-Llama1B-Phi4MiniInstruct

read -r -d '' COMMAND <<EOF
export WANDB_API_KEY=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
export HF_TOKEN=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

export HF_HOME=/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_genai/users/avenkateshha/hf
export CUDA_DEVICE_MAX_CONNECTIONS=1
export HF_DATASETS_CACHE=/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_genai/users/avenkateshha/hf_datasets_cache

export CROSS_TOK_DEBUG_DIR=/lustre/fsw/portfolios/coreai/users/avenkateshha/nemo_rl/RL/x_token/debug_dump
export NCCL_DEBUG=INFO

uv run /lustre/fsw/portfolios/coreai/users/avenkateshha/nemo_rl/RL/examples/run_off_policy_distillation_arrow_with_eval.py \
  --config /lustre/fsw/portfolios/coreai/users/avenkateshha/nemo_rl/RL/examples/configs/cross_tokenizer_off_policy_arrow.yaml \
  cluster.num_nodes=${NUM_ACTOR_NODES} \
  distillation.num_prompts_per_step=32 \
  policy.train_global_batch_size=32 \
  teacher.train_global_batch_size=32 \
  teacher.model_name=microsoft/Phi-4-mini-instruct \
  teacher.tokenizer.name=microsoft/Phi-4-mini-instruct \
  token_aligner.projection_matrix_path=cross_tokenizer_data/projection_map_Llama-3.2_to_Phi-4-mini-instruct_multitoken_top_32_double_special.pt \
  distillation.use_ipc=true \
  distillation.max_num_steps=80000 \
  eval.val_period=1000
EOF

export COMMAND

# Ray logs go to $BASE_LOG_DIR/$SLURM_JOB_ID-logs (see ray.sub). Use x_token/ under current dir.
export BASE_LOG_DIR="$(pwd)/x_token"

DEP_OPT=""
PREV_JOBID=""

MY_CONTAINER="/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_genai/users/avenkateshha/nemo_rl/nemo-rl.sqsh"

for (( i = 1; i <= ${N_CALLS}; i++ ))
do
  if [ -n "$PREV_JOBID" ]; then
    DEP_OPT="--dependency=afterany:${PREV_JOBID}"
  fi
  echo "Submitting job ${i}${PREV_JOBID:+ with dependency on jobid ${PREV_JOBID}}"
  export CONTAINER="${MY_CONTAINER}"
  export MOUNTS="/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_genai/users/avenkateshha:/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_genai/users/avenkateshha,/lustre/fsw/portfolios/coreai/users/avenkateshha/nemo_rl:/lustre/fsw/portfolios/coreai/users/avenkateshha/nemo_rl,/lustre/fsw/portfolios/llmservice/users/sdiao/data:/lustre/fsw/portfolios/llmservice/users/sdiao/data"
  OUTPUT=$(sbatch \
    ${DEP_OPT} \
    --nodes=${NUM_ACTOR_NODES} \
    --account=coreai_dlalgo_genai \
    --job-name=nemo-rl.${EXP_NAME} \
    --partition=batch_short \
    --time=2:0:0 \
    --gres=gpu:8 \
    ray.sub)
  PREV_JOBID="$(cut -d' ' -f4 <<< "$OUTPUT")"
done
