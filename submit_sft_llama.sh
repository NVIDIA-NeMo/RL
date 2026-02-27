# Run SFT (Llama 3.2 1B) on CLIMB Arrow dataset from the root of NeMo RL repo
# Run 5 times: bash submit_sft_llama.sh -n 5
# NUM_ACTOR_NODES=2
NUM_ACTOR_NODES=1


# default: single submission; use -n to repeat
N_CALLS=1
while getopts "n:" opt; do
  case $opt in
    n) N_CALLS=$OPTARG;;
  esac
done

EXP_NAME=SFT-Arrow-Llama

read -r -d '' COMMAND <<EOF
export WANDB_API_KEY=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
export HF_TOKEN=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

export HF_HOME=/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_genai/users/avenkateshha/hf
export CUDA_DEVICE_MAX_CONNECTIONS=1
export HF_DATASETS_CACHE=/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_genai/users/avenkateshha/hf_datasets_cache

# Run SFT (Llama 3.2 1B, Arrow dataset)
uv run /lustre/fsw/portfolios/coreai/users/avenkateshha/nemo_rl/RL/examples/run_sft.py \
  --config /lustre/fsw/portfolios/coreai/users/avenkateshha/nemo_rl/RL/examples/configs/llama_sft_arrow.yaml \
  cluster.num_nodes=${NUM_ACTOR_NODES}
EOF

export COMMAND

# Ray logs go to $BASE_LOG_DIR/$SLURM_JOB_ID-logs (see ray.sub). Use llama-sft/ under current dir.
export BASE_LOG_DIR="$(pwd)/llama-sft"

DEP_OPT=""
PREV_JOBID=""

MY_CONTAINER="/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_genai/users/avenkateshha/nemo_rl/nemo-rl.sqsh"

for (( i = 1; i <= ${N_CALLS}; i++ ))
do
  if [ -n "$PREV_JOBID" ]; then
    DEP_OPT="--dependency=afterany:${PREV_JOBID}"
  fi
  echo "Submitting job ${i}${PREV_JOBID:+ with dependency on jobid ${PREV_JOBID}}"
  # Export variables so sbatch passes them to the job script
  export CONTAINER="${MY_CONTAINER}"
  # Mount fs1 (HF cache), fsw repo (code), and fsw data (arrow files)
  export MOUNTS="/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_genai/users/avenkateshha:/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_genai/users/avenkateshha,/lustre/fsw/portfolios/coreai/users/avenkateshha/nemo_rl:/lustre/fsw/portfolios/coreai/users/avenkateshha/nemo_rl,/lustre/fsw/portfolios/llmservice/users/sdiao/data:/lustre/fsw/portfolios/llmservice/users/sdiao/data"
  OUTPUT=$(sbatch \
    ${DEP_OPT} \
    --nodes=${NUM_ACTOR_NODES} \
    --account=coreai_dlalgo_genai \
    --job-name=nemo-rl.${EXP_NAME} \
    --partition=batch \
    --time=1:30:0 \
    --gres=gpu:8 \
    ray.sub)
  PREV_JOBID="$(cut -d' ' -f4 <<< "$OUTPUT")"
done
