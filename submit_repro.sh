# Run from the root of NeMo RL repo
NUM_ACTOR_NODES=2

# default: single submission; use -n to repeat
N_CALLS=1
while getopts "n:" opt; do
  case $opt in
    n) N_CALLS=$OPTARG;;
  esac
done

# grpo_math_8b uses Llama-3.1-8B-Instruct model

EXP_NAME=OPD-Reproduce

read -r -d '' COMMAND <<EOF
export WANDB_API_KEY=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
export HF_TOKEN=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

export HF_HOME=/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_genai/users/avenkateshha/hf
export CUDA_DEVICE_MAX_CONNECTIONS=1
export HF_DATASETS_CACHE=/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_genai/users/avenkateshha/hf_datasets_cache


uv run /lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_genai/users/avenkateshha/nemo_rl/RL/examples/run_distillation_math.py \
 --config /lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_genai/users/avenkateshha/nemo_rl/RL/examples/configs/dist.yaml \
 cluster.num_nodes=${NUM_ACTOR_NODES}
EOF

export COMMAND

DEP_OPT="--dependency=afterany:6841564"
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
  # Mount the fs1 path
  export MOUNTS="/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_genai/users/avenkateshha:/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_genai/users/avenkateshha"
  OUTPUT=$(sbatch \
    ${DEP_OPT} \
    --nodes=${NUM_ACTOR_NODES} \
    --account=coreai_dlalgo_genai \
    --job-name=nemo-rl.${EXP_NAME} \
    --partition=batch\
    --time=4:0:0 \
    --gres=gpu:8 \
    ray.sub)
  PREV_JOBID="$(cut -d' ' -f4 <<< "$OUTPUT")"
done