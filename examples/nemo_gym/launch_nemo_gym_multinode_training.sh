# ----- PARAMETERS -----
# WANDB_API_KEY, EXP_NAME, NUM_ACTOR_NODES, REPO_LOCATION, CONTAINER_IMAGE_PATH, SLURM_ACCOUNT, SLURM_PARTITION

# ray.sub needs to be launched from the NeMo-RL root directory
cd $REPO_LOCATION

# Construct the command
read -r -d '' COMMAND <<EOF
cd ${REPO_LOCATION}

HF_HOME=$PWD/.cache/ \
WANDB_API_KEY=$WANDB_API_KEY \
uv run python examples/nemo_gym/run_grpo_nemo_gym.py \
    ++cluster.num_nodes=$NUM_ACTOR_NODES \
    ++logger.wandb.name=$EXP_NAME \
    ++logger.log_dir=results/$EXP_NAME \
    ++checkpointing.checkpoint_dir=results/$EXP_NAME \
    $@
EOF

echo -e "Running command:\n$COMMAND"

mount=$(findmnt -n -o TARGET --target .)

COMMAND=$COMMAND \
CONTAINER=$CONTAINER_IMAGE_PATH \
MOUNTS=$mount:$mount \
sbatch \
    --nodes=$NUM_ACTOR_NODES \
    --account=$SLURM_ACCOUNT \
    --partition=$SLURM_PARTITION \
    --time=4:0:0 \
    --job-name=$EXP_NAME \
    --gres=gpu:8 \
    ray.sub
