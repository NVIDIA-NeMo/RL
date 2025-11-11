#!/bin/bash

# ================ EDIT THESE VALUES AS NEEDED ================
# Number of nodes for your job
NUM_ACTOR_NODES=16
DATE=$(date +%Y%m%d)
CONFIG_PATH="experiments/grpo_acemath_rl_mcore.yaml"

IMPORTANCE_SAMPLING_CORRECTION=False
EXP_SUFFIX="grpo-acemath-mcore_tp4-vllm_tp2-perform_initialization_True"
# EXP_SUFFIX="grpo-acemath-mcore_tp1-vllm_tp2-perform_initialization_True"
# ============================================================

# Set up paths and names based on experiment suffix
CHECKPOINT_DIR="results/${EXP_SUFFIX}"
WANDB_NAME="${EXP_SUFFIX}"
SNAPSHOT_DIR="code_snapshots/${EXP_SUFFIX}"

echo "Submitting job with experiment suffix: ${EXP_SUFFIX}"
echo "Checkpoint directory: ${CHECKPOINT_DIR}"

# Create a code snapshot if it doesn't already exist
if [ ! -d "${SNAPSHOT_DIR}" ]; then
  # Create snapshots directory if it doesn't exist
  mkdir -p code_snapshots
  
  echo "Creating new code snapshot in $SNAPSHOT_DIR"
  
  # Create the snapshot directory
  mkdir -p $SNAPSHOT_DIR
  
  # Copy only git-tracked files to the snapshot directory
  echo "Copying git-tracked files..."
  
  # Ensure submodules are initialized and up to date
  git submodule update --init --recursive

  # Create a temporary file with a null-terminated list of git-tracked files (including submodules)
  GIT_FILES_LIST=$(mktemp)
  if git ls-files -z --recurse-submodules > "$GIT_FILES_LIST" 2>/dev/null; then
    :
  else
    # Fallback for older git versions without --recurse-submodules
    git ls-files -z > "$GIT_FILES_LIST"
    git submodule foreach --quiet --recursive 'git ls-files -z | sed -z "s|^|$path/|"' >> "$GIT_FILES_LIST"
  fi

  # Copy the git-tracked files to the snapshot directory (reads null-terminated list)
  rsync -a --from0 --files-from="$GIT_FILES_LIST" ./ "$SNAPSHOT_DIR"/

  # Remove the temporary file
  rm "$GIT_FILES_LIST"
  
  echo "Snapshot creation complete"
else
  echo "Using existing code snapshot in $SNAPSHOT_DIR"
fi

# Change to the snapshot directory
cd $SNAPSHOT_DIR

# grpo_math_8b uses Llama-3.1-8B-Instruct model
HF_HOME=/lustre/fs1/portfolios/coreai/users/ffrujeri/hf_home \
RAY_DEDUP_LOGS=0 \
COMMAND="uv pip install -e .; \
VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 NRL_FORCE_REBUILD_VENVS=true uv run experiments/run.py \
    --config=${CONFIG_PATH} \
    loss_fn.use_importance_sampling_correction=${IMPORTANCE_SAMPLING_CORRECTION} \
    cluster.num_nodes=${NUM_ACTOR_NODES} \
    cluster.gpus_per_node=8 \
    logger.wandb_enabled=True \
    logger.wandb.name='${WANDB_NAME}' \
    ++logger.wandb.id='${WANDB_NAME}' \
    logger.wandb.project='nemo-rl-acemath-rl' \
    checkpointing.checkpoint_dir='${PWD}/${CHECKPOINT_DIR}' \
    ++policy.generation.vllm_kwargs.compilation_config.use_inductor=False" \
CONTAINER=/lustre/fsw/portfolios/coreai/users/terryk/enroot-images/gitlab-master.nvidia.com/terryk/images/nemo-rl:main-c249efc8.squashfs \
MOUNTS="/lustre:/lustre:ro,$PWD:$PWD" \
sbatch \
    --nodes=${NUM_ACTOR_NODES} \
    --account=coreai_dlalgo_nemorl \
    --job-name=$(whoami)-${EXP_SUFFIX} \
    --partition=batch \
    --time=4:0:0 \
    --gres=gpu:8 \
    ray.sub | tee /dev/stderr | grep -o '[0-9]\+' > latest_job_id.txt

JOB_ID=$(cat latest_job_id.txt)
echo "Job submitted with ID: $JOB_ID"

# Return to original directory
cd - > /dev/null