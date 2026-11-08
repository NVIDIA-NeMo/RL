#!/bin/bash
#SBATCH --nodes=3
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=128
#SBATCH --gres=gpu:8
#SBATCH --exclusive
#SBATCH --account=coreai_dlalgo_nemorl
#SBATCH --partition=batch
#SBATCH --time=04:00:00
#SBATCH --job-name=yuekaiz-pr5090-stream-la0
set -euo pipefail
export PR3_REPO=/lustre/fsw/portfolios/coreai/users/yuekaiz/single_controller/RL-pr5090-streaming
cd "$PR3_REPO"
export CONTAINER=/lustre/fsw/portfolios/coreai/users/yuekaiz/containers/nemo_rl.1030.sqsh
export MOUNTS=/lustre:/lustre
export BASE_LOG_DIR="$PR3_REPO/validation/pr3/slurm"
export GPUS_PER_NODE=8 CPUS_PER_WORKER=128 OMP_NUM_THREADS=4
export ENROOT_ROOTFS_WRITABLE=1
export PR3_MODE="${PR3_MODE:-train}"
export PR3_TARGET_STEPS="${PR3_TARGET_STEPS:-300}"
if [[ "$PR3_MODE" == train || "$PR3_MODE" == exploratory-train ]]; then
  export PR3_CHECKPOINT_ROOT=validation/pr3/results/yuekaiz-qwen35-streaming-la0-300/checkpoints
  if [[ "$PR3_MODE" == exploratory-train ]]; then
    export PR3_CHECKPOINT_ROOT=validation/pr3/results/yuekaiz-qwen35-streaming-la0-300-exploratory/checkpoints
  fi
  # Metadata-only check before starting containers for a continuation segment.
  PR3_COMPLETED_STEPS=$(python3 - <<'PY'
import json
import os
from pathlib import Path
root = Path(os.environ['PR3_CHECKPOINT_ROOT'])
completed = max((json.loads(path.read_text()).get('total_steps', 0)
                 for path in root.glob('step_*/training_info.json')), default=0)
minimum = int(os.environ.get('PR3_MIN_RESUME_STEP', '0'))
if completed < minimum:
    raise RuntimeError(f'Resume requires checkpoint >= {minimum}, found {completed}')
print(completed)
PY
  )
  if (( PR3_COMPLETED_STEPS >= PR3_TARGET_STEPS )); then
    echo "Streaming LA0 already completed $PR3_TARGET_STEPS updates."
    exit 0
  fi
fi
export COMMAND="bash '$PR3_REPO/validation/pr3/run.sh'"
bash ray.sub
