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
if [[ "$PR3_MODE" == train ]]; then
  # Metadata-only check before starting containers for a continuation segment.
  if python3 - <<'PY'
import json
from pathlib import Path
root = Path('validation/pr3/results/yuekaiz-qwen35-streaming-la0-300/checkpoints')
completed = any(json.loads(path.read_text()).get('total_steps', 0) >= 300
                for path in root.glob('step_*/training_info.json'))
raise SystemExit(0 if completed else 1)
PY
  then
    echo 'Streaming LA0 already completed 300 updates.'
    exit 0
  fi
fi
export COMMAND="bash '$PR3_REPO/validation/pr3/run.sh'"
bash ray.sub
