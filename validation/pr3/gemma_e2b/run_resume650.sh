#!/bin/bash
# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
set -euo pipefail
set +x
source /lustre/fsw/portfolios/coreai/users/yuekaiz/single_controller/RL-pr5090-streaming/validation/pr3/environment.sh
[[ "$USER" == root && -n "${SLURM_JOB_ID:-}" ]]
exec 9>validation/pr3/gemma_e2b/slurm/driver.lock
flock -n 9
nvidia-smi --query-gpu=name --format=csv,noheader
uv run --no-project --python /opt/nemo_rl_venv/bin/python python validation/pr3/gemma_e2b/preflight_resume.py
export WANDB_API_KEY
WANDB_API_KEY=$(< /lustre/fsw/portfolios/coreai/users/yuekaiz/.wandb_nvidia)
export WANDB_BASE_URL=https://api.wandb.ai WANDB_MODE=online
export WANDB_DIR="$PR3_REPO/validation/pr3/gemma_e2b"
unset WANDB_SERVICE _WANDB_SERVICE
uv run --no-project --python /opt/nemo_rl_venv/bin/python python examples/run_grpo_single_controller.py \
  --config validation/pr3/gemma_e2b/resume650.yaml
uv run --no-project --python /opt/nemo_rl_venv/bin/python python - <<'PY'
import json
from pathlib import Path
from validation.pr3.gemma_e2b.plot import audit_checkpoint
path=Path('validation/pr3/gemma_e2b/results/streaming-la0/checkpoints/latest_checkpoint_status.json')
step=json.loads(path.read_text())['last_checkpoint_step']
audit_checkpoint(step)
print(f'Completed Gemma segment at durable checkpoint{step}',flush=True)
PY
if [[ $(uv run --no-project --python /opt/nemo_rl_venv/bin/python python -c "import json; print(json.load(open('validation/pr3/gemma_e2b/results/streaming-la0/checkpoints/latest_checkpoint_status.json'))['last_checkpoint_step'])") == 650 ]]; then
  uv run --no-project --python /opt/nemo_rl_venv/bin/python python validation/pr3/gemma_e2b/plot.py --max-step 650
fi
