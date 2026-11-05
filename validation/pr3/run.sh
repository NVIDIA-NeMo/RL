#!/bin/bash
set -euo pipefail
set +x
source /lustre/fsw/portfolios/coreai/users/yuekaiz/single_controller/RL-pr5090-streaming/validation/pr3/environment.sh
[[ "$USER" == root && -n "${SLURM_JOB_ID:-}" ]]
if [[ "${PR3_MODE:-train}" == parity || "${PR3_MODE:-train}" == normalizer-control ]]; then
  uv run --no-project --python /opt/nemo_rl_venv/bin/python python validation/pr3/run_ep16_parity.py
  exit
fi
uv run --no-project --python /opt/nemo_rl_venv/bin/python python - <<'PY'
import hashlib
import json
from pathlib import Path
report = json.loads(Path('validation/pr3/results/ep16-fixed-input/parity.json').read_text())
assert report['passed'] and report['recipe_optimizer']
for name, expected in report['source_file_sha256'].items():
    assert hashlib.sha256(Path(name).read_bytes()).hexdigest() == expected, name
PY
# Read the requested account's key without echoing it or passing it on the CLI.
export WANDB_API_KEY
WANDB_API_KEY=$(< /lustre/fsw/portfolios/coreai/users/yuekaiz/.wandb_nvidia)
export WANDB_BASE_URL=https://api.wandb.ai WANDB_MODE=online
export WANDB_DIR="$PR3_REPO/validation/pr3"
unset WANDB_SERVICE _WANDB_SERVICE
uv run --no-project --python /opt/nemo_rl_venv/bin/python python examples/run_grpo_single_controller.py \
  --config validation/pr3/streaming_qwen35.yaml
