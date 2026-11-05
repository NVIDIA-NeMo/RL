#!/bin/bash
set -euo pipefail
set +x
source /lustre/fsw/portfolios/coreai/users/yuekaiz/single_controller/RL-pr5090-streaming/validation/pr3/environment.sh
[[ "$USER" == root && -n "${SLURM_JOB_ID:-}" ]]
if [[ "${PR3_MODE:-train}" == parity || "${PR3_MODE:-train}" == normalizer-control ]]; then
  uv run --no-project --python /opt/nemo_rl_venv/bin/python python validation/pr3/run_ep16_parity.py
  exit
fi
if [[ "${PR3_MODE:-train}" == exploratory-train ]]; then
  # Explicit research lane: preserve the failed original gate and use a
  # separate run identity. The default qualification path stays unchanged.
  report_file=validation/pr3/results/ep16-normalizer-control/normalizer-control.json
  config_file=validation/pr3/streaming_qwen35_exploratory.yaml
elif [[ "${PR3_MODE:-train}" == train ]]; then
  report_file=validation/pr3/results/ep16-fixed-input/parity.json
  config_file=validation/pr3/streaming_qwen35.yaml
else
  echo "Unknown PR3 mode: $PR3_MODE" >&2
  exit 2
fi
uv run --no-project --python /opt/nemo_rl_venv/bin/python python - "$report_file" "${PR3_MODE:-train}" <<'PY'
import hashlib
import json
import sys
from pathlib import Path
report = json.loads(Path(sys.argv[1]).read_text())
assert report['passed']
if sys.argv[2] == 'exploratory-train':
    assert report['contract'] == 'normalizer_placement_diagnostic_only'
    assert report['releases_training'] is False
else:
    assert report['recipe_optimizer']
    assert report['contract'] == 'bf16_gradient_and_optimizer_oracle_v2'
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
  --config "$config_file"
