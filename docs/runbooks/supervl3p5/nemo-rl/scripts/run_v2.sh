#!/usr/bin/env bash
set -euo pipefail
script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source "$script_dir/env.sh"
config=${SUPER_CONFIG:-$script_dir/../configs/supervl3p5-v2-production.yaml}
for path in "$MM_TRAINER_DATA_PATH" "$MM_TRAINER_MODEL_PATH/config.json" "$MM_TRAINER_MODEL_PATH/chat_template.jinja" "$config"; do
  [[ -s "$path" ]] || { echo "Missing or empty file: $path" >&2; exit 1; }
done
[[ -d "$GYM_EXTRA_DIR" && -x "$DRIVER_PYTHON" ]] || { echo "Missing Gym extras or driver Python" >&2; exit 1; }
mkdir -p "$MM_TRAINER_RESULTS_DIR" "$SUPER_CACHE_DIR"
cd "$RL_DIR"
exec uv run --no-sync --python "$DRIVER_PYTHON" python \
  examples/run_grpo_single_controller.py --config "$config" "$@"
