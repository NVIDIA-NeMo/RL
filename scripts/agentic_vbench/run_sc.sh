#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Preflight the gym-native Agentic-VBench inputs (pinned benchmark checkout, Harbor
# runtime, Docker client, task-container backend, judge credentials) and submit
# SingleController GRPO with token capture. Hydra overrides pass through.
#   CONTEXT_COMPACTION=true AGENTIC_VBENCH_BACKEND_DIR=... bash scripts/agentic_vbench/run_sc.sh [overrides]
set -euo pipefail
script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
code_dir=$(cd "$script_dir/../.." && pwd)
gym_dir=$code_dir/3rdparty/Gym-workspace/Gym
export LAUNCH_BENCHMARK=agentic_vbench
export AVB_DATA_ROOT=${AVB_DATA_ROOT:-$code_dir/data/agentic_vbench}
export AGENTIC_VBENCH_ROOT=${AGENTIC_VBENCH_ROOT:-$AVB_DATA_ROOT/agentic-vbench}
export AGENTIC_VBENCH_HARBOR_PYTHON=${AGENTIC_VBENCH_HARBOR_PYTHON:-$AVB_DATA_ROOT/harbor-venv/bin/python}
export AGENTIC_VBENCH_DOCKER_CLI_DIR=${AGENTIC_VBENCH_DOCKER_CLI_DIR:-$AVB_DATA_ROOT/docker-client}
export AGENTIC_VBENCH_BACKEND=${AGENTIC_VBENCH_BACKEND:-remote}
export AGENTIC_VBENCH_JUDGE_PROTOCOL=${AGENTIC_VBENCH_JUDGE_PROTOCOL-nvinference-hybrid}
credentials_file=${AGENTIC_VBENCH_CREDENTIALS_FILE:-$HOME/.config/agentic-vbench/nvinference.env}
tasks=${AGENTIC_VBENCH_TASKS:-all}
train_repeats=${TRAIN_REPEATS:-1}
train_limit=${TRAIN_LIMIT:-}

expected_revision=$(python3 -c "import sys; sys.path.insert(0, '$gym_dir'); from responses_api_agents.agentic_vbench_agent.core import REVISION; print(REVISION)")
[[ -d $AGENTIC_VBENCH_ROOT/.git ]] || { echo "Missing pinned benchmark checkout at $AGENTIC_VBENCH_ROOT (link_shared_assets.sh --benchmark ...)" >&2; exit 1; }
revision=$(git -C "$AGENTIC_VBENCH_ROOT" rev-parse HEAD)
[[ $revision == "$expected_revision" ]] || { echo "Benchmark checkout is at $revision, expected $expected_revision" >&2; exit 1; }
[[ -x $AGENTIC_VBENCH_HARBOR_PYTHON ]] || { echo "Missing Harbor Python at $AGENTIC_VBENCH_HARBOR_PYTHON (setup_harbor.sh or link_shared_assets.sh)" >&2; exit 1; }
"$AGENTIC_VBENCH_HARBOR_PYTHON" -c 'import importlib.metadata as m, dotenv; assert m.version("harbor") == "0.6.6", m.version("harbor")'
[[ -x $AGENTIC_VBENCH_DOCKER_CLI_DIR/docker && -x $AGENTIC_VBENCH_DOCKER_CLI_DIR/cli-plugins/docker-compose ]] || { echo "Docker client missing under $AGENTIC_VBENCH_DOCKER_CLI_DIR" >&2; exit 1; }
if [[ -n $AGENTIC_VBENCH_JUDGE_PROTOCOL ]]; then
  # Repurpose verifiers judge through NVIDIA Inference Hub. The key travels in the job
  # environment (judge_api_key reads NVINFERENCE_API_KEY); the file itself is not needed
  # inside the container and is never echoed.
  if [[ -s $credentials_file ]]; then
    set -a; source "$credentials_file"; set +a
  fi
  [[ -n ${NVINFERENCE_API_KEY:-${INFERENCE_API_KEY:-}} ]] || { echo "Judge protocol $AGENTIC_VBENCH_JUDGE_PROTOCOL needs NVINFERENCE_API_KEY (or $credentials_file)" >&2; exit 1; }
  export AGENTIC_VBENCH_CREDENTIALS_FILE=""
fi
if [[ ${DRY_RUN:-0} != 1 ]]; then
  : "${AGENTIC_VBENCH_BACKEND_DIR:?Set AGENTIC_VBENCH_BACKEND_DIR to the directory where serve_rootless_backend.sh publishes client.env}"
  [[ -s $AGENTIC_VBENCH_BACKEND_DIR/client.env ]] || { echo "No task backend yet: $AGENTIC_VBENCH_BACKEND_DIR/client.env (submit_backend.sh)" >&2; exit 1; }
  (
    set -a; source "$AGENTIC_VBENCH_BACKEND_DIR/client.env"; set +a
    export DOCKER_CERT_PATH=$AGENTIC_VBENCH_BACKEND_DIR PATH=$AGENTIC_VBENCH_DOCKER_CLI_DIR:$PATH DOCKER_CONFIG=$AGENTIC_VBENCH_DOCKER_CLI_DIR
    docker info --format '{{.ServerVersion}}' >/dev/null || { echo 'Task backend is unreachable from this host' >&2; exit 1; }
    mapfile -t existing_containers < <(docker ps -aq)
    if (( ${#existing_containers[@]} )); then
      echo "Task backend is not empty: ${#existing_containers[@]} container(s) at $AGENTIC_VBENCH_BACKEND_DIR" >&2
      docker ps -a --format '{{.ID}} {{.Names}} {{.Status}}' >&2
      echo 'Use a dedicated empty backend or remove only containers from the terminated run before submitting.' >&2
      exit 1
    fi
  )
fi
export AGENTIC_VBENCH_BACKEND_DIR=${AGENTIC_VBENCH_BACKEND_DIR:-}
train_name=${tasks// /_}
export TRAIN_PATH=${TRAIN_PATH:-$AVB_DATA_ROOT/train/${train_name}${train_limit:+_first${train_limit}}_x${train_repeats}.jsonl}
export VAL_PATH=${VAL_PATH:-$TRAIN_PATH}
if [[ ! -s $TRAIN_PATH ]]; then
  python3 "$script_dir/build_train_dataset.py" --benchmark-root "$AGENTIC_VBENCH_ROOT" --tasks "$tasks" \
    ${train_limit:+--limit "$train_limit"} --repeats "$train_repeats" --output "$TRAIN_PATH"
fi
# A non-empty cache is not sufficient: older datasets on this branch used the
# deprecated agent_ref schema and can silently route the wrong Gym source.  Rebuild
# the expected rows in memory and compare the complete public row payload before
# allocating GPUs.
python3 - "$gym_dir" "$AGENTIC_VBENCH_ROOT" "$TRAIN_PATH" "$tasks" "$train_limit" "$train_repeats" <<'PY'
import json
import sys
from pathlib import Path

gym_root, benchmark_root, train_path, selection, raw_limit, raw_repeats = sys.argv[1:]
sys.path.insert(0, gym_root)
from responses_api_agents.agentic_vbench_agent.core import dataset_rows, inventory

expected = dataset_rows(inventory(Path(benchmark_root)), selection)
if raw_limit:
    expected = expected[: int(raw_limit)]
expected = [
    {**row, "task_source": "agentic_vbench_agent"}
    for _ in range(int(raw_repeats))
    for row in expected
]
with Path(train_path).open() as handle:
    actual = [json.loads(line) for line in handle if line.strip()]
if actual != expected:
    raise SystemExit(
        f"Stale/mismatched Agentic VBench dataset: {train_path}; "
        "remove it or select a new TRAIN_PATH so it can be regenerated"
    )
print(f"Validated {len(actual)} Agentic VBench rows against the pinned benchmark")
PY
export TRAINING_DEPS_SETUP_SCRIPT=/opt/nemo-rl/tools/ensure_agentic_vbench_deps.sh
# Leave GYM_VENV_DIR unset unless the caller chose one. launch_sc.sh then uses
# the submitter-owned, prewarmed cc-pr4200 cache under PERSISTENT_CACHE instead
# of silently selecting an empty checkout-local directory.
export BASE_NAME=${BASE_NAME:-agentic_vbench_sc_$(date -u +%Y%m%d-%H%M%S)}
export WANDB_RUN_NAME=${WANDB_RUN_NAME:-$BASE_NAME}
exec bash "$script_dir/launch_sc.sh" "$@"
