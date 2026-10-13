#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Build/validate ClawEval prompt-only data and launch an effectively uncapped
# SingleController GRPO job through the shared Harbor/OpenCode transport.
set -euo pipefail

script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
code_dir=$(cd "$script_dir/../.." && pwd)

export CLAW_EVAL_SOURCE=${CLAW_EVAL_SOURCE:-/scratch/fsw/portfolios/nemotron/projects/nemotron_n4_omni/users/arushig/claw-eval}
export CLAW_EVAL_ORIGINAL_TASK_ROOT=${CLAW_EVAL_ORIGINAL_TASK_ROOT:-$CLAW_EVAL_SOURCE/data_generation/runs/kimi_video_scaled_100/generated/tasks}
export CLAW_EVAL_BRIEF_TASK_ROOT=${CLAW_EVAL_BRIEF_TASK_ROOT:-$CLAW_EVAL_SOURCE/data_generation/runs/kimi_video_scaled_100_brief_eval/generated/tasks}
export CLAW_EVAL_B_TASK_ROOT=${CLAW_EVAL_B_TASK_ROOT:-$CLAW_EVAL_SOURCE/data_generation/runs/kimi_video_brief_1000/generated/tasks}
export CLAW_EVAL_SFT=${CLAW_EVAL_SFT:-/scratch/fsw/portfolios/nemotron/projects/nemotron_n4_omni/users/ehosseiniasl/github_repos/ehosseiniasl_vlm2_claweval/data/claw_kimi_all_passing_631_opencode_v3/energon_sft.jsonl}
export CLAW_EVAL_DATA_ROOT=${CLAW_EVAL_DATA_ROOT:-$code_dir/data/claweval}
export CLAW_EVAL_HARBOR_PYTHON=${CLAW_EVAL_HARBOR_PYTHON:-$code_dir/data/agentic_vbench/harbor-venv/bin/python}
export CLAW_EVAL_DOCKER_CLI_DIR=${CLAW_EVAL_DOCKER_CLI_DIR:-$code_dir/data/agentic_vbench/docker-client}
export CLAW_EVAL_BACKEND=${CLAW_EVAL_BACKEND:-remote}
export CLAW_EVAL_BACKEND_DIR=${CLAW_EVAL_BACKEND_DIR:-${AGENTIC_VBENCH_BACKEND_DIR:-}}
export CLAW_EVAL_CREDENTIALS_FILE=${CLAW_EVAL_CREDENTIALS_FILE:-/lustre/fsw/portfolios/nemotron/users/ehosseiniasl/codex/credentials.env}

context_compaction=${CONTEXT_COMPACTION:-false}
case $context_compaction in true|false) ;; *) echo "ERROR: CONTEXT_COMPACTION must be true or false" >&2; exit 1;; esac
subagents=${CLAW_EVAL_SUBAGENTS_ENABLED:-false}
case $subagents in true|false) ;; *) echo "ERROR: CLAW_EVAL_SUBAGENTS_ENABLED must be true or false" >&2; exit 1;; esac
# Pin OpenCode explicitly. Leaving these null keeps OpenCode's own defaults on,
# which is unsafe for append-only and subagent capture. Framework-owned context
# reconstruction remains controlled independently by CONTEXT_COMPACTION.
export CLAW_EVAL_COMPACTION_AUTO=${CLAW_EVAL_COMPACTION_AUTO:-$context_compaction}
export CLAW_EVAL_COMPACTION_PRUNE=${CLAW_EVAL_COMPACTION_PRUNE:-false}
if [[ $subagents == true ]]; then
  export CLAW_EVAL_TRAINING_MODE=${CLAW_EVAL_TRAINING_MODE:-subagent}
  [[ $CLAW_EVAL_TRAINING_MODE == subagent ]] || { echo "ERROR: subagents require CLAW_EVAL_TRAINING_MODE=subagent" >&2; exit 1; }
  [[ $context_compaction == true ]] || { echo "ERROR: subagent trace training requires CONTEXT_COMPACTION=true" >&2; exit 1; }
  [[ $CLAW_EVAL_COMPACTION_AUTO == false && $CLAW_EVAL_COMPACTION_PRUNE == false ]] || {
    echo "ERROR: subagent trace training requires OpenCode compaction and pruning off" >&2
    exit 1
  }
fi

for required in "$CLAW_EVAL_SOURCE" "$CLAW_EVAL_ORIGINAL_TASK_ROOT" "$CLAW_EVAL_BRIEF_TASK_ROOT" "$CLAW_EVAL_B_TASK_ROOT" "$CLAW_EVAL_SFT"; do
  [[ -e $required ]] || { echo "ERROR: missing ClawEval input: $required" >&2; exit 1; }
done
[[ -x $CLAW_EVAL_HARBOR_PYTHON ]] || { echo "ERROR: Harbor Python missing: $CLAW_EVAL_HARBOR_PYTHON" >&2; exit 1; }
[[ -s $CLAW_EVAL_CREDENTIALS_FILE ]] || { echo "ERROR: credentials file missing: $CLAW_EVAL_CREDENTIALS_FILE" >&2; exit 1; }
set -a
# shellcheck disable=SC1090
source "$CLAW_EVAL_CREDENTIALS_FILE"
set +a
export CLAW_EVAL_JUDGE_BASE_URL=${CLAW_EVAL_JUDGE_BASE_URL:-https://inference-api.nvidia.com/v1}
export CLAW_EVAL_JUDGE_MODEL=${CLAW_EVAL_JUDGE_MODEL:-gcp/google/gemini-3-flash-preview}
export CLAW_EVAL_JUDGE_API_KEY_ENV=${CLAW_EVAL_JUDGE_API_KEY_ENV:-NV_INFERENCE_API_KEY}
[[ $CLAW_EVAL_JUDGE_API_KEY_ENV =~ ^[A-Za-z_][A-Za-z0-9_]*$ ]] || {
  echo "ERROR: CLAW_EVAL_JUDGE_API_KEY_ENV is not a valid environment-variable name" >&2
  exit 1
}
[[ -n ${!CLAW_EVAL_JUDGE_API_KEY_ENV:-} ]] || {
  echo "ERROR: ClawEval native grading needs $CLAW_EVAL_JUDGE_API_KEY_ENV" >&2
  exit 1
}

export TRAIN_PATH=${TRAIN_PATH:-$CLAW_EVAL_DATA_ROOT/train/opencode_v3_all_619.jsonl}
export VAL_PATH=${VAL_PATH:-$TRAIN_PATH}
if [[ ! -s $TRAIN_PATH ]]; then
  python3 "$script_dir/build_train_dataset.py" \
    --sft "$CLAW_EVAL_SFT" \
    --task-root "original_100=$CLAW_EVAL_ORIGINAL_TASK_ROOT" \
    --task-root "brief_100=$CLAW_EVAL_BRIEF_TASK_ROOT" \
    --task-root "brief_1000=$CLAW_EVAL_B_TASK_ROOT" \
    --output "$TRAIN_PATH"
fi
python3 - "$code_dir/3rdparty/Gym-workspace/Gym" "$CLAW_EVAL_SFT" "$CLAW_EVAL_ORIGINAL_TASK_ROOT" "$CLAW_EVAL_BRIEF_TASK_ROOT" "$CLAW_EVAL_B_TASK_ROOT" "$TRAIN_PATH" <<'PY'
import json
import sys
from pathlib import Path

gym_root, sft, original, brief, brief_1000, train_path = sys.argv[1:]
sys.path.insert(0, gym_root)
from responses_api_agents.claweval_agent.core import dataset_rows, inventory

tasks = inventory({"original_100": original, "brief_100": brief, "brief_1000": brief_1000})
expected = [{**row, "task_source": "claweval_agent"} for row in dataset_rows(sft, tasks)]
with Path(train_path).open() as handle:
    actual = [json.loads(line) for line in handle if line.strip()]
if actual != expected:
    raise SystemExit(
        f"Stale/mismatched ClawEval dataset: {train_path}; "
        "remove it or select a new TRAIN_PATH so it can be regenerated"
    )
print(f"Validated {len(actual)} prompt-only ClawEval rows against all three task-package variants")
PY

# The schema requires a finite integer. One million is a production horizon,
# while launch_sc explicitly disables the dataset-derived epoch clamp.
export MAX_STEPS=${MAX_STEPS:-1000000}
export CONFIG_IN_CONTAINER=${CONFIG_IN_CONTAINER:-examples/configs/recipes/vlm/vlm_grpo-nemotron-super-omni-120ba12b-8n8g-megatron-claweval-sc.v2.yaml}
export TRAINING_DEPS_SETUP_SCRIPT=/opt/nemo-rl/tools/ensure_agentic_vbench_deps.sh
export BASE_NAME=${BASE_NAME:-claweval_sc_$(date -u +%Y%m%d-%H%M%S)}
export WANDB_PROJECT=nemorl-v2-subagent-comapction-debug
export WANDB_ENTITY=${WANDB_ENTITY:-adlr}
export CLAW_EVAL_OUTPUT_ROOT=${CLAW_EVAL_OUTPUT_ROOT:-$code_dir/results/$BASE_NAME/episodes}

# Compatibility aliases used by the shared transport launcher.
export AGENTIC_VBENCH_BACKEND_DIR=$CLAW_EVAL_BACKEND_DIR
export AGENTIC_VBENCH_DOCKER_CLI_DIR=$CLAW_EVAL_DOCKER_CLI_DIR
export AGENTIC_VBENCH_HARBOR_PYTHON=$CLAW_EVAL_HARBOR_PYTHON
export AGENTIC_VBENCH_CREDENTIALS_FILE=$CLAW_EVAL_CREDENTIALS_FILE
export AGENTIC_VBENCH_JUDGE_PROTOCOL=

exec bash "$code_dir/scripts/agentic_vbench/launch_sc.sh" "$@"
