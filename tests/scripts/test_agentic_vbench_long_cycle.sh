#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail

repo_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT

# Exercise the real launcher in a minimal synthetic checkout. This keeps the
# contract test independent of whether optional git submodules were initialized
# in the caller's worktree; DRY_RUN never executes these placeholder sources.
fixture=$tmp/repo
mkdir -p \
  "$fixture/scripts/agentic_vbench" \
  "$fixture/examples" \
  "$fixture/3rdparty/Gym-workspace/Gym/responses_api_agents/agentic_vbench_agent" \
  "$fixture/3rdparty/Megatron-Bridge-workspace/Megatron-Bridge/3rdparty/Megatron-LM"
cp "$repo_dir/scripts/agentic_vbench/launch_sc.sh" "$fixture/scripts/agentic_vbench/launch_sc.sh"
printf '# placeholder\n' >"$fixture/recipe.yaml"
printf '# placeholder\n' >"$fixture/examples/run_grpo_single_controller.py"
printf '# placeholder\n' >"$fixture/3rdparty/Gym-workspace/Gym/responses_api_agents/agentic_vbench_agent/app.py"
printf '# placeholder\n' >"$fixture/3rdparty/Megatron-Bridge-workspace/Megatron-Bridge/.fixture"
printf '# placeholder\n' >"$fixture/3rdparty/Megatron-Bridge-workspace/Megatron-Bridge/3rdparty/Megatron-LM/.fixture"
printf '#!/usr/bin/env bash\n' >"$fixture/ray.sub"

model=$tmp/model
mkdir -p "$model"
printf '{}\n' >"$model/config.json"
printf '{"weight_map":{"dummy":"model-00001-of-00001.safetensors"}}\n' >"$model/model.safetensors.index.json"
printf 'x\n' >"$model/model-00001-of-00001.safetensors"
printf '{}\n' >"$model/tokenizer.json"
printf '{}\n' >"$model/processor_config.json"
printf '{}\n' >"$tmp/train.jsonl"

run_dry() {
  local results=$1
  shift
  env \
    DRY_RUN=1 \
    MODEL_PATH="$model" \
    TRAIN_PATH="$tmp/train.jsonl" \
    VAL_PATH="$tmp/train.jsonl" \
    RESULTS_DIR="$results" \
    BASE_NAME=agentic_vbench_long_cycle_contract \
    CONFIG_IN_CONTAINER=recipe.yaml \
    MAX_STEPS=1000000 \
    SLURM_QOS=normal \
    SLURM_TIME_LIMIT=7-00:00:00 \
    MIN_SLURM_TIME_SECONDS=86400 \
    MIN_CHECKPOINT_BUDGET_SECONDS=86400 \
    REQUIRE_RESUMABLE_JOB_CYCLES=true \
    JOB_CYCLES=6 \
    "$@" \
    bash "$fixture/scripts/agentic_vbench/launch_sc.sh"
}

run_dry "$tmp/results" >"$tmp/output"

grep -Fqx 'time_limit=7-00:00:00 checkpoint_must_save_by=06:23:40:00' "$tmp/output"
grep -Fqx 'allocation_boundary_seconds=604800 controller_boundary_seconds=603600 grpo_horizon_steps=1000000 continuation_array=0-5%1' "$tmp/output"
grep -F -- '--time=7-00:00:00' "$tmp/output" >/dev/null
grep -F -- '--array=0-5%1' "$tmp/output" >/dev/null
grep -F -- '--qos=normal' "$tmp/output" >/dev/null
grep -F 'grpo.max_num_steps="1000000"' "$tmp/results/training_command.sh" >/dev/null
grep -F 'grpo.max_num_epochs=null' "$tmp/results/training_command.sh" >/dev/null
grep -F 'checkpointing.checkpoint_must_save_by="06:23:40:00"' "$tmp/results/training_command.sh" >/dev/null

if run_dry "$tmp/too-short" SLURM_TIME_LIMIT=02:00:00 >/dev/null 2>&1; then
  echo 'launcher accepted a cycle below MIN_SLURM_TIME_SECONDS' >&2
  exit 1
fi

if run_dry "$tmp/short-controller" CHECKPOINT_MUST_SAVE_BY=00:01:40:00 >/dev/null 2>&1; then
  echo 'launcher accepted a controller boundary below MIN_CHECKPOINT_BUDGET_SECONDS' >&2
  exit 1
fi

if run_dry "$tmp/no-continuation" JOB_CYCLES=1 >/dev/null 2>&1; then
  echo 'launcher accepted a non-resumable one-cycle production launch' >&2
  exit 1
fi

echo 'agentic-vbench long-cycle contracts: PASS'
