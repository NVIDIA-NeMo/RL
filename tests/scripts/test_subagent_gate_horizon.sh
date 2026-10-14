#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail

repo_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT

# Follow the real wrapper chain but stop immediately before benchmark preflight.
# This records only the non-secret fields needed by the contract test.
cat >"$tmp/bash" <<'FAKE_BASH'
#!/bin/bash
set -euo pipefail
target=$1
if [[ $target == */run_sc.sh ]]; then
  shift
  printf 'MAX_STEPS=%s\n' "${MAX_STEPS:-unset}"
  printf 'SLURM_QOS=%s\n' "${SLURM_QOS:-unset}"
  printf 'SLURM_TIME_LIMIT=%s\n' "${SLURM_TIME_LIMIT:-unset}"
  printf 'JOB_CYCLES=%s\n' "${JOB_CYCLES:-unset}"
  printf 'MIN_SLURM_TIME_SECONDS=%s\n' "${MIN_SLURM_TIME_SECONDS:-unset}"
  printf 'MIN_CHECKPOINT_BUDGET_SECONDS=%s\n' "${MIN_CHECKPOINT_BUDGET_SECONDS:-unset}"
  printf 'REQUIRE_RESUMABLE_JOB_CYCLES=%s\n' "${REQUIRE_RESUMABLE_JOB_CYCLES:-unset}"
  printf 'NUM_NODES=%s\n' "${NUM_NODES:-unset}"
  printf 'NUM_GEN_NODES=%s\n' "${NUM_GEN_NODES:-unset}"
  printf 'SEGMENT_SIZE=%s\n' "${SEGMENT_SIZE:-unset}"
  printf 'WANDB_PROJECT=%s\n' "${WANDB_PROJECT:-unset}"
  printf 'NRL_ROUTER_REPLAY_VALIDATE=%s\n' "${NRL_ROUTER_REPLAY_VALIDATE:-unset}"
  printf 'NRL_R3_TRACE=%s\n' "${NRL_R3_TRACE:-unset}"
  [[ -n ${AGENTIC_VBENCH_DELEGATION_PROMPT:-} ]] && printf 'DELEGATION_PROMPT_SET=true\n'
  [[ -n ${AGENTIC_VBENCH_SUBAGENT_PROMPT:-} ]] && printf 'SUBAGENT_PROMPT_SET=true\n'
  for arg do printf 'ARG=%s\n' "$arg"; done
  exit 0
fi
exec /bin/bash "$@"
FAKE_BASH
chmod +x "$tmp/bash"

assert_horizon() {
  local wrapper=$1 output=$2 actual
  MODEL_PATH=/checkpoint PATH="$tmp:$PATH" /bin/bash "$repo_dir/$wrapper" \
    grpo.max_num_steps=20 grpo.max_num_epochs=1 >"$output"
  grep -Fqx 'MAX_STEPS=1000000' "$output"
  actual=$(grep '^ARG=grpo.max_num_steps=' "$output" | tail -1 || true)
  [[ $actual == ARG=grpo.max_num_steps=1000000 ]] || {
    echo "$wrapper resolved max_num_steps as: ${actual:-missing}" >&2
    sed -n '/^ARG=/p' "$output" >&2
    return 1
  }
  actual=$(grep '^ARG=grpo.max_num_epochs=' "$output" | tail -1 || true)
  [[ $actual == ARG=grpo.max_num_epochs=null ]] || {
    echo "$wrapper resolved max_num_epochs as: ${actual:-missing}" >&2
    sed -n '/^ARG=/p' "$output" >&2
    return 1
  }

  if MAX_STEPS=20 MODEL_PATH=/checkpoint PATH="$tmp:$PATH" /bin/bash \
    "$repo_dir/$wrapper" >/dev/null 2>&1; then
    echo "$wrapper accepted a 20-step production horizon" >&2
    return 1
  fi
}

assert_horizon scripts/agentic_vbench/debug_subagents_sync_8n.sh "$tmp/agentic-sync.out"
assert_horizon scripts/agentic_vbench/debug_subagents_async_16n.sh "$tmp/agentic-async.out"
assert_horizon scripts/claweval/train_subagents_sync_8n.sh "$tmp/claw-sync.out"
assert_horizon scripts/claweval/train_subagents_async_16n.sh "$tmp/claw-async.out"

for output in \
  "$tmp/agentic-sync.out" \
  "$tmp/agentic-async.out" \
  "$tmp/claw-sync.out" \
  "$tmp/claw-async.out"; do
  grep -Fqx 'SLURM_QOS=normal' "$output"
  grep -Fqx 'SLURM_TIME_LIMIT=24:00:00' "$output"
  grep -Fqx 'JOB_CYCLES=6' "$output"
  grep -Fqx 'MIN_SLURM_TIME_SECONDS=86400' "$output"
  grep -Fqx 'MIN_CHECKPOINT_BUDGET_SECONDS=82800' "$output"
  grep -Fqx 'REQUIRE_RESUMABLE_JOB_CYCLES=true' "$output"
done
grep -Fqx 'NUM_NODES=8' "$tmp/agentic-sync.out"
grep -Fqx 'NUM_GEN_NODES=4' "$tmp/agentic-sync.out"
grep -Fqx 'SEGMENT_SIZE=4' "$tmp/agentic-sync.out"
grep -Fqx 'WANDB_PROJECT=nemorl-v2-subagent-comapction-debug' "$tmp/agentic-sync.out"
grep -Fqx 'NRL_ROUTER_REPLAY_VALIDATE=1' "$tmp/agentic-sync.out"
grep -Fqx 'NRL_R3_TRACE=1' "$tmp/agentic-sync.out"
grep -Fqx 'DELEGATION_PROMPT_SET=true' "$tmp/agentic-sync.out"
grep -Fqx 'SUBAGENT_PROMPT_SET=true' "$tmp/agentic-sync.out"
grep -Fqx 'ARG=+policy.router_replay.enabled=true' "$tmp/agentic-sync.out"
grep -Fqx 'ARG=async_rl.sampler.name=in_order' "$tmp/agentic-sync.out"
grep -Fqx 'ARG=+env.nemo_gym.training_mode=subagent' "$tmp/agentic-sync.out"

recipe=$repo_dir/examples/configs/recipes/vlm/vlm_grpo-nemotron-super-omni-120ba12b-16n8g-megatron-agentic-vbench-sc-256k.v1.yaml
grep -Fqx '  max_num_epochs: null' "$recipe"
grep -Fqx '  max_num_steps: 1000000' "$recipe"

echo 'subagent gate horizon contracts: PASS'
