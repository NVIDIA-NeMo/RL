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
  printf 'SLURM_ACCOUNT=%s\n' "${SLURM_ACCOUNT:-unset}"
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

sync_policy_checkpoint=$tmp/sync-policy
mkdir -p "$sync_policy_checkpoint/iter_0000000"
printf '0\n' >"$sync_policy_checkpoint/latest_checkpointed_iteration.txt"
printf '{}\n' >"$sync_policy_checkpoint/iter_0000000/run_config.yaml"

assert_horizon() {
  local wrapper=$1 output=$2 actual
  AGENTIC_VBENCH_BACKEND_DIR=/dedicated-agentic-backend \
    CLAW_EVAL_BACKEND_DIR=/dedicated-claw-backend \
    SYNC_POLICY_CHECKPOINT="$sync_policy_checkpoint" \
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

  if MAX_STEPS=20 AGENTIC_VBENCH_BACKEND_DIR=/dedicated-agentic-backend \
    CLAW_EVAL_BACKEND_DIR=/dedicated-claw-backend \
    SYNC_POLICY_CHECKPOINT="$sync_policy_checkpoint" \
    MODEL_PATH=/checkpoint PATH="$tmp:$PATH" /bin/bash \
    "$repo_dir/$wrapper" >/dev/null 2>&1; then
    echo "$wrapper accepted a 20-step production horizon" >&2
    return 1
  fi
}

assert_horizon scripts/agentic_vbench/debug_subagents_sync_8n.sh "$tmp/agentic-sync.out"
assert_horizon scripts/agentic_vbench/debug_subagents_async_16n.sh "$tmp/agentic-async.out"
assert_horizon scripts/claweval/train_subagents_sync_8n.sh "$tmp/claw-sync.out"
assert_horizon scripts/claweval/train_subagents_async_16n.sh "$tmp/claw-async.out"

if env -u AGENTIC_VBENCH_BACKEND_DIR \
  SYNC_POLICY_CHECKPOINT="$sync_policy_checkpoint" \
  MODEL_PATH=/checkpoint PATH="$tmp:$PATH" /bin/bash \
  "$repo_dir/scripts/agentic_vbench/debug_subagents_async_16n.sh" \
  >/dev/null 2>&1; then
  echo 'Agentic async accepted an implicit/default backend' >&2
  exit 1
fi
if env -u CLAW_EVAL_BACKEND_DIR \
  SYNC_POLICY_CHECKPOINT="$sync_policy_checkpoint" \
  MODEL_PATH=/checkpoint PATH="$tmp:$PATH" /bin/bash \
  "$repo_dir/scripts/claweval/train_subagents_async_16n.sh" \
  >/dev/null 2>&1; then
  echo 'ClawEval async accepted an implicit/default backend' >&2
  exit 1
fi

if env -u SYNC_POLICY_CHECKPOINT \
  AGENTIC_VBENCH_BACKEND_DIR=/dedicated-agentic-backend \
  MODEL_PATH=/checkpoint PATH="$tmp:$PATH" /bin/bash \
  "$repo_dir/scripts/agentic_vbench/debug_subagents_async_16n.sh" \
  >/dev/null 2>&1; then
  echo 'Agentic async accepted a missing sync policy donor' >&2
  exit 1
fi

assert_rejects_bad_donor() {
  local wrapper=$1 backend_assignment=$2 donor=$3
  if env "$backend_assignment" SYNC_POLICY_CHECKPOINT="$donor" \
    MODEL_PATH=/checkpoint PATH="$tmp:$PATH" /bin/bash \
    "$repo_dir/$wrapper" >/dev/null 2>&1; then
    echo "$wrapper accepted invalid sync policy donor: $donor" >&2
    exit 1
  fi
}

missing_tracker=$tmp/sync-policy-missing-tracker
mkdir -p "$missing_tracker/iter_0000000"
printf '{}\n' >"$missing_tracker/iter_0000000/run_config.yaml"
empty_tracker=$tmp/sync-policy-empty-tracker
mkdir -p "$empty_tracker/iter_0000000"
: >"$empty_tracker/latest_checkpointed_iteration.txt"
printf '{}\n' >"$empty_tracker/iter_0000000/run_config.yaml"
malformed_tracker=$tmp/sync-policy-malformed-tracker
mkdir -p "$malformed_tracker/iter_0000000"
printf 'not-an-iteration\n' >"$malformed_tracker/latest_checkpointed_iteration.txt"
printf '{}\n' >"$malformed_tracker/iter_0000000/run_config.yaml"
missing_tracked_config=$tmp/sync-policy-missing-tracked-config
mkdir -p "$missing_tracked_config/iter_0000000"
printf '7\n' >"$missing_tracked_config/latest_checkpointed_iteration.txt"
printf '{}\n' >"$missing_tracked_config/iter_0000000/run_config.yaml"

for wrapper_and_backend in \
  'scripts/agentic_vbench/debug_subagents_async_16n.sh AGENTIC_VBENCH_BACKEND_DIR=/dedicated-agentic-backend' \
  'scripts/claweval/train_subagents_async_16n.sh CLAW_EVAL_BACKEND_DIR=/dedicated-claw-backend'; do
  read -r wrapper backend_assignment <<<"$wrapper_and_backend"
  assert_rejects_bad_donor "$wrapper" "$backend_assignment" "$tmp/does-not-exist"
  assert_rejects_bad_donor "$wrapper" "$backend_assignment" "$missing_tracker"
  assert_rejects_bad_donor "$wrapper" "$backend_assignment" "$empty_tracker"
  assert_rejects_bad_donor "$wrapper" "$backend_assignment" "$malformed_tracker"
  assert_rejects_bad_donor "$wrapper" "$backend_assignment" "$missing_tracked_config"
done
if env -u SYNC_POLICY_CHECKPOINT \
  CLAW_EVAL_BACKEND_DIR=/dedicated-claw-backend \
  MODEL_PATH=/checkpoint PATH="$tmp:$PATH" /bin/bash \
  "$repo_dir/scripts/claweval/train_subagents_async_16n.sh" \
  >/dev/null 2>&1; then
  echo 'ClawEval async accepted a missing sync policy donor' >&2
  exit 1
fi

for output in \
  "$tmp/agentic-sync.out" \
  "$tmp/agentic-async.out" \
  "$tmp/claw-sync.out" \
  "$tmp/claw-async.out"; do
  grep -Fqx 'SLURM_ACCOUNT=nemotron_n4_omni' "$output"
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

for output in "$tmp/agentic-async.out" "$tmp/claw-async.out"; do
  grep -Fqx "ARG=+checkpointing.pretrained_checkpoint.path=$sync_policy_checkpoint" "$output"
  grep -Fqx 'ARG=+checkpointing.pretrained_checkpoint.format=megatron_bridge' "$output"
  [[ $(grep -Fc 'ARG=+checkpointing.pretrained_checkpoint.path=' "$output") == 1 ]]
  [[ $(grep -Fc 'ARG=+checkpointing.pretrained_checkpoint.format=' "$output") == 1 ]]
done

for wrapper_and_backend in \
  'scripts/agentic_vbench/debug_subagents_async_16n.sh AGENTIC_VBENCH_BACKEND_DIR=/dedicated-agentic-backend' \
  'scripts/claweval/train_subagents_async_16n.sh CLAW_EVAL_BACKEND_DIR=/dedicated-claw-backend'; do
  read -r wrapper backend_assignment <<<"$wrapper_and_backend"
  if env "$backend_assignment" SYNC_POLICY_CHECKPOINT="$sync_policy_checkpoint" \
    MODEL_PATH=/checkpoint PATH="$tmp:$PATH" /bin/bash "$repo_dir/$wrapper" \
    +checkpointing.pretrained_checkpoint.path=/wrong-caller-donor \
    >/dev/null 2>&1; then
    echo "$wrapper accepted a caller-supplied donor override" >&2
    exit 1
  fi
done
if grep -Fq 'checkpointing.pretrained_checkpoint' "$tmp/agentic-sync.out" ||
  grep -Fq 'checkpointing.pretrained_checkpoint' "$tmp/claw-sync.out"; then
  echo 'A sync gate unexpectedly received the async donor override' >&2
  exit 1
fi

recipe=$repo_dir/examples/configs/recipes/vlm/vlm_grpo-nemotron-super-omni-120ba12b-16n8g-megatron-agentic-vbench-sc-256k.v1.yaml
grep -Fqx '  max_num_epochs: null' "$recipe"
grep -Fqx '  max_num_steps: 1000000' "$recipe"

echo 'subagent gate horizon contracts: PASS'
