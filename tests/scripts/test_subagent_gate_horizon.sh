#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail

repo_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
# Snapshot creation hashes hundreds of MiB across nested repositories. Keep the
# disposable fixture on node-local storage even when the caller exports a
# shared-filesystem TMPDIR (Codex and some Slurm shells do this by default).
tmp=$(mktemp -d /tmp/nemorl-subagent-gate.XXXXXX)
trap 'rm -rf "$tmp"' EXIT
super_checkpoint=/lustre/fsw/portfolios/nemotron/users/ehosseiniasl/checkpoints/super35-journey-mopd2-identity-upsampling-from-my-step30-yifuw-001_boosted_mtp
container=/lustre/fsw/portfolios/nemotron/users/arushig/images/nemo-rl-cc-pr4200.sqsh
agentic_data=$repo_dir/data/agentic_vbench/train/assembly_first4_x4.jsonl
claw_data=$repo_dir/data/claweval/train/opencode_v3_all_619.jsonl
agentic_delegation='contract-test agentic delegation prompt'
agentic_subagent='contract-test agentic subagent prompt'
claw_delegation='contract-test claw delegation prompt'
claw_subagent='contract-test claw subagent prompt'

# Follow the real wrapper chain but stop immediately before benchmark preflight.
# This captures non-secret contract fields without submitting a Slurm job.
cat >"$tmp/bash" <<'FAKE_BASH'
#!/bin/bash
set -euo pipefail
target=$1
if [[ $target == */run_sc.sh ]]; then
  shift
  for variable in \
    MAX_STEPS SLURM_ACCOUNT SLURM_QOS SLURM_TIME_LIMIT JOB_CYCLES \
    MIN_SLURM_TIME_SECONDS MIN_CHECKPOINT_BUDGET_SECONDS \
    REQUIRE_RESUMABLE_JOB_CYCLES NUM_NODES GPUS_PER_NODE NUM_GEN_NODES \
    SEGMENT_SIZE MODEL_PATH CONFIG_IN_CONTAINER SAVE_PERIOD \
    CHECKPOINT_KEEP_TOP_K RESULTS_DIR CHECKPOINT_ROOT SYNC_GATE_ATTESTATION \
    SYNC_DCP_MANIFEST_SHA256 CONTEXT_COMPACTION \
    AGENTIC_VBENCH_COMPACTION_AUTO AGENTIC_VBENCH_COMPACTION_PRUNE \
    AGENTIC_VBENCH_TRAINING_MODE AGENTIC_VBENCH_SUBAGENTS_ENABLED \
    CLAW_EVAL_COMPACTION_AUTO CLAW_EVAL_COMPACTION_PRUNE \
    CLAW_EVAL_TRAINING_MODE CLAW_EVAL_SUBAGENTS_ENABLED WANDB_PROJECT \
    NEMO_RL_TMPE_OUTLIER_THRESHOLD RUN_TMPE_OUTLIER_TESTS \
    NRL_ROUTER_REPLAY_VALIDATE NRL_R3_TRACE IMMUTABLE_SOURCE_SNAPSHOT \
    REQUIRE_FRESH_PROVENANCE TRAIN_PATH VAL_PATH; do
    printf '%s=%s\n' "$variable" "${!variable:-unset}"
  done
  [[ -n ${AGENTIC_VBENCH_DELEGATION_PROMPT:-${CLAW_EVAL_DELEGATION_PROMPT:-}} ]] &&
    printf 'DELEGATION_PROMPT_SET=true\n'
  [[ -n ${AGENTIC_VBENCH_SUBAGENT_PROMPT:-${CLAW_EVAL_SUBAGENT_PROMPT:-}} ]] &&
    printf 'SUBAGENT_PROMPT_SET=true\n'
  for arg do printf 'ARG=%s\n' "$arg"; done
  exit 0
fi
exec /bin/bash "$@"
FAKE_BASH
chmod +x "$tmp/bash"

# Generic wrapper tests use a schema-complete fake report. Exact production
# launchers reject this substitution and pin the canonical auditor by hash.
cat >"$tmp/fake_health_auditor.py" <<'FAKE_AUDITOR'
#!/usr/bin/env python3
import json
import sys
from pathlib import Path

required = {
    "completed_optimizer_steps",
    "finite_training_metrics",
    "reward_learning_signal",
    "optimizer_update_applied",
    "clean_step_admission_metrics",
    "post_admission_role_training",
    "raw_tmpe_bound",
    "no_tmpe_masking",
    "diagnostic_step_alignment",
    "tmpe_diagnostics",
    "diagnostic_aggregate_correlation",
    "learner_evaluated_parent_and_child_roles",
    "rollout_parent_and_child_roles",
    "finite_episode_rewards",
    "episode_integrity",
    "episode_failures",
    "finalized_checkpoint",
    "tmpe_masking_config",
    "no_empty_rejection_poison_warnings",
}
print(json.dumps({
    "schema_version": 2,
    "audit_mode": "read_only",
    "result_directory": str(Path(sys.argv[1]).resolve()),
    "requirements": {
        "minimum_completed_optimizer_steps": 20,
        "raw_tmpe_upper_bound": 1.03,
    },
    "overall_status": "PASS",
    "checks": [{"name": name, "status": "PASS"} for name in sorted(required)],
}))
FAKE_AUDITOR
export SYNC_HEALTH_AUDITOR=$tmp/fake_health_auditor.py

# Remote ClawEval must fail before dataset construction when its backend is not
# live or its Docker client is absent.
if CLAW_EVAL_BACKEND=remote CLAW_EVAL_BACKEND_DIR="$tmp/dead-claw-backend" \
  /bin/bash "$repo_dir/scripts/claweval/run_sc.sh" >"$tmp/dead-claw.out" 2>&1; then
  echo 'ClawEval accepted a missing remote backend client.env' >&2
  exit 1
fi
grep -Fq 'live ClawEval backend missing:' "$tmp/dead-claw.out"
mkdir -p "$tmp/incomplete-claw-backend" "$tmp/no-docker-cli"
printf 'DOCKER_HOST=tcp://invalid.example:1\n' >"$tmp/incomplete-claw-backend/client.env"
if CLAW_EVAL_BACKEND=remote CLAW_EVAL_BACKEND_DIR="$tmp/incomplete-claw-backend" \
  CLAW_EVAL_DOCKER_CLI_DIR="$tmp/no-docker-cli" \
  /bin/bash "$repo_dir/scripts/claweval/run_sc.sh" >"$tmp/incomplete-claw.out" 2>&1; then
  echo 'ClawEval accepted a remote backend without its Docker client' >&2
  exit 1
fi
grep -Fq 'ClawEval Docker client missing:' "$tmp/incomplete-claw.out"

create_fixture() {
  local root=$1 benchmark=$2 config=$3 train_data=$4 delegation=$5 subagent=$6
  local snapshot=$root/runtime_source manifest=$root/runtime_source_manifest.json
  mkdir -p "$root"
  python3 "$repo_dir/tools/runtime_source_snapshot.py" \
    --create --repo "$repo_dir" --destination "$snapshot" --manifest "$manifest" \
    >/dev/null
  python3 - "$root" "$benchmark" "$config" "$train_data" "$delegation" \
    "$subagent" "$repo_dir" "$super_checkpoint" "$container" <<'FIXTURE_PY'
import hashlib
import json
import pickle
import subprocess
import sys
from pathlib import Path

(
    raw_root,
    benchmark,
    config,
    raw_train_data,
    delegation,
    subagent,
    raw_repo,
    raw_model,
    raw_container,
) = sys.argv[1:]
root = Path(raw_root).resolve()
repo = Path(raw_repo).resolve()
model = str(Path(raw_model).resolve())
container = Path(raw_container).resolve()
train_data = Path(raw_train_data).resolve()
snapshot = root / "runtime_source"
manifest_path = root / "runtime_source_manifest.json"
manifest = json.loads(manifest_path.read_text())


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def identity(path: Path, *, hashed: bool = True) -> dict[str, object]:
    path = path.resolve()
    stat = path.stat()
    result = {"path": str(path), "size_bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns}
    if hashed:
        result["sha256"] = sha256(path)
    return result


contracts = {
    "agentic_vbench": {
        "num_prompts": 4,
        "max_turns": 32,
        "task_selection": "assembly",
        "train_limit": 4,
        "train_repeats": 4,
    },
    "claweval": {
        "num_prompts": 8,
        "max_turns": 40,
        "task_selection": "all",
        "train_limit": None,
        "train_repeats": 1,
    },
}
contract = contracts[benchmark]
source_commit = subprocess.check_output(
    ["git", "-C", str(repo), "rev-parse", "HEAD"], text=True
).strip()
submodule_paths = (
    "3rdparty/Gym-workspace/Gym",
    "3rdparty/Megatron-Bridge-workspace/Megatron-Bridge",
    "3rdparty/Megatron-Bridge-workspace/Megatron-Bridge/3rdparty/Megatron-LM",
)
submodule_commits = {
    path: subprocess.check_output(
        ["git", "-C", str(repo / path), "rev-parse", "HEAD"], text=True
    ).strip()
    for path in submodule_paths
}

weights = root / "checkpoints" / "step_20" / "policy" / "weights"
iteration = weights / "iter_0000000"
iteration.mkdir(parents=True)
(root / "checkpoints" / "latest_checkpoint_status.json").write_text(
    json.dumps({"last_checkpoint_step": 20, "last_successful_ckpt_save_completion": 1.0})
)
(root / "checkpoints" / "step_20" / "training_info.json").write_text(
    json.dumps({"current_step": 20, "total_steps": 20, "trainer_version": 20})
)
(root / "checkpoints" / "step_20" / "config.yaml").write_text("grpo: {}\n")
(weights / "latest_checkpointed_iteration.txt").write_text("0\n")
(weights / "latest_train_state.pt").write_bytes(b"latest-state")
(iteration / ".metadata").write_bytes(
    pickle.dumps({"storage_data": [{"relative_path": "__0_0.distcp"}]}, protocol=4)
)
(iteration / "__0_0.distcp").write_bytes(b"distributed-payload")
(iteration / "metadata.json").write_text(
    json.dumps({"sharded_backend": "torch_dist", "sharded_backend_version": 1})
)
(iteration / "run_config.yaml").write_text("model: synthetic\n")
(iteration / "train_state.pt").write_bytes(b"train-state")
(root / "training_command.sh").write_text(
    'run_grpo grpo.max_num_steps=1000000 grpo.max_num_epochs=null '
    'checkpointing.save_period=20 checkpointing.keep_top_k=null '
    'token_capture.context_compaction=true '
    'logger.wandb.project="nemorl-v2-subagent-comapction-debug"\n'
)
provenance = {
    "schema_version": 2,
    "benchmark": benchmark,
    "source_commit": source_commit,
    "gym_commit": submodule_commits[submodule_paths[0]],
    "bridge_commit": submodule_commits[submodule_paths[1]],
    "mcore_commit": submodule_commits[submodule_paths[2]],
    "result_directory": str(root),
    "checkpoint_directory": str(root / "checkpoints"),
    "model_path": model,
    "config": config,
    "config_identity": identity(snapshot / config),
    "train_data": identity(train_data),
    "validation_data": identity(train_data),
    "container": identity(container, hashed=False),
    "num_nodes": 8,
    "num_generation_nodes": 4,
    "gpus_per_node": 8,
    "segment_size": 4,
    "num_prompts": contract["num_prompts"],
    "num_generations": 4,
    "max_num_steps": 1000000,
    "max_num_epochs": None,
    "save_period": 20,
    "keep_top_k": None,
    "context_compaction": True,
    "opencode_compaction_auto": False,
    "opencode_compaction_prune": False,
    "subagents_enabled": True,
    "training_mode": "subagent",
    "wandb_project": "nemorl-v2-subagent-comapction-debug",
    "wandb_cluster_name": "nsc-svg-slurm-1",
    "immutable_source_snapshot": True,
    "runtime_snapshot_directory": str(snapshot),
    "runtime_source_manifest": str(manifest_path),
    "runtime_manifest_sha256": manifest["runtime_manifest_sha256"],
    "max_turns": contract["max_turns"],
    "context_tokens": 262144,
    "output_tokens": 8192,
    "tool_output_max_lines": 400,
    "tool_output_max_bytes": 8192,
    "max_train_row_tokens": 65536,
    "task_selection": contract["task_selection"],
    "train_limit": contract["train_limit"],
    "train_repeats": contract["train_repeats"],
    "delegation_prompt_sha256": hashlib.sha256(delegation.encode()).hexdigest(),
    "subagent_prompt_sha256": hashlib.sha256(subagent.encode()).hexdigest(),
}
(root / "launch_provenance.json").write_text(json.dumps(provenance))
FIXTURE_PY
}

agentic_config=examples/configs/recipes/vlm/vlm_grpo-nemotron-super-omni-120ba12b-16n8g-megatron-agentic-vbench-sc-256k.v1.yaml
claw_config=examples/configs/recipes/vlm/vlm_grpo-nemotron-super-omni-120ba12b-8n8g-megatron-claweval-sc.v2.yaml
sync_result_dir=$tmp/agentic-sync-result
claw_sync_result_dir=$tmp/claw-sync-result
create_fixture "$sync_result_dir" agentic_vbench "$agentic_config" "$agentic_data" \
  "$agentic_delegation" "$agentic_subagent"
create_fixture "$claw_sync_result_dir" claweval "$claw_config" "$claw_data" \
  "$claw_delegation" "$claw_subagent"
sync_policy_checkpoint=$sync_result_dir/checkpoints/step_20/policy/weights
claw_sync_policy_checkpoint=$claw_sync_result_dir/checkpoints/step_20/policy/weights

export AGENTIC_VBENCH_DELEGATION_PROMPT=$agentic_delegation
export AGENTIC_VBENCH_SUBAGENT_PROMPT=$agentic_subagent
export CLAW_EVAL_DELEGATION_PROMPT=$claw_delegation
export CLAW_EVAL_SUBAGENT_PROMPT=$claw_subagent

assert_horizon() {
  local wrapper=$1 output=$2 donor=$3
  AGENTIC_VBENCH_BACKEND_DIR=/dedicated-agentic-backend \
    CLAW_EVAL_BACKEND_DIR=/dedicated-claw-backend \
    SYNC_POLICY_CHECKPOINT="$donor" MODEL_PATH="$super_checkpoint" \
    PATH="$tmp:$PATH" /bin/bash "$repo_dir/$wrapper" \
    grpo.max_num_steps=20 grpo.max_num_epochs=1 >"$output"
  grep -Fqx 'MAX_STEPS=1000000' "$output"
  grep -Fqx 'ARG=grpo.max_num_steps=1000000' "$output"
  grep -Fqx 'ARG=grpo.max_num_epochs=null' "$output"
  if MAX_STEPS=20 AGENTIC_VBENCH_BACKEND_DIR=/dedicated-agentic-backend \
    CLAW_EVAL_BACKEND_DIR=/dedicated-claw-backend \
    SYNC_POLICY_CHECKPOINT="$donor" MODEL_PATH="$super_checkpoint" \
    PATH="$tmp:$PATH" /bin/bash "$repo_dir/$wrapper" >/dev/null 2>&1; then
    echo "$wrapper accepted a 20-step production horizon" >&2
    exit 1
  fi
}

assert_horizon scripts/agentic_vbench/debug_subagents_sync_8n.sh "$tmp/agentic-sync.out" "$sync_policy_checkpoint"
assert_horizon scripts/agentic_vbench/debug_subagents_async_16n.sh "$tmp/agentic-async.out" "$sync_policy_checkpoint"
assert_horizon scripts/claweval/train_subagents_sync_8n.sh "$tmp/claw-sync.out" "$claw_sync_policy_checkpoint"
assert_horizon scripts/claweval/train_subagents_async_16n.sh "$tmp/claw-async.out" "$claw_sync_policy_checkpoint"

for output in "$tmp/agentic-sync.out" "$tmp/agentic-async.out" "$tmp/claw-sync.out" "$tmp/claw-async.out"; do
  grep -Fqx 'SLURM_ACCOUNT=nemotron_n4_omni' "$output"
  grep -Fqx 'SLURM_QOS=normal' "$output"
  grep -Fqx 'SLURM_TIME_LIMIT=24:00:00' "$output"
  grep -Fqx 'JOB_CYCLES=6' "$output"
  grep -Fqx 'MAX_STEPS=1000000' "$output"
done
grep -Fqx "ARG=+checkpointing.pretrained_checkpoint.path=$sync_policy_checkpoint" "$tmp/agentic-async.out"
grep -Fqx "ARG=+checkpointing.pretrained_checkpoint.path=$claw_sync_policy_checkpoint" "$tmp/claw-async.out"
grep -Eq '^SYNC_DCP_MANIFEST_SHA256=[0-9a-f]{64}$' "$tmp/agentic-async.out"
grep -Eq '^SYNC_DCP_MANIFEST_SHA256=[0-9a-f]{64}$' "$tmp/claw-async.out"

# Exact Gate 1 launchers own all critical identities and reject injections.
unset SYNC_HEALTH_AUDITOR AGENTIC_VBENCH_DELEGATION_PROMPT \
  AGENTIC_VBENCH_SUBAGENT_PROMPT CLAW_EVAL_DELEGATION_PROMPT \
  CLAW_EVAL_SUBAGENT_PROMPT
BASE_NAME=agentic-exact-contract-$$ PATH="$tmp:$PATH" /bin/bash \
  "$repo_dir/scripts/agentic_vbench/launch_subagents_super_sync20.sh" \
  >"$tmp/agentic-exact-sync.out"
CLAW_EVAL_BACKEND_DIR=/dedicated-claw-backend BASE_NAME=claw-exact-contract-$$ \
  PATH="$tmp:$PATH" /bin/bash \
  "$repo_dir/scripts/claweval/launch_subagents_super_sync20.sh" \
  >"$tmp/claw-exact-sync.out"
for output in "$tmp/agentic-exact-sync.out" "$tmp/claw-exact-sync.out"; do
  grep -Fqx 'IMMUTABLE_SOURCE_SNAPSHOT=true' "$output"
  grep -Fqx 'REQUIRE_FRESH_PROVENANCE=true' "$output"
  grep -Fqx 'WANDB_PROJECT=nemorl-v2-subagent-comapction-debug' "$output"
  grep -Fqx 'SAVE_PERIOD=20' "$output"
  grep -Fqx 'CHECKPOINT_KEEP_TOP_K=null' "$output"
done
grep -Fqx "TRAIN_PATH=$agentic_data" "$tmp/agentic-exact-sync.out"
grep -Fqx "TRAIN_PATH=$claw_data" "$tmp/claw-exact-sync.out"

if /bin/bash "$repo_dir/scripts/agentic_vbench/launch_subagents_super_sync20.sh" \
  grpo.max_num_steps=20 >/dev/null 2>&1; then
  echo 'Exact Super sync20 launcher accepted a caller Hydra override' >&2
  exit 1
fi
if RESULTS_DIR=/wrong /bin/bash \
  "$repo_dir/scripts/agentic_vbench/launch_subagents_super_sync20.sh" \
  >/dev/null 2>&1; then
  echo 'Exact Super sync20 launcher accepted a caller-owned result directory' >&2
  exit 1
fi

# Exact Gate 2 must reject a substituted auditor even when the fake emits PASS.
if SYNC_HEALTH_AUDITOR=$tmp/fake_health_auditor.py \
  AGENTIC_VBENCH_BACKEND_DIR=/dedicated-agentic-backend \
  SYNC_POLICY_CHECKPOINT="$sync_policy_checkpoint" /bin/bash \
  "$repo_dir/scripts/agentic_vbench/launch_subagents_super_async16.sh" \
  >/dev/null 2>&1; then
  echo 'Exact Super async16 launcher accepted a substituted health auditor' >&2
  exit 1
fi

if CHECKPOINT_DIR=/wrong-resume AGENTIC_VBENCH_BACKEND_DIR=/dedicated-agentic-backend \
  SYNC_POLICY_CHECKPOINT="$sync_policy_checkpoint" MODEL_PATH="$super_checkpoint" \
  PATH="$tmp:$PATH" /bin/bash \
  "$repo_dir/scripts/agentic_vbench/debug_subagents_async_16n.sh" \
  >/dev/null 2>&1; then
  echo 'Agentic async accepted an externally selected resume directory' >&2
  exit 1
fi
if AGENTIC_VBENCH_BACKEND_DIR=/dedicated-agentic-backend \
  SYNC_POLICY_CHECKPOINT="$sync_policy_checkpoint" MODEL_PATH="$super_checkpoint" \
  PATH="$tmp:$PATH" /bin/bash "$repo_dir/scripts/agentic_vbench/debug_subagents_async_16n.sh" \
  checkpointing.checkpoint_dir=/wrong-resume >/dev/null 2>&1; then
  echo 'Agentic async accepted a caller checkpoint-directory override' >&2
  exit 1
fi

assert_rejects_bad_donor() {
  local wrapper=$1 backend_assignment=$2 donor=$3
  if env "$backend_assignment" SYNC_POLICY_CHECKPOINT="$donor" \
    MODEL_PATH="$super_checkpoint" PATH="$tmp:$PATH" /bin/bash \
    "$repo_dir/$wrapper" >/dev/null 2>&1; then
    echo "$wrapper accepted invalid sync policy donor: $donor" >&2
    exit 1
  fi
}

wrong_step_donor=$tmp/step_19/policy/weights
mkdir -p "$wrong_step_donor/iter_0000000"
printf '0\n' >"$wrong_step_donor/latest_checkpointed_iteration.txt"
printf '{}\n' >"$wrong_step_donor/iter_0000000/run_config.yaml"
missing_tracker=$tmp/missing-tracker/step_20/policy/weights
mkdir -p "$missing_tracker/iter_0000000"
printf '{}\n' >"$missing_tracker/iter_0000000/run_config.yaml"
for wrapper_and_backend in \
  'scripts/agentic_vbench/debug_subagents_async_16n.sh AGENTIC_VBENCH_BACKEND_DIR=/dedicated-agentic-backend' \
  'scripts/claweval/train_subagents_async_16n.sh CLAW_EVAL_BACKEND_DIR=/dedicated-claw-backend'; do
  read -r wrapper backend_assignment <<<"$wrapper_and_backend"
  assert_rejects_bad_donor "$wrapper" "$backend_assignment" "$tmp/does-not-exist"
  assert_rejects_bad_donor "$wrapper" "$backend_assignment" "$wrong_step_donor"
  assert_rejects_bad_donor "$wrapper" "$backend_assignment" "$missing_tracker"
done

recipe=$repo_dir/examples/configs/recipes/vlm/vlm_grpo-nemotron-super-omni-120ba12b-16n8g-megatron-agentic-vbench-sc-256k.v1.yaml
grep -Fqx '  max_num_epochs: null' "$recipe"
grep -Fqx '  max_num_steps: 1000000' "$recipe"

echo 'subagent gate horizon and handoff contracts: PASS'
