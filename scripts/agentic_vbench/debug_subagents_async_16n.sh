#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Second production gate for subagent training. Run only after the 8-node sync
# gate keeps every observed raw TMPE value at or below 1.03.
set -euo pipefail

script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
repo_dir=$(cd "$script_dir/../.." && pwd)

: "${AGENTIC_VBENCH_BACKEND_DIR:?Set AGENTIC_VBENCH_BACKEND_DIR to the dedicated live async backend}"
: "${SYNC_POLICY_CHECKPOINT:?Set SYNC_POLICY_CHECKPOINT to the qualified sync step_20/policy/weights directory}"
if [[ -n ${EXTRA_OVERRIDES:-} ]]; then
  echo "ERROR: async qualification requires EXTRA_OVERRIDES to be empty" >&2
  exit 2
fi
for reserved_variable in RESULTS_DIR CHECKPOINT_ROOT CHECKPOINT_DIR TOKEN_CAPTURE_DIR AGENTIC_VBENCH_OUTPUT_ROOT WANDB_RUN_ID; do
  if [[ -n ${!reserved_variable+x} ]]; then
    echo "ERROR: async qualification owns $reserved_variable; unset it before launch" >&2
    exit 2
  fi
done
[[ -d $SYNC_POLICY_CHECKPOINT ]] || {
  echo "ERROR: SYNC_POLICY_CHECKPOINT is not a directory: $SYNC_POLICY_CHECKPOINT" >&2
  exit 2
}
sync_policy_checkpoint_real=$(realpath "$SYNC_POLICY_CHECKPOINT")
case $sync_policy_checkpoint_real in
  */step_20/policy/weights) ;;
  *)
    echo "ERROR: async qualification requires the exact sync step_20/policy/weights donor: $sync_policy_checkpoint_real" >&2
    exit 2
    ;;
esac
SYNC_POLICY_CHECKPOINT=$sync_policy_checkpoint_real
[[ -s $SYNC_POLICY_CHECKPOINT/latest_checkpointed_iteration.txt ]] || {
  echo "ERROR: sync policy donor lacks latest_checkpointed_iteration.txt: $SYNC_POLICY_CHECKPOINT" >&2
  exit 2
}
sync_iteration=
IFS= read -r sync_iteration <"$SYNC_POLICY_CHECKPOINT/latest_checkpointed_iteration.txt" || true
if [[ $sync_iteration == release ]]; then
  sync_iteration_dir=release
elif [[ $sync_iteration =~ ^[0-9]+$ ]]; then
  printf -v sync_iteration_dir 'iter_%07d' "$((10#$sync_iteration))"
else
  echo "ERROR: sync policy donor has an invalid checkpoint tracker: $sync_iteration" >&2
  exit 2
fi
[[ -s $SYNC_POLICY_CHECKPOINT/$sync_iteration_dir/run_config.yaml ]] || {
  echo "ERROR: tracked sync policy iteration lacks run_config.yaml: $SYNC_POLICY_CHECKPOINT/$sync_iteration_dir" >&2
  exit 2
}
for override in "$@"; do
  case $override in
    checkpointing.pretrained_checkpoint* | +checkpointing.pretrained_checkpoint* | ++checkpointing.pretrained_checkpoint* | '~checkpointing.pretrained_checkpoint'*)
      echo "ERROR: the async gate owns checkpointing.pretrained_checkpoint; set only SYNC_POLICY_CHECKPOINT" >&2
      exit 2
      ;;
    checkpointing.checkpoint_dir* | +checkpointing.checkpoint_dir* | ++checkpointing.checkpoint_dir* | '~checkpointing.checkpoint_dir'* | policy.model_name* | +policy.model_name* | ++policy.model_name* | '~policy.model_name'* | cluster.num_nodes* | +cluster.num_nodes* | ++cluster.num_nodes* | '~cluster.num_nodes'* | cluster.gpus_per_node* | +cluster.gpus_per_node* | ++cluster.gpus_per_node* | '~cluster.gpus_per_node'*)
      echo "ERROR: the async gate owns checkpoint/model/topology identity: $override" >&2
      exit 2
      ;;
  esac
done

super_checkpoint=/lustre/fsw/portfolios/nemotron/users/ehosseiniasl/checkpoints/super35-journey-mopd2-identity-upsampling-from-my-step30-yifuw-001_boosted_mtp
export MODEL_PATH=${MODEL_PATH:-$super_checkpoint}
[[ $(realpath "$MODEL_PATH") == $(realpath "$super_checkpoint") ]] || {
  echo "ERROR: async qualification requires the exact Super-GA checkpoint: $super_checkpoint" >&2
  exit 2
}
export GENERATION_TENSOR_PARALLEL_SIZE=${GENERATION_TENSOR_PARALLEL_SIZE:-4}
# Gate 2 keeps the exact Gate 1 prompts; only sampler topology changes.
debug_delegation_prompt="Your first action must be exactly one task tool call before bash read or any other tool. "
debug_delegation_prompt+="Delegate the complete video assembly task to the general subagent. Include the original storyboard slot requirements and require the child to inspect the supplied clips choose every slot create solution.mp4 and create a truthful solution.json before returning. Require solution.json to be a top-level JSON object with a segments list and never a bare JSON list. "
debug_delegation_prompt+="When the child returns do not repeat its visual analysis. Immediately use one bash tool call to verify both output files exist and use Python to assert solution.json parses as a top-level JSON object with a segments list containing only objects. If either artifact is missing or invalid call the general subagent once more with a repair request that forbids new visual inspection and requires immediate creation from existing work in at most four tool calls. "
debug_delegation_prompt+="Keep every response short. Never end while a required deliverable is missing. After verification succeeds return one brief final sentence."
export AGENTIC_VBENCH_DELEGATION_PROMPT=${AGENTIC_VBENCH_DELEGATION_PROMPT:-"${debug_delegation_prompt}"}
debug_subagent_prompt="Complete the delegated AgenticVBench task end to end and create both required deliverables before returning. Begin with tools and do not narrate candidate comparisons. "
debug_subagent_prompt+="Use one batched ffprobe command and one batched ffmpeg -hide_banner -loglevel error -nostats frame extraction workflow while suppressing routine output. Create no more than four labeled overview JPEGs under /workspace/work/subagent_contact_sheets. Each JPEG must be at most 1280 by 720 pixels and each candidate must have four chronological thumbnails with its numeric clip ID. Read every overview exactly once in one parallel read batch. After that read batch never extract or read another frame and never create diagnostic or comparison images. "
debug_subagent_prompt+="Choose the best full clip for every storyboard slot silently. Your very next response after the overview read results must call bash to create /workspace/output/solution.mp4 and /workspace/output/solution.json. Concatenate each selected full clip once. Use source ffprobe durations for truthful contiguous JSON ranges and do not tune timestamps frame counts or sub-frame MP4 duration rounding. The JSON must be a top-level JSON object with a segments list and never a bare JSON list. "
debug_subagent_prompt+="Do not use todowrite and do not retry cosmetic labels. Keep private selection reasoning under 200 words. Use one final bash call to verify both artifacts exist and that solution.json parses. Then return only a brief completion result."
export AGENTIC_VBENCH_SUBAGENT_PROMPT=${AGENTIC_VBENCH_SUBAGENT_PROMPT:-"${debug_subagent_prompt}"}
source_commit=$(git -C "$repo_dir" rev-parse HEAD)
sync_result_dir=$(realpath "$SYNC_POLICY_CHECKPOINT/../../../..")
sync_health_auditor=${SYNC_HEALTH_AUDITOR:-$repo_dir/../../nemo-rl-resources/recipes/subagent/audit_single_controller_health.py}
[[ -s $sync_health_auditor ]] || {
  echo "ERROR: sync health auditor is missing: $sync_health_auditor" >&2
  exit 2
}
actual_auditor_sha256=$(sha256sum "$sync_health_auditor" | cut -d' ' -f1)
if [[ -n ${SYNC_HEALTH_AUDITOR_SHA256:-} && $actual_auditor_sha256 != "$SYNC_HEALTH_AUDITOR_SHA256" ]]; then
  echo "ERROR: sync health auditor hash mismatch: $actual_auditor_sha256" >&2
  exit 2
fi
expected_train_data=${TRAIN_PATH:-$repo_dir/data/agentic_vbench/train/assembly_first4_x4.jsonl}
expected_validation_data=${VAL_PATH:-$expected_train_data}
expected_container=${CONTAINER:-/lustre/fsw/portfolios/nemotron/users/arushig/images/nemo-rl-cc-pr4200.sqsh}
delegation_prompt_sha256=$(printf '%s' "$AGENTIC_VBENCH_DELEGATION_PROMPT" | sha256sum | cut -d' ' -f1)
subagent_prompt_sha256=$(printf '%s' "$AGENTIC_VBENCH_SUBAGENT_PROMPT" | sha256sum | cut -d' ' -f1)
handoff_tmp_dir=$(mktemp -d)
trap 'rm -rf "$handoff_tmp_dir"' EXIT
sync_audit_json=$handoff_tmp_dir/sync_health_audit.json
python3 "$sync_health_auditor" "$sync_result_dir" \
  --min-steps 20 --tmpe-threshold 1.03 --json >"$sync_audit_json"
handoff_json=$handoff_tmp_dir/handoff_attestation.json
python3 "$repo_dir/tools/validate_subagent_sync_handoff.py" \
  --result-dir "$sync_result_dir" \
  --weights-dir "$SYNC_POLICY_CHECKPOINT" \
  --audit-json "$sync_audit_json" \
  --benchmark agentic_vbench \
  --expected-source-commit "$source_commit" \
  --expected-repo-dir "$repo_dir" \
  --expected-model "$super_checkpoint" \
  --expected-train-data "$expected_train_data" \
  --expected-validation-data "$expected_validation_data" \
  --expected-container "$expected_container" \
  --expected-delegation-prompt-sha256 "$delegation_prompt_sha256" \
  --expected-subagent-prompt-sha256 "$subagent_prompt_sha256" \
  --expected-generation-tensor-parallel-size "$GENERATION_TENSOR_PARALLEL_SIZE" \
  --health-auditor-sha256 "$actual_auditor_sha256" \
  --expected-wandb-project nemorl-v2-subagent-comapction-debug \
  --min-steps 20 --tmpe-threshold 1.03 >"$handoff_json"
handoff_dir=$sync_result_dir/handoff_attestations
mkdir -p "$handoff_dir"
handoff_destination=$handoff_dir/agentic_vbench-step20-to-${source_commit}.json
cp "$handoff_json" "$handoff_destination.tmp"
mv "$handoff_destination.tmp" "$handoff_destination"
export SYNC_GATE_ATTESTATION=$handoff_destination
SYNC_DCP_MANIFEST_SHA256=$(python3 - "$handoff_destination" <<'DCP_DIGEST_PY'
import json
import sys
from pathlib import Path

payload = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
print(payload["dcp_sha256_manifest"]["manifest_sha256"])
DCP_DIGEST_PY
)
export SYNC_DCP_MANIFEST_SHA256
export SLURM_ACCOUNT=${SLURM_ACCOUNT:-nemotron_n4_omni}
export CONFIG_IN_CONTAINER=${CONFIG_IN_CONTAINER:-examples/configs/recipes/vlm/vlm_grpo-nemotron-super-omni-120ba12b-16n8g-megatron-agentic-vbench-sc-256k.v1.yaml}
export NUM_NODES=${NUM_NODES:-16}
export NUM_GEN_NODES=${NUM_GEN_NODES:-8}
export SEGMENT_SIZE=${SEGMENT_SIZE:-8}
export NUM_PROMPTS=${NUM_PROMPTS:-8}
export NUM_GENERATIONS=${NUM_GENERATIONS:-4}
export MAX_STEPS=${MAX_STEPS:-1000000}
export TRAIN_REPEATS=${TRAIN_REPEATS:-4}
# Keep the first async comparison on the same four-task causal-gate manifest as
# sync v19. Eight prompt groups draw repeated rows from that unchanged set.
export TRAIN_LIMIT=${TRAIN_LIMIT:-4}
export SAVE_PERIOD=${SAVE_PERIOD:-1}
export CHECKPOINT_KEEP_TOP_K=${CHECKPOINT_KEEP_TOP_K:-2}
export JOB_CYCLES=${JOB_CYCLES:-6}
export SLURM_QOS=${SLURM_QOS:-normal}
export SLURM_TIME_LIMIT=${SLURM_TIME_LIMIT:-24:00:00}
export MIN_SLURM_TIME_SECONDS=${MIN_SLURM_TIME_SECONDS:-86400}
export MIN_CHECKPOINT_BUDGET_SECONDS=${MIN_CHECKPOINT_BUDGET_SECONDS:-82800}
export REQUIRE_RESUMABLE_JOB_CYCLES=${REQUIRE_RESUMABLE_JOB_CYCLES:-true}
export AGENTIC_VBENCH_MAX_TURNS=${AGENTIC_VBENCH_MAX_TURNS:-32}
# Match the qualified sync gate's native OpenCode observation bounds. Full tool
# output remains on disk and verifier semantics are unchanged.
export AGENTIC_VBENCH_TOOL_OUTPUT_MAX_LINES=${AGENTIC_VBENCH_TOOL_OUTPUT_MAX_LINES:-400}
export AGENTIC_VBENCH_TOOL_OUTPUT_MAX_BYTES=${AGENTIC_VBENCH_TOOL_OUTPUT_MAX_BYTES:-8192}
export BASE_NAME=${BASE_NAME:-agentic_vbench_v2_subagents_async_16n_$(date -u +%Y%m%d-%H%M%S)-$$}
[[ $BASE_NAME =~ ^[A-Za-z0-9._-]+$ ]] || {
  echo "ERROR: BASE_NAME must contain only letters, digits, dot, underscore, or dash" >&2
  exit 2
}
export RESULTS_DIR=$repo_dir/results/$BASE_NAME
[[ ! -e $RESULTS_DIR ]] || {
  echo "ERROR: async qualification requires a fresh result directory: $RESULTS_DIR" >&2
  exit 2
}
export CHECKPOINT_ROOT=$RESULTS_DIR
export WANDB_RUN_NAME=${WANDB_RUN_NAME:-$BASE_NAME}
export WANDB_PROJECT=nemorl-v2-subagent-comapction-debug
export WANDB_ENTITY=${WANDB_ENTITY:-adlr}
# Super-GA is MoE: preserve generation-time expert routes when Megatron scores
# and trains stale async rows. Trace the first two steps; never mask bad TMPE.
export NRL_ROUTER_REPLAY_VALIDATE=${NRL_ROUTER_REPLAY_VALIDATE:-1}
export NRL_R3_TRACE=${NRL_R3_TRACE:-1}
export NRL_R3_TRACE_STEPS=${NRL_R3_TRACE_STEPS:-2}
export NRL_R3_TRACE_SAMPLES=${NRL_R3_TRACE_SAMPLES:-2}
export NRL_R3_TRACE_MICROBATCHES=${NRL_R3_TRACE_MICROBATCHES:-2}
export NRL_R3_TRACE_VERIFY_FORWARD=${NRL_R3_TRACE_VERIFY_FORWARD:-1}
export NRL_R3_TRACE_DIR=${NRL_R3_TRACE_DIR:-$repo_dir/results/$BASE_NAME/r3_trace}

inflight_groups=$((NUM_PROMPTS * 3))
# ReadyFirst's max_staleness_versions bounds dispatch-ahead batches. Completed
# late stragglers remain selectable by design, so live qualification must report
# the realized `lag=` metric instead of treating this knob as a hard age filter.
exec bash "$script_dir/smoke_subagents_fast_step.sh" \
  "policy.generation.vllm_cfg.tensor_parallel_size=${GENERATION_TENSOR_PARALLEL_SIZE}" \
  +policy.router_replay.enabled=true \
  async_rl.sampler.name=ready_first \
  ~async_rl.sampler.max_lookahead_versions \
  +async_rl.sampler.max_staleness_versions=2 \
  "async_rl.min_groups_for_streaming_train=${NUM_PROMPTS}" \
  "async_rl.max_inflight_prompts=${inflight_groups}" \
  "async_rl.max_buffered_rollouts=${inflight_groups}" \
  "$@" \
  "+checkpointing.pretrained_checkpoint.path=${SYNC_POLICY_CHECKPOINT}" \
  +checkpointing.pretrained_checkpoint.format=megatron_bridge
