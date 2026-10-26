#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Submit AgenticVBench SingleController GRPO (token capture, optional context
# compaction) on Slurm with the container built for this checkout.
# Adapted from pivot-rl-video/RL/scripts/run_crop_video_pivot.sh.
# DRY_RUN=1 writes the driver/setup commands and previews the dependency chain.
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
code_dir="$(cd "${script_dir}/../.." && pwd)"

export SLURM_ACCOUNT="${SLURM_ACCOUNT:-nemotron_n4_omni}"
export SLURM_PARTITION="${SLURM_PARTITION:-batch_long}"
# This launcher submits exclusive GPU nodes. Let each nested Ray srun use the
# memory already reserved by its parent job instead of Slurm's tiny step default.
export NRL_RAY_SRUN_MEM="${NRL_RAY_SRUN_MEM:-0}"

env_file="${ENV_FILE:-${HOME}/.venv}"
if [[ -f "${env_file}" ]]; then
  set -a
  # shellcheck disable=SC1090
  source "${env_file}"
  set +a
fi
export WANDB_MODE=online

# The unified-teacher base ran in containers/nemo-rl.nightly-gym.20260930.sqsh, an
# aarch64 image (GB200, 4 GPUs/node) that cannot start on the x86 B200 nodes here;
# the base lockfile differs from cc-pr4200's only in Ray (2.56.1 vs 2.58.0, and
# this Gym accepts >=2.56.1), so the x86 cc-pr4200 image is the equivalent.
container="${CONTAINER:-/lustre/fsw/portfolios/nemotron/users/arushig/images/nemo-rl-cc-pr4200.sqsh}"
config_in_container="${CONFIG_IN_CONTAINER:-examples/configs/recipes/vlm/vlm_grpo-nemotron-super-omni-120ba12b-8n8g-megatron-agentic-vbench-sc.v2.yaml}"
: "${MODEL_PATH:?MODEL_PATH must point to a complete Hugging Face checkpoint visible on every node}"
model_path="${MODEL_PATH}"
export MODEL_PATH="${model_path}"
export RAY_TMPDIR="${RAY_TMPDIR:-/tmp}"
: "${TRAIN_PATH:?TRAIN_PATH is required}"
export VAL_PATH="${VAL_PATH:-${TRAIN_PATH}}"
[[ -s "${TRAIN_PATH}" ]] || { echo "ERROR: TRAIN_PATH must exist before submission: ${TRAIN_PATH}" >&2; exit 1; }
extra_cli=""
if (( $# )); then
  printf -v extra_cli '%q ' "$@"
fi
tokenizer_chat_template="${TOKENIZER_CHAT_TEMPLATE:-default}"
vllm_chat_template="${VLLM_CHAT_TEMPLATE:-null}"
vllm_load_format="${VLLM_LOAD_FORMAT:-auto}"
# Compile/JIT caches contain user-owned logs and lock files. Sharing another
# user's cache can read successfully during preflight and then fail only on a
# worker's first FlashInfer or vLLM write. Default to the submitter's shared
# Lustre namespace; PERSISTENT_CACHE remains available for an explicit cache.
cache_user=${USER:-$(id -un)}
persistent_cache="${PERSISTENT_CACHE:-/lustre/fsw/portfolios/nemotron/users/${cache_user}/nemo_rl_cache}"
gym_venv_dir="${GYM_VENV_DIR:-${persistent_cache}/gym_venvs_cc_pr4200}"
slurm_time_limit="${SLURM_TIME_LIMIT:-04:00:00}"
# The controller stops a cycle early once elapsed + mean step time would pass
# checkpointing.checkpoint_must_save_by; NeMo-RL's default is 3h45 regardless of
# the job's limit, which made a 12 h cycle quit after one 2 h step. Derive it from
# the Slurm limit minus a shutdown margin (DD:HH:MM:SS); CHECKPOINT_MUST_SAVE_BY overrides.
_limit_to_seconds() {  # accepts HH:MM:SS, D-HH:MM:SS, D-HH, MM:SS, MM
  local t=$1 d=0 hms
  if [[ $t == *-* ]]; then d=${t%%-*}; hms=${t#*-}; else hms=$t; fi
  IFS=: read -r a b c <<<"$hms"
  if [[ -z $b ]]; then hms="00:$a:00"; elif [[ -z $c ]]; then [[ $t == *-* ]] && hms="$a:$b:00" || hms="00:$a:$b"; fi
  IFS=: read -r h m s <<<"$hms"
  echo $(( d*86400 + 10#$h*3600 + 10#$m*60 + 10#$s ))
}
_seconds_to_ddhhmmss() { printf '%02d:%02d:%02d:%02d' $(($1/86400)) $(($1%86400/3600)) $(($1%3600/60)) $(($1%60)); }
_deadline_to_seconds() {  # accepts the controller's DD:HH:MM:SS format
  local raw=$1 d h m s extra
  IFS=: read -r d h m s extra <<<"$raw"
  [[ -z $extra && $d =~ ^[0-9]+$ && $h =~ ^[0-9]+$ && $m =~ ^[0-9]+$ && $s =~ ^[0-9]+$ ]] || return 1
  echo $(( 10#$d*86400 + 10#$h*3600 + 10#$m*60 + 10#$s ))
}
slurm_time_seconds=$(_limit_to_seconds "$slurm_time_limit")
min_slurm_time_seconds=${MIN_SLURM_TIME_SECONDS:-0}
case "$min_slurm_time_seconds" in
  ''|*[!0-9]*) echo "ERROR: MIN_SLURM_TIME_SECONDS must be a non-negative integer" >&2; exit 1 ;;
esac
if (( slurm_time_seconds < min_slurm_time_seconds )); then
  echo "ERROR: Slurm allocation ${slurm_time_limit} is shorter than the required ${min_slurm_time_seconds}s resumable cycle" >&2
  exit 1
fi
_budget_s=$(( slurm_time_seconds - ${CHECKPOINT_MARGIN_SECONDS:-1200} ))
(( _budget_s < 600 )) && _budget_s=600
checkpoint_must_save_by="${CHECKPOINT_MUST_SAVE_BY:-$(_seconds_to_ddhhmmss "$_budget_s")}"
min_checkpoint_budget_seconds=${MIN_CHECKPOINT_BUDGET_SECONDS:-0}
case "$min_checkpoint_budget_seconds" in
  ''|*[!0-9]*) echo "ERROR: MIN_CHECKPOINT_BUDGET_SECONDS must be a non-negative integer" >&2; exit 1 ;;
esac
checkpoint_budget_seconds=none
if [[ $checkpoint_must_save_by != null ]]; then
  checkpoint_budget_seconds=$(_deadline_to_seconds "$checkpoint_must_save_by") || {
    echo "ERROR: CHECKPOINT_MUST_SAVE_BY must be null or DD:HH:MM:SS" >&2
    exit 1
  }
  if (( checkpoint_budget_seconds < min_checkpoint_budget_seconds )); then
    echo "ERROR: controller checkpoint boundary ${checkpoint_must_save_by} is shorter than the required ${min_checkpoint_budget_seconds}s" >&2
    exit 1
  fi
fi

# Context compaction: "true" enables NeMo-RL's framework-owned context
# reconstruction (token_capture.context_compaction). OpenCode's own compaction is
# pinned separately through AGENTIC_VBENCH_COMPACTION_AUTO (read by the Gym agent
# config); ordinary capture cannot train a rewritten context, so it must be off there.
context_compaction="${CONTEXT_COMPACTION:-false}"
case "${context_compaction}" in true|false) ;; *) echo "ERROR: CONTEXT_COMPACTION must be true or false" >&2; exit 1;; esac
export AGENTIC_VBENCH_COMPACTION_AUTO="${AGENTIC_VBENCH_COMPACTION_AUTO:-${context_compaction}}"
export AGENTIC_VBENCH_COMPACTION_PRUNE="${AGENTIC_VBENCH_COMPACTION_PRUNE:-false}"
if [[ "${context_compaction}" != true && ( "${AGENTIC_VBENCH_COMPACTION_AUTO}" == true || "${AGENTIC_VBENCH_COMPACTION_PRUNE}" == true ) ]]; then
  echo "ERROR: OpenCode compaction/pruning requires CONTEXT_COMPACTION=true" >&2
  exit 1
fi

run_id="${RUN_ID:-$(date -u +%Y%m%d-%H%M%S)}"
base_name="${BASE_NAME:-agentic_vbench_sc_step120_${run_id}}"
candidate_name="${base_name}_${SLURM_ACCOUNT}_${SLURM_PARTITION//,/_}"
results_dir="${RESULTS_DIR:-${code_dir}/results/${base_name}}"
checkpoint_root="${CHECKPOINT_ROOT:-${results_dir}}"
checkpoint_dir="${CHECKPOINT_DIR:-${checkpoint_root}/checkpoints}"
token_capture_dir="${TOKEN_CAPTURE_DIR:-${results_dir}/token_capture}"
# TorchInductor's generated Python records absolute cubin paths. A submit-shell
# TMPDIR on shared Lustre therefore lets ranks on different nodes race through
# one cache and can leave a graph pointing at a missing cubin. Keep compiler
# caches run-scoped and node-local. The uv standalone-Python bin directory also
# supplies python3-config, which the static Triton launcher invokes via make.
job_local_cache_id=$(printf '%s' "${base_name}" | sha256sum | cut -c1-16)
job_local_cache_root="${JOB_LOCAL_CACHE_ROOT:-/tmp/nemo_rl_${cache_user}_${job_local_cache_id}}"
export TMPDIR="${JOB_TMPDIR:-/tmp}"
export TORCHINDUCTOR_CACHE_DIR="${TORCHINDUCTOR_CACHE_DIR_OVERRIDE:-${job_local_cache_root}/torchinductor}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR_OVERRIDE:-${job_local_cache_root}/triton}"
export CUDA_CACHE_PATH="${CUDA_CACHE_PATH_OVERRIDE:-${job_local_cache_root}/cuda}"
python_toolchain_bin="${PYTHON_TOOLCHAIN_BIN:-/opt/uv/python/cpython-3.13.14-linux-x86_64-gnu/bin}"
export PATH="${python_toolchain_bin}:${PATH}"
# Harbor episodes (trial artifacts, runner logs, recorded model calls) per run.
export AGENTIC_VBENCH_OUTPUT_ROOT="${AGENTIC_VBENCH_OUTPUT_ROOT:-${results_dir}/episodes}"
slurm_log_dir="${results_dir}/slurm"
job_cycles="${JOB_CYCLES:-1}"
case "$job_cycles" in
  ''|*[!0-9]*) echo "ERROR: JOB_CYCLES must be a positive integer" >&2; exit 1 ;;
esac
(( job_cycles >= 1 )) || { echo "ERROR: JOB_CYCLES must be at least 1" >&2; exit 1; }
case "${REQUIRE_RESUMABLE_JOB_CYCLES:-false}" in
  true) (( job_cycles >= 2 )) || { echo "ERROR: resumable training requires JOB_CYCLES >= 2" >&2; exit 1; } ;;
  false) ;;
  *) echo "ERROR: REQUIRE_RESUMABLE_JOB_CYCLES must be true or false" >&2; exit 1 ;;
esac
num_nodes="${NUM_NODES:-16}"
gpus_per_node="${GPUS_PER_NODE:-8}"
num_gen_nodes="${NUM_GEN_NODES:-8}"
# Nodes per NVLink domain for topology-aware placement (8 for 16-node, 4 for 8-node runs).
segment_size="${SEGMENT_SIZE:-8}"
num_prompts="${NUM_PROMPTS:-8}"
num_generations="${NUM_GENERATIONS:-4}"
generation_tensor_parallel_size="${GENERATION_TENSOR_PARALLEL_SIZE:-}"
if [[ -n $generation_tensor_parallel_size && ! $generation_tensor_parallel_size =~ ^[1-9][0-9]*$ ]]; then
  echo "ERROR: GENERATION_TENSOR_PARALLEL_SIZE must be a positive integer when set" >&2
  exit 1
fi
# Keep the production GRPO horizon. The controller still stops at the data-epoch
# or Slurm/checkpoint boundary; this must not become a one-step smoke cap.
max_steps="${MAX_STEPS:-1000000}"
save_period="${SAVE_PERIOD:-10}"
checkpoint_keep_top_k="${CHECKPOINT_KEEP_TOP_K:-0}"
checkpoint_ft_save_period="${CHECKPOINT_FT_SAVE_PERIOD:-10}"
checkpoint_ft_keep_latest_k="${CHECKPOINT_FT_KEEP_LATEST_K:-5}"
train_global_batch_size=$((num_prompts * num_generations))
num_storage_units="${TQ_NUM_STORAGE_UNITS:-$((2 * num_nodes))}"

if (( num_gen_nodes <= 0 || num_gen_nodes >= num_nodes )); then
  echo "ERROR: NUM_GEN_NODES must be between 1 and NUM_NODES-1" >&2
  exit 1
fi
num_train_nodes=$((num_nodes - num_gen_nodes))
if (( num_train_nodes % segment_size != 0 || num_gen_nodes % segment_size != 0 )); then
  echo "ERROR: SEGMENT_SIZE must divide both policy and generation node counts" >&2
  exit 1
fi

if [[ ! -f "${container}" ]]; then
  if [[ "${DRY_RUN:-0}" == "1" ]]; then
    echo "WARNING: container not found yet: ${container} (build it with scripts/agentic_vbench/container/build_container.sbatch)" >&2
  else
    echo "ERROR: container not found: ${container} (build it with scripts/agentic_vbench/container/build_container.sbatch)" >&2
    exit 1
  fi
fi
[[ -f "${model_path}/config.json" ]] || { echo "ERROR: checkpoint not found: ${model_path}" >&2; exit 1; }
for model_file in config.json model.safetensors.index.json tokenizer.json; do
  [[ -s "${model_path}/${model_file}" ]] || { echo "ERROR: missing model file: ${model_file}" >&2; exit 1; }
done
if [[ ! -s "${model_path}/processor_config.json" && ! -s "${model_path}/preprocessor_config.json" ]]; then
  echo "ERROR: checkpoint needs processor_config.json or preprocessor_config.json" >&2
  exit 1
fi
python3 - "${model_path}" <<'CHECKPOINT_PY'
import json
import sys
from pathlib import Path
root = Path(sys.argv[1])
index = json.loads((root / "model.safetensors.index.json").read_text())
missing = [name for name in set(index["weight_map"].values()) if not (root / name).is_file() or (root / name).stat().st_size == 0]
if missing:
    raise SystemExit(f"Missing or empty checkpoint shards: {missing}")
CHECKPOINT_PY
[[ -f "${code_dir}/${config_in_container}" ]] || { echo "ERROR: recipe not found: ${code_dir}/${config_in_container}" >&2; exit 1; }
[[ -f "${code_dir}/3rdparty/Gym-workspace/Gym/responses_api_agents/agentic_vbench_agent/app.py" ]] || {
  echo "ERROR: AgenticVBench agent not in this clone's Gym" >&2
  exit 1
}
[[ -f "${code_dir}/examples/run_grpo_single_controller.py" ]] || { echo "ERROR: SingleController entrypoint missing" >&2; exit 1; }
if [[ "${WANDB_MODE:-online}" != "offline" ]]; then
  [[ -n "${WANDB_API_KEY:-}" ]] || { echo "ERROR: WANDB_API_KEY is not set for online logging" >&2; exit 1; }
fi

mkdir -p \
  "${results_dir}" \
  "${checkpoint_dir}" \
  "${token_capture_dir}" \
  "${AGENTIC_VBENCH_OUTPUT_ROOT}" \
  "${slurm_log_dir}" \
  "${persistent_cache}/huggingface" \
  "${persistent_cache}/vllm_compile_cache_cc" \
  "${persistent_cache}/flashinfer_cubins" \
  "${persistent_cache}/flashinfer_workspace" \
  "${persistent_cache}/megatron_ckpt_cache_cc" \
  "${persistent_cache}/hf_config_locks" \
  "${gym_venv_dir}"

wandb_run_id="${WANDB_RUN_ID:-$(printf '%s' "${base_name}" | sha256sum | cut -c1-16)}"
wandb_project="${WANDB_PROJECT:-megatron-omni-v3}"
wandb_run_name="${WANDB_RUN_NAME:-${base_name}}"
wandb_cluster_name="${WANDB_CLUSTER_NAME:-${SLURM_CLUSTER_NAME:-}}"
if [[ -z "${wandb_cluster_name}" ]]; then
  wandb_cluster_name="$(scontrol show config | awk '$1 == "ClusterName" {print $3}')"
fi
case "${wandb_cluster_name}" in
  cw-dfw-*) wandb_cluster_name=dfw ;;
esac
[[ "${wandb_cluster_name}" =~ ^[A-Za-z0-9._-]+$ ]] || {
  echo "ERROR: WANDB_CLUSTER_NAME must contain only letters, digits, dot, underscore, or dash" >&2
  exit 1
}
case "${wandb_run_name}" in
  *"_cluster-${wandb_cluster_name}") ;;
  *) wandb_run_name="${wandb_run_name}_cluster-${wandb_cluster_name}" ;;
esac
wandb_group="${WANDB_RUN_GROUP:-cluster-${wandb_cluster_name}}"
wandb_cluster_tag="cluster-${wandb_cluster_name}"

# Exact gates execute a committed snapshot rather than a mutable shared checkout.
# The snapshot is assembled from the superproject plus the three nested runtime
# repositories and is verified again inside the allocation before model load.
immutable_source_snapshot=${IMMUTABLE_SOURCE_SNAPSHOT:-false}
case "$immutable_source_snapshot" in
  true|false) ;;
  *) echo "ERROR: IMMUTABLE_SOURCE_SNAPSHOT must be true or false" >&2; exit 1 ;;
esac
runtime_code_dir=$code_dir
runtime_snapshot_dir=
runtime_source_manifest=
runtime_manifest_sha256=
if [[ $immutable_source_snapshot == true ]]; then
  runtime_snapshot_dir=$results_dir/runtime_source
  runtime_source_manifest=$results_dir/runtime_source_manifest.json
  python3 "$code_dir/tools/runtime_source_snapshot.py" \
    --create --repo "$code_dir" --destination "$runtime_snapshot_dir" \
    --manifest "$runtime_source_manifest" >/dev/null
  runtime_manifest_sha256=$(python3 - "$runtime_source_manifest" <<'MANIFEST_DIGEST_PY'
import json
import sys
from pathlib import Path

payload = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
print(payload["runtime_manifest_sha256"])
MANIFEST_DIGEST_PY
  )
  runtime_code_dir=$runtime_snapshot_dir
fi

# Persist the exact source/data/topology identity before submission.  The
# sync-to-async handoff validator requires this file so a directory merely
# named step_20 cannot be mistaken for the qualified donor.
source_commit=$(git -C "${code_dir}" rev-parse HEAD)
source_branch=$(git -C "${code_dir}" symbolic-ref --quiet --short HEAD 2>/dev/null || echo DETACHED)
gym_commit=$(git -C "${code_dir}/3rdparty/Gym-workspace/Gym" rev-parse HEAD)
bridge_commit=$(git -C "${code_dir}/3rdparty/Megatron-Bridge-workspace/Megatron-Bridge" rev-parse HEAD)
mcore_commit=$(git -C "${code_dir}/3rdparty/Megatron-Bridge-workspace/Megatron-Bridge/3rdparty/Megatron-LM" rev-parse HEAD)
source_worktree_dirty=false
[[ -z $(git -C "${code_dir}" status --porcelain --untracked-files=no) ]] || source_worktree_dirty=true
case ${LAUNCH_BENCHMARK:-} in
 claweval)
  launch_benchmark=$LAUNCH_BENCHMARK
  launch_compaction_auto=${CLAW_EVAL_COMPACTION_AUTO:-false}
  launch_compaction_prune=${CLAW_EVAL_COMPACTION_PRUNE:-false}
  launch_subagents=${CLAW_EVAL_SUBAGENTS_ENABLED:-false}
  launch_training_mode=${CLAW_EVAL_TRAINING_MODE:-standard}
  launch_max_turns=${CLAW_EVAL_MAX_TURNS:-40}
  launch_context_tokens=${CLAW_EVAL_CONTEXT_TOKENS:-49152}
  launch_output_tokens=${CLAW_EVAL_OUTPUT_TOKENS:-8192}
  launch_tool_output_max_lines=${CLAW_EVAL_TOOL_OUTPUT_MAX_LINES:-}
  launch_tool_output_max_bytes=${CLAW_EVAL_TOOL_OUTPUT_MAX_BYTES:-}
  launch_max_train_row_tokens=${CLAW_EVAL_MAX_ROW_TOKENS:-}
  launch_task_selection=all
  launch_train_limit=
  launch_train_repeats=1
  launch_delegation_prompt=${CLAW_EVAL_DELEGATION_PROMPT:-}
  launch_subagent_prompt=${CLAW_EVAL_SUBAGENT_PROMPT:-}
  ;;
 agentic_vbench)
  launch_benchmark=$LAUNCH_BENCHMARK
  launch_compaction_auto=${AGENTIC_VBENCH_COMPACTION_AUTO:-false}
  launch_compaction_prune=${AGENTIC_VBENCH_COMPACTION_PRUNE:-false}
  launch_subagents=${AGENTIC_VBENCH_SUBAGENTS_ENABLED:-false}
  launch_training_mode=${AGENTIC_VBENCH_TRAINING_MODE:-standard}
  launch_max_turns=${AGENTIC_VBENCH_MAX_TURNS:-32}
  launch_context_tokens=${AGENTIC_VBENCH_CONTEXT_TOKENS:-49152}
  launch_output_tokens=${AGENTIC_VBENCH_OUTPUT_TOKENS:-8192}
  launch_tool_output_max_lines=${AGENTIC_VBENCH_TOOL_OUTPUT_MAX_LINES:-}
  launch_tool_output_max_bytes=${AGENTIC_VBENCH_TOOL_OUTPUT_MAX_BYTES:-}
  launch_max_train_row_tokens=${SUBAGENT_MAX_ROW_TOKENS:-}
  launch_task_selection=${AGENTIC_VBENCH_TASKS:-all}
  launch_train_limit=${TRAIN_LIMIT:-}
  launch_train_repeats=${TRAIN_REPEATS:-1}
  launch_delegation_prompt=${AGENTIC_VBENCH_DELEGATION_PROMPT:-}
  launch_subagent_prompt=${AGENTIC_VBENCH_SUBAGENT_PROMPT:-}
  ;;
 *)
  echo "ERROR: LAUNCH_BENCHMARK must be agentic_vbench or claweval" >&2
  exit 1
  ;;
esac
provenance_path=${results_dir}/launch_provenance.json
case ${REQUIRE_FRESH_PROVENANCE:-false} in
  true)
    [[ ! -e $provenance_path ]] || {
      echo "ERROR: refusing to overwrite launch provenance: $provenance_path" >&2
      exit 1
    }
    ;;
  false) ;;
  *) echo "ERROR: REQUIRE_FRESH_PROVENANCE must be true or false" >&2; exit 1 ;;
esac
python3 - \
  "${provenance_path}" \
  "${launch_benchmark}" "${source_commit}" "${source_branch}" \
  "${source_worktree_dirty}" "${gym_commit}" "${bridge_commit}" \
  "${mcore_commit}" "${results_dir}" "${checkpoint_dir}" "${model_path}" \
  "${config_in_container}" "${runtime_code_dir}/${config_in_container}" \
  "${TRAIN_PATH}" "${VAL_PATH}" "${container}" \
  "${num_nodes}" "${num_gen_nodes}" "${gpus_per_node}" "${segment_size}" \
  "${num_prompts}" "${num_generations}" "${generation_tensor_parallel_size}" \
  "${max_steps}" "${save_period}" \
  "${checkpoint_keep_top_k}" "${checkpoint_ft_save_period}" \
  "${checkpoint_ft_keep_latest_k}" "${context_compaction}" \
  "${launch_compaction_auto}" "${launch_compaction_prune}" \
  "${launch_subagents}" "${launch_training_mode}" "${wandb_project}" \
  "${wandb_run_id}" "${wandb_run_name}" "${wandb_cluster_name}" \
  "${SYNC_POLICY_CHECKPOINT:-}" "${SYNC_GATE_ATTESTATION:-}" \
  "${immutable_source_snapshot}" "${runtime_snapshot_dir}" \
  "${runtime_source_manifest}" "${runtime_manifest_sha256}" \
  "${launch_max_turns}" "${launch_context_tokens}" "${launch_output_tokens}" \
  "${launch_tool_output_max_lines}" "${launch_tool_output_max_bytes}" \
  "${launch_max_train_row_tokens}" "${launch_task_selection}" \
  "${launch_train_limit}" "${launch_train_repeats}" \
  "${launch_delegation_prompt}" "${launch_subagent_prompt}" <<'PROVENANCE_PY'
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

(
    output,
    benchmark,
    source_commit,
    source_branch,
    source_worktree_dirty,
    gym_commit,
    bridge_commit,
    mcore_commit,
    result_directory,
    checkpoint_directory,
    model_path,
    config,
    config_path,
    train_path,
    validation_path,
    container_path,
    num_nodes,
    num_generation_nodes,
    gpus_per_node,
    segment_size,
    num_prompts,
    num_generations,
    generation_tensor_parallel_size,
    max_num_steps,
    save_period,
    keep_top_k,
    ft_save_period,
    ft_keep_latest_k,
    context_compaction,
    opencode_compaction_auto,
    opencode_compaction_prune,
    subagents_enabled,
    training_mode,
    wandb_project,
    wandb_run_id,
    wandb_run_name,
    wandb_cluster_name,
    sync_policy_checkpoint,
    sync_gate_attestation,
    immutable_source_snapshot,
    runtime_snapshot_directory,
    runtime_source_manifest,
    runtime_manifest_sha256,
    max_turns,
    context_tokens,
    output_tokens,
    tool_output_max_lines,
    tool_output_max_bytes,
    max_train_row_tokens,
    task_selection,
    train_limit,
    train_repeats,
    delegation_prompt,
    subagent_prompt,
) = sys.argv[1:]


def boolean(value: str) -> bool:
    if value not in {"true", "false"}:
        raise SystemExit(f"invalid provenance boolean: {value!r}")
    return value == "true"


def resolved(value: str) -> str:
    return str(Path(value).resolve())


def optional_int(value: str) -> int | None:
    return int(value) if value else None


def file_identity(value: str, *, hash_contents: bool = True) -> dict[str, object]:
    path = Path(value).resolve(strict=True)
    stat_result = path.stat()
    identity: dict[str, object] = {
        "path": str(path),
        "size_bytes": stat_result.st_size,
        "mtime_ns": stat_result.st_mtime_ns,
    }
    if hash_contents:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            while chunk := handle.read(8 * 1024 * 1024):
                digest.update(chunk)
        identity["sha256"] = digest.hexdigest()
    return identity


def prompt_sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


payload = {
    "schema_version": 2,
    "created_at_utc": datetime.now(timezone.utc).isoformat(),
    "benchmark": benchmark,
    "source_commit": source_commit,
    "source_branch": source_branch,
    "source_worktree_dirty": boolean(source_worktree_dirty),
    "gym_commit": gym_commit,
    "bridge_commit": bridge_commit,
    "mcore_commit": mcore_commit,
    "result_directory": resolved(result_directory),
    "checkpoint_directory": resolved(checkpoint_directory),
    "model_path": resolved(model_path),
    "config": config,
    "config_identity": file_identity(config_path),
    "train_data": file_identity(train_path),
    "validation_data": file_identity(validation_path),
    # The 50-GB squashfs is pinned by canonical path plus immutable stat identity;
    # hashing it at every launch would add avoidable shared-filesystem load.
    "container": file_identity(container_path, hash_contents=False),
    "num_nodes": int(num_nodes),
    "num_generation_nodes": int(num_generation_nodes),
    "gpus_per_node": int(gpus_per_node),
    "segment_size": int(segment_size),
    "num_prompts": int(num_prompts),
    "num_generations": int(num_generations),
    "generation_tensor_parallel_size": optional_int(
        generation_tensor_parallel_size
    ),
    "max_num_steps": int(max_num_steps),
    "max_num_epochs": None,
    "save_period": int(save_period),
    "keep_top_k": None if keep_top_k == "null" else int(keep_top_k),
    "ft_save_period": int(ft_save_period),
    "ft_keep_latest_k": int(ft_keep_latest_k),
    "context_compaction": boolean(context_compaction),
    "opencode_compaction_auto": boolean(opencode_compaction_auto),
    "opencode_compaction_prune": boolean(opencode_compaction_prune),
    "subagents_enabled": boolean(subagents_enabled),
    "training_mode": training_mode,
    "wandb_project": wandb_project,
    "wandb_run_id": wandb_run_id,
    "wandb_run_name": wandb_run_name,
    "wandb_cluster_name": wandb_cluster_name,
    "sync_policy_checkpoint": resolved(sync_policy_checkpoint)
    if sync_policy_checkpoint
    else None,
    "sync_gate_attestation": resolved(sync_gate_attestation)
    if sync_gate_attestation
    else None,
    "immutable_source_snapshot": boolean(immutable_source_snapshot),
    "runtime_snapshot_directory": resolved(runtime_snapshot_directory)
    if runtime_snapshot_directory
    else None,
    "runtime_source_manifest": resolved(runtime_source_manifest)
    if runtime_source_manifest
    else None,
    "runtime_manifest_sha256": runtime_manifest_sha256 or None,
    "max_turns": int(max_turns),
    "context_tokens": int(context_tokens),
    "output_tokens": int(output_tokens),
    "tool_output_max_lines": optional_int(tool_output_max_lines),
    "tool_output_max_bytes": optional_int(tool_output_max_bytes),
    "max_train_row_tokens": optional_int(max_train_row_tokens),
    "task_selection": task_selection,
    "train_limit": optional_int(train_limit),
    "train_repeats": int(train_repeats),
    "delegation_prompt_sha256": prompt_sha256(delegation_prompt),
    "subagent_prompt_sha256": prompt_sha256(subagent_prompt),
}
destination = Path(output)
temporary = destination.with_suffix(destination.suffix + ".tmp")
temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
temporary.replace(destination)
PROVENANCE_PY

if [[ -n ${SYNC_POLICY_CHECKPOINT:-} ]]; then
  : "${SYNC_GATE_ATTESTATION:?async donor requires SYNC_GATE_ATTESTATION}"
  : "${SYNC_DCP_MANIFEST_SHA256:?async donor requires SYNC_DCP_MANIFEST_SHA256}"
  [[ $SYNC_DCP_MANIFEST_SHA256 =~ ^[0-9a-f]{64}$ ]] || {
    echo "ERROR: SYNC_DCP_MANIFEST_SHA256 must be 64 lowercase hexadecimal characters" >&2
    exit 1
  }
fi

base_mounts="/lustre:/lustre"
[[ ! -d /scratch ]] || base_mounts+=",/scratch:/scratch"
selective_mounts="${runtime_code_dir}/nemo_rl:/opt/nemo-rl/nemo_rl"
selective_mounts+=",${runtime_code_dir}/examples:/opt/nemo-rl/examples"
selective_mounts+=",${runtime_code_dir}/tools:/opt/nemo-rl/tools"
selective_mounts+=",${runtime_code_dir}/scripts:/opt/nemo-rl/scripts"
selective_mounts+=",${runtime_code_dir}/tests:/opt/nemo-rl/tests"
selective_mounts+=",${runtime_code_dir}/pyproject.toml:/opt/nemo-rl/pyproject.toml"
selective_mounts+=",${runtime_code_dir}/uv.lock:/opt/nemo-rl/uv.lock"
selective_mounts+=",${runtime_code_dir}/3rdparty/Gym-workspace/Gym:/opt/nemo-rl/3rdparty/Gym-workspace/Gym"
selective_mounts+=",${runtime_code_dir}/3rdparty/Megatron-Bridge-workspace/Megatron-Bridge:/opt/nemo-rl/3rdparty/Megatron-Bridge-workspace/Megatron-Bridge"
# The mounts above shadow the image's copies: an uninitialized submodule would hide
# the installed Gym or Megatron-Bridge/Megatron-LM (ModuleNotFoundError: megatron.bridge
# at worker creation, after the GPUs are allocated). Fail here instead.
for sub in 3rdparty/Gym-workspace/Gym 3rdparty/Megatron-Bridge-workspace/Megatron-Bridge 3rdparty/Megatron-Bridge-workspace/Megatron-Bridge/3rdparty/Megatron-LM; do
  [[ -n "$(ls -A "${runtime_code_dir}/${sub}" 2>/dev/null)" ]] || { echo "ERROR: ${sub} is empty in the selected runtime source" >&2; exit 1; }
done
export MOUNTS="${base_mounts},${selective_mounts}${EXTRA_MOUNTS:+,${EXTRA_MOUNTS}}"
export CONTAINER="${container}"
export CONTAINER_ENV_VARS="${CONTAINER_ENV_VARS:+${CONTAINER_ENV_VARS},}PATH,PYTHONPATH,LD_LIBRARY_PATH"
export BASE_LOG_DIR="${slurm_log_dir}"
export RAY_SUB_PATH="${code_dir}/ray.sub"
# Sync Ray's own session logs (gcs_server/raylet) to the shared log dir often enough
# that an early GCS death still leaves evidence behind.
export RAY_LOG_SYNC_FREQUENCY="${RAY_LOG_SYNC_FREQUENCY:-15}"
export GPUS_PER_NODE="${gpus_per_node}"
export SANDBOX_COMMAND=""
export SANDBOX_CONTAINER=""
export RAY_ENABLE_UV_RUN_RUNTIME_ENV=0
export NEMO_RL_VENV_DIR=${NEMO_RL_VENV_DIR:-/opt/ray_venvs}
export NEMO_GYM_VENV_DIR="${gym_venv_dir}"
export NRL_FORCE_REBUILD_VENVS=false
export NRL_IGNORE_VERSION_MISMATCH="${NRL_IGNORE_VERSION_MISMATCH:-1}"
export NRL_WG_USE_RAY_REF=1
export NRL_MEGATRON_CHECKPOINT_DIR="${persistent_cache}/megatron_ckpt_cache_cc"
export MEGATRON_CONFIG_LOCK_DIR="${persistent_cache}/hf_config_locks"
export VLLM_CACHE_ROOT="${persistent_cache}/vllm_compile_cache_cc"
export DG_JIT_CACHE_DIR="${persistent_cache}/vllm_compile_cache_cc/deep_gemm"
export VLLM_DEEP_GEMM_WARMUP=skip
export FLASHINFER_CUBIN_DIR="${persistent_cache}/flashinfer_cubins"
export FLASHINFER_WORKSPACE_BASE="${persistent_cache}/flashinfer_workspace"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HOME="${persistent_cache}/huggingface"
export HF_MODULES_CACHE="${persistent_cache}/huggingface/modules/${base_name}"
export NRL_LOCAL_MCORE="${job_local_cache_root}/megatron-core"
# All actors must use this checkout's Bridge/MCore pair. Stage MCore locally
# before Ray starts so any dataset helper builds leave the source tree intact.
export PYTHONPATH="/opt/nemo-rl:/opt/nemo-rl/3rdparty/Gym-workspace/Gym:/opt/nemo-rl/3rdparty/Megatron-Bridge-workspace/Megatron-Bridge/src:${NRL_LOCAL_MCORE}:${PYTHONPATH:-}"

read -r -d '' SETUP_COMMAND <<'SETUP_EOF' || true
set -euo pipefail
cd /opt/nemo-rl
export PATH=${PYTHON_TOOLCHAIN_BIN:-/opt/uv/python/cpython-3.13.14-linux-x86_64-gnu/bin}:${PATH}
export UV_PROJECT_ENVIRONMENT=/opt/nemo_rl_venv
export UV_LINK_MODE=copy
# Image venvs may symlink into /root/.cache/uv. Keep that store visible and
# write new uv cache entries under this allocation's writable scratch instead.
export UV_CACHE_DIR="${TMPDIR}/uv"
export NRL_CONTAINER=1
export RAY_USAGE_STATS_ENABLED=0
# The driver venv has no nemo_gym extra; the driver imports Gym's config helpers from the source tree.
if [[ ! -f ${NRL_LOCAL_MCORE}/megatron/core/__init__.py ]]; then
  mkdir -p "${NRL_LOCAL_MCORE}"
  cp -a /opt/nemo-rl/3rdparty/Megatron-Bridge-workspace/Megatron-Bridge/3rdparty/Megatron-LM/. "${NRL_LOCAL_MCORE}/"
fi
mkdir -p "${TMPDIR}" "${TORCHINDUCTOR_CACHE_DIR}" "${TRITON_CACHE_DIR}" "${CUDA_CACHE_PATH}"
command -v python3-config >/dev/null
uv_bin=$(command -v uv || echo /usr/local/bin/uv)
test -x "${uv_bin}"
bash "${TRAINING_DEPS_SETUP_SCRIPT:-/opt/nemo-rl/tools/ensure_agentic_vbench_deps.sh}"
"${uv_bin}" run --no-sync python -c 'import sys; assert sys.version_info >= (3, 13, 14); import ray, transformers, nemo_rl.algorithms.single_controller, nemo_rl.environments.nemo_gym'
generation_vllm_python=${NEMO_RL_VENV_DIR}/nemo_rl.models.generation.vllm.vllm_worker_async.VllmAsyncGenerationWorker/bin/python
test -x "${generation_vllm_python}"
"${generation_vllm_python}" -c 'import vllm; from nemo_rl.models.generation.vllm.vllm_worker_async import VllmAsyncGenerationWorker; from nemo_rl.models.generation.capture_context import decide_capture_input'
megatron_python=${NEMO_RL_VENV_DIR}/nemo_rl.models.policy.workers.megatron_policy_worker.MegatronPolicyWorker/bin/python
test -x "${megatron_python}"
"${megatron_python}" -c 'import megatron.core, transformer_engine'
SETUP_EOF
export SETUP_COMMAND

read -r -d '' COMMAND <<COMMAND_EOF || true
set -euo pipefail
cd /opt/nemo-rl
export PATH=${python_toolchain_bin}:\${PATH}
export UV_PROJECT_ENVIRONMENT=/opt/nemo_rl_venv
export UV_LINK_MODE=copy
export UV_CACHE_DIR="\${TMPDIR}/uv"
export NRL_CONTAINER=1
export RAY_USAGE_STATS_ENABLED=0
export PYTHONPATH=/opt/nemo-rl:/opt/nemo-rl/3rdparty/Gym-workspace/Gym:\${PYTHONPATH:-}
uv_bin=\$(command -v uv || echo /usr/local/bin/uv)
test -x "\${uv_bin}"
if [[ -n "${runtime_source_manifest}" ]]; then
  "\${uv_bin}" run --no-sync python tools/runtime_source_snapshot.py \
    --verify --allow-extra --root /opt/nemo-rl --manifest "${runtime_source_manifest}" \
    >"${results_dir}/runtime_source_verification-\${SLURM_JOB_ID:-local}-\${SLURM_ARRAY_TASK_ID:-0}.json"
fi
if [[ "\${RUN_TMPE_OUTLIER_TESTS:-0}" == 1 ]]; then
  "\${uv_bin}" run --no-sync pytest -q tests/unit/single_controller/test_tmpe_outlier_dump.py
fi
export HF_HOME=${HF_HOME}
export HF_MODULES_CACHE=${HF_MODULES_CACHE}
export RAY_ENABLE_UV_RUN_RUNTIME_ENV=0
export NEMO_RL_VENV_DIR=${NEMO_RL_VENV_DIR}
export NEMO_GYM_VENV_DIR=${gym_venv_dir}
export NRL_FORCE_REBUILD_VENVS=false
# Harbor/Podman scratch for this job's episodes, on the node running the Gym servers.
export AGENTIC_VBENCH_RUNTIME_ROOT=/tmp/avb-runtime-\${SLURM_JOB_ID:-local}-\$\$
mkdir -p "\${AGENTIC_VBENCH_RUNTIME_ROOT}"
# Import nemo_rl first in every fresh process so its Transformers dynamic-module
# symlink-cache compatibility patch is active before trust_remote_code executes.
"\${uv_bin}" run --no-sync python -c "import nemo_rl; from transformers import AutoConfig, AutoProcessor, AutoTokenizer; p='${model_path}'; AutoConfig.from_pretrained(p, trust_remote_code=True); AutoProcessor.from_pretrained(p, trust_remote_code=True, use_fast=True); AutoTokenizer.from_pretrained(p, trust_remote_code=True, use_fast=True); print('Prewarmed HF dynamic modules cache')"
# Populate the cache with the exact Transformers environment used by vLLM.
# A second fresh process verifies imports before concurrent actor model loads.
for prewarm_pass in populate verify; do
  ${NEMO_RL_VENV_DIR}/nemo_rl.models.generation.vllm.vllm_worker_async.VllmAsyncGenerationWorker/bin/python -c "import nemo_rl; from transformers import AutoConfig, AutoProcessor, AutoTokenizer; p='${model_path}'; c=AutoConfig.from_pretrained(p, trust_remote_code=True); AutoProcessor.from_pretrained(p, trust_remote_code=True, use_fast=True); AutoTokenizer.from_pretrained(p, trust_remote_code=True, use_fast=True); print('vLLM prewarm: \${prewarm_pass}', type(c).__module__, type(c).__name__)"
done
# Restore native replay and rollout ownership alongside model/optimizer state.
# SKIP_REPLAY_RESTORE=1 is an explicit recovery option for incompatible replay
# snapshots: preserve the sidecars as .disabled and refill from fresh prompts.
if [[ "${SKIP_REPLAY_RESTORE:-0}" == "1" ]]; then
  # rollout_recovery.pt must go with it: the controller refuses a recovery file
  # whose TQ snapshot is not being restored.
  for meta in "${checkpoint_dir}"/step_*/replay_buffer_metadata.pt "${checkpoint_dir}"/step_*/rollout_recovery.pt; do
    [[ -f "\$meta" ]] && mv "\$meta" "\$meta.disabled" && echo "[launch_sc] disabled replay restore: \$meta"
  done
fi
if [[ -n "${SYNC_POLICY_CHECKPOINT:-}" ]]; then
  sync_iteration=\$(<"${SYNC_POLICY_CHECKPOINT:-}/latest_checkpointed_iteration.txt")
  if [[ "\$sync_iteration" == release ]]; then
    sync_iteration_dir=release
  elif [[ "\$sync_iteration" =~ ^[0-9]+$ ]]; then
    printf -v sync_iteration_dir 'iter_%07d' "\$((10#\$sync_iteration))"
  else
    echo "ERROR: invalid async donor iteration marker: \$sync_iteration" >&2
    exit 1
  fi
  megatron_python=/opt/ray_venvs/nemo_rl.models.policy.workers.megatron_policy_worker.MegatronPolicyWorker/bin/python
  "\$megatron_python" tools/verify_torch_dcp_checkpoint.py \
    --checkpoint-dir "${SYNC_POLICY_CHECKPOINT:-}/\$sync_iteration_dir" \
    --sha256 --expected-manifest-sha256 "${SYNC_DCP_MANIFEST_SHA256:-}" \
    >"${results_dir}/sync_dcp_revalidation-\${SLURM_JOB_ID:-local}-\${SLURM_ARRAY_TASK_ID:-0}.json"
fi
"\${uv_bin}" run --no-sync python examples/run_grpo_single_controller.py \
  --config "${config_in_container}" \
  checkpointing.checkpoint_dir="${checkpoint_dir}" \
  logger.log_dir="${results_dir}/logs" \
  logger.wandb_enabled=true \
  logger.wandb.project="${wandb_project}" \
  logger.wandb.name="${wandb_run_name}" \
  ++logger.wandb.entity="${WANDB_ENTITY:-adlr}" \
  ++logger.wandb.id="${wandb_run_id}" \
  ++logger.wandb.resume=allow \
  ++logger.wandb.group="${wandb_group}" \
  ++logger.wandb.tags="[${wandb_cluster_tag}]" \
  ++logger.wandb.config.cluster_name="${wandb_cluster_name}" \
  ++env.nemo_gym.uv_venv_dir="${gym_venv_dir}" \
  env.nemo_gym.skip_venv_if_present=true \
  policy.model_name="${model_path}" \
  ++policy.generation.vllm_cfg.load_format="${vllm_load_format}" \
  ++policy.tokenizer.chat_template="${tokenizer_chat_template}" \
  ++policy.generation.vllm_cfg.http_server_serving_chat_kwargs.chat_template="${vllm_chat_template}" \
  data.train.data_path="${TRAIN_PATH}" \
  data.validation.data_path="${VAL_PATH}" \
  grpo.num_prompts_per_step="${num_prompts}" \
  grpo.num_generations_per_prompt="${num_generations}" \
  grpo.max_num_epochs=null \
  grpo.max_num_steps="${max_steps}" \
  checkpointing.checkpoint_must_save_by="${checkpoint_must_save_by}" \
  policy.train_global_batch_size="${train_global_batch_size}" \
  checkpointing.save_period="${save_period}" \
  checkpointing.keep_top_k="${checkpoint_keep_top_k}" \
  ++checkpointing.ft_save_period="${checkpoint_ft_save_period}" \
  ++checkpointing.ft_keep_latest_k="${checkpoint_ft_keep_latest_k}" \
  token_capture.enabled=true \
  token_capture.context_compaction="${context_compaction}" \
  token_capture.capture_dir="${token_capture_dir}" \
  data_plane.simple.num_storage_units="${num_storage_units}" \
  cluster.num_nodes="${num_nodes}" \
  cluster.gpus_per_node="${gpus_per_node}" \
  ++cluster.segment_size="${segment_size}" \
  policy.generation.colocated.resources.num_nodes="${num_gen_nodes}" \
  policy.generation.colocated.resources.gpus_per_node="${gpus_per_node}" ${EXTRA_OVERRIDES:-} ${extra_cli}
COMMAND_EOF
export COMMAND

# The cluster's OccupiedIdleGPUsJobReaper cancels jobs whose GPUs look idle;
# model loading, Harbor container setup and the first long-episode wave keep the
# training GPUs idle for a whole generation phase. Preserve the supported
# exemption when callers add a descriptive label or additional JSON metadata.
# JSON braces inside ${VAR:-...} append stray braces to a supplied override.
default_sbatch_comment='{"OccupiedIdleGPUsJobReaper":{"exemptIdleTimeMins":"240","reason":"data_loading","description":"Model load, task-container setup and agentic rollouts; training GPUs wait a whole generation phase"}}'
sbatch_comment=$(python3 - "$default_sbatch_comment" "${SBATCH_COMMENT:-}" <<'PY'
import json
import sys

comment = json.loads(sys.argv[1])
custom = sys.argv[2].strip()
if custom.startswith("{"):
    overrides = json.loads(custom)
    reaper = overrides.pop("OccupiedIdleGPUsJobReaper", {})
    if not isinstance(reaper, dict):
        raise ValueError("SBATCH_COMMENT OccupiedIdleGPUsJobReaper must be an object")
    comment["OccupiedIdleGPUsJobReaper"].update(reaper)
    comment.update(overrides)
elif custom:
    comment["description"] = custom
print(json.dumps(comment, separators=(",", ":")))
PY
)
submit_args=(
  --parsable
  --comment="${sbatch_comment}"
  --nodes="${num_nodes}"
  --gres="gpu:${gpus_per_node}"
  --exclusive
  --time="${slurm_time_limit}"
  --account="${SLURM_ACCOUNT}"
  --partition="${SLURM_PARTITION}"
  --job-name="${candidate_name}"
  --output="${slurm_log_dir}/%j-${SLURM_ACCOUNT}.out"
  --error="${slurm_log_dir}/%j-${SLURM_ACCOUNT}.out"
)
if [[ -n ${SLURM_QOS:-} ]]; then
  submit_args+=(--qos="${SLURM_QOS}")
fi

echo "candidate=${candidate_name}"
echo "sbatch_comment=${sbatch_comment}"
echo "time_limit=${slurm_time_limit} checkpoint_must_save_by=${checkpoint_must_save_by}"
echo "account=${SLURM_ACCOUNT} partition=${SLURM_PARTITION} qos=${SLURM_QOS:-default} nodes=${num_nodes} gpus_per_node=${gpus_per_node} generation_nodes=${num_gen_nodes} segment_size=${segment_size} prompts=${num_prompts} generations=${num_generations} steps=${max_steps} cycles=${job_cycles} save_period=${save_period} keep_top_k=${checkpoint_keep_top_k} tokenizer_chat_template=${tokenizer_chat_template} vllm_chat_template=${vllm_chat_template} vllm_load_format=${vllm_load_format}"
echo "allocation_boundary_seconds=${slurm_time_seconds} controller_boundary_seconds=${checkpoint_budget_seconds} grpo_horizon_steps=${max_steps} continuation_chain=afterany cycles=${job_cycles}"
echo "slurm_dependency=${SLURM_DEPENDENCY:-singleton}"
echo "context_compaction=${context_compaction}"
if [[ $launch_benchmark == claweval ]]; then
  echo "claweval_compaction_auto=${CLAW_EVAL_COMPACTION_AUTO:-null} claweval_prune=${CLAW_EVAL_COMPACTION_PRUNE:-null} training_mode=${CLAW_EVAL_TRAINING_MODE:-null} subagents=${CLAW_EVAL_SUBAGENTS_ENABLED:-false} max_turns=${CLAW_EVAL_MAX_TURNS:-40} context_tokens=${CLAW_EVAL_CONTEXT_TOKENS:-49152} output_tokens=${CLAW_EVAL_OUTPUT_TOKENS:-8192}"
else
  echo "agentic_vbench_compaction_auto=${AGENTIC_VBENCH_COMPACTION_AUTO} agentic_vbench_prune=${AGENTIC_VBENCH_COMPACTION_PRUNE} max_turns=${AGENTIC_VBENCH_MAX_TURNS:-32} context_tokens=${AGENTIC_VBENCH_CONTEXT_TOKENS:-49152} output_tokens=${AGENTIC_VBENCH_OUTPUT_TOKENS:-8192} tool_output_max_lines=${AGENTIC_VBENCH_TOOL_OUTPUT_MAX_LINES:-native} tool_output_max_bytes=${AGENTIC_VBENCH_TOOL_OUTPUT_MAX_BYTES:-native} tasks=${AGENTIC_VBENCH_TASKS:-all}"
fi
echo "episodes=${AGENTIC_VBENCH_OUTPUT_ROOT}"
echo "container=${container}"
echo "config=${config_in_container}"
echo "model=${model_path}"
echo "train=${TRAIN_PATH}"
echo "validation=${VAL_PATH}"
echo "results=${results_dir}"
echo "checkpoints=${checkpoint_dir}"
echo "token_capture=${token_capture_dir}"
echo "job_local_cache_root=${job_local_cache_root}"
# Record the AgenticVBench environment the Gym agent will read, for reproducibility.
env | grep -E '^(AGENTIC_VBENCH_|AVB_DATA_ROOT|CLAW_EVAL_|TRAIN_PATH|VAL_PATH|MODEL_PATH|CONTEXT_COMPACTION|NEMO_RL_TMPE_|NRL_MEGATRON_|WANDB_MODE|WANDB_PROJECT|WANDB_ENTITY|TMPDIR|TORCHINDUCTOR_CACHE_DIR|TRITON_CACHE_DIR|CUDA_CACHE_PATH)' | grep -v -i 'key\|token\|secret' | sort > "${results_dir}/gym_environment.env"

# Archive the exact executable commands for both real submissions and dry runs.
# These contain paths and Hydra settings but no credential values.
printf '%s\n' "${COMMAND}" > "${results_dir}/training_command.sh"
printf '%s\n' "${SETUP_COMMAND}" > "${results_dir}/setup_command.sh"
bash -n "${results_dir}/training_command.sh"
bash -n "${results_dir}/setup_command.sh"

if [[ "${DRY_RUN:-0}" == "1" ]]; then
  echo "driver_preview=${results_dir}/training_command.sh"
else
  cycle_receipts=$(mktemp "${results_dir}/job_cycles.XXXXXX")
  echo "job_cycle_receipts=${cycle_receipts}"
fi

# Array throttling does not express predecessor completion and can leave every
# pending cycle showing Priority. Chain standalone jobs with afterany to resume
# from the last checkpoint even after a failed allocation or walltime limit.
cycle_dependency=${SLURM_DEPENDENCY:-singleton}
for ((cycle = 1; cycle <= job_cycles; cycle++)); do
  cycle_args=("${submit_args[@]}" "--dependency=${cycle_dependency}" "${code_dir}/ray.sub")
  if [[ "${DRY_RUN:-0}" == "1" ]]; then
    printf 'sbatch '
    printf '%q ' "${cycle_args[@]}"
    printf '\n'
    cycle_dependency="afterany:<cycle-${cycle}-job-id>"
    continue
  fi
  cycle_response=$(sbatch "${cycle_args[@]}")
  cycle_job_id=${cycle_response%%;*}
  if [[ ! ${cycle_job_id} =~ ^[0-9]+$ ]]; then
    printf '%s\n' "${cycle_response}" >"${cycle_receipts}.unexpected-response"
    echo "ERROR: sbatch returned no numeric job ID; inspect ${cycle_receipts}.unexpected-response before submitting again" >&2
    exit 1
  fi
  printf '%s\t%s\t%s\n' "${cycle}" "${cycle_job_id}" "${cycle_dependency}" >>"${cycle_receipts}"
  echo "cycle=${cycle}/${job_cycles} job_id=${cycle_job_id} dependency=${cycle_dependency}"
  cycle_dependency="afterany:${cycle_job_id}"
done
