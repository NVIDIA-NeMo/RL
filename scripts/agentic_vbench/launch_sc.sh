#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Submit AgenticVBench SingleController GRPO (token capture, optional context
# compaction) on Slurm with the container built for this checkout.
# Adapted from pivot-rl-video/RL/scripts/run_crop_video_pivot.sh.
# DRY_RUN=1 writes the driver/setup commands and prints the sbatch line only.
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
code_dir="$(cd "${script_dir}/../.." && pwd)"

export SLURM_ACCOUNT="${SLURM_ACCOUNT:-nemotron_n4_omni}"
export SLURM_PARTITION="${SLURM_PARTITION:-batch_long}"

env_file="${ENV_FILE:-${HOME}/.venv}"
if [[ -f "${env_file}" ]]; then
  set -a
  # shellcheck disable=SC1090
  source "${env_file}"
  set +a
fi
if [[ -z "${WANDB_API_KEY:-}" ]]; then
  echo "WANDB_API_KEY not found; using WANDB_MODE=offline." >&2
  export WANDB_MODE=offline
fi

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
persistent_cache="${PERSISTENT_CACHE:-/lustre/fsw/portfolios/nemotron/users/arushig/nemo_rl_cache}"
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
_budget_s=$(( $(_limit_to_seconds "$slurm_time_limit") - ${CHECKPOINT_MARGIN_SECONDS:-1200} ))
(( _budget_s < 600 )) && _budget_s=600
checkpoint_must_save_by="${CHECKPOINT_MUST_SAVE_BY:-$(_seconds_to_ddhhmmss "$_budget_s")}"

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
# Harbor episodes (trial artifacts, runner logs, recorded model calls) per run.
export AGENTIC_VBENCH_OUTPUT_ROOT="${AGENTIC_VBENCH_OUTPUT_ROOT:-${results_dir}/episodes}"
slurm_log_dir="${results_dir}/slurm"
job_cycles="${JOB_CYCLES:-1}"
num_nodes="${NUM_NODES:-16}"
gpus_per_node="${GPUS_PER_NODE:-8}"
num_gen_nodes="${NUM_GEN_NODES:-8}"
# Nodes per NVLink domain for topology-aware placement (8 for 16-node, 4 for 8-node runs).
segment_size="${SEGMENT_SIZE:-8}"
num_prompts="${NUM_PROMPTS:-8}"
num_generations="${NUM_GENERATIONS:-4}"
max_steps="${MAX_STEPS:-20}"
save_period="${SAVE_PERIOD:-10}"
checkpoint_keep_top_k="${CHECKPOINT_KEEP_TOP_K:-2}"
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

base_mounts="/lustre:/lustre,/scratch:/scratch"
selective_mounts="${code_dir}/nemo_rl:/opt/nemo-rl/nemo_rl"
selective_mounts+=",${code_dir}/examples:/opt/nemo-rl/examples"
selective_mounts+=",${code_dir}/tools:/opt/nemo-rl/tools"
selective_mounts+=",${code_dir}/scripts:/opt/nemo-rl/scripts"
selective_mounts+=",${code_dir}/3rdparty/Gym-workspace/Gym:/opt/nemo-rl/3rdparty/Gym-workspace/Gym"
selective_mounts+=",${code_dir}/3rdparty/Megatron-Bridge-workspace/Megatron-Bridge:/opt/nemo-rl/3rdparty/Megatron-Bridge-workspace/Megatron-Bridge"
# The mounts above shadow the image's copies: an uninitialized submodule would hide
# the installed Gym or Megatron-Bridge/Megatron-LM (ModuleNotFoundError: megatron.bridge
# at worker creation, after the GPUs are allocated). Fail here instead.
for sub in 3rdparty/Gym-workspace/Gym 3rdparty/Megatron-Bridge-workspace/Megatron-Bridge 3rdparty/Megatron-Bridge-workspace/Megatron-Bridge/3rdparty/Megatron-LM; do
  [[ -n "$(ls -A "${code_dir}/${sub}" 2>/dev/null)" ]] || { echo "ERROR: ${sub} is empty; run: git submodule update --init --recursive ${sub%%/3rdparty/Megatron-LM}" >&2; exit 1; }
done
export MOUNTS="${base_mounts},${selective_mounts}${EXTRA_MOUNTS:+,${EXTRA_MOUNTS}}"
export CONTAINER="${container}"
export BASE_LOG_DIR="${slurm_log_dir}"
export RAY_SUB_PATH="${code_dir}/ray.sub"
# Sync Ray's own session logs (gcs_server/raylet) to the shared log dir often enough
# that an early GCS death still leaves evidence behind.
export RAY_LOG_SYNC_FREQUENCY="${RAY_LOG_SYNC_FREQUENCY:-15}"
export GPUS_PER_NODE="${gpus_per_node}"
export SANDBOX_COMMAND=""
export SANDBOX_CONTAINER=""
export RAY_ENABLE_UV_RUN_RUNTIME_ENV=0
export NEMO_RL_VENV_DIR=/opt/ray_venvs
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

read -r -d '' SETUP_COMMAND <<'SETUP_EOF' || true
set -euo pipefail
cd /opt/nemo-rl
export UV_PROJECT_ENVIRONMENT=/opt/nemo_rl_venv
export UV_LINK_MODE=copy
export NRL_CONTAINER=1
export RAY_USAGE_STATS_ENABLED=0
# The driver venv has no nemo_gym extra; the driver imports Gym's config helpers from the source tree.
export PYTHONPATH=/opt/nemo-rl:/opt/nemo-rl/3rdparty/Gym-workspace/Gym:${PYTHONPATH:-}
uv_bin=$(command -v uv || echo /usr/local/bin/uv)
test -x "${uv_bin}"
bash "${TRAINING_DEPS_SETUP_SCRIPT:-/opt/nemo-rl/tools/ensure_agentic_vbench_deps.sh}"
"${uv_bin}" run --no-sync python -c 'import sys; assert sys.version_info >= (3, 13, 14); import ray, transformers, nemo_rl.algorithms.single_controller, nemo_rl.environments.nemo_gym'
generation_vllm_python=/opt/ray_venvs/nemo_rl.models.generation.vllm.vllm_worker_async.VllmAsyncGenerationWorker/bin/python
test -x "${generation_vllm_python}"
"${generation_vllm_python}" -c 'import vllm; from nemo_rl.models.generation.vllm.vllm_worker_async import VllmAsyncGenerationWorker; from nemo_rl.models.generation.capture_context import decide_capture_input'
megatron_python=/opt/ray_venvs/nemo_rl.models.policy.workers.megatron_policy_worker.MegatronPolicyWorker/bin/python
test -x "${megatron_python}"
"${megatron_python}" -c 'import megatron.core, transformer_engine'
SETUP_EOF
export SETUP_COMMAND

read -r -d '' COMMAND <<COMMAND_EOF || true
set -euo pipefail
cd /opt/nemo-rl
export UV_PROJECT_ENVIRONMENT=/opt/nemo_rl_venv
export UV_LINK_MODE=copy
export NRL_CONTAINER=1
export RAY_USAGE_STATS_ENABLED=0
export PYTHONPATH=/opt/nemo-rl:/opt/nemo-rl/3rdparty/Gym-workspace/Gym:\${PYTHONPATH:-}
uv_bin=\$(command -v uv || echo /usr/local/bin/uv)
test -x "\${uv_bin}"
export HF_HOME=${HF_HOME}
export HF_MODULES_CACHE=${HF_MODULES_CACHE}
export RAY_ENABLE_UV_RUN_RUNTIME_ENV=0
export NEMO_RL_VENV_DIR=/opt/ray_venvs
export NEMO_GYM_VENV_DIR=${gym_venv_dir}
export NRL_FORCE_REBUILD_VENVS=false
# Harbor/Podman scratch for this job's episodes, on the node running the Gym servers.
export AGENTIC_VBENCH_RUNTIME_ROOT=/tmp/avb-runtime-\${SLURM_JOB_ID:-local}-\$\$
mkdir -p "\${AGENTIC_VBENCH_RUNTIME_ROOT}"
"\${uv_bin}" run --no-sync python -c "from transformers import AutoConfig, AutoProcessor, AutoTokenizer; p='${model_path}'; AutoConfig.from_pretrained(p, trust_remote_code=True); AutoProcessor.from_pretrained(p, trust_remote_code=True, use_fast=True); AutoTokenizer.from_pretrained(p, trust_remote_code=True, use_fast=True); print('Prewarmed HF dynamic modules cache')"
# Populate the cache with the exact Transformers environment used by vLLM.
# A second fresh process verifies imports before concurrent actor model loads.
for prewarm_pass in populate verify; do
  /opt/ray_venvs/nemo_rl.models.generation.vllm.vllm_worker_async.VllmAsyncGenerationWorker/bin/python -c "from transformers import AutoConfig, AutoProcessor, AutoTokenizer; p='${model_path}'; c=AutoConfig.from_pretrained(p, trust_remote_code=True); AutoProcessor.from_pretrained(p, trust_remote_code=True, use_fast=True); AutoTokenizer.from_pretrained(p, trust_remote_code=True, use_fast=True); print('vLLM prewarm: \${prewarm_pass}', type(c).__module__, type(c).__name__)"
done
# SKIP_REPLAY_RESTORE=1 (default): resume weights + dataloader cursor only. The
# native TQ snapshot restore logs "dtype mismatch: existing=int64, incoming=
# float32" for both partitions and the first step trained on restored rows had a
# token_mult_prob_error of 1140 (run 2209083/2210161 step 2). Without
# replay_buffer_metadata.pt the controller takes its documented empty-buffer
# path and regenerates the buffered groups (costs about one step per restart).
if [[ "${SKIP_REPLAY_RESTORE:-1}" == "1" ]]; then
  # rollout_recovery.pt must go with it: the controller refuses a recovery file
  # whose TQ snapshot is not being restored.
  for meta in "${checkpoint_dir}"/step_*/replay_buffer_metadata.pt "${checkpoint_dir}"/step_*/rollout_recovery.pt; do
    [[ -f "\$meta" ]] && mv "\$meta" "\$meta.disabled" && echo "[launch_sc] disabled replay restore: \$meta"
  done
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
  ++env.nemo_gym.uv_venv_dir="${gym_venv_dir}" \
  env.nemo_gym.skip_venv_if_present=true \
  policy.model_name="${model_path}" \
  policy.generation.vllm_cfg.load_format="${vllm_load_format}" \
  policy.tokenizer.chat_template="${tokenizer_chat_template}" \
  policy.generation.vllm_cfg.http_server_serving_chat_kwargs.chat_template="${vllm_chat_template}" \
  data.train.data_path="${TRAIN_PATH}" \
  data.validation.data_path="${VAL_PATH}" \
  grpo.num_prompts_per_step="${num_prompts}" \
  grpo.num_generations_per_prompt="${num_generations}" \
  grpo.max_num_steps="${max_steps}" \
  checkpointing.checkpoint_must_save_by="${checkpoint_must_save_by}" \
  policy.train_global_batch_size="${train_global_batch_size}" \
  checkpointing.save_period="${save_period}" \
  checkpointing.keep_top_k="${checkpoint_keep_top_k}" \
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

last_array_task=$((job_cycles - 1))
# The cluster's OccupiedIdleGPUsJobReaper cancels jobs whose GPUs look idle;
# model loading, Harbor container setup and the first long-episode wave keep the
# generation GPUs idle for well over an hour. Declare the exemption up front.
sbatch_comment="${SBATCH_COMMENT:-{\"OccupiedIdleGPUsJobReaper\":{\"exemptIdleTimeMins\":\"240\",\"reason\":\"data_loading\",\"description\":\"Model load, task-container setup and agentic rollouts keep GPUs idle for up to 240 minutes (training GPUs wait a whole generation phase)\"}}}"
submit_args=(
  --parsable
  --comment="${sbatch_comment}"
  --nodes="${num_nodes}"
  --gres="gpu:${gpus_per_node}"
  --exclusive
  --time="${slurm_time_limit}"
  --dependency=singleton
  --array="0-${last_array_task}%1"
  --account="${SLURM_ACCOUNT}"
  --partition="${SLURM_PARTITION}"
  --job-name="${candidate_name}"
  --output="${slurm_log_dir}/%A_%a-${SLURM_ACCOUNT}.out"
  --error="${slurm_log_dir}/%A_%a-${SLURM_ACCOUNT}.out"
)
if [[ -n ${SLURM_QOS:-} ]]; then
  submit_args+=(--qos="${SLURM_QOS}")
fi
submit_args+=("${code_dir}/ray.sub")

echo "candidate=${candidate_name}"
echo "sbatch_comment=${sbatch_comment}"
echo "time_limit=${slurm_time_limit} checkpoint_must_save_by=${checkpoint_must_save_by}"
echo "account=${SLURM_ACCOUNT} partition=${SLURM_PARTITION} qos=${SLURM_QOS:-default} nodes=${num_nodes} gpus_per_node=${gpus_per_node} generation_nodes=${num_gen_nodes} segment_size=${segment_size} prompts=${num_prompts} generations=${num_generations} steps=${max_steps} cycles=${job_cycles} save_period=${save_period} keep_top_k=${checkpoint_keep_top_k} tokenizer_chat_template=${tokenizer_chat_template} vllm_chat_template=${vllm_chat_template} vllm_load_format=${vllm_load_format}"
echo "context_compaction=${context_compaction} opencode_compaction_auto=${AGENTIC_VBENCH_COMPACTION_AUTO} opencode_prune=${AGENTIC_VBENCH_COMPACTION_PRUNE} max_turns=${AGENTIC_VBENCH_MAX_TURNS:-32} context_tokens=${AGENTIC_VBENCH_CONTEXT_TOKENS:-49152} output_tokens=${AGENTIC_VBENCH_OUTPUT_TOKENS:-8192} tasks=${AGENTIC_VBENCH_TASKS:-all}"
echo "episodes=${AGENTIC_VBENCH_OUTPUT_ROOT}"
echo "container=${container}"
echo "config=${config_in_container}"
echo "model=${model_path}"
echo "train=${TRAIN_PATH}"
echo "validation=${VAL_PATH}"
echo "results=${results_dir}"
echo "checkpoints=${checkpoint_dir}"
echo "token_capture=${token_capture_dir}"
# Record the AgenticVBench environment the Gym agent will read, for reproducibility.
env | grep -E '^(AGENTIC_VBENCH_|AVB_DATA_ROOT|TRAIN_PATH|VAL_PATH|MODEL_PATH|CONTEXT_COMPACTION|NRL_MEGATRON_|WANDB_MODE)' | grep -v -i 'key\|token\|secret' | sort > "${results_dir}/avb_environment.env"

if [[ "${DRY_RUN:-0}" == "1" ]]; then
  printf '%s\n' "${COMMAND}" > "${results_dir}/training_command.sh"
  printf '%s\n' "${SETUP_COMMAND}" > "${results_dir}/setup_command.sh"
  bash -n "${results_dir}/training_command.sh"
  bash -n "${results_dir}/setup_command.sh"
  echo "driver_preview=${results_dir}/training_command.sh"
  printf 'sbatch '
  printf '%q ' "${submit_args[@]}"
  printf '\n'
  exit 0
fi

sbatch "${submit_args[@]}"
