#!/usr/bin/env bash
# Self-contained launcher for combined CAPRL video + SA-V tracking GRPO.
# The code, recipe, Gym verifier, tokenizer/template contract, and media
# alignment checks all come from this checkout.
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
code_dir="$(cd "${script_dir}/../.." && pwd)"

export SLURM_ACCOUNT="${SLURM_ACCOUNT:-nemotron_omni_vision}"
export SLURM_PARTITION="${SLURM_PARTITION:-batch_long}"

env_file="${ENV_FILE:-/lustre/fsw/portfolios/nemotron/users/ehosseiniasl/.codex/credentials.env}"
if [[ ! -f "${env_file}" ]]; then
  env_file="${HOME}/.venv"
fi
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

container="${CONTAINER:-/scratch/fsw/portfolios/nemotron/projects/nemotron_omni_vision/users/pulkitk/tracking/images/nemo-rl:super35_20260901_prefetched_venvs_arm64.squashfs}"
config_in_container="${CONFIG_IN_CONTAINER:-examples/configs/recipes/vlm/vlm_grpo_videoqa_super_profile_band.yaml}"
model_path="${MODEL_PATH:-/scratch/fsw/portfolios/nemotron/projects/nemotron_omni_vision/users/pulkitk/tracking/weights/full_generalist_12500_stage2_0828_iter3159}"
data_path="${DATA_PATH:-/lustre/fsw/portfolios/nemotron/users/arushig/nemo_gym_rl_video_0803/nemo_rl/results/combined_sav_caprl_20260822/train_sav_all_tracks_plus_caprl_exclude6215_cluster_paths.jsonl}"
tokenizer_chat_template="${TOKENIZER_CHAT_TEMPLATE:-default}"
vllm_chat_template="${VLLM_CHAT_TEMPLATE:-null}"
persistent_cache="${PERSISTENT_CACHE:-/scratch/fsw/portfolios/nemotron/projects/nemotron_omni_vision/users/ehosseiniasl/nemo_rl_cache}"
require_complete_megatron_cache="${REQUIRE_COMPLETE_MEGATRON_CACHE:-false}"
slurm_time_limit="${SLURM_TIME_LIMIT:-14:00:00}"

run_id="${RUN_ID:-$(date -u +%Y%m%d-%H%M%S)}"
base_name="${BASE_NAME:-async_grpo_super35_latest_combined_sav_caprl_${run_id}}"
candidate_name="${base_name}_${SLURM_ACCOUNT}_${SLURM_PARTITION//,/_}"
results_dir="${RESULTS_DIR:-${code_dir}/results/${base_name}}"
run_cache_dir="${RUN_CACHE_DIR:-${persistent_cache}/runs/${base_name}}"
gym_venv_dir="${GYM_VENV_DIR:-${persistent_cache}/gym_venvs_derisk}"
hf_home="${RUN_HF_HOME:-${run_cache_dir}/huggingface}"
hf_modules_cache="${RUN_HF_MODULES_CACHE:-${hf_home}/modules}"
hf_config_lock_dir="${MEGATRON_CONFIG_LOCK_DIR:-${run_cache_dir}/hf_config_locks}"
vllm_cache_dir="${VLLM_CACHE_ROOT:-${run_cache_dir}/vllm_compile}"
flashinfer_cubin_dir="${FLASHINFER_CUBIN_DIR:-${run_cache_dir}/flashinfer_cubins}"
flashinfer_workspace_dir="${FLASHINFER_WORKSPACE_BASE:-${run_cache_dir}/flashinfer_workspace}"
torch_cache_dir="${TORCH_HOME:-${run_cache_dir}/torch}"
triton_cache_dir="${TRITON_CACHE_DIR:-/tmp/nemo_rl_triton_${base_name}}"
torchinductor_cache_dir="${TORCHINDUCTOR_CACHE_DIR:-/tmp/nemo_rl_torchinductor_${base_name}}"
megatron_checkpoint_dir="${MEGATRON_CHECKPOINT_DIR:-${run_cache_dir}/megatron_ckpt_cache}"
slurm_log_dir="${results_dir}/slurm"
job_cycles="${JOB_CYCLES:-20}"
num_nodes="${NUM_NODES:-32}"
gpus_per_node="${GPUS_PER_NODE:-4}"
num_gen_nodes="${NUM_GEN_NODES:-16}"
segment_size="${SEGMENT_SIZE:-8}"
num_prompts="${NUM_PROMPTS:-128}"
num_generations="${NUM_GENERATIONS:-16}"
max_steps="${MAX_STEPS:-100000}"
save_period="${SAVE_PERIOD:-5}"
checkpoint_keep_top_k="${CHECKPOINT_KEEP_TOP_K:-2}"
in_flight_weight_updates="${IN_FLIGHT_WEIGHT_UPDATES:-true}"
recompute_kv_cache_after_weight_updates="${RECOMPUTE_KV_CACHE_AFTER_WEIGHT_UPDATES:-false}"
length_penalty_enabled="${LENGTH_PENALTY_ENABLED:-true}"
profile_band_enabled="${PROFILE_BAND_ENABLED:-true}"
router_replay_enabled="${ROUTER_REPLAY_ENABLED:-false}"
router_replay_transport="${ROUTER_REPLAY_TRANSPORT:-}"
load_replay_buffer="${LOAD_REPLAY_BUFFER:-}"
use_leave_one_out_baseline="${USE_LEAVE_ONE_OUT_BASELINE:-false}"
use_leave_one_out_std="${USE_LEAVE_ONE_OUT_STD:-false}"
async_grpo_enabled="${ASYNC_GRPO_ENABLED:-true}"
max_trajectory_age_steps="${MAX_TRAJECTORY_AGE_STEPS:-1}"
freeze_moe_router="${FREEZE_MOE_ROUTER:-true}"
moe_router_load_balancing_type="${MOE_ROUTER_LOAD_BALANCING_TYPE:-none}"
moe_router_bias_update_rate="${MOE_ROUTER_BIAS_UPDATE_RATE:-0.0}"
train_global_batch_size=$((num_prompts * num_generations))

if [[ -z "${router_replay_transport}" ]]; then
  if [[ "${router_replay_enabled}" == "true" ]]; then
    router_replay_transport=ray
  else
    router_replay_transport=inline
  fi
fi
if [[ -z "${load_replay_buffer}" ]]; then
  if [[ "${router_replay_enabled}" == "true" && "${router_replay_transport}" == "ray" ]]; then
    # A new Slurm window starts a new Ray cluster, so references saved by the
    # previous cluster are not durable. Resume model/optimizer/dataloader state,
    # but regenerate the async lookahead buffer.
    load_replay_buffer=false
  else
    load_replay_buffer=true
  fi
fi

if (( num_gen_nodes <= 0 || num_gen_nodes >= num_nodes )); then
  echo "ERROR: NUM_GEN_NODES must be between 1 and NUM_NODES-1" >&2
  exit 1
fi
num_train_nodes=$((num_nodes - num_gen_nodes))
if (( num_train_nodes % segment_size != 0 || num_gen_nodes % segment_size != 0 )); then
  echo "ERROR: SEGMENT_SIZE must divide both policy and generation node counts" >&2
  exit 1
fi
# This recipe fixes TP=2, PP=1, and CP=1, so policy DP is the number of
# training GPUs divided by two.  Reject an incompatible node split before
# sbatch; otherwise async GRPO discovers it only after generating all 2048
# first-step sequences.
policy_model_parallel_size=2
policy_world_size=$((num_train_nodes * gpus_per_node))
if (( policy_world_size % policy_model_parallel_size != 0 )); then
  echo "ERROR: policy GPU count ${policy_world_size} is not divisible by recipe model-parallel size ${policy_model_parallel_size}" >&2
  exit 1
fi
policy_dp_size=$((policy_world_size / policy_model_parallel_size))
if (( train_global_batch_size % policy_dp_size != 0 )); then
  echo "ERROR: rollout batch ${train_global_batch_size} is not divisible by policy DP ${policy_dp_size} (${num_train_nodes} training nodes). Choose a compatible NUM_GEN_NODES; for NUM_NODES=20 use NUM_GEN_NODES=12, and for NUM_NODES=32 use NUM_GEN_NODES=16." >&2
  exit 1
fi

[[ -f "${container}" ]] || { echo "ERROR: container not found: ${container}" >&2; exit 1; }
[[ -f "${model_path}/config.json" ]] || { echo "ERROR: checkpoint not found: ${model_path}" >&2; exit 1; }
[[ -s "${data_path}" ]] || { echo "ERROR: dataset not found: ${data_path}" >&2; exit 1; }
[[ -f "${code_dir}/${config_in_container}" ]] || { echo "ERROR: recipe not found" >&2; exit 1; }
[[ -f "${code_dir}/3rdparty/Gym-workspace/Gym/resources_servers/sav_tracks/app.py" ]] || {
  echo "ERROR: SA-V tracks verifier not in this clone's Gym" >&2
  exit 1
}
if [[ "${require_complete_megatron_cache}" == "true" ]]; then
  complete_cache_found=false
  shopt -s nullglob
  for iter_dir in "${megatron_checkpoint_dir}"/*/iter_0000000; do
    cache_root="${iter_dir%/iter_0000000}"
    if [[ -s "${cache_root}/latest_checkpointed_iteration.txt" ]] \
      && [[ -s "${iter_dir}/run_config.yaml" ]] \
      && { [[ -s "${iter_dir}/metadata.json" ]] || [[ -s "${iter_dir}/.metadata" ]]; }; then
      complete_cache_found=true
      break
    fi
  done
  shopt -u nullglob
  [[ "${complete_cache_found}" == "true" ]] || {
    echo "ERROR: no finalized Megatron conversion under ${megatron_checkpoint_dir}" >&2
    exit 1
  }
fi
if [[ "${WANDB_MODE:-online}" != "offline" ]]; then
  [[ -n "${WANDB_API_KEY:-}" ]] || { echo "ERROR: WANDB_API_KEY is not set for online logging" >&2; exit 1; }
fi

mkdir -p \
  "${results_dir}" \
  "${slurm_log_dir}" \
  "${run_cache_dir}" \
  "${hf_home}" \
  "${hf_modules_cache}" \
  "${hf_config_lock_dir}" \
  "${vllm_cache_dir}" \
  "${flashinfer_cubin_dir}" \
  "${flashinfer_workspace_dir}" \
  "${torch_cache_dir}" \
  "${megatron_checkpoint_dir}" \
  "${gym_venv_dir}"

wandb_run_id="${WANDB_RUN_ID:-$(printf '%s' "${base_name}" | sha256sum | cut -c1-16)}"
wandb_project="${WANDB_PROJECT:-Nemotron-omni-RL-debug}"
cluster_name="${CLUSTER_NAME:-${SLURM_CLUSTER_NAME:-aws-cmh-slurm-1-v1}}"
wandb_run_name="${WANDB_RUN_NAME:-${cluster_name}_${base_name}}"

base_mounts="/lustre:/lustre,/scratch:/scratch,/home:/home"
selective_mounts="${code_dir}/nemo_rl:/opt/nemo-rl/nemo_rl"
selective_mounts+=",${code_dir}/examples:/opt/nemo-rl/examples"
selective_mounts+=",${code_dir}/tools:/opt/nemo-rl/tools"
selective_mounts+=",${code_dir}/3rdparty/Gym-workspace/Gym:/opt/nemo-rl/3rdparty/Gym-workspace/Gym"
selective_mounts+=",${code_dir}/3rdparty/Megatron-Bridge-workspace/Megatron-Bridge:/opt/nemo-rl/3rdparty/Megatron-Bridge-workspace/Megatron-Bridge"
export MOUNTS="${base_mounts},${selective_mounts}"
export CONTAINER="${container}"
export BASE_LOG_DIR="${slurm_log_dir}"
export RAY_SUB_PATH="${code_dir}/ray.sub"
export GPUS_PER_NODE="${gpus_per_node}"
export SANDBOX_COMMAND=""
export SANDBOX_CONTAINER=""
export RAY_ENABLE_UV_RUN_RUNTIME_ENV=0
export NEMO_RL_VENV_DIR=/opt/ray_venvs
export NEMO_GYM_VENV_DIR="${gym_venv_dir}"
export NRL_FORCE_REBUILD_VENVS=false
export NRL_IGNORE_VERSION_MISMATCH=1
export NRL_WG_USE_RAY_REF=1
export NRL_MEGATRON_CHECKPOINT_DIR="${megatron_checkpoint_dir}"
export MEGATRON_CONFIG_LOCK_DIR="${hf_config_lock_dir}"
export VLLM_CACHE_ROOT="${vllm_cache_dir}"
export DG_JIT_CACHE_DIR="${vllm_cache_dir}/deep_gemm"
export VLLM_DEEP_GEMM_WARMUP=skip
export FLASHINFER_CUBIN_DIR="${flashinfer_cubin_dir}"
export FLASHINFER_WORKSPACE_BASE="${flashinfer_workspace_dir}"
export TORCH_HOME="${torch_cache_dir}"
export TRITON_CACHE_DIR="${triton_cache_dir}"
export TORCHINDUCTOR_CACHE_DIR="${torchinductor_cache_dir}"
export NEMO_RL_VIDEO_MEDIA_ROOT=/
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HOME="${hf_home}"
export HF_MODULES_CACHE="${hf_modules_cache}"
export NRL_MODEL_PATH="${model_path}"

read -r -d '' SETUP_COMMAND <<'SETUP_EOF' || true
set -euo pipefail
cd /opt/nemo-rl
export UV_PROJECT_ENVIRONMENT=/opt/nemo_rl_venv
export UV_LINK_MODE=copy
export NRL_CONTAINER=1
export RAY_USAGE_STATS_ENABLED=0
uv_bin=/root/.local/bin/uv
test -x "${uv_bin}"
export PYTHONPATH=/opt/nemo-rl:/opt/nemo-rl/3rdparty/Gym-workspace/Gym:/opt/nemo-rl/3rdparty/Megatron-Bridge-workspace/Megatron-Bridge/src:/opt/nemo-rl/3rdparty/Megatron-Bridge-workspace/Megatron-Bridge/3rdparty/Megatron-LM:${PYTHONPATH:-}
"${uv_bin}" run --no-sync python -c 'import sys; assert sys.version_info >= (3, 13, 14); import ray, megatron.core, nemo_rl.algorithms.grpo, nemo_rl.environments.nemo_gym; from transformers import AutoConfig, AutoProcessor, AutoTokenizer'
generation_vllm_python=/opt/ray_venvs/nemo_rl.models.generation.vllm.vllm_worker_async.VllmAsyncGenerationWorker/bin/python
test -x "${generation_vllm_python}"
"${generation_vllm_python}" -c 'import vllm; from nemo_rl.models.generation.vllm.vllm_worker_async import VllmAsyncGenerationWorker'
"${generation_vllm_python}" -c 'import os; from transformers import AutoConfig, AutoProcessor, AutoTokenizer; p=os.environ["NRL_MODEL_PATH"]; AutoConfig.from_pretrained(p, trust_remote_code=True); AutoProcessor.from_pretrained(p, trust_remote_code=True, use_fast=True); AutoTokenizer.from_pretrained(p, trust_remote_code=True, use_fast=True); print("Prewarmed vLLM HF dynamic modules cache")'
SETUP_EOF
export SETUP_COMMAND

read -r -d '' COMMAND <<COMMAND_EOF || true
set -euo pipefail
cd /opt/nemo-rl
export UV_PROJECT_ENVIRONMENT=/opt/nemo_rl_venv
export UV_LINK_MODE=copy
export NRL_CONTAINER=1
export RAY_USAGE_STATS_ENABLED=0
uv_bin=/root/.local/bin/uv
test -x "\${uv_bin}"
export PYTHONPATH=/opt/nemo-rl:/opt/nemo-rl/3rdparty/Gym-workspace/Gym:/opt/nemo-rl/3rdparty/Megatron-Bridge-workspace/Megatron-Bridge/src:/opt/nemo-rl/3rdparty/Megatron-Bridge-workspace/Megatron-Bridge/3rdparty/Megatron-LM:\${PYTHONPATH:-}
export HF_HOME=${HF_HOME}
export HF_MODULES_CACHE=${HF_MODULES_CACHE}
export VLLM_CACHE_ROOT=${VLLM_CACHE_ROOT}
export DG_JIT_CACHE_DIR=${DG_JIT_CACHE_DIR}
export FLASHINFER_CUBIN_DIR=${FLASHINFER_CUBIN_DIR}
export FLASHINFER_WORKSPACE_BASE=${FLASHINFER_WORKSPACE_BASE}
export TORCH_HOME=${TORCH_HOME}
export TRITON_CACHE_DIR=${TRITON_CACHE_DIR}
export TORCHINDUCTOR_CACHE_DIR=${TORCHINDUCTOR_CACHE_DIR}
export RAY_ENABLE_UV_RUN_RUNTIME_ENV=0
export NEMO_RL_VENV_DIR=/opt/ray_venvs
export NEMO_GYM_VENV_DIR=${gym_venv_dir}
export NRL_FORCE_REBUILD_VENVS=false
export NEMO_RL_VIDEO_MEDIA_ROOT=/
"\${uv_bin}" run --no-sync python -c "from transformers import AutoConfig, AutoProcessor, AutoTokenizer; p='${model_path}'; AutoConfig.from_pretrained(p, trust_remote_code=True); AutoProcessor.from_pretrained(p, trust_remote_code=True, use_fast=True); AutoTokenizer.from_pretrained(p, trust_remote_code=True, use_fast=True); print('Prewarmed HF dynamic modules cache')"
"\${uv_bin}" run --no-sync python examples/nemo_gym/run_grpo_nemo_gym.py \
  --config "${config_in_container}" \
  checkpointing.checkpoint_dir="${results_dir}/checkpoints" \
  logger.log_dir="${results_dir}/logs" \
  logger.wandb_enabled=true \
  logger.wandb.project="${wandb_project}" \
  logger.wandb.name="${wandb_run_name}" \
  ++logger.wandb.entity=adlr \
  ++logger.wandb.id="${wandb_run_id}" \
  ++logger.wandb.resume=allow \
  ++env.nemo_gym.uv_venv_dir="${gym_venv_dir}" \
  env.nemo_gym.skip_venv_if_present=true \
  policy.model_name="${model_path}" \
  policy.tokenizer.chat_template="${tokenizer_chat_template}" \
  policy.generation.vllm_cfg.http_server_serving_chat_kwargs.chat_template="${vllm_chat_template}" \
  data.train.data_path="${data_path}" \
  data.validation.data_path="${data_path}" \
  grpo.num_prompts_per_step="${num_prompts}" \
  grpo.num_generations_per_prompt="${num_generations}" \
  grpo.max_num_steps="${max_steps}" \
  policy.train_global_batch_size="${train_global_batch_size}" \
  checkpointing.save_period="${save_period}" \
  checkpointing.keep_top_k="${checkpoint_keep_top_k}" \
  grpo.async_grpo.enabled="${async_grpo_enabled}" \
  grpo.async_grpo.max_trajectory_age_steps="${max_trajectory_age_steps}" \
  grpo.async_grpo.in_flight_weight_updates="${in_flight_weight_updates}" \
  grpo.async_grpo.recompute_kv_cache_after_weight_updates="${recompute_kv_cache_after_weight_updates}" \
  grpo.length_penalty.default.enabled="${length_penalty_enabled}" \
  grpo.length_penalty.profile_band.enabled="${profile_band_enabled}" \
  grpo.use_leave_one_out_baseline="${use_leave_one_out_baseline}" \
  ++grpo.adv_estimator.use_leave_one_out_baseline="${use_leave_one_out_baseline}" \
  ++grpo.adv_estimator.use_leave_one_out_std="${use_leave_one_out_std}" \
  policy.megatron_cfg.freeze_moe_router="${freeze_moe_router}" \
  policy.megatron_cfg.moe_router_load_balancing_type="${moe_router_load_balancing_type}" \
  policy.megatron_cfg.moe_router_bias_update_rate="${moe_router_bias_update_rate}" \
  policy.router_replay.enabled="${router_replay_enabled}" \
  ++policy.router_replay.transport="${router_replay_transport}" \
  ++checkpointing.load_replay_buffer="${load_replay_buffer}" \
  cluster.num_nodes="${num_nodes}" \
  cluster.gpus_per_node="${gpus_per_node}" \
  cluster.segment_size="${segment_size}" \
  policy.generation.colocated.resources.num_nodes="${num_gen_nodes}" \
  policy.generation.colocated.resources.gpus_per_node="${gpus_per_node}" ${EXTRA_OVERRIDES:-}
COMMAND_EOF
export COMMAND

last_array_task=$((job_cycles - 1))
submit_args=(
  --parsable
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
  "${code_dir}/ray.sub"
)

echo "candidate=${candidate_name}"
echo "account=${SLURM_ACCOUNT} partition=${SLURM_PARTITION} nodes=${num_nodes} gpus_per_node=${gpus_per_node} training_nodes=${num_train_nodes} generation_nodes=${num_gen_nodes} policy_dp=${policy_dp_size} segment_size=${segment_size} prompts=${num_prompts} generations=${num_generations} steps=${max_steps} cycles=${job_cycles} save_period=${save_period} keep_top_k=${checkpoint_keep_top_k} async=${async_grpo_enabled} age=${max_trajectory_age_steps} length_penalty=${length_penalty_enabled} profile_band=${profile_band_enabled} router_replay=${router_replay_enabled} router_transport=${router_replay_transport} load_replay_buffer=${load_replay_buffer} leave_one_out=${use_leave_one_out_baseline} leave_one_out_std=${use_leave_one_out_std} freeze_moe_router=${freeze_moe_router} moe_load_balancing=${moe_router_load_balancing_type} moe_bias_rate=${moe_router_bias_update_rate} tokenizer_chat_template=${tokenizer_chat_template} vllm_chat_template=${vllm_chat_template}"
echo "container=${container}"
echo "results=${results_dir}"
echo "run_cache=${run_cache_dir}"
echo "gym_venv_cache=${gym_venv_dir}"
echo "hf_cache=${hf_home}"
echo "vllm_cache=${vllm_cache_dir}"
echo "megatron_checkpoint_cache=${megatron_checkpoint_dir} require_complete=${require_complete_megatron_cache}"

if [[ "${DRY_RUN:-0}" == "1" ]]; then
  printf 'sbatch '
  printf '%q ' "${submit_args[@]}"
  printf '\n'
  exit 0
fi

sbatch "${submit_args[@]}"
