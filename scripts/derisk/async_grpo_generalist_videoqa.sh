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
config_in_container="examples/configs/recipes/vlm/vlm_grpo_videoqa_super_profile_band.yaml"
model_path="${MODEL_PATH:-/scratch/fsw/portfolios/nemotron/projects/nemotron_omni_vision/users/pulkitk/tracking/weights/full_generalist_12500_stage2_0828_iter3159}"
data_path="${DATA_PATH:-/lustre/fsw/portfolios/nemotron/users/arushig/nemo_gym_rl_video_0803/nemo_rl/results/combined_sav_caprl_20260822/train_sav_all_tracks_plus_caprl_exclude6215_cluster_paths.jsonl}"
tokenizer_chat_template="${TOKENIZER_CHAT_TEMPLATE:-default}"
vllm_chat_template="${VLLM_CHAT_TEMPLATE:-null}"
persistent_cache="${PERSISTENT_CACHE:-/scratch/fsw/portfolios/nemotron/projects/nemotron_omni_vision/users/ehosseiniasl/nemo_rl_cache}"
slurm_time_limit="${SLURM_TIME_LIMIT:-14:00:00}"

run_id="${RUN_ID:-$(date -u +%Y%m%d-%H%M%S)}"
# Gym environments are created lazily and are not safe for independent cold
# starts to populate concurrently. Keep one cache across this run's array
# windows while isolating it from other experiments.
gym_venv_dir="${GYM_VENV_DIR:-${persistent_cache}/gym_venvs_derisk/${run_id}}"
base_name="${BASE_NAME:-async_grpo_super35_latest_combined_sav_caprl_${run_id}}"
candidate_name="${base_name}_${SLURM_ACCOUNT}_${SLURM_PARTITION//,/_}"
results_dir="${RESULTS_DIR:-${code_dir}/results/${base_name}}"
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
train_global_batch_size=$((num_prompts * num_generations))

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
if [[ "${WANDB_MODE:-online}" != "offline" ]]; then
  [[ -n "${WANDB_API_KEY:-}" ]] || { echo "ERROR: WANDB_API_KEY is not set for online logging" >&2; exit 1; }
fi

mkdir -p \
  "${results_dir}" \
  "${slurm_log_dir}" \
  "${persistent_cache}/huggingface" \
  "${persistent_cache}/vllm_compile_cache_derisk" \
  "${persistent_cache}/flashinfer_cubins" \
  "${persistent_cache}/flashinfer_workspace" \
  "${persistent_cache}/megatron_ckpt_cache_derisk" \
  "${persistent_cache}/hf_config_locks" \
  "${gym_venv_dir}"

wandb_run_id="${WANDB_RUN_ID:-$(printf '%s' "${base_name}" | sha256sum | cut -c1-16)}"
wandb_project="${WANDB_PROJECT:-Nemotron-omni-RL-debug}"
cluster_name="${CLUSTER_NAME:-${SLURM_CLUSTER_NAME:-aws-cmh-slurm-1-v1}}"
wandb_run_name="${WANDB_RUN_NAME:-${cluster_name}_${base_name}}"

base_mounts="/lustre:/lustre,/scratch:/scratch"
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
export NRL_MEGATRON_CHECKPOINT_DIR="${persistent_cache}/megatron_ckpt_cache_derisk"
export MEGATRON_CONFIG_LOCK_DIR="${persistent_cache}/hf_config_locks"
export VLLM_CACHE_ROOT="${persistent_cache}/vllm_compile_cache_derisk"
export DG_JIT_CACHE_DIR="${persistent_cache}/vllm_compile_cache_derisk/deep_gemm"
export VLLM_DEEP_GEMM_WARMUP=skip
export FLASHINFER_CUBIN_DIR="${persistent_cache}/flashinfer_cubins"
export FLASHINFER_WORKSPACE_BASE="${persistent_cache}/flashinfer_workspace"
export NEMO_RL_VIDEO_MEDIA_ROOT=/
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
uv_bin=/root/.local/bin/uv
test -x "${uv_bin}"
export PYTHONPATH=/opt/nemo-rl:/opt/nemo-rl/3rdparty/Gym-workspace/Gym:/opt/nemo-rl/3rdparty/Megatron-Bridge-workspace/Megatron-Bridge/src:/opt/nemo-rl/3rdparty/Megatron-Bridge-workspace/Megatron-Bridge/3rdparty/Megatron-LM:${PYTHONPATH:-}
"${uv_bin}" run --no-sync python -c 'import sys; assert sys.version_info >= (3, 13, 14); import ray, transformers, megatron.core, nemo_rl.algorithms.grpo, nemo_rl.environments.nemo_gym'
generation_vllm_python=/opt/ray_venvs/nemo_rl.models.generation.vllm.vllm_worker_async.VllmAsyncGenerationWorker/bin/python
test -x "${generation_vllm_python}"
"${generation_vllm_python}" -c 'import vllm; from nemo_rl.models.generation.vllm.vllm_worker_async import VllmAsyncGenerationWorker'
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
  grpo.async_grpo.in_flight_weight_updates="${in_flight_weight_updates}" \
  grpo.async_grpo.recompute_kv_cache_after_weight_updates="${recompute_kv_cache_after_weight_updates}" \
  grpo.length_penalty.default.enabled="${length_penalty_enabled}" \
  grpo.length_penalty.profile_band.enabled="${profile_band_enabled}" \
  policy.router_replay.enabled="${router_replay_enabled}" \
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
echo "account=${SLURM_ACCOUNT} partition=${SLURM_PARTITION} nodes=${num_nodes} gpus_per_node=${gpus_per_node} training_nodes=${num_train_nodes} generation_nodes=${num_gen_nodes} policy_dp=${policy_dp_size} segment_size=${segment_size} prompts=${num_prompts} generations=${num_generations} steps=${max_steps} cycles=${job_cycles} save_period=${save_period} keep_top_k=${checkpoint_keep_top_k} length_penalty=${length_penalty_enabled} profile_band=${profile_band_enabled} router_replay=${router_replay_enabled} tokenizer_chat_template=${tokenizer_chat_template} vllm_chat_template=${vllm_chat_template}"
echo "container=${container}"
echo "results=${results_dir}"

if [[ "${DRY_RUN:-0}" == "1" ]]; then
  printf 'sbatch '
  printf '%q ' "${submit_args[@]}"
  printf '\n'
  exit 0
fi

sbatch "${submit_args[@]}"
