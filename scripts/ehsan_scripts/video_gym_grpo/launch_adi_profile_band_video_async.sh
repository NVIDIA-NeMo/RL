#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
code_dir="$(cd "${script_dir}/../../.." && pwd)"

: "${SLURM_ACCOUNT:?Set SLURM_ACCOUNT for this scheduler candidate}"
: "${SLURM_PARTITION:?Set SLURM_PARTITION for this scheduler candidate}"

container="${CONTAINER:-/lustre/fsw/portfolios/llmservice/users/ehosseiniasl/github_repos/nemorl/RL-3604/.cache/containers/nemo-rl-main_20260807-super-ultra-omni-oci-flat-x86_64.sqsh}"
config_in_container="examples/configs/recipes/vlm/vlm_grpo-nemotron-omni-30ba3b-16n8g-megatron-tp2ep16-async-gym-video-adi-profile-band.v1.yaml"
model_path="/lustre/fsw/portfolios/llmservice/users/arushig/workspace/output/generalist-49k-video-mpo-20260812-113202/checkpoints/tp_1_hf/iter_0000125/mcore_to_hf"
data_path="/lustre/fsw/portfolios/llmservice/users/arushig/nemo_gym_rl_video_0803/nemo-rl/results/video_frame_cache_caprl_passrate_n5_easy_to_hard_lt60s_20260806_f64/stable_split_95_5/train_exclude_line6215.jsonl"

run_id="${RUN_ID:-$(date -u +%Y%m%d-%H%M%S)}"
base_name="${BASE_NAME:-vg16a_adi_rpb_${run_id}}"
candidate_name="${base_name}_${SLURM_ACCOUNT}_${SLURM_PARTITION//,/_}"
results_dir="${RESULTS_DIR:-${code_dir}/results/${base_name}}"
slurm_log_dir="${results_dir}/slurm"
race_root="${RACE_ROOT:-${results_dir}/scheduler_race}"
race_claim_dir="${RACE_CLAIM_DIR:-${race_root}/claim}"
job_cycles="${JOB_CYCLES:-20}"

[[ -f "${container}" ]] || { echo "ERROR: container not found: ${container}" >&2; exit 1; }
[[ -f "${model_path}/config.json" ]] || { echo "ERROR: checkpoint not found: ${model_path}" >&2; exit 1; }
[[ -s "${data_path}" ]] || { echo "ERROR: dataset not found: ${data_path}" >&2; exit 1; }
[[ -f "${code_dir}/${config_in_container}" ]] || { echo "ERROR: recipe not found" >&2; exit 1; }
[[ "${SLURM_ACCOUNT}" != "nemotron_edge_text" ]] || { echo "ERROR: nemotron_edge_text is paused by user request" >&2; exit 1; }
if [[ "${WANDB_MODE:-online}" != "offline" ]]; then
  [[ -n "${WANDB_API_KEY:-}" ]] || { echo "ERROR: WANDB_API_KEY is not set for online logging" >&2; exit 1; }
fi

mkdir -p "${results_dir}" "${slurm_log_dir}" "${race_root}" "${code_dir}/.cache/huggingface"

wandb_run_id="${WANDB_RUN_ID:-$(printf '%s' "${base_name}" | sha256sum | cut -c1-16)}"
wandb_project="${WANDB_PROJECT:-Nemotron-omni-RL-debug}"
wandb_run_name="${WANDB_RUN_NAME:-vg16a_adi_rpb_mpo125}"

base_mounts="/lustre:/lustre"
selective_mounts="${code_dir}/nemo_rl:/opt/nemo-rl/nemo_rl"
selective_mounts+=",${code_dir}/examples:/opt/nemo-rl/examples"
selective_mounts+=",${code_dir}/tools:/opt/nemo-rl/tools"
selective_mounts+=",${script_dir}/adi_profile_band_ray_entrypoint:/usr/local/bin/ray"
selective_mounts+=",${script_dir}/adi_profile_band_uv_entrypoint:/usr/local/bin/uv"
selective_mounts+=",${code_dir}/3rdparty/Gym-workspace/Gym:/opt/nemo-rl/3rdparty/Gym-workspace/Gym"
selective_mounts+=",${code_dir}/3rdparty/Megatron-Bridge-workspace/Megatron-Bridge:/opt/nemo-rl/3rdparty/Megatron-Bridge-workspace/Megatron-Bridge"
selective_mounts+=",/lustre/fsw/portfolios/llmservice/users/ehosseiniasl/github_repos/nemorl/RL-3604/3rdparty/Megatron-Bridge-workspace/Megatron-Bridge/3rdparty/Megatron-LM:/opt/nemo-rl/3rdparty/Megatron-Bridge-workspace/Megatron-Bridge/3rdparty/Megatron-LM"
export MOUNTS="${base_mounts},${selective_mounts}"
export CONTAINER="${container}"
export BASE_LOG_DIR="${slurm_log_dir}"
export RAY_SUB_PATH="${code_dir}/ray.sub"
export RACE_CLAIM_DIR="${race_claim_dir}"
export GPUS_PER_NODE=8
export RAY_ENABLE_UV_RUN_RUNTIME_ENV=0
export NEMO_RL_VENV_DIR=/opt/ray_venvs
export NEMO_GYM_VENV_DIR=/opt/gym_venvs
export NRL_FORCE_REBUILD_VENVS=false
export NRL_IGNORE_VERSION_MISMATCH=1
export NEMO_GYM_VLLM_TOKEN_ID_SOURCE=native
export NEMO_GYM_VLLM_RETURN_TOKEN_IDS=1
export NEMO_RL_VIDEO_MEDIA_ROOT=/lustre/fsw/portfolios/llmservice/users/arushig/nemo_gym_rl_video_0803/nemo-rl/results/video_frame_cache_caprl_passrate_n5_easy_to_hard_lt60s_20260806_f64
export VLLM_VIDEO_LOADER_BACKEND=nemotron_vl
export VLLM_NANO_VL_POLICY_IMAGE_RESIZE=1
export VLLM_NANO_VL_POLICY_VIDEO_RESIZE=1
export NRL_VIDEO_SFT_MIN_FRAMES=64
export NRL_VIDEO_SFT_MAX_FRAMES=64
export NRL_VIDEO_PROMPT_STYLE=sft_v2_grouped
export NRL_VIDEO_SAMPLING_STYLE=nemotron_vl
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HOME="${code_dir}/.cache/huggingface"
export PATH=/opt/nemo_rl_venv/bin:/root/.local/bin:/cm/shared/apps/slurm/current/bin:/usr/local/nvidia/bin:/usr/local/cuda/bin:/usr/local/mpi/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:/usr/local/ucx/bin:/opt/amazon/efa/bin
export LD_LIBRARY_PATH=/opt/amazon/ofi-nccl/lib:/opt/amazon/efa/lib:/opt/nemo_rl_venv/lib/python3.13/site-packages/nvidia/cudnn/lib:/usr/local/cuda/compat/lib:/usr/local/nvidia/lib:/usr/local/nvidia/lib64
export LIBRARY_PATH=/usr/local/cuda/lib64/stubs:/usr/local/cuda/lib64/stubs:
export UV_PROJECT_ENVIRONMENT=/opt/nemo_rl_venv
export UV_LINK_MODE=copy
export NRL_CONTAINER=1
export RAY_USAGE_STATS_ENABLED=0
export CUDA_HOME=/usr/local/cuda
export CPLUS_INCLUDE_PATH=/usr/local/cuda/include/cccl
export CUDNN_HOME=/opt/nemo_rl_venv/lib/python3.13/site-packages/nvidia/cudnn
export CUDNN_PATH=/opt/nemo_rl_venv/lib/python3.13/site-packages/nvidia/cudnn
export NVIDIA_VISIBLE_DEVICES=all
export NVIDIA_DRIVER_CAPABILITIES=compute,utility,video
export NVIDIA_REQUIRE_CUDA='cuda>=9.0'
export TORCH_CUDA_ARCH_LIST=9.0

read -r -d '' SETUP_COMMAND <<'SETUP_EOF' || true
set -euo pipefail
cd /opt/nemo-rl
export PATH=/opt/nemo_rl_venv/bin:/root/.local/bin:/usr/local/nvidia/bin:/usr/local/cuda/bin:/usr/local/mpi/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:/usr/local/ucx/bin:/opt/amazon/efa/bin
export LD_LIBRARY_PATH=/opt/amazon/ofi-nccl/lib:/opt/amazon/efa/lib:/opt/nemo_rl_venv/lib/python3.13/site-packages/nvidia/cudnn/lib:/usr/local/cuda/compat/lib:/usr/local/nvidia/lib:/usr/local/nvidia/lib64
export LIBRARY_PATH=/usr/local/cuda/lib64/stubs:/usr/local/cuda/lib64/stubs:
export UV_PROJECT_ENVIRONMENT=/opt/nemo_rl_venv
export UV_LINK_MODE=copy
export NRL_CONTAINER=1
export RAY_USAGE_STATS_ENABLED=0
export CUDA_HOME=/usr/local/cuda
export CPLUS_INCLUDE_PATH=/usr/local/cuda/include/cccl
export CUDNN_HOME=/opt/nemo_rl_venv/lib/python3.13/site-packages/nvidia/cudnn
export CUDNN_PATH=/opt/nemo_rl_venv/lib/python3.13/site-packages/nvidia/cudnn
export NVIDIA_VISIBLE_DEVICES=all
export NVIDIA_DRIVER_CAPABILITIES=compute,utility,video
export NVIDIA_REQUIRE_CUDA='cuda>=9.0'
export TORCH_CUDA_ARCH_LIST=9.0
uv_bin=/root/.local/bin/uv
test -x "${uv_bin}"
export PYTHONPATH=/opt/nemo-rl:/opt/nemo-rl/3rdparty/Gym-workspace/Gym:/opt/nemo-rl/3rdparty/Megatron-Bridge-workspace/Megatron-Bridge/src:/opt/nemo-rl/3rdparty/Megatron-Bridge-workspace/Megatron-Bridge/3rdparty/Megatron-LM:${PYTHONPATH:-}

"${uv_bin}" run --no-sync python -c 'import sys; assert sys.version_info >= (3, 13, 14); import ray, transformers, megatron.core, nemo_rl.algorithms.grpo, nemo_rl.environments.nemo_gym'
generation_vllm_python=/opt/ray_venvs/nemo_rl.models.generation.vllm.vllm_worker_async.VllmAsyncGenerationWorker/bin/python
test -x "${generation_vllm_python}"
"${generation_vllm_python}" -c 'import vllm; from nemo_rl.models.generation.vllm.vllm_worker_async import VllmAsyncGenerationWorker'
gym_model_python=/opt/gym_venvs/responses_api_models/vllm_model/.venv/bin/python
test -x "${gym_model_python}"
"${gym_model_python}" -c 'import openai; from responses_api_models.vllm_model.app import VLLMModel'
SETUP_EOF
export SETUP_COMMAND

read -r -d '' COMMAND <<COMMAND_EOF || true
set -euo pipefail
cd /opt/nemo-rl
export PATH=/opt/nemo_rl_venv/bin:/root/.local/bin:/usr/local/nvidia/bin:/usr/local/cuda/bin:/usr/local/mpi/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:/usr/local/ucx/bin:/opt/amazon/efa/bin
export LD_LIBRARY_PATH=/opt/amazon/ofi-nccl/lib:/opt/amazon/efa/lib:/opt/nemo_rl_venv/lib/python3.13/site-packages/nvidia/cudnn/lib:/usr/local/cuda/compat/lib:/usr/local/nvidia/lib:/usr/local/nvidia/lib64
export LIBRARY_PATH=/usr/local/cuda/lib64/stubs:/usr/local/cuda/lib64/stubs:
export UV_PROJECT_ENVIRONMENT=/opt/nemo_rl_venv
export UV_LINK_MODE=copy
export NRL_CONTAINER=1
export RAY_USAGE_STATS_ENABLED=0
export CUDA_HOME=/usr/local/cuda
export CPLUS_INCLUDE_PATH=/usr/local/cuda/include/cccl
export CUDNN_HOME=/opt/nemo_rl_venv/lib/python3.13/site-packages/nvidia/cudnn
export CUDNN_PATH=/opt/nemo_rl_venv/lib/python3.13/site-packages/nvidia/cudnn
export NVIDIA_VISIBLE_DEVICES=all
export NVIDIA_DRIVER_CAPABILITIES=compute,utility,video
export NVIDIA_REQUIRE_CUDA='cuda>=9.0'
export TORCH_CUDA_ARCH_LIST=9.0
uv_bin=/root/.local/bin/uv
test -x "\${uv_bin}"
export PYTHONPATH=/opt/nemo-rl:/opt/nemo-rl/3rdparty/Gym-workspace/Gym:/opt/nemo-rl/3rdparty/Megatron-Bridge-workspace/Megatron-Bridge/src:/opt/nemo-rl/3rdparty/Megatron-Bridge-workspace/Megatron-Bridge/3rdparty/Megatron-LM:\${PYTHONPATH:-}
export HF_HOME=${HF_HOME}
export RAY_ENABLE_UV_RUN_RUNTIME_ENV=0
export NEMO_RL_VENV_DIR=/opt/ray_venvs
export NEMO_GYM_VENV_DIR=/opt/gym_venvs
export NRL_FORCE_REBUILD_VENVS=false
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
  ++env.nemo_gym.uv_venv_dir=/opt/gym_venvs \
  env.nemo_gym.skip_venv_if_present=true \
  policy.model_name="${model_path}" \
  data.train.data_path="${data_path}" \
  data.validation.data_path="${data_path}"
COMMAND_EOF
export COMMAND

last_array_task=$((job_cycles - 1))
submit_args=(
  --parsable
  --nodes=16
  --gres=gpu:8
  --exclusive
  --time=04:00:00
  --array="0-${last_array_task}%1"
  --account="${SLURM_ACCOUNT}"
  --partition="${SLURM_PARTITION}"
  --job-name="${candidate_name}"
  --output="${slurm_log_dir}/%A_%a-${SLURM_ACCOUNT}-${SLURM_PARTITION//,/_}.out"
  --error="${slurm_log_dir}/%A_%a-${SLURM_ACCOUNT}-${SLURM_PARTITION//,/_}.out"
  "${script_dir}/adi_profile_band_race_ray.sub"
)

echo "candidate=${candidate_name}"
echo "account=${SLURM_ACCOUNT} partition=${SLURM_PARTITION} nodes=16 cycles=${job_cycles}"
echo "container=${container}"
echo "results=${results_dir}"

if [[ "${DRY_RUN:-0}" == "1" ]]; then
  printf 'sbatch '
  printf '%q ' "${submit_args[@]}"
  printf '\n'
  exit 0
fi

/cm/shared/apps/slurm/current/bin/sbatch "${submit_args[@]}"
