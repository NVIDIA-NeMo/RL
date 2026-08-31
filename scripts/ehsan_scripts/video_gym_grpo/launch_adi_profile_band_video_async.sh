#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
code_dir="$(cd "${script_dir}/../../.." && pwd)"

: "${SLURM_ACCOUNT:?Set SLURM_ACCOUNT for this scheduler candidate}"
: "${SLURM_PARTITION:?Set SLURM_PARTITION for this scheduler candidate}"

container="${CONTAINER:-/lustre/fsw/portfolios/llmservice/projects/llmservice_nemo_mlops/users/ehosseiniasl/github_repos/nemorl/RL-3604/.cache/containers/nemo-rl-main_20260807-super-ultra-omni-oci-flat-x86_64.sqsh}"
config_in_container="${CONFIG_IN_CONTAINER:-examples/configs/recipes/vlm/vlm_grpo-nemotron-omni-30ba3b-h100-16n8g-megatron-tp2ep16-async-gym-video-adi-profile-band.v1.yaml}"
model_path="${MODEL_PATH:-/lustre/fsw/portfolios/llmservice/users/arushig/workspace/output/generalist-49k-video-mpo-20260812-113202/checkpoints/tp_1_hf/iter_0000125/mcore_to_hf}"
data_path="${DATA_PATH:-/lustre/fsw/portfolios/llmservice/users/arushig/nemo_gym_rl_video_0803/nemo-rl/results/video_frame_cache_caprl_passrate_n5_easy_to_hard_lt60s_20260806_f64/stable_split_95_5/train_exclude_line6215.jsonl}"
dss_cache_dir="${NVDATASET_CACHE_DIR:-/lustre/fsw/portfolios/llmservice/users/ehosseiniasl/.cache}"
caprl_video_data_path="${CAPRL_VIDEO_DATA_PATH:-${data_path}}"
sav_tracking_data_path="${SAV_TRACKING_DATA_PATH:-${dss_cache_dir}/segment_anything_video/v4/jsonls/sav_all_tracks_rl_images_dfw.v1.jsonl}"
combined_smoke_data_path="${COMBINED_SMOKE_DATA_PATH:-${dss_cache_dir}/segment_anything_video/v4/jsonls/combined_caprl_sav_smoke_dfw.v2.jsonl}"
if [[ "${config_in_container}" == *dss-f32* ]]; then
  video_num_frames="${VIDEO_NUM_FRAMES:-32}"
else
  video_num_frames="${VIDEO_NUM_FRAMES:-64}"
fi

run_id="${RUN_ID:-$(date -u +%Y%m%d-%H%M%S)}"
base_name="${BASE_NAME:-s35v_rpb_${run_id}}"
candidate_name="${base_name}_${SLURM_ACCOUNT}_${SLURM_PARTITION//,/_}"
results_dir="${RESULTS_DIR:-${code_dir}/results/${base_name}}"
slurm_log_dir="${results_dir}/slurm"
race_root="${RACE_ROOT:-${results_dir}/scheduler_race}"
race_claim_dir="${RACE_CLAIM_DIR:-${race_root}/claim}"
job_cycles="${JOB_CYCLES:-20}"
job_nodes="${JOB_NODES:-16}"
job_time="${JOB_TIME:-04:00:00}"
job_dependency="${JOB_DEPENDENCY:-}"
max_num_steps="${MAX_NUM_STEPS:-100000}"

[[ -f "${container}" ]] || { echo "ERROR: container not found: ${container}" >&2; exit 1; }
[[ -f "${model_path}/config.json" ]] || { echo "ERROR: checkpoint not found: ${model_path}" >&2; exit 1; }
[[ -f "${code_dir}/${config_in_container}" ]] || { echo "ERROR: recipe not found" >&2; exit 1; }
if [[ "${config_in_container}" == *sav-caprl* ]]; then
  if [[ "${config_in_container}" == *dss-f32* ]]; then
    : "${DSS_CAPRL_VIDEO_DATA_PATH:?Set DSS_CAPRL_VIDEO_DATA_PATH to the materialized CAPRL f32 JSONL}"
    : "${DSS_SAV_TRACKING_DATA_PATH:?Set DSS_SAV_TRACKING_DATA_PATH to the materialized absolute-path SAV JSONL}"
    [[ -s "${DSS_CAPRL_VIDEO_DATA_PATH}" ]] || { echo "ERROR: CAPRL DSS dataset not found: ${DSS_CAPRL_VIDEO_DATA_PATH}" >&2; exit 1; }
    [[ -s "${DSS_SAV_TRACKING_DATA_PATH}" ]] || { echo "ERROR: SAV DSS dataset not found: ${DSS_SAV_TRACKING_DATA_PATH}" >&2; exit 1; }
  elif [[ "${config_in_container}" == *2n8g* ]]; then
    [[ -s "${combined_smoke_data_path}" ]] || { echo "ERROR: combined smoke dataset not found: ${combined_smoke_data_path}" >&2; exit 1; }
  else
    [[ -s "${caprl_video_data_path}" ]] || { echo "ERROR: CAPRL dataset not found: ${caprl_video_data_path}" >&2; exit 1; }
    [[ -s "${sav_tracking_data_path}" ]] || { echo "ERROR: SAV dataset not found: ${sav_tracking_data_path}" >&2; exit 1; }
  fi
  sav_gym_config="${code_dir}/3rdparty/Gym-workspace/Gym/resources_servers/sav_tracks/configs/sav_tracks.yaml"
  [[ -s "${sav_gym_config}" ]] || {
    echo "ERROR: SAV Gym verifier wiring is missing: ${sav_gym_config}" >&2
    exit 1
  }
else
  [[ -s "${data_path}" ]] || { echo "ERROR: dataset not found: ${data_path}" >&2; exit 1; }
fi
if [[ "${SLURM_ACCOUNT}" == "nemotron_edge_text" && "${ALLOW_EDGE_TEXT:-0}" != "1" ]]; then
  echo "ERROR: nemotron_edge_text requires ALLOW_EDGE_TEXT=1" >&2
  exit 1
fi
if [[ "${WANDB_MODE:-online}" != "offline" ]]; then
  [[ -n "${WANDB_API_KEY:-}" ]] || { echo "ERROR: WANDB_API_KEY is not set for online logging" >&2; exit 1; }
fi

mkdir -p "${results_dir}" "${slurm_log_dir}" "${race_root}" "${code_dir}/.cache/huggingface"

wandb_run_id="${WANDB_RUN_ID:-$(printf '%s' "${base_name}" | sha256sum | cut -c1-16)}"
wandb_project="${WANDB_PROJECT:-Nemotron-omni-RL-debug}"
if [[ "${config_in_container}" == *nemotron-super-omni* ]]; then
  wandb_model_name=super
else
  wandb_model_name=nano
fi
if [[ "${config_in_container}" == *sav-caprl* ]]; then
  wandb_workload_name=sav-caprl-rpb
else
  wandb_workload_name=video-rpb
fi
wandb_run_name="${WANDB_RUN_NAME:-dfw-h100-${wandb_model_name}-${wandb_workload_name}}"

base_mounts="/lustre:/lustre"
selective_mounts="${code_dir}/nemo_rl:/opt/nemo-rl/nemo_rl"
selective_mounts+=",${code_dir}/examples:/opt/nemo-rl/examples"
selective_mounts+=",${code_dir}/tools:/opt/nemo-rl/tools"
selective_mounts+=",${script_dir}/adi_profile_band_ray_entrypoint:/usr/local/bin/ray"
selective_mounts+=",${script_dir}/adi_profile_band_uv_entrypoint:/usr/local/bin/uv"
selective_mounts+=",${code_dir}/3rdparty/Gym-workspace/Gym:/opt/nemo-rl/3rdparty/Gym-workspace/Gym"
selective_mounts+=",${code_dir}/3rdparty/Megatron-Bridge-workspace/Megatron-Bridge:/opt/nemo-rl/3rdparty/Megatron-Bridge-workspace/Megatron-Bridge"
selective_mounts+=",${code_dir}/3rdparty/Megatron-Bridge-workspace/Megatron-Bridge/3rdparty/Megatron-LM:/opt/nemo-rl/3rdparty/Megatron-Bridge-workspace/Megatron-Bridge/3rdparty/Megatron-LM"
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
export NVDATASET_CACHE_DIR="${dss_cache_dir}"
export NEMO_RL_VIDEO_MEDIA_ROOT=/lustre
export VLLM_VIDEO_LOADER_BACKEND=nemotron_vl
export VLLM_NANO_VL_POLICY_IMAGE_RESIZE=1
export VLLM_NANO_VL_POLICY_VIDEO_RESIZE=1
export NRL_VIDEO_SFT_MIN_FRAMES="${video_num_frames}"
export NRL_VIDEO_SFT_MAX_FRAMES="${video_num_frames}"
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
export NVDATASET_CACHE_DIR=${dss_cache_dir}
export CAPRL_VIDEO_DATA_PATH=${caprl_video_data_path}
export SAV_TRACKING_DATA_PATH=${sav_tracking_data_path}
export COMBINED_SMOKE_DATA_PATH=${combined_smoke_data_path}
export DSS_CAPRL_VIDEO_DATA_PATH=${DSS_CAPRL_VIDEO_DATA_PATH:-}
export DSS_SAV_TRACKING_DATA_PATH=${DSS_SAV_TRACKING_DATA_PATH:-}
export RAY_ENABLE_UV_RUN_RUNTIME_ENV=0
export NEMO_RL_VENV_DIR=/opt/ray_venvs
export NEMO_GYM_VENV_DIR=/opt/gym_venvs
export NRL_FORCE_REBUILD_VENVS=false
config_args=()
if [[ "${config_in_container}" != *sav-caprl* ]]; then
  config_args+=(
    data.train.data_path="${data_path}"
    data.validation.data_path="${data_path}"
  )
fi
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
  "\${config_args[@]}" \
  grpo.max_num_steps="${max_num_steps}"
COMMAND_EOF
export COMMAND

last_array_task=$((job_cycles - 1))
submit_args=(
  --parsable
  --nodes="${job_nodes}"
  --gres=gpu:8
  --exclusive
  --time="${job_time}"
  --array="0-${last_array_task}%1"
  --account="${SLURM_ACCOUNT}"
  --partition="${SLURM_PARTITION}"
  --job-name="${candidate_name}"
  --output="${slurm_log_dir}/%A_%a-${SLURM_ACCOUNT}-${SLURM_PARTITION//,/_}.out"
  --error="${slurm_log_dir}/%A_%a-${SLURM_ACCOUNT}-${SLURM_PARTITION//,/_}.out"
)
if [[ -n "${job_dependency}" ]]; then
  submit_args+=(--dependency="${job_dependency}")
fi
submit_args+=("${script_dir}/adi_profile_band_race_ray.sub")

echo "candidate=${candidate_name}"
echo "account=${SLURM_ACCOUNT} partition=${SLURM_PARTITION} nodes=${job_nodes} cycles=${job_cycles} max_steps=${max_num_steps}"
echo "dependency=${job_dependency:-none}"
echo "container=${container}"
echo "results=${results_dir}"

if [[ "${DRY_RUN:-0}" == "1" ]]; then
  printf 'sbatch '
  printf '%q ' "${submit_args[@]}"
  printf '\n'
  exit 0
fi

/cm/shared/apps/slurm/current/bin/sbatch "${submit_args[@]}"
