#!/bin/bash
set -euo pipefail

mode=${1:?Usage: submit.sh async|sync bf16-storage|default|option-a|option-b [test-only]}
arm=${2:?Usage: submit.sh async|sync bf16-storage|default|option-a|option-b [test-only]}
action=${3:-submit}
[[ "$mode" == async || "$mode" == sync ]]
[[ "$arm" == bf16-storage || "$arm" == default || "$arm" == option-a || "$arm" == option-b ]]
[[ "$action" == submit || "$action" == test-only ]]

if [[ "$arm" == option-a ]]; then
  echo "WARNING: Option A uses an unpatched cuBLASLt grouped-GEMM path (NVBUG 6815125). Check numerical metrics before using results." >&2
fi

: "${CONTAINER:?Set smoke-validated immutable nightly image}"
: "${SOURCE_ARCHIVE:?Set immutable source archive}"
: "${SOURCE_COMMIT:?Set expected source commit}"
: "${RESULT_ROOT:?Set shared results directory}"
: "${WANDB_API_KEY:?W&B cloud logging must be enabled}"

account=${SLURM_ACCOUNT:-nemotron_sw_post}
max_steps=${MAX_STEPS:-20}
repo=$(git rev-parse --show-toplevel)
if [[ "$action" == submit ]]; then
  git -C "$repo" -c fetch.recurseSubmodules=false pull --ff-only
fi
test "$(git -C "$repo" rev-parse HEAD)" = "$SOURCE_COMMIT"
test -z "$(git -C "$repo" status --porcelain --untracked-files=no --ignore-submodules=none)"

case "$arm" in
  bf16-storage) config="../lightning_main_20261006/${mode}-mxfp8.yaml" ;;
  default) config="${mode}-mxfp8-train.yaml" ;;
  option-a) config="${mode}-mxfp8-train-option-a.yaml" ;;
  option-b) config="${mode}-mxfp8-train-option-b.yaml" ;;
esac
name="lightning-pr4353-${mode}-${arm}-${max_steps}step"
run_root="${RESULT_ROOT}/${name}"
local_root="/raid/scratch/${USER}/nr-pr4353-${SOURCE_COMMIT:0:10}-${mode}-${arm}"
source_root="${local_root}/source"
hf_source="/lustre/fsw/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/${USER}/hf_home"
model_cache=models--nvidia--NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16
segment=8
[[ "$mode" != async ]] || segment=4

if [[ "$action" == submit ]]; then
  mkdir -p "$run_root"
fi
export GPUS_PER_NODE=4 CPUS_PER_WORKER=144 DEDICATED_RAY_HEAD=0
export CONTAINER_REMAP_ROOT=1 BASE_LOG_DIR="$run_root" RAY_TMPDIR=/tmp
export SLURM_COMMAND_PATH=/cm/local/apps/slurm/25.11/bin
export PATH="${SLURM_COMMAND_PATH}:${PATH}"
export MOUNTS="/lustre:/lustre,/home:/home,/raid/scratch:/raid/scratch,/home/${USER}/.netrc:/root/.netrc"
export SETUP_COMMAND="set -euo pipefail
mkdir -p ${source_root} ${local_root}/hf/hub ${local_root}/uv ${local_root}/vllm ${local_root}/inductor ${local_root}/triton
tar -xf ${SOURCE_ARCHIVE} -C ${source_root}
rsync -a --ignore-existing ${hf_source}/hub/${model_cache}/ ${local_root}/hf/hub/${model_cache}/"
export COMMAND="set -euo pipefail
ulimit -c 0
cd ${source_root}
export PYTHONPATH=${source_root}:${source_root}/3rdparty/Megatron-Bridge-workspace/Megatron-Bridge/src:${source_root}/3rdparty/Megatron-Bridge-workspace/Megatron-Bridge/3rdparty/Megatron-LM
export HF_HOME=${local_root}/hf HF_HUB_CACHE=${local_root}/hf/hub HUGGINGFACE_HUB_CACHE=${local_root}/hf/hub
export HF_DATASETS_CACHE=${hf_source}/datasets HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export NRL_MEGATRON_CHECKPOINT_DIR=${RESULT_ROOT}/checkpoints-${SOURCE_COMMIT}
export NEMO_RL_VENV_DIR=/opt/ray_venvs NRL_FORCE_REBUILD_VENVS=false FLA_TILELANG=0
export UV_CACHE_DIR=${local_root}/uv VLLM_CACHE_ROOT=${local_root}/vllm TORCHINDUCTOR_CACHE_DIR=${local_root}/inductor TRITON_CACHE_DIR=${local_root}/triton
export PYTHONPYCACHEPREFIX=${local_root}/pycache RAY_TMPDIR=/tmp
unset NRL_IGNORE_VERSION_MISMATCH PYTHONOPTIMIZE
/opt/nemo_rl_venv/bin/python tools/config_cli.py expand experiments/lightning_pr4353_20261007/${config} >/dev/null
/opt/nemo_rl_venv/bin/python examples/run_grpo.py --config experiments/lightning_pr4353_20261007/${config} grpo.max_num_steps=${max_steps} logger.log_dir=${run_root}/metrics logger.wandb.name=${name}"

args=(--nodes=8 --gres=gpu:4 --exclusive --mem=0 --account="$account" --partition=batch --time=04:00:00
  --segment="$segment" --job-name="${account}.${name}" --output="${run_root}/slurm-%j.out"
  --comment='{"OccupiedIdleGPUsJobReaper":{"exemptIdleTimeMins":"120","reason":"model_loading","description":"Lightning MXFP8 model initialization"}}'
  --dependency=)
if [[ -n "${AFTEROK_JOB_ID:-}" ]]; then
  args+=(--dependency="afterok:${AFTEROK_JOB_ID}")
fi
if [[ "$action" == test-only ]]; then
  args+=(--test-only)
fi
printf 'source=%s\ncontainer=%s\nconfig=%s\narm=%s\n' "$SOURCE_COMMIT" "$CONTAINER" "$config" "$arm"
exec sbatch "${args[@]}" "$repo/ray.sub"
