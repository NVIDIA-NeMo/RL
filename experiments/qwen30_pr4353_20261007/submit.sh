#!/bin/bash
set -euo pipefail

arm=${1:?Usage: submit.sh bf16-bf16|bf16-mxfp8|mxfp8-default|mxfp8-param-false|mxfp8-option-b|mxfp8-option-b-param-false [test-only]}
action=${2:-submit}
[[ "$action" == submit || "$action" == test-only ]]
case "$arm" in
  bf16-bf16) config=async-bf16-bf16.yaml ;;
  bf16-mxfp8) config=async-bf16-mxfp8.yaml ;;
  mxfp8-default) config=async-mxfp8-train.yaml ;;
  mxfp8-param-false) config=async-mxfp8-train.yaml ;;
  mxfp8-option-b) config=async-mxfp8-train-option-b.yaml ;;
  mxfp8-option-b-param-false) config=async-mxfp8-train-option-b.yaml ;;
  *) echo "Unknown arm: $arm" >&2; exit 2 ;;
esac

: "${CONTAINER:?Set smoke-validated immutable nightly image}"
: "${SOURCE_ARCHIVE:?Set immutable source archive}"
: "${SOURCE_COMMIT:?Set expected source commit}"
: "${RESULT_ROOT:?Set shared results directory}"
: "${WANDB_API_KEY:?W&B cloud logging must be enabled}"

repo=$(git -C "$(dirname "${BASH_SOURCE[0]}")" rev-parse --show-toplevel)
launch_commit=${LAUNCH_COMMIT:-$SOURCE_COMMIT}
test "$(git -C "$repo" rev-parse HEAD)" = "$launch_commit"
test -z "$(git -C "$repo" status --porcelain --untracked-files=no --ignore-submodules=none)"

max_steps=${MAX_STEPS:-20}
account=${SLURM_ACCOUNT:-coreai_dlalgo_nemorl}
name="qwen30-pr4353-async-${arm}-${max_steps}step${RUN_SUFFIX:+-${RUN_SUFFIX}}"
attention_backend=${VLLM_ATTENTION_BACKEND:-}
attention_override=""
attention_suffix=""
if [[ -n "$attention_backend" ]]; then
  case "$attention_backend" in
    TRITON_ATTN) ;;
    *) echo "Unsupported attention backend: $attention_backend" >&2; exit 2 ;;
  esac
  attention_override="+policy.generation.vllm_kwargs.attention_backend=${attention_backend}"
  attention_suffix="-${attention_backend,,}"
  name="${name}${attention_suffix}"
fi
run_root="${RESULT_ROOT}/${name}"
local_root="/raid/scratch/${USER}/nr-qwen30-${SOURCE_COMMIT:0:10}-${arm}${attention_suffix}"
source_root="${local_root}/source"
te_config_file="${source_root}/experiments/lightning_pr4353_20261007/te-routed-mxfp8.yaml"
te_config_override=""
if [[ "$arm" == mxfp8-* ]]; then
  te_config_override="policy.megatron_cfg.te_precision_config_file=${te_config_file}"
fi
param_override=""
if [[ "$arm" == mxfp8-param-false || "$arm" == mxfp8-option-b-param-false ]]; then
  param_override="policy.megatron_cfg.fp8_cfg.fp8_param=false"
fi
hf_source="/lustre/fsw/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/${USER}/hf_home"
model_cache=models--Qwen--Qwen3-30B-A3B

if [[ "$action" == submit ]]; then
  test -f "$SOURCE_ARCHIVE"
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
test -f ${te_config_file}
export PYTHONPATH=${source_root}:${source_root}/3rdparty/Megatron-Bridge-workspace/Megatron-Bridge/src:${source_root}/3rdparty/Megatron-Bridge-workspace/Megatron-Bridge/3rdparty/Megatron-LM
export HF_HOME=${local_root}/hf HF_HUB_CACHE=${local_root}/hf/hub HUGGINGFACE_HUB_CACHE=${local_root}/hf/hub
export HF_DATASETS_CACHE=${hf_source}/datasets HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export NRL_MEGATRON_CHECKPOINT_DIR=${RESULT_ROOT}/checkpoints-${SOURCE_COMMIT}
export NEMO_RL_VENV_DIR=/opt/ray_venvs NRL_FORCE_REBUILD_VENVS=false FLA_TILELANG=0
export UV_CACHE_DIR=${local_root}/uv VLLM_CACHE_ROOT=${local_root}/vllm TORCHINDUCTOR_CACHE_DIR=${local_root}/inductor TRITON_CACHE_DIR=${local_root}/triton
export PYTHONPYCACHEPREFIX=${local_root}/pycache RAY_TMPDIR=/tmp
unset NRL_IGNORE_VERSION_MISMATCH PYTHONOPTIMIZE
/opt/nemo_rl_venv/bin/python tools/config_cli.py expand experiments/qwen30_pr4353_20261007/${config} >/dev/null
/opt/nemo_rl_venv/bin/python examples/run_grpo.py --config experiments/qwen30_pr4353_20261007/${config} ${te_config_override} ${param_override} ${attention_override} grpo.max_num_steps=${max_steps} logger.log_dir=${run_root}/metrics logger.wandb.name=${name}"

args=(--nodes=4 --gres=gpu:4 --exclusive --mem=0 --account="$account" --partition=batch --time=04:00:00
  --segment=2 --job-name="${account}.${name}" --output="${run_root}/slurm-%j.out"
  --comment='{"OccupiedIdleGPUsJobReaper":{"exemptIdleTimeMins":"120","reason":"model_loading","description":"Qwen3 MXFP8 model initialization"}}')
if [[ "$action" == test-only ]]; then
  args+=(--test-only)
fi
printf 'launcher=%s\nsource=%s\ncontainer=%s\nconfig=%s\narm=%s\nattention=%s\n' \
  "$launch_commit" "$SOURCE_COMMIT" "$CONTAINER" "$config" "$arm" "${attention_backend:-auto}"
exec sbatch "${args[@]}" "$repo/ray.sub"
