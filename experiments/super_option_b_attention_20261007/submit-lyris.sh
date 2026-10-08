#!/usr/bin/env bash
set -euo pipefail

backend=${1:?Usage: submit-lyris.sh flashinfer|triton [test-only]}
action=${2:-submit}
[[ "$action" == submit || "$action" == test-only ]]
case "$backend" in
  flashinfer) attention_override="" ;;
  triton) attention_override="+policy.generation.vllm_kwargs.attention_backend=TRITON_ATTN" ;;
  *) echo "Unknown attention backend: $backend" >&2; exit 2 ;;
esac

: "${CONTAINER:?Set the vLLM 0.29 nightly image}"
: "${SOURCE_ARCHIVE:?Set the immutable source archive}"
: "${SOURCE_COMMIT:?Set the expected source commit}"
: "${RESULT_ROOT:?Set the shared result directory}"
: "${WANDB_API_KEY:?W&B cloud logging must be enabled}"

repo=$(git -C "$(dirname "${BASH_SOURCE[0]}")" rev-parse --show-toplevel)
if [[ "$action" == submit ]]; then
  git -C "$repo" -c fetch.recurseSubmodules=false pull --ff-only
fi
test "$(git -C "$repo" rev-parse HEAD)" = "$SOURCE_COMMIT"
test -z "$(git -C "$repo" status --porcelain --untracked-files=no --ignore-submodules=none)"
test -f "$SOURCE_ARCHIVE"
test -f "$CONTAINER"

account=${SLURM_ACCOUNT:-coreai_dlalgo_llm}
max_steps=${MAX_STEPS:-20}
name="super-optionb-${backend}-${max_steps}step${RUN_SUFFIX:+-${RUN_SUFFIX}}"
run_root="${RESULT_ROOT}/${name}"
local_root="/raid/scratch/${USER}/nr-super-optb-${SOURCE_COMMIT:0:10}-${backend}"
source_root="${local_root}/source"
model_root="/raid/scratch/${USER}/nr-super-model"
hf_source="/lustre/fsw/coreai_dlalgo_llm/users/${USER}/hf_home"
model_cache="models--nvidia--NVIDIA-Nemotron-3-Super-120B-A12B-BF16"
te_config="${source_root}/experiments/lightning_pr4353_20261007/te-routed-mxfp8.yaml"
config="experiments/super_option_b_attention_20261007/async-option-b.yaml"

if [[ "$action" == submit ]]; then
  mkdir -p "$run_root"
fi
export GPUS_PER_NODE=4 CPUS_PER_WORKER=144 DEDICATED_RAY_HEAD=0
export CONTAINER_REMAP_ROOT=1 BASE_LOG_DIR="$run_root" RAY_TMPDIR=/tmp
export MOUNTS="/lustre:/lustre,/home:/home,/raid/scratch:/raid/scratch"
export SETUP_COMMAND="set -euo pipefail
mkdir -p ${source_root} ${model_root}/hf/hub/${model_cache} ${local_root}/uv ${local_root}/vllm ${local_root}/inductor ${local_root}/triton
tar -xf ${SOURCE_ARCHIVE} -C ${source_root}
if [[ ! -f ${model_root}/hf/.super-cache-ready ]]; then
  rsync -a --ignore-existing ${hf_source}/hub/${model_cache}/ ${model_root}/hf/hub/${model_cache}/
  touch ${model_root}/hf/.super-cache-ready
fi"
export COMMAND="set -euo pipefail
ulimit -c 0
cd ${source_root}
export PYTHONPATH=${source_root}:/opt/nemo-rl/3rdparty/Megatron-Bridge-workspace/Megatron-Bridge/src:/opt/nemo-rl/3rdparty/Megatron-Bridge-workspace/Megatron-Bridge/3rdparty/Megatron-LM
export HF_HOME=${model_root}/hf HF_HUB_CACHE=${model_root}/hf/hub HUGGINGFACE_HUB_CACHE=${model_root}/hf/hub
export HF_DATASETS_CACHE=${hf_source}/datasets HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export NRL_MEGATRON_CHECKPOINT_DIR=${RESULT_ROOT}/checkpoints-${SOURCE_COMMIT}
export NEMO_RL_VENV_DIR=/opt/ray_venvs NRL_FORCE_REBUILD_VENVS=false FLA_TILELANG=0
export UV_CACHE_DIR=${local_root}/uv VLLM_CACHE_ROOT=${local_root}/vllm TORCHINDUCTOR_CACHE_DIR=${local_root}/inductor TRITON_CACHE_DIR=${local_root}/triton
export PYTHONPYCACHEPREFIX=${local_root}/pycache RAY_TMPDIR=/tmp
unset NRL_IGNORE_VERSION_MISMATCH PYTHONOPTIMIZE
/opt/nemo_rl_venv/bin/python tools/config_cli.py expand ${config} >/dev/null
/opt/nemo_rl_venv/bin/python examples/run_grpo.py --config ${config} policy.megatron_cfg.te_precision_config_file=${te_config} ${attention_override} grpo.max_num_steps=${max_steps} logger.log_dir=${run_root}/metrics logger.wandb.name=${name}"

args=(--nodes=32 --exclusive --mem=0 --account="$account" --partition=gb200
  --qos=user-restrictions --time=04:00:00 --segment=8
  --job-name="${account}.${name}" --output="${run_root}/slurm-%j.out")
if [[ "$action" == test-only ]]; then
  args+=(--test-only)
fi
printf 'source=%s\ncontainer=%s\nconfig=%s\nbackend=%s\n' \
  "$SOURCE_COMMIT" "$CONTAINER" "$config" "$backend"
exec sbatch "${args[@]}" "$repo/ray.sub"
