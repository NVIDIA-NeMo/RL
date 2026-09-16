#!/bin/bash

set -euo pipefail

variant=${1:-all}
case "${variant}" in
  legacy|nccl_only|nccl_full|all) ;;
  *)
    echo "Usage: $0 {legacy|nccl_only|nccl_full|all}" >&2
    exit 2
    ;;
esac

repo_root=$(git rev-parse --show-toplevel)
branch_sha=$(git -C "${repo_root}" rev-parse HEAD)
if [[ -n $(git -C "${repo_root}" status --short) ]]; then
  echo "Refusing to submit a dirty worktree." >&2
  exit 1
fi

account=${SLURM_ACCOUNT:-nemotron_n4_post}
partition=${PARTITION:-batch}
container=${CONTAINER:-/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/sna/containers/nemo_rl_nightly.sqsh}
hf_home=${HF_HOME:-/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/sna/hf_home}
output_root=${OUTPUT_ROOT:-/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/sna/experiments/qwen30-refit-cumulative-20260916}
wandb_project=${WANDB_PROJECT:-nemo-rl-mxfp8-refit-ablation}
recipe=examples/configs/recipes/llm/performance/grpo-qwen3-30ba3b-4n4g-async-1off-mxfp8-rollout.yaml

if [[ ! -f "${container}" ]]; then
  echo "Container does not exist: ${container}" >&2
  exit 1
fi
if [[ -z ${WANDB_API_KEY:-} ]]; then
  echo "WANDB_API_KEY is not set." >&2
  exit 1
fi

submit_arm() {
  local arm=$1
  local transport
  local batched_shuffle
  local route_cache

  case "${arm}" in
    legacy)
      transport=null
      batched_shuffle=0
      route_cache=false
      ;;
    nccl_only)
      transport=nccl_reshard
      batched_shuffle=0
      route_cache=false
      ;;
    nccl_full)
      transport=nccl_reshard
      batched_shuffle=1
      route_cache=true
      ;;
  esac

  local run_name="qwen30-refit-${arm}-20s-20260916"
  local output_dir="${output_root}/${arm}"
  mkdir -p "${output_dir}"

  cat >"${output_dir}/manifest.txt" <<EOF
source_sha=${branch_sha}
container=$(readlink -f "${container}")
recipe=${recipe}
arm=${arm}
refit_transport=${transport}
nrl_mxfp8_batched_shuffle=${batched_shuffle}
refit_cache_loader_routes=${route_cache}
refit_prequantize=false
refit_persistent_ipc_buffers=false
container_python_setup=research/qwen30_refit_cumulative_ablation/repair_container_python.sh
steps=20
steady_window=2-19
account=${account}
partition=${partition}
EOF

  export CONTAINER="${container}"
  export PATH="/cm/local/apps/slurm/current/bin:${PATH}"
  export GPUS_PER_NODE=4
  export CPUS_PER_WORKER=144
  export BASE_LOG_DIR="${output_dir}"
  export HF_HOME="${hf_home}"
  export HF_DATASETS_CACHE="${hf_home}/datasets"
  export NRL_MXFP8_BATCHED_SHUFFLE="${batched_shuffle}"
  export RAY_LOG_SYNC_FREQUENCY=300
  export SETUP_COMMAND="bash /opt/nemo-rl/research/qwen30_refit_cumulative_ablation/repair_container_python.sh"
  export MOUNTS="/lustre:/lustre,${repo_root}/nemo_rl:/opt/nemo-rl/nemo_rl,${repo_root}/examples:/opt/nemo-rl/examples,${repo_root}/research/qwen30_refit_cumulative_ablation:/opt/nemo-rl/research/qwen30_refit_cumulative_ablation,${repo_root}/3rdparty/Megatron-Bridge-workspace/Megatron-Bridge:/opt/nemo-rl/3rdparty/Megatron-Bridge-workspace/Megatron-Bridge"
  export COMMAND="cd /opt/nemo-rl && if [[ -f 3rdparty/vllm/nemo-rl.env ]]; then source 3rdparty/vllm/nemo-rl.env; fi && uv run --no-sync examples/run_grpo.py --config ${recipe} grpo.max_num_steps=20 policy.generation.refit_transport=${transport} policy.generation.vllm_cfg.refit_prequantize=false policy.generation.vllm_cfg.refit_cache_loader_routes=${route_cache} policy.refit_persistent_ipc_buffers=false checkpointing.enabled=false logger.log_dir=${output_dir}/logs logger.wandb_enabled=true logger.wandb.project=${wandb_project} logger.wandb.name=${run_name}"

  local sbatch_args=(
    --nodes=4
    --account="${account}"
    --job-name="${account}.${run_name}"
    --partition="${partition}"
    --time=04:00:00
    --gres=gpu:4
    --segment=2
    --chdir="${output_dir}"
    --output="${output_dir}/slurm-%j.out"
  )
  if [[ ${DRY_RUN:-0} == 1 ]]; then
    sbatch --test-only "${sbatch_args[@]}" "${repo_root}/ray.sub"
  else
    sbatch --parsable "${sbatch_args[@]}" "${repo_root}/ray.sub"
  fi
}

if [[ ${variant} == all ]]; then
  submit_arm legacy
  submit_arm nccl_only
  submit_arm nccl_full
else
  submit_arm "${variant}"
fi
