#!/bin/bash

set -euo pipefail

mode=${1:-submit}
case "${mode}" in
  submit) submit_mode=(--parsable) ;;
  test-only) submit_mode=(--test-only) ;;
  *) printf 'Usage: %s [submit|test-only]\n' "$0" >&2; exit 2 ;;
esac

work_root=/lustre/fsw/portfolios/coreai/projects/coreai_chef_posttrain/users/sna
repo=${work_root}/experiments/pr2964-20step-20260807/RL
experiment_root=${work_root}/experiments/pr2964-20step-20260807
run_name=${RUN_NAME_OVERRIDE:-qwen3-30ba3b-4n8g-sync-hybridep-pr2964-20step}
run_root=${experiment_root}/runs/${run_name}
container=${CONTAINER_OVERRIDE:-${work_root}/containers/nemo-rl-nightly-20260807/nemo_rl_nightly_20260808_504915.sqsh}
hf_home=${work_root}/.cache/huggingface
wheel=${HYBRID_EP_WHEEL_OVERRIDE:-${experiment_root}/deepep-wheels/17cfb817bccec3a9c247013360cc550c2bac441e-sm100-504834/deep_ep-1.2.1+17cfb81-cp313-cp313-linux_x86_64.whl}
wheel_sha256=${HYBRID_EP_WHEEL_SHA256_OVERRIDE:-82487725cc4a384374530fe7a031ef138d1737621d04a3850060b3271b9e5f99}
overlay=/tmp/nemo-rl-hybridep-17cf
job_dependency=${JOB_DEPENDENCY:-}
slurm_exclude=${SLURM_EXCLUDE:-}

test "$(git -C "${repo}" rev-parse HEAD)" = a028b33bcde0ef8aeb9fcc626a2e0c57fb568d2f
git -C "${repo}" merge-base --is-ancestor 60a10b4f54c2754d44150771a06260fe9e8b186f HEAD
git -C "${repo}" merge-base --is-ancestor a9aaa395c37963a9fd8a7320d61a516c7b714e57 HEAD
test -z "$(git -C "${repo}" status --porcelain --untracked-files=no --ignore-submodules=untracked)"
if git -C "${repo}" submodule status --recursive | grep -qE '^[+-U]'; then
  printf 'Submodules do not match the pinned gitlinks.\n' >&2
  exit 2
fi
printf '%s  %s\n' "${wheel_sha256}" "${wheel}" | sha256sum --check --status
mkdir -p "${run_root}/ray" "${run_root}/training"

driver_args=(
  /opt/nemo_rl_venv/bin/python
  examples/run_grpo.py
  --config examples/configs/recipes/llm/performance/grpo-qwen3-30ba3b-4n8g.yaml
  grpo.max_num_steps=20
  cluster.num_nodes=4
  cluster.gpus_per_node=8
  cluster.segment_size=4
  policy.megatron_cfg.moe_token_dispatcher_type=flex
  ++policy.megatron_cfg.moe_flex_dispatcher_backend=hybridep
  ++policy.megatron_cfg.moe_hybridep_num_sms=32
  "++policy.megatron_cfg.env_vars.NUM_OF_HYBRID_EP_RANKS_PER_NVLINK_DOMAIN='8'"
  "++policy.megatron_cfg.env_vars.NUM_OF_TOKENS_PER_CHUNK_COMBINE_API='128'"
  "++policy.megatron_cfg.env_vars.NVLINK_DOMAIN_SIZE='8'"
  "++policy.megatron_cfg.env_vars.USE_MNNVL='0'"
  checkpointing.enabled=false
  "logger.log_dir=${run_root}/training"
  logger.wandb_enabled=true
  logger.wandb.project=sna-hybridep-b200
  "logger.wandb.name=${run_name}"
  logger.monitor_gpus=true
  logger.tensorboard_enabled=true
)
printf -v COMMAND '%q ' "${driver_args[@]}"

read -r -d '' SETUP_COMMAND <<EOF || true
set -euo pipefail
overlay=${overlay}
wheel=${wheel}
expected_sha256=${wheel_sha256}
[[ "\${overlay}" == /tmp/nemo-rl-hybridep-* ]]
test "\$(sha256sum "\${wheel}" | cut -d' ' -f1)" = "\${expected_sha256}"
rm -rf -- "\${overlay}"
mkdir -p "\${overlay}"
unset UV_CONFIG_FILE
UV_NO_CONFIG=1 uv pip install --python /opt/nemo_rl_venv/bin/python --target "\${overlay}" --reinstall --no-deps --no-index "\${wheel}"
PYTHONPATH="\${overlay}" /opt/nemo_rl_venv/bin/python -c 'import deep_ep, deep_ep_cpp, hybrid_ep_cpp; print(deep_ep.__file__); print(deep_ep_cpp.__file__); print(hybrid_ep_cpp.__file__)'
EOF

export COMMAND SETUP_COMMAND
export CONTAINER="${container}"
export MOUNTS=/lustre:/lustre
export BASE_LOG_DIR="${run_root}/ray"
export GPUS_PER_NODE=8
export HF_HOME="${hf_home}"
export HF_DATASETS_CACHE="${hf_home}/datasets"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export WANDB_MODE=offline
export NCCL_NVLS_ENABLE=0
export NEMO_RL_PY_EXECUTABLES_SYSTEM=0
export NEMO_RL_VENV_DIR=/opt/ray_venvs
export NRL_FORCE_REBUILD_VENVS=false
export NRL_IGNORE_VERSION_MISMATCH=1
export PYTHONPATH="${overlay}:${repo}:${repo}/3rdparty/Megatron-Bridge-workspace/Megatron-Bridge/src:${repo}/3rdparty/Megatron-Bridge-workspace/Megatron-Bridge/3rdparty/Megatron-LM"

cd "${repo}"
slurm_extra_args=()
if [[ -n "${slurm_exclude}" ]]; then
  slurm_extra_args+=(--exclude="${slurm_exclude}")
fi
job_output=$(sbatch "${submit_mode[@]}" \
  "${slurm_extra_args[@]}" \
  --export=ALL \
  --nodes=4 \
  --gpus-per-node=8 \
  --exclusive \
  --segment=4 \
  --account=coreai_chef_posttrain \
  --partition=batch_long \
  --time=08:00:00 \
  --job-name="coreai_chef_posttrain.${run_name}" \
  --output="${run_root}/slurm-%j.out" \
  --dependency="${job_dependency}" \
  ray.sub)
printf '%s\n' "${job_output}"

if [[ "${mode}" == submit ]]; then
  printf 'job_id=%s\nrun_name=%s\nvalidation_head=%s\npr2964_head=%s\npr3436_head=%s\ndeepep_commit=%s\ndeepep_wheel=%s\ndeepep_wheel_sha256=%s\ncontainer=%s\njob_dependency=%s\nslurm_exclude=%s\n' \
    "${job_output}" "${run_name}" "$(git rev-parse HEAD)" \
    60a10b4f54c2754d44150771a06260fe9e8b186f \
    a9aaa395c37963a9fd8a7320d61a516c7b714e57 \
    17cfb817bccec3a9c247013360cc550c2bac441e "${wheel}" "${wheel_sha256}" \
    "${container}" "${job_dependency}" "${slurm_exclude}" \
    > "${run_root}/submission.env"
fi
