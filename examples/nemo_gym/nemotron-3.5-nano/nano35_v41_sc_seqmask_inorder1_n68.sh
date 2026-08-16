#!/usr/bin/env bash
set -euo pipefail

# Fresh V2 convergence run with sequence-level logprob-error masking.
# Four-hour continuations share EXP_NAME/CHECKPOINT_DIR; dependencies are set
# by the submitter, not by this single-job launcher.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export CONTAINER=/lustre/fsw/portfolios/coreai/users/yifuw/enroot-images/gitlab-master.nvidia.com/yifuw/images/nemo-rl:main_ultra_recipes_prebaked_venvs_20260730.squashfs

unset GENRM_BASE_URL NL2BASH_BASE_URL
export EXTERNAL_JUDGES=1
export GENRM_REPLICAS=2
export GENRM_TENSOR_PARALLEL_SIZE=8
export NL2BASH_REPLICAS=8
export NL2BASH_TENSOR_PARALLEL_SIZE=4
export EXTERNAL_VLLM_SEGMENT_SIZE=2

export EXP_NAME="${EXP_NAME:-akamehra-nano35-v41-v2-sc-inorder1-seqmask2-fresh-n68-t8-g40-gym8-j12-pps128-gbs2048}"
export SLURM_ACCOUNT="${SLURM_ACCOUNT:-nemotron_sw_post}"
export SLURM_PARTITION="${SLURM_PARTITION:-batch}"
export WALLTIME="${WALLTIME:-4:00:00}"
export JOB_REAPER_EXEMPT_MINS="${JOB_REAPER_EXEMPT_MINS:-120}"
export CHECKPOINTING_SAVE_BY="${CHECKPOINTING_SAVE_BY:-00:03:35:00}"

export NUM_TRAIN_NODES=8
export NUM_GEN_NODES=40
export NUM_GYM_NODES=8
export SEGMENT_SIZE=2

export SAMPLER=in_order
export MAX_LOOKAHEAD_VERSIONS=1
export _NUM_PROMPTS_PER_STEP=128
export STREAM_MIN_GROUPS=32
export NUM_STORAGE_UNITS=8
export BUFFER_RETENTION_MULTIPLIER=1
export REFIT_TRANSPORT=nccl_reshard

export USE_SNAPSHOT=0
export SANDBOX_CONTAINER=
unset NRL_IGNORE_VERSION_MISMATCH NRL_FORCE_REBUILD_VENVS PYTHONPATH

exec bash "${SCRIPT_DIR}/nano35_dolphin_launch_sc.sh" \
  grpo.num_prompts_per_step=128 \
  policy.train_global_batch_size=2048 \
  +policy.megatron_cfg.reproduce_per_chunk_grad_bug=false \
  checkpointing.enabled=true \
  checkpointing.ft_save_period=5 \
  checkpointing.metric_name=null \
  "$@"
