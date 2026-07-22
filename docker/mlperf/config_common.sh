# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

export DGXNGPU=4

# Recipe path inside the container (conf/ is baked to /opt/nemo-rl/qwen_35/configs/)
export RECIPE=${RECIPE:-qwen_35/configs/grpo_qwen35_397b_swe_openhands_async_gbs256.yaml}

export MLPERF_TARGET_ACCURACY=${MLPERF_TARGET_ACCURACY:-1.0}
export MLPERF_BENCHMARK_NAME=qwen35_397b_grpo_swe

# Idle-GPU reaper exemption: async GRPO idles the training pool while the
# replay buffer fills from slow SWE rewards. sbatch reads SBATCH_COMMENT at
# submission; an explicit --comment overrides it.
export SLURM_IDLE_EXEMPT_MINS=${SLURM_IDLE_EXEMPT_MINS:-120}
export SBATCH_COMMENT=${SBATCH_COMMENT:-"{\"OccupiedIdleGPUsJobReaper\":{\"exemptIdleTimeMins\":\"${SLURM_IDLE_EXEMPT_MINS}\",\"reason\":\"rl-rollout-warmup\",\"description\":\"NeMo-RL GRPO: training GPUs idle during rollout/SWE-reward buffer-fill\"}}"}

# Sync node-local /tmp/ray logs to the shared log dir (crash tracebacks
# survive job teardown)
export RAY_LOG_SYNC_FREQUENCY=${RAY_LOG_SYNC_FREQUENCY:-120}

export NCCL_TEST=${NCCL_TEST:-0}
