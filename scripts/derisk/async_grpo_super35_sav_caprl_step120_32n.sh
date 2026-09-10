#!/usr/bin/env bash
# Submit the matched 32-node combined SA-V + CAPRL step-120 experiment.
#
# R3 run (default):
#   R3_ENABLED=true bash scripts/derisk/async_grpo_super35_sav_caprl_step120_32n.sh
#
# Non-R3 control:
#   R3_ENABLED=false bash scripts/derisk/async_grpo_super35_sav_caprl_step120_32n.sh
#
# Both modes use the same checkpoint, data, topology, legacy LOO estimator,
# profile-band reward shaping, and frozen MoE router. Each invocation submits a
# 20-task singleton Slurm array and resumes training across job windows.
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

r3_enabled="${R3_ENABLED:-true}"
case "${r3_enabled}" in
  true)
    r3_tag="r3"
    router_replay_overrides="++policy.router_replay.enabled=true ++policy.router_replay.transport=ray ++checkpointing.load_replay_buffer=false"
    ;;
  false)
    r3_tag="nonr3"
    router_replay_overrides="++policy.router_replay.enabled=false"
    ;;
  *)
    echo "ERROR: R3_ENABLED must be true or false, got: ${r3_enabled}" >&2
    exit 2
    ;;
esac

run_stamp="${RUN_ID:-$(date -u +%Y%m%d-%H%M%S)}"
cluster_name="${SLURM_CLUSTER_NAME:-aws-cmh-slurm-1-v1}"
experiment="super35_0910_latest_pulkit_video_lnfix_legacy_loo_pb_async_${r3_tag}_frozen_src_text_r3_step120_32n_${run_stamp}"

export RUN_ID="${run_stamp}"
export BASE_NAME="${BASE_NAME:-async_grpo_super35_derisk_sav_caprl_${experiment}}"
export WANDB_PROJECT="${WANDB_PROJECT:-Nemotron-omni-RL-debug}"
export WANDB_RUN_NAME="${WANDB_RUN_NAME:-${cluster_name}_${experiment}}"

export CONTAINER="${CONTAINER:-/scratch/fsw/portfolios/nemotron/projects/nemotron_omni_vision/users/ehosseiniasl/images/rl-gym.67009223-gym_ln_fix.sqsh}"
export MODEL_PATH="${MODEL_PATH:-/lustre/fsw/portfolios/nemotron/users/arushig/workspace/output/stage2_text_rl_with_r3_step120}"
export DATA_PATH="${DATA_PATH:-/lustre/fsw/portfolios/nemotron/users/arushig/nemo_gym_rl_video_0803/nemo_rl/results/combined_sav_caprl_20260822/train_sav_all_tracks_plus_caprl_exclude6215_cluster_paths.jsonl}"

export PERSISTENT_CACHE="${PERSISTENT_CACHE:-/scratch/fsw/portfolios/nemotron/projects/nemotron_omni_vision/users/${USER}/nemo_rl_cache/super35_0910_step120_32n}"
export GYM_VENV_DIR="${GYM_VENV_DIR:-${PERSISTENT_CACHE}/gym_venvs_${r3_tag}}"
export PREFETCH_GYM_VENVS="${PREFETCH_GYM_VENVS:-true}"

export JOB_CYCLES="${JOB_CYCLES:-20}"
export NUM_NODES="${NUM_NODES:-32}"
export NUM_GEN_NODES="${NUM_GEN_NODES:-16}"
export GPUS_PER_NODE="${GPUS_PER_NODE:-4}"
export SEGMENT_SIZE="${SEGMENT_SIZE:-8}"
export NUM_PROMPTS="${NUM_PROMPTS:-128}"
export NUM_GENERATIONS="${NUM_GENERATIONS:-16}"
export MAX_STEPS="${MAX_STEPS:-1000000}"
export SAVE_PERIOD="${SAVE_PERIOD:-5}"
export CHECKPOINT_KEEP_TOP_K="${CHECKPOINT_KEEP_TOP_K:-2}"
export SLURM_TIME_LIMIT="${SLURM_TIME_LIMIT:-14:00:00}"

export TOKENIZER_CHAT_TEMPLATE="${TOKENIZER_CHAT_TEMPLATE:-default}"
export VLLM_CHAT_TEMPLATE="${VLLM_CHAT_TEMPLATE:-null}"
export IN_FLIGHT_WEIGHT_UPDATES="${IN_FLIGHT_WEIGHT_UPDATES:-true}"
export RECOMPUTE_KV_CACHE_AFTER_WEIGHT_UPDATES="${RECOMPUTE_KV_CACHE_AFTER_WEIGHT_UPDATES:-false}"
export LENGTH_PENALTY_ENABLED="${LENGTH_PENALTY_ENABLED:-true}"
export PROFILE_BAND_ENABLED="${PROFILE_BAND_ENABLED:-true}"

user_extra_overrides="${EXTRA_OVERRIDES:-}"
export EXTRA_OVERRIDES="grpo.async_grpo.enabled=true grpo.async_grpo.max_trajectory_age_steps=1 ${router_replay_overrides} policy.megatron_cfg.freeze_moe_router=true policy.megatron_cfg.moe_router_load_balancing_type=none policy.megatron_cfg.moe_router_bias_update_rate=0.0 ${user_extra_overrides}"

exec "${script_dir}/async_grpo_generalist_videoqa.sh" "$@"
