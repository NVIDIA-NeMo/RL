#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export BASE_NAME="${BASE_NAME:-super35_0908_combined_peter3941_text_rl_r3_step120_corrected_loo_pb_async_r3_ray_frozen_20260908-v1}"
export MODEL_PATH="${MODEL_PATH:-/scratch/fsw/portfolios/nemotron/projects/nemotron_omni_vision/users/ehosseiniasl/checkpoints/grpo_superv3_5_rlvr_v43_broad_falcon_r3-oci-hsg-20260905-r1/step_120/hf}"
export NUM_NODES="${NUM_NODES:-20}"
export NUM_GEN_NODES="${NUM_GEN_NODES:-12}"
export SEGMENT_SIZE="${SEGMENT_SIZE:-4}"
export GPUS_PER_NODE="${GPUS_PER_NODE:-4}"
export JOB_CYCLES="${JOB_CYCLES:-20}"
export MAX_STEPS="${MAX_STEPS:-100000}"
export NUM_PROMPTS="${NUM_PROMPTS:-128}"
export NUM_GENERATIONS="${NUM_GENERATIONS:-16}"
export SAVE_PERIOD="${SAVE_PERIOD:-5}"
export CHECKPOINT_KEEP_TOP_K="${CHECKPOINT_KEEP_TOP_K:-2}"

export ASYNC_GRPO_ENABLED=true
export MAX_TRAJECTORY_AGE_STEPS=1
export ROUTER_REPLAY_ENABLED=true
export ROUTER_REPLAY_TRANSPORT=ray
export LOAD_REPLAY_BUFFER=false
export USE_LEAVE_ONE_OUT_BASELINE=true
export PROFILE_BAND_ENABLED=true
export LENGTH_PENALTY_ENABLED=true
export FREEZE_MOE_ROUTER=true
export MOE_ROUTER_LOAD_BALANCING_TYPE=none
export MOE_ROUTER_BIAS_UPDATE_RATE=0.0
export WANDB_PROJECT=Nemotron-omni-RL-debug

exec bash "${script_dir}/async_grpo_generalist_videoqa.sh"
