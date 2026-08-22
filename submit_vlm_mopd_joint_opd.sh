#!/usr/bin/env bash
set -euo pipefail
# Joint text+vision MOPD on the super-v3.5-posttraining stack (HSG, 30n4g).
# ONE student (vlm_student), TWO teachers routed per-sample by Gym agent name:
#   vision rows (string_match_simple_agent) -> VLM_TEACHER_PATH
#   all 24 text agents                       -> TEXT_TEACHER_PATH (default alias)
# Data: joint interleaved jsonl (text curriculum order preserved, vision rows
# Bresenham-interleaved ~2:1; 285,344 rows total, built 2026-08-21).
# Forked from submit_vlm_mopd_babyvision_refmatch.sh (last working omni submit).

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export EXP_NAME="${EXP_NAME:-mopd-super-omni-joint-opd-hsg-001}"
export CONFIG_PATH="${CONFIG_PATH:-examples/configs/recipes/vlm/mopd-nemotron-super-omni-joint-opd-30n4g-hsg.v1.yaml}"
export MODEL_PATH="${MODEL_PATH:-/lustre/fsw/portfolios/llmservice/users/adithyare/code/main_branch_unified_opd/vlm_run_resources/vlm_student}"
# vision teacher (routes string_match_simple_agent)
export VLM_TEACHER_PATH="${VLM_TEACHER_PATH:-/lustre/fsw/portfolios/llmservice/users/adithyare/code/main_branch_unified_opd/vlm_run_resources/vlm_teacher}"
# text teacher (default alias "general": every text agent falls back to it)
export TEXT_TEACHER_PATH="${TEXT_TEACHER_PATH:-/lustre/fsw/portfolios/llmservice/users/yianz/projects/nemotron3/3.5_nano/copied_scripts/geshen/results/super_exp_rlvr_v41/step_70/policy/conversion/hf}"
export TRAIN_PATH="${TRAIN_PATH:-/lustre/fs1/portfolios/llmservice/projects/llmservice_modelalignment_ppo/users/adithyare/code/main_branch_unified_opd/vlm_run_resources/vlm_training_data/joint_opd_dolphin_v41_curriculum_plus_babyvision_interleaved.jsonl}"
export VAL_PATH="${VAL_PATH:-${TRAIN_PATH}}"
# Container: branch-matched pipeline image, proven on the omni 002-refmatch run.
export CONTAINER="${CONTAINER:-/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/yifuw/images/rl-super35-gym.f1b9b7d-63436537.20260818.sqsh}"
# Gym venvs on Lustre: joint run needs the text agents' venvs on top of
# string_match — first primer builds them into this shared dir.
export GYM_VENV_DIR="${GYM_VENV_DIR:-/lustre/fsw/portfolios/llmservice/users/adithyare/code/main_branch_unified_opd/cache/v3p5_omni_gym_venvs}"
mkdir -p "${GYM_VENV_DIR}"
# Sandbox nginx flakes on cluster DNS at job start. Proven retry wrapper:
export SANDBOX_COMMAND="${SANDBOX_COMMAND:-for _a in 1 2 3 4 5 6; do /start-with-nginx.sh; echo \"[RETRY] sandbox start attempt \$_a failed (transient DNS?); cleaning up and retrying\"; pkill -9 -f \"[u]vicorn\" 2>/dev/null; pkill -9 -x nginx 2>/dev/null; sleep 20; done}"
export SANDBOX_CONTAINER="${SANDBOX_CONTAINER:-/lustre/fsw/portfolios/llmservice/users/geshen/mopd_nano_fast/images/nemo-skills-sandbox-no-sync.sqsh}"
export PERSISTENT_CACHE="${PERSISTENT_CACHE:-/lustre/fsw/portfolios/llmservice/users/adithyare/code/main_branch_unified_opd/cache/v3p5_omni}"
export SLURM_ACCOUNT="${SLURM_ACCOUNT:-nemotron_n4_post}"
export GPUS_PER_NODE="${GPUS_PER_NODE:-4}"   # HSG GB200: 4 GPUs/node
export SLURM_PARTITION="${SLURM_PARTITION:-batch}"
export WANDB_PROJ="${WANDB_PROJ:-unified_opd_tests_main_hsg}"
export SLURM_TIME_LIMIT="${SLURM_TIME_LIMIT:-4:0:0}"

# teacher routing: strip trailing slashes, then override both aliases
while [[ "${VLM_TEACHER_PATH}" == */ && "${VLM_TEACHER_PATH}" != "/" ]]; do
    VLM_TEACHER_PATH="${VLM_TEACHER_PATH%/}"
done
while [[ "${TEXT_TEACHER_PATH}" == */ && "${TEXT_TEACHER_PATH}" != "/" ]]; do
    TEXT_TEACHER_PATH="${TEXT_TEACHER_PATH%/}"
done
teacher_overrides="on_policy_distillation.teacher_model_by_agent_name.string_match_simple_agent=${VLM_TEACHER_PATH} on_policy_distillation.teacher_model_by_agent_name.general=${TEXT_TEACHER_PATH}"
export EXTRA_HYDRA_ARGS="${EXTRA_HYDRA_ARGS:+${EXTRA_HYDRA_ARGS} }${teacher_overrides}"

export EXTRA_MOUNTS="${EXTRA_MOUNTS:-/lustre:/lustre}"
mkdir -p "${PERSISTENT_CACHE}"
exec "${SCRIPT_DIR}/examples/nemo_gym/nemotron-3-super-omni/super_omni_launch.sh"
