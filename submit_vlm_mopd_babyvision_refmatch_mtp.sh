#!/usr/bin/env bash
set -euo pipefail
# Babyvision-reasoning MOPD on the UPDATED super-v3.5-posttraining stack (HSG, 26n4g).
# Ported from RL_omni_mopd_test_stacked 2026-08-20. OMNI runs only — text MOPD
# uses the submit_super_* scripts / pipeline repo launcher instead.
# Self-contained: thin wrapper over this repo's own super_omni_launch.sh,
# following the run_mopd_circle_count.sh pattern. No references to other stacks.
# Data: babyvision_reasoning local_images jsonl — image paths verified resolvable on HSG (2026-08-17).

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export EXP_NAME="${EXP_NAME:-mopd-super-omni-babyvision-hsg-003-refmatch-mtp}"
export CONFIG_PATH="${CONFIG_PATH:-examples/configs/recipes/vlm/mopd-nemotron-super-omni-babyvision-26n4g-hsg.v3-refmatch-mtp.yaml}"
export MODEL_PATH="${MODEL_PATH:-/lustre/fsw/portfolios/llmservice/users/adithyare/code/main_branch_unified_opd/vlm_run_resources/vlm_student}"
export TEACHER_MODEL_PATH="${TEACHER_MODEL_PATH:-/lustre/fsw/portfolios/llmservice/users/adithyare/code/main_branch_unified_opd/vlm_run_resources/vlm_teacher}"
export TRAIN_PATH="${TRAIN_PATH:-/lustre/fs1/portfolios/llmservice/projects/llmservice_modelalignment_ppo/users/adithyare/code/main_branch_unified_opd/vlm_run_resources/vlm_training_data/babyvision_reasoning_rep1_gym_random_shuf.local_images.jsonl}"
export VAL_PATH="${VAL_PATH:-${TRAIN_PATH}}"
# Container: the branch-matched pipeline image (built at f1b9b7d, same lock as
# this repo). UNVALIDATED for omni/VLM runs — first primer must watch venv sync
# and RADIO/vision deps; fall back to yifuw omni_mopd_20260814 image if broken.
export CONTAINER="${CONTAINER:-/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/yifuw/images/rl-super35-gym.f1b9b7d-63436537.20260818.sqsh}"
# Gym venvs on Lustre: the pipeline image may not bake a string_match venv.
export GYM_VENV_DIR="${GYM_VENV_DIR:-/lustre/fsw/portfolios/llmservice/users/adithyare/code/main_branch_unified_opd/cache/v3p5_omni_gym_venvs}"
mkdir -p "${GYM_VENV_DIR}"
# Sandbox nginx flakes on cluster DNS at job start (host-not-found in upstream;
# hit on every stack incl. this one, job 6265291). Proven retry wrapper:
export SANDBOX_COMMAND="${SANDBOX_COMMAND:-for _a in 1 2 3 4 5 6; do /start-with-nginx.sh; echo \"[RETRY] sandbox start attempt \$_a failed (transient DNS?); cleaning up and retrying\"; pkill -9 -f \"[u]vicorn\" 2>/dev/null; pkill -9 -x nginx 2>/dev/null; sleep 20; done}"
export SANDBOX_CONTAINER="${SANDBOX_CONTAINER:-/lustre/fsw/portfolios/llmservice/users/geshen/mopd_nano_fast/images/nemo-skills-sandbox-no-sync.sqsh}"
export PERSISTENT_CACHE="${PERSISTENT_CACHE:-/lustre/fsw/portfolios/llmservice/users/adithyare/code/main_branch_unified_opd/cache/v3p5_omni}"
export SLURM_ACCOUNT="${SLURM_ACCOUNT:-nemotron_n4_post}"
export GPUS_PER_NODE="${GPUS_PER_NODE:-4}"   # HSG GB200: 4 GPUs/node (ray.sub defaults to 8 and its GRES preflight hard-fails)
export SLURM_PARTITION="${SLURM_PARTITION:-batch}"
export WANDB_PROJ="${WANDB_PROJ:-unified_opd_tests_main_hsg}"
export SLURM_TIME_LIMIT="${SLURM_TIME_LIMIT:-4:0:0}"

# teacher: babyvision rows use string_match_simple_agent
while [[ "${TEACHER_MODEL_PATH}" == */ && "${TEACHER_MODEL_PATH}" != "/" ]]; do
    TEACHER_MODEL_PATH="${TEACHER_MODEL_PATH%/}"
done
teacher_override="on_policy_distillation.teacher_model_by_agent_name.string_match_simple_agent=${TEACHER_MODEL_PATH}"
export EXTRA_HYDRA_ARGS="${EXTRA_HYDRA_ARGS:+${EXTRA_HYDRA_ARGS} }${teacher_override}"

export EXTRA_MOUNTS="${EXTRA_MOUNTS:-/lustre:/lustre}"
mkdir -p "${PERSISTENT_CACHE}"
exec "${SCRIPT_DIR}/examples/nemo_gym/nemotron-3-super-omni/super_omni_launch.sh"
