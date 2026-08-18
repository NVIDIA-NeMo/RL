#!/usr/bin/env bash
set -euo pipefail
# =============================================================================
# RLVR on HSG — adithyare/gdpo-lenpen-no-truncation-main branch
#
# Thin wrapper over examples/nemo_gym/nemotron-3-super/super_launch.sh, in the
# style of RL-super-v3p5-posttraining/submit_super_32n_131k_*.sh.
#
# Geometry (config stage1_rlvr_hsg_32n4g.v1.yaml, derived from stage1_rlvr.yaml):
#   32 nodes x 4 GPUs (HSG GB200) = 16 train (TP4/CP8/EP16, DP2)
#                                 + 12 gen  (vLLM TP8 -> 6 engines)
#                                 +  4 gym  (nl2bash judge: Qwen3-235B FP8, TP8xDP2)
#   GBS 2048 = 128 prompts x 16 gens, seqlen 65536 (prod science, scaled fleet)
#   GenRM / safety-judge / jailbreak servers REMOVED (not in the training data).
#
# Defaults target venkats' RLVR setup (v18mix60-iter4500-mtp student + ccre_math
# data, 14 agents — all covered by this config's config_paths).
#
# 1h + 6h protocol:
#   WALLTIME_OVERRIDE=1:00:00 PARTITION_OVERRIDE=batch ./submit_rlvr_hsg_32n.sh
#   WALLTIME_OVERRIDE=6:00:00 PARTITION_OVERRIDE=batch_long \
#     SLURM_EXTRA_DEPENDENCY=afterany:<primer_jobid> ./submit_rlvr_hsg_32n.sh
#
# UNVALIDATED (first-primer checklist):
#   - CONTAINER: yifuw's omni 20260814 image is the closest lock-date match for
#     this main-based branch, but has NOT been exercised with it. Watch the uv
#     sync / venv stage of the first run.
#   - Gym venvs: the image's /opt/gym_venvs was baked for the omni env set; the
#     super servers here may be missing -> cold build into GYM_VENV_DIR below
#     (marker+flock semantics were fixed upstream of this branch's Gym pin, but
#     verify; prebuild recommended before production).
# =============================================================================

REPO="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "${REPO}"

export EXP_NAME="${EXP_NAME:-rlvr-lenpen-super-v18mtp-hsg-001}"
export CONFIG_PATH="${CONFIG_PATH:-examples/nemo_gym/nemotron-3-super/stage1_rlvr_hsg_32n4g.v1.yaml}"

# Student: venkats' v18mix60 iter4500 MTP checkpoint (old-schema wrapper export)
export MODEL_PATH="${MODEL_PATH:-/lustre/fsw/portfolios/llmservice/users/venkats/training_actual_0603/conv_wrappers/super-v18mix60-iter4500-mtp/evals/hf}"

# Data: venkats' 14-agent hard mix + ccre_math additions (269,158 rows)
export TRAIN_PATH="${TRAIN_PATH:-/lustre/fs1/portfolios/llmservice/projects/llmservice_modelalignment_ppo/users/venkats/reward_profiling/rlvr_profile_v18mix60_4500/filtered/v18_4500_hard_agents16_termV20refresh_ccre_math.len65k.shuf.train.jsonl}"
export VAL_PATH="${VAL_PATH:-${TRAIN_PATH}}"

# Containers: training image UNVALIDATED for this branch (see header); sandbox proven on HSG.
export CONTAINER="${CONTAINER:-/lustre/fsw/portfolios/coreai/users/yifuw/enroot-images/gitlab-master.nvidia.com/yifuw/images/nemo-rl:omni_mopd_20260814_prefetched_venvs_arm64.squashfs}"
export SANDBOX_CONTAINER="${SANDBOX_CONTAINER:-/lustre/fs1/portfolios/llmservice/projects/llmservice_modelalignment_ppo/users/geshen/containers/nemo-skills-sandbox-no-sync.sqsh}"

# Sandbox nginx DNS flake armor (proven on v3p5/baseline/omni stacks): retry
# /start-with-nginx.sh up to 6x; pkill patterns are self-immune (-x nginx, [u]vicorn).
export SANDBOX_COMMAND="${SANDBOX_COMMAND:-for _a in 1 2 3 4 5 6; do /start-with-nginx.sh; echo \"[RETRY] sandbox start attempt \$_a failed (transient DNS?); cleaning up and retrying\"; pkill -9 -f \"[u]vicorn\" 2>/dev/null; pkill -9 -x nginx 2>/dev/null; sleep 20; done}"

export PERSISTENT_CACHE="${PERSISTENT_CACHE:-${REPO}/cache}"
mkdir -p "${PERSISTENT_CACHE}"
# Gym venvs on Lustre (writable), NOT /opt/gym_venvs: the omni image's baked
# venvs don't cover the super server set. skip_venv_if_present reuses built ones.
export GYM_VENV_DIR="${GYM_VENV_DIR:-${PERSISTENT_CACHE}/gym_venvs}"

export GPUS_PER_NODE="${GPUS_PER_NODE:-4}"          # HSG GB200: 4 GPUs/node (ray.sub reads this)
export SLURM_ACCOUNT="${SLURM_ACCOUNT:-nemotron_sw_post}"
export SLURM_PARTITION="${PARTITION_OVERRIDE:-batch}"
export SLURM_TIME_LIMIT="${WALLTIME_OVERRIDE:-1:00:00}"

export WANDB_PROJ="${WANDB_PROJ:-unified_opd_tests_main_hsg}"
export EXTRA_MOUNTS="${EXTRA_MOUNTS:-/lustre:/lustre}"

# MTP student: enable speculative decoding in vLLM (weights arrive via refit)
export ENABLE_MTP_INFERENCE="${ENABLE_MTP_INFERENCE:-1}"

# W&B under the nvidia entity + anything else, appended last so it wins
export EXTRA_HYDRA_ARGS="++logger.wandb.entity=nvidia ${EXTRA_HYDRA_ARGS:-}"

exec bash examples/nemo_gym/nemotron-3-super/super_launch.sh "$@"
