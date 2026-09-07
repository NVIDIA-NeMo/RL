#!/usr/bin/env bash
# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

set -euo pipefail

: "${WANDB_API_KEY:?WANDB_API_KEY must be set in the environment}"
: "${GITLAB_PAT:?GITLAB_PAT must be set in the environment}"

MODE="${1:-dry-run}"
case "${MODE}" in
    dry-run|smoke|production) ;;
    *) echo "Usage: $0 [dry-run|smoke|production]" >&2; exit 2 ;;
esac

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(realpath "${SCRIPT_DIR}/../..")
SOURCE_DATA="${SOURCE_DATA:-/lustre/fs1/portfolios/llmservice/projects/llmservice_modelalignment_ppo/users/venkats/reward_profiling/rlvr_profile_v18mix60_4500/filtered/v18_4500_hard_agents16_termV20refresh_ccre_math.len65k.shuf.train.jsonl}"
DATA_DIR="${DATA_DIR:-${REPO_ROOT}/results/tropd-super35/data}"
if [[ -n "${TRAIN_PATH:-}" ]]; then
    if [[ ! -r "${TRAIN_PATH}" ]]; then
        echo "Requested training data is not readable: ${TRAIN_PATH}" >&2
        exit 1
    fi
else
    TRAIN_PATH="${DATA_DIR}/v18_4500_hard_agents13.no_math_judge.first1920.jsonl"
    python3 "${SCRIPT_DIR}/build_proxy_dataset.py" "${SOURCE_DATA}" "${TRAIN_PATH}"
fi

RUN_STAMP="${RUN_STAMP:-$(date -u +%Y%m%dT%H%M%SZ)}"
RUN_NAMESPACE="${TROPD_RUN_NAMESPACE:-tropd-super35-v24-to-oraclev1-alpha0p2-${RUN_STAMP}}"
if [[ "${MODE}" == "smoke" ]]; then
    RUN_NAMESPACE="${RUN_NAMESPACE}-smoke"
fi

export EXP_NAME="${RUN_NAMESPACE}"
export CONFIG_PATH="${SCRIPT_DIR}/tropd-super35.yaml"
export TRAIN_PATH
unset VAL_PATH || true
export MODEL_PATH="/lustre/fs1/portfolios/llmservice/projects/llmservice_modelalignment_ppo/users/venkats/training_actual_0603/super_n4_post/conv_wrappers/super-v24_1mix60-iter6000/evals/hf"
export CONTAINER="${CONTAINER:-/lustre/fsw/portfolios/coreai/users/yifuw/enroot-images/gitlab-master.nvidia.com/dl/joc/nemo-ci/main-mirror/rl-gym:pipe.64391373.squashfs}"
export SANDBOX_CONTAINER="${SANDBOX_CONTAINER:-/lustre/fsw/portfolios/llmservice/users/geshen/mopd_nano_fast/images/nemo-skills-sandbox-no-sync.sqsh}"
export SANDBOX_COMMAND="${SANDBOX_COMMAND:-/start-with-nginx.sh}"
for image in "${CONTAINER}" "${SANDBOX_CONTAINER}"; do
    if [[ ! -r "${image}" ]]; then
        echo "Required container image is not readable: ${image}" >&2
        exit 1
    fi
done
export RESULTS_DIR="${RESULTS_DIR:-${REPO_ROOT}/results/tropd-super35/runs/${RUN_NAMESPACE}}"
export PERSISTENT_CACHE="${PERSISTENT_CACHE:-${RESULTS_DIR}/cache}"
export WANDB_PROJ="kbhardwaj-tropd-super"
export SLURM_ACCOUNT="${SLURM_ACCOUNT:-nemotron_n4_post}"
DEFAULT_EXCLUDE_NODES="nvl72d183-T06,nvl72d042-T11,nvl72130-T14,nvl72131-T02"
if [[ -z "${EXCLUDE_NODES:-}" ]]; then
    # Keep the shared bad-node list portable across HSG partitions/clusters:
    # sbatch rejects an exclusion if even one named node is not registered.
    IFS=',' read -r -a exclude_candidates <<< "${DEFAULT_EXCLUDE_NODES}"
    registered_excludes=()
    for node in "${exclude_candidates[@]}"; do
        if scontrol show node "${node}" >/dev/null 2>&1; then
            registered_excludes+=("${node}")
        else
            echo "Skipping unregistered excluded node on this cluster: ${node}" >&2
        fi
    done
    EXCLUDE_NODES=$(IFS=','; echo "${registered_excludes[*]}")
fi
export EXCLUDE_NODES
export SLURM_PARTITION="batch_long"
export SLURM_QOS="normal"
export SLURM_TIME_LIMIT="23:59:59"
export SBATCH_NUM_NODES=32
export GPUS_PER_NODE=4
export CPUS_PER_WORKER=144
export SLURM_SEGMENT=2
export EXTRA_MOUNTS="/lustre:/lustre"
export ENABLE_MTP_INFERENCE=0
# The non-colocated collective refit uses these controls instead of
# policy.refit_buffer_size_gb. Optimizer state leaves less than 3 GiB free on
# the training ranks, so use one small staging buffer to avoid a post-step OOM.
export NRL_REFIT_BUFFER_MEMORY_RATIO="${NRL_REFIT_BUFFER_MEMORY_RATIO:-0.004}"
export NRL_REFIT_NUM_BUFFERS="${NRL_REFIT_NUM_BUFFERS:-1}"
export SKIP_CODE_SNAPSHOT=true

EXTRA_HYDRA_ARGS="logger.wandb.entity=nvidia"
if [[ "${MODE}" == "production" ]]; then
    # The explicit production dataset already has more than the 51,200 rows
    # consumed by 100 steps, so train over it once instead of triplicating it.
    EXTRA_HYDRA_ARGS+=" data.train.repeat=1"
elif [[ "${MODE}" == "smoke" ]]; then
    # Keep the production recipe at 32 nodes/196k/GBS512, but make the runtime
    # gate small enough to backfill quickly. This still loads student,
    # generation, and teacher models; performs one TROPD optimizer step; and
    # writes both checkpoint and diagnostic artifacts.
    export SBATCH_NUM_NODES=8
    export SLURM_PARTITION=batch
    export SLURM_TIME_LIMIT=02:00:00
    EXTRA_HYDRA_ARGS+=" grpo.max_num_steps=1 checkpointing.save_period=1 checkpointing.ft_save_period=1"
    EXTRA_HYDRA_ARGS+=" cluster.num_nodes=7 env.nemo_gym.num_gpu_nodes=1"
    EXTRA_HYDRA_ARGS+=" policy.generation.colocated.resources.num_nodes=2"
    EXTRA_HYDRA_ARGS+=" policy.max_total_sequence_length=65536"
    EXTRA_HYDRA_ARGS+=" policy.megatron_cfg.context_parallel_size=4 policy.megatron_cfg.expert_model_parallel_size=8"
    EXTRA_HYDRA_ARGS+=" on_policy_distillation.non_colocated_teachers.default_teacher_cfg.context_parallel_size=1"
    EXTRA_HYDRA_ARGS+=" on_policy_distillation.non_colocated_teachers.default_teacher_cfg.expert_model_parallel_size=4"
    EXTRA_HYDRA_ARGS+=" on_policy_distillation.non_colocated_teachers.default_teacher_cfg.num_nodes=1"
    EXTRA_HYDRA_ARGS+=" grpo.num_generations_per_prompt=1 policy.train_global_batch_size=32"
fi
export EXTRA_HYDRA_ARGS

if [[ "${MODE}" == "dry-run" || "${SUBMIT_DRY_RUN:-false}" == "true" ]]; then
    export DRY_RUN=true
else
    export DRY_RUN=false
fi

cd "${REPO_ROOT}"
exec bash examples/nemo_gym/nemotron-3-super/super_launch.sh
