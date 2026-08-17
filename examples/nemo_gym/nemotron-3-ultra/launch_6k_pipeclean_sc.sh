#!/bin/bash
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

# =============================================================================
# launch_6k_pipeclean_sc.sh
#
# SingleController variant of launch_6k_pipeclean.sh: same 6K-GPU Ultra
# pipeclean, driven by run_grpo_single_controller.py with streaming
# forward/backward and the TransferQueue data plane.
#
# Shape (GB200 NVL72, 4 GPUs/node), unchanged from the legacy pipeclean so the
# two are comparable:
#   NeMo RL: Training 512 / vLLM 960 / Gym 48
#   External judges: GenRM 16 nodes / NL2Bash 2 nodes
#
# Smaller-scale SC runs at a 4:1 generation-to-training ratio were still
# generation-bound, spending 71-73% of the step in exposed_generation. If this
# split starves, shift nodes with NUM_TRAIN_NODES / NUM_GEN_NODES /
# NUM_GYM_NODES (the total must stay a multiple of SEGMENT_SIZE=16).
#
# NO CHECKPOINTING. The SC path raises if checkpointing.enabled is true, so the
# run cannot resume across the wall clock and Slurm singleton buys nothing
# here. Size NRL_MAX_STEPS to fit one allocation.
#
# Usage — the site block below supplies model, data, container and Slurm
# defaults for the GB200 cluster this recipe was validated on, identical to
# launch_6k_pipeclean.sh, so the two runs differ only in the SC wiring:
#
#   WANDB_API_KEY=$WANDB_API_KEY NRL_MAX_STEPS=10 \
#   bash examples/nemo_gym/nemotron-3-ultra/launch_6k_pipeclean_sc.sh
#
# On any other cluster, export the site variables yourself (they are all
# ${VAR:-default} and every one of them wins over the default).
#
# Optional:
#   NRL_MAX_STEPS=4              # short pipeclean
#   WALLTIME=4:00:00
#   CONTEXT_PARALLEL_SIZE=16     # default; raise only if CP=16 still OOMs
#   STREAM_MIN_GROUPS=256        # async_rl.min_groups_for_streaming_train
#   NUM_STORAGE_UNITS=64         # data_plane.num_storage_units
#   REFIT_TRANSPORT=null         # fall back to the full-tensor NCCL broadcast
#   DRY_RUN=1
#   NUM_TRAIN_NODES / NUM_GEN_NODES / NUM_GYM_NODES  # override the 6K split
#
# Extra positional args are forwarded as Hydra overrides to ultra_launch.sh.
# =============================================================================
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)

# Default config — callers may still override CONFIG_PATH explicitly.
export CONFIG_PATH="${CONFIG_PATH:-${SCRIPT_DIR}/pipeclean_6k_sc.yaml}"

# The SC driver. data_plane.enabled=true (set in the config) is mandatory for it.
export TRAIN_ENTRYPOINT="${TRAIN_ENTRYPOINT:-./examples/run_grpo_single_controller.py}"

# =============================================================================
# Site defaults — identical to launch_6k_pipeclean.sh
# =============================================================================
# Kept byte-identical to the baseline wrapper on purpose: the SC comparison is
# only meaningful if both runs read the same checkpoint, blend and judges.
# =============================================================================
export MODEL_PATH="${MODEL_PATH:-/lustre/fsw/portfolios/llmservice/users/jiaqiz/models/ultra_stage2sft_step300}"
export TRAIN_PATH="${TRAIN_PATH:-/lustre/fsw/portfolios/llmservice/users/jiaqiz/data/gym/rl-data-tools/blends/curriculum_v35_inescapable-sawfly.train.efforts0p15_qamathcode.jsonl}"
export VAL_PATH="${VAL_PATH:-${TRAIN_PATH}}"

# Must carry vLLM 0.25.1 in the RL venvs to match this branch's code; a
# pre-bump image fails at import with "cannot import name ServingTokenization".
#
# Both images are Lustre-striped copies (lfs setstripe -c -1 -S 16M, 350 OSTs)
# of the upstream squashfs files. The originals sit on 8 OSTs, and ~1500 nodes
# extracting an 87 GB image from 8 OSTs is a read storm: pyxis fails with
# "failed to create container filesystem" / "could not wait for child:
# Interrupted system call" in the thousands, and the job dies before the Ray
# head comes up. Striping cut that to a handful of failures. Refresh the copies
# when the upstream image changes:
#   mkdir -p <dir> && lfs setstripe -c -1 -S 16M <dir> && cp <upstream> <dir>/
# The sandbox copy pins the dc43f3e build that nemo-skills-sandbox-latest.sqsh
# resolved to, so the sandbox version no longer drifts between runs.
export CONTAINER="${CONTAINER:-/lustre/fsw/portfolios/llmservice/users/sauramishra/images-striped/nemo-rl-nightly-20260806-sandbox.squashfs}"
export SANDBOX_CONTAINER="${SANDBOX_CONTAINER:-/lustre/fsw/portfolios/llmservice/users/sauramishra/images-striped/nemo-skills-sandbox-dc43f3e.sqsh}"
export EXTRA_MOUNTS="${EXTRA_MOUNTS:-/lustre:/lustre}"
export PERSISTENT_CACHE="${PERSISTENT_CACHE:-/lustre/fsw/portfolios/llmservice/users/${USER}/.cache/nemotron_ultra}"
export HF_HOME="${HF_HOME:-/lustre/fsw/portfolios/llmservice/users/${USER}/hf_home}"

export GENRM_MODEL="${GENRM_MODEL:-/lustre/fsw/portfolios/llmservice/users/ansubramania/models/qwen235b_principle_comparison_genrm_step1230}"
export NL2BASH_JUDGE_MODEL="${NL2BASH_JUDGE_MODEL:-/lustre/fsw/portfolios/llmservice/users/ansubramania/models/Qwen3-235B-A22B-Instruct-2507-FP8}"
export SAFETY_JUDGE_MODEL="${SAFETY_JUDGE_MODEL:-/lustre/fsw/portfolios/llmservice/users/ansubramania/super_v3/model_checkpoints/Nemotron-Content-Safety-Reasoning-4B}"

# Keep judge deployment identical to launch_6k_pipeclean.sh so the comparison
# isolates the SingleController path.
export EXTERNAL_JUDGES="${EXTERNAL_JUDGES:-1}"
export GENRM_REPLICAS="${GENRM_REPLICAS:-16}"
export GENRM_TENSOR_PARALLEL_SIZE="${GENRM_TENSOR_PARALLEL_SIZE:-4}"
export GENRM_REASONING_PARSER_NAME="${GENRM_REASONING_PARSER_NAME:-deepseek_r1}"
export GENRM_ENABLE_EXPERT_PARALLEL="${GENRM_ENABLE_EXPERT_PARALLEL:-0}"
export NL2BASH_REPLICAS="${NL2BASH_REPLICAS:-2}"
export NL2BASH_TENSOR_PARALLEL_SIZE="${NL2BASH_TENSOR_PARALLEL_SIZE:-4}"
export EXTERNAL_VLLM_SEGMENT_SIZE="${EXTERNAL_VLLM_SEGMENT_SIZE:-2}"
export EXTERNAL_VLLM_SKIP_PREFLIGHT="${EXTERNAL_VLLM_SKIP_PREFLIGHT:-1}"

export SLURM_ACCOUNT="${SLURM_ACCOUNT:-nemotron_sw_pre}"
export SLURM_PARTITION="${SLURM_PARTITION:-batch}"
# A 6K allocation may additionally need SLURM_QOS, SLURM_RESERVATION and
# EXCLUDE_NODES; see launch_6k_pipeclean.sh.

# 6K node split, identical to the legacy pipeclean. Callers may override.
export NUM_TRAIN_NODES="${NUM_TRAIN_NODES:-512}"
export NUM_GEN_NODES="${NUM_GEN_NODES:-960}"
export NUM_GYM_NODES="${NUM_GYM_NODES:-48}"

# CP=16 is baked into pipeclean_6k.yaml; allow an override for memory experiments.
CONTEXT_PARALLEL_SIZE="${CONTEXT_PARALLEL_SIZE:-16}"

# The two SC knobs worth sweeping without editing the config. Lowering
# STREAM_MIN_GROUPS starts the optimizer step earlier on partial cohorts;
# NUM_STORAGE_UNITS is the untuned one at this data volume.
STREAM_MIN_GROUPS="${STREAM_MIN_GROUPS:-256}"
NUM_STORAGE_UNITS="${NUM_STORAGE_UNITS:-64}"

# Shard-to-shard weight refit, on by default in this variant. It is still
# experimental, so keep the escape hatch one env var away: REFIT_TRANSPORT=null
# restores the full-tensor broadcast that pipeclean_6k.yaml uses.
REFIT_TRANSPORT="${REFIT_TRANSPORT:-nccl_reshard}"

# Sensible defaults for short 6K hero / pipeclean allocations when unset.
export EXP_NAME="${EXP_NAME:-ultra-6k-pipeclean-sc}"
export WALLTIME="${WALLTIME:-4:00:00}"

exec bash "${SCRIPT_DIR}/ultra_launch.sh" \
  "policy.megatron_cfg.context_parallel_size=${CONTEXT_PARALLEL_SIZE}" \
  "async_rl.min_groups_for_streaming_train=${STREAM_MIN_GROUPS}" \
  "data_plane.num_storage_units=${NUM_STORAGE_UNITS}" \
  "policy.generation.refit_transport=${REFIT_TRANSPORT}" \
  "$@"
