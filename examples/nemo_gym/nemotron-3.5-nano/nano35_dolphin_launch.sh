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

set -euo pipefail

# =============================================================================
# nano35_dolphin_launch.sh
#
# Nemotron 3.5 Nano — RLVR, legacy async-1, honest-dolphin warm start.
# Reproduces the internal reference run
#   geshen-ultra-rl-nano-honest-dolphin-v10-iter6000-mopd-rlvr
# (launch_nano_honest_dolphin.sh + grpo_ultra_512n4g_bf16.yaml on
#  nemo-rl-internal @ 97c55ee2) on public NeMo-RL main.
#
# This is a thin wrapper over examples/nemo_gym/nemotron-3-ultra/ultra_launch.sh.
# That launcher is fully parameterised and already handles code snapshotting,
# persistent-cache seeding, container mounts, Ray/Gym orchestration, and the
# OccupiedIdleGPUsJobReaper --comment exemption — so we set environment and
# delegate rather than forking 800+ lines.
#
# Usage. Judges run one of two ways; see the GenRM section below for the
# trade-off between them.
#
#   Against a warm out-of-band GenRM pool (default; must already be serving):
#     GENRM_BASE_URL=http://<lb-host>:9213/v1 \
#       bash examples/nemo_gym/nemotron-3.5-nano/nano35_dolphin_launch.sh
#
#   Hosting GenRM and the NL2Bash judge inside the job, as the 6K recipe does:
#     EXTERNAL_JUDGES=1 \
#       bash examples/nemo_gym/nemotron-3.5-nano/nano35_dolphin_launch.sh
#
#   DRY_RUN=1 GENRM_BASE_URL=... bash .../nano35_dolphin_launch.sh   # inspect only
#
# Extra Hydra overrides are forwarded verbatim:
#   GENRM_BASE_URL=... bash .../nano35_dolphin_launch.sh grpo.max_num_steps=2
# =============================================================================

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "${REPO_ROOT}"   # ultra_launch.sh derives PROJECT_ROOT from $PWD

# -----------------------------------------------------------------------------
# GenRM hosting. Two shapes, and this recipe supports both.
#
# EXTERNAL_JUDGES=1 is the 6K shape: GenRM and the NL2Bash judge are
# co-scheduled in a second Slurm hetgroup inside this job. The allocation
# wrapper brings up every replica plus one load balancer per pool, waits for
# health, then substitutes the resolved URLs into the driver command. The job
# owns its judges, so the reward model lands in provenance and no run can
# outlive, mismatch, or be starved by a pool it does not control.
#
# EXTERNAL_JUDGES=0 (default) keeps GenRM on a separately managed warm pool
# reached through GENRM_BASE_URL, and leaves the NL2Bash judge in the Gym pool.
# Partition `batch` caps at 4 h, so a warm pool amortizes the 470 GB bf16
# Qwen3-235B load across a whole chain of jobs. The in-job path instead pays
# that load once per job — about ten minutes before training starts, measured
# on the 6K run — in exchange for being self-contained.
#
# To stand up the out-of-band pool (copy the dir first — it holds .lb_pid_*,
# logs/ and a flock'd registry, so running someone else's in place collides):
#
#   cp -r /lustre/fs1/portfolios/llmservice/projects/llmservice_modelalignment_ppo/\
# users/geshen/mopd_nano_fast/genrm_serving  <your-dir>/genrm_serving
#   cd <your-dir>/genrm_serving
#   MODEL=/lustre/fsw/portfolios/llmservice/users/ansubramania/models/qwen235b_principle_comparison_genrm_step1230 \
#   ACCOUNT=nemotron_sw_post PARTITION=batch_long TIME=1-12:00:00 \
#   LB_PORT=9213 GENRM_GROUP_ID=nano35_dolphin \
#     ./genrm_server_manager.sh launch N
#   ./genrm_server_manager.sh url
#
# NOTE: that script's default MODEL is the *ultra* GenRM (step_720) — override it.
# Each worker is 2 nodes x 4 GPUs at TP=8, separate from this job's 64 nodes.
#
# The `model` field must equal the pool's --served-model-name ("model" in
# genrm_worker.sh). ultra_launch.sh sets base_url XOR model, never both, so the
# name is pinned in rlvr_dolphin.yaml instead of passed here.
# -----------------------------------------------------------------------------
export EXTERNAL_JUDGES="${EXTERNAL_JUDGES:-0}"
_DEFAULT_GYM_NODES=16
if [[ "${EXTERNAL_JUDGES}" == "1" ]]; then
  # Both pools reproduce the control run's serving shape, not just its GPU count,
  # so only the judges' location changes. GenRM is 2 replicas at TP=8 spanning
  # two nodes each, exactly as genrm_worker.sh ran them in the warm pool, filling
  # the same four nodes. NL2Bash is eight TP=4 replicas, equalling the TP4 x DP8
  # deployment it replaces in Gym.
  #
  # GenRM's TP is the one number worth not "simplifying" to a single node. The
  # checkpoint is 438 GB, so it is resident once per replica: 4 x TP4 would hold
  # four copies (1752 GB) on the same 16 GPUs where 2 x TP8 holds two (876 GB),
  # and the difference comes straight out of KV cache -- roughly 1.1 TB against
  # 2.0 TB pool-wide. Four replicas would buy more schedulers and more concurrent
  # slots, but GenRM's long prompts are KV-bound, so the control's shape wins.
  export GENRM_MODEL="${GENRM_MODEL:-/lustre/fsw/portfolios/llmservice/users/ansubramania/models/qwen235b_principle_comparison_genrm_step1230}"
  export GENRM_REPLICAS="${GENRM_REPLICAS:-2}"
  export GENRM_TENSOR_PARALLEL_SIZE="${GENRM_TENSOR_PARALLEL_SIZE:-8}"
  # The warm pool serves this checkpoint through the ultra_v3 parser plugin, so
  # carry both the plugin and its name over rather than vLLM's built-ins.
  export GENRM_REASONING_PARSER="${GENRM_REASONING_PARSER:-/lustre/fsw/portfolios/llmservice/users/lvega/evals/ultra_v3_reasoning_parser.py}"
  export GENRM_REASONING_PARSER_NAME="${GENRM_REASONING_PARSER_NAME:-ultra_v3}"
  # The control's genrm_worker.sh passes --enable-expert-parallel; the 6K
  # deployment of this checkpoint does not. Follow the control.
  export GENRM_ENABLE_EXPERT_PARALLEL="${GENRM_ENABLE_EXPERT_PARALLEL:-1}"
  export NL2BASH_REPLICAS="${NL2BASH_REPLICAS:-8}"
  export NL2BASH_TENSOR_PARALLEL_SIZE="${NL2BASH_TENSOR_PARALLEL_SIZE:-4}"
  export EXTERNAL_VLLM_SEGMENT_SIZE="${EXTERNAL_VLLM_SEGMENT_SIZE:-2}"
  # Gym drops to 8 nodes because the NL2Bash judge vacates exactly the 8 it was
  # filling: TP=4 puts one replica on each four-GPU node, and DP=8 means eight
  # of them. Every Gym node that was not serving NL2Bash stays, so the safety
  # judge and the CPU-side env servers are untouched.
  #
  # That keeps the GPU allocation comparable to the control run. Akash's
  # baseline was a 64-node job (5931924: 8 train + 40 gen + 16 gym) alongside a
  # 4-node GenRM pool of its own (5931683 and 5931688, 2 nodes each, spanning
  # the full training window) -- 68 nodes, 272 GB200 GPUs. This path is
  # 8 + 40 + 8 in hetgroup 0 plus 12 service nodes, which is the same 68.
  _DEFAULT_GYM_NODES=8
  #
  # CHECK THIS FIRST IF A POOL FAILS TO START. serve_vllm_on_ray.py imports
  # nemo_rl before vLLM's serve CLI, so pools must run the RL venv; they cannot
  # use the Gym venv that rlvr_dolphin.yaml deliberately gives the in-Gym
  # NL2Bash judge. On a vLLM 0.25 container that RL venv pairs vLLM with the
  # openai release uv.lock pins, and `vllm serve` dies importing NamespaceTool
  # from openai.types.responses (job 5943331). Judges share nothing with the
  # trainer, so the fix is to serve them from an image whose RL venv is
  # self-consistent rather than to match CONTAINER. Both pools default to
  # CONTAINER in ultra_launch.sh and are overridable independently:
  #   GENRM_CONTAINER=<image> NL2BASH_CONTAINER=<image>
else
  : "${GENRM_BASE_URL:?GENRM_BASE_URL must point to the external GenRM /v1 endpoint (or set EXTERNAL_JUDGES=1 to host GenRM in-job)}"
  export GENRM_BASE_URL
  unset GENRM_MODEL
fi

# -----------------------------------------------------------------------------
# Experiment identity
# EXP_NAME drives the W&B run name, the singleton job name, and the checkpoint
# and log dirs — so changing it starts a *new* run rather than resuming.
# -----------------------------------------------------------------------------
export EXP_NAME="${EXP_NAME:-akamehra-nano35-honest-dolphin-v10-iter6000-rlvr-async1-tp4_cp4_ep8_pp1_gpp16_pps128_gbs2048}"
export CONFIG_PATH="${CONFIG_PATH:-examples/nemo_gym/nemotron-3.5-nano/rlvr_dolphin.yaml}"

# -----------------------------------------------------------------------------
# Model and data
# -----------------------------------------------------------------------------
# honest-dolphin SFT v10 (closethink unmask, from midtrain 100B LC), iter_0006000.
export MODEL_PATH="${MODEL_PATH:-/lustre/fsw/portfolios/llmservice/users/venkats/training_actual_0603/nano_n3_post/checkpoints/nano-3.5-sft-v10-closethink-unmask-orig6k-from-midtrain-100B-lc-lr2e-5/eval/iter_0006000/hf}"

# trusty_viper: 199,680 prompts / 24 agent families, carrying agent_ref per row.
# VAL_PATH intentionally equals TRAIN_PATH, as in the reference — validation is
# effectively disabled (grpo.val_period is very large) because the genrm cohort
# envs are train-only and would hang under eval.
_BLEND="${_BLEND:-/lustre/fs1/portfolios/llmservice/projects/llmservice_modelalignment_ppo/users/geshen/rl-data-tools/blends/curriculum_honest_dolphin_v41_trusty_viper.train.jsonl}"
export TRAIN_PATH="${TRAIN_PATH:-${_BLEND}}"
export VAL_PATH="${VAL_PATH:-${_BLEND}}"

# -----------------------------------------------------------------------------
# Judge checkpoints. Where they run depends on EXTERNAL_JUDGES above: the safety
# judge is always served from the Gym pool, and the NL2Bash judge joins it there
# unless the in-job hetgroup is enabled, in which case Gym skips its local
# launch and proxies to the pool's load balancer instead.
# -----------------------------------------------------------------------------
export NL2BASH_JUDGE_MODEL="${NL2BASH_JUDGE_MODEL:-/lustre/fsw/portfolios/llmservice/users/ansubramania/models/Qwen3-235B-A22B-Instruct-2507-FP8}"
export SAFETY_JUDGE_MODEL="${SAFETY_JUDGE_MODEL:-/lustre/fsw/portfolios/llmservice/users/ansubramania/super_v3/model_checkpoints/Nemotron-Content-Safety-Reasoning-4B}"

# -----------------------------------------------------------------------------
# Containers
# Built 2026-08-14 with prefetched venvs, on vllm 0.25.1.
#
# The image supplies Megatron, and that is why this pin has to track the repo.
# ultra_launch.sh overlays only nemo_rl/, examples/configs and Gym from the
# worktree, so megatron.core always comes from the container. `46ab18ce1 ci: bump
# Megatron-Bridge to 0c565c9a0 (#3568)` landed on main 2026-08-11 and made
# megatron_policy_worker.py import FullyShardedDataParallelV1, which no image
# built before that date has -- including the 2026-07-30 one that used to be the
# default here and the 2026-08-07 main build. Both fail at
# `from megatron.core.distributed.fsdp.mcore_fsdp_adapter import ...` while the
# Megatron workers come up, ~20 min in (job 6231494). To check a candidate before
# spending an allocation on it:
#
#   unsquashfs -l <image> | grep mcore_fsdp_adapter
#   unsquashfs -d /tmp/probe -f <image> \
#     opt/nemo-rl/3rdparty/Megatron-Bridge-workspace/Megatron-Bridge/3rdparty/\
# Megatron-LM/megatron/core/distributed/fsdp/mcore_fsdp_adapter.py
#
# Keep vllm >= 0.25.1 when moving this pin: vllm_worker_async.py imports
# ServingTokenization from vllm.entrypoints.serve.tokenize.serving, added in
# 0.25.x, and a 0.20.0 image kills every generation worker at engine init ~25 min
# in, after the judges and the Megatron workers have already loaded (jobs 5942981
# and 5981739).
#
# The image's vllm and its nvidia-cutlass-dsl also have to agree, which is why
# this is the 08-14 build and not the 08-15 one that first got us past the
# Megatron import. The 08-15 image carries vllm 0.25.1+precompiled, whose
# vllm/vllm_flash_attn/cute/utils.py adds
# `from cutlass._mlir_helpers.arith import recast_type`; the cutlass-dsl 4.5.2
# it ships exposes that module only as cutlass.base_dsl._mlir_helpers, so the
# import raises ModuleNotFoundError. utils.py is imported lazily, from the FA4
# branch of flash_attn_varlen_func, so nothing surfaces until the first real
# attention forward: engine init and the weight sync both succeed, then every
# TP worker dies on the first rollout wave, EngineCore goes down, every
# generation 500s and the infra budget aborts the run ~3 min into serving (jobs
# 6233682 and 6234556, which is also where the scheduler's req_id_to_index
# KeyError came from -- collateral from the workers dying mid-batch, not a
# cause). FA4 is the default on these GB200s, so there is no getting to it.
# Add to the candidate probe above:
#
#   unsquashfs -d /tmp/probe -f <image> \
#     opt/ray_venvs/nemo_rl.models.generation.vllm.vllm_worker_async.\
# VllmAsyncGenerationWorker/lib/python3.13/site-packages/vllm/vllm_flash_attn/\
# cute/utils.py
#   # follow the symlink into /root/.cache/uv, then:
#   grep -n 'from cutlass' <extracted utils.py>   # must not name _mlir_helpers
#
# The 07-30 image passes that check but fails the Megatron one; 08-14 is the
# only build in yifuw's directory that passes both.
# -----------------------------------------------------------------------------
if [[ -z "${CONTAINER:-}" ]]; then
  : "${LUSTRE:?Set LUSTRE or provide CONTAINER explicitly}"
  export CONTAINER="${LUSTRE}/enroot-images/nvcr.io/nvidian/nemo-rl:nightly-gym.squashfs"
else
  export CONTAINER
fi

# -----------------------------------------------------------------------------
# Sandbox process DISABLED — this is what killed jobs 5726250 and 5732681.
#
# ray.sub launches the nemo-skills sandbox on every node with
# `--kill-on-bad-exit=1` (ray.sub:967), unlike the Ray worker step which uses
# `--kill-on-bad-exit=0` (ray.sub:1016). So a single node failing during sandbox
# startup makes srun tear down all 64 sandbox tasks; ray.sub then sees its
# background sandbox srun die and exits. At 64 nodes that happened on 3 of 3
# attempts, on different nodes each time (nvl72133-T01, nvl72141-T07,
# nvl72126-T05/T17) — node exclusion cannot fix it.
#
# Only three Gym servers use the sandbox: competitive_coding_challenges (not in
# our blend), math_formal_lean and ns_tools. Both of the latter are in
# config_paths and are LEFT THERE deliberately: they construct their sandbox
# client lazily (math_formal_lean/app.py:387 builds it, :444 uses it inside a
# request handler; ns_tools only holds sandbox_host/port as config), so they
# boot fine without a sandbox and only contact it if a request routes to them.
# The dolphin blend routes to neither, so the sandbox is never needed.
#
# ray.sub gates everything sandbox-related on SANDBOX_CONTAINER && SANDBOX_COMMAND
# both being non-empty (ray.sub:559) — the ports dir, the 64-instance ready wait,
# and the srun itself. The unmodified ultra launcher replaces an empty
# SANDBOX_COMMAND with its default, so keep SANDBOX_CONTAINER empty instead.
#
# Side benefit: no 16 GB sandbox image extracted on 64 nodes, so faster startup.
# To re-enable (e.g. if a future blend uses Lean4), explicitly set
# SANDBOX_CONTAINER to the sandbox image path.
# -----------------------------------------------------------------------------
# Unset -> historical default; set-but-EMPTY -> sandbox sidecar disabled
# (the dolphin blend never routes to the sandbox-backed servers, see above).
if [[ -z "${SANDBOX_CONTAINER+x}" ]]; then
  : "${LUSTRE:?Set LUSTRE or provide SANDBOX_CONTAINER explicitly}"
  export SANDBOX_CONTAINER="${LUSTRE}/enroot-images/nvcr.io/nvidian/nemo-rl:skills-sandbox-latest.squashfs"
else
  export SANDBOX_CONTAINER
fi

# -----------------------------------------------------------------------------
# Caches
# PERSISTENT_CACHE must be set explicitly: ultra_launch.sh requires it, and the
# internal reference's derivation (/lustre/fsw/portfolios/${ACCOUNT%%_*}/users/$USER)
# would resolve to /lustre/fsw/portfolios/nemotron/... which is read-only for us.
# HF_HOME is a *sibling* of the cache, not inside it, because the launcher purges
# vllm_compile_cache* under PERSISTENT_CACHE on every submission.
# HF_HOME also decides where the HF->Megatron conversion of the 62 GB checkpoint
# lands (get_megatron_checkpoint_dir falls back to $HF_HOME/nemo_rl), so keeping
# it on Lustre means the conversion is done once, not once per job.
# -----------------------------------------------------------------------------
export PERSISTENT_CACHE="${PERSISTENT_CACHE:-/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_llm/users/akamehra/.cache/nano35-dolphin}"
export HF_HOME="${HF_HOME:-/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_llm/users/akamehra/hf_home}"

# -----------------------------------------------------------------------------
# Container mounts — REQUIRED.
# ultra_launch.sh starts MOUNTS empty and only appends three source overlays
# (nemo_rl, examples/configs, Gym). It never mounts /lustre, so without this the
# container cannot see the checkpoint, the blend jsonl, the judge models,
# HF_HOME or PERSISTENT_CACHE. The internal reference hardcoded this mount.
# -----------------------------------------------------------------------------
export MOUNTS="${MOUNTS:-/lustre:/lustre}"

# -----------------------------------------------------------------------------
# Do not write bytecode into the Lustre-mounted source.
#
# nemo_rl is bind-mounted from Lustre into every container. Without this, all 64
# nodes write .pyc back into that shared tree (194 files appeared during earlier
# runs, including generation/__pycache__/interfaces.cpython-313.pyc). That is
# metadata churn on the exact directories every node is importing from.
#
# Job 5742619 died when Ray unpickled VllmAsyncGenerationWorker on one node:
# __init__.py executed from the mount, then its sibling interfaces.py was not
# found — a per-node directory-view inconsistency, not a missing file. Reads
# alone are far safer than reads plus concurrent writes.
# -----------------------------------------------------------------------------
export PYTHONDONTWRITEBYTECODE="${PYTHONDONTWRITEBYTECODE:-1}"

# -----------------------------------------------------------------------------
# examples/nemo_gym mount — REQUIRED, and the reason job 5730369 died with
#   FileNotFoundError: /opt/nemo-rl/examples/nemo_gym/nemotron-3.5-nano/rlvr_dolphin.yaml
#
# ultra_launch.sh overlays only nemo_rl, examples/configs and Gym. Everything
# else under /opt/nemo-rl — including examples/nemo_gym, where this recipe and
# the nemotron-3-ultra base config live — comes from the container image.
# Two things are therefore invisible without this mount:
#   1. rlvr_dolphin.yaml itself (written here, never in any image), and
#   2. nemotron-3-ultra/student_rlvr1.yaml, which it inherits — that landed on
#      main in 64cb9f985 (Jul 29-30) but this image was built Jul 26, so the
#      image genuinely does not contain it (verified by inspecting the image).
# Mounting the directory fixes both, and also removes a version skew: nemo_rl
# would otherwise come from this checkout while examples/nemo_gym came from a
# Jul-26 image.
#
# Scoped to this one directory rather than mounting the repo root over
# /opt/nemo-rl, to avoid shadowing anything the image builds in place.
# -----------------------------------------------------------------------------
_NEMO_GYM_MOUNT="${REPO_ROOT}/examples/nemo_gym:/opt/nemo-rl/examples/nemo_gym"
if [[ -n "${EXTRA_MOUNTS:-}" ]]; then
  export EXTRA_MOUNTS="${EXTRA_MOUNTS},${_NEMO_GYM_MOUNT}"
else
  export EXTRA_MOUNTS="${_NEMO_GYM_MOUNT}"
fi

# -----------------------------------------------------------------------------
# Snapshotting is OFF because tools/code_snapshot.sh copies only *git-tracked*
# files, and this recipe is untracked (`?? examples/nemo_gym/nemotron-3.5-nano/`).
# With snapshotting on, the mount above would point into a snapshot that does
# not contain the config. To restore frozen provenance, `git add` the recipe and
# set USE_SNAPSHOT=1.
# -----------------------------------------------------------------------------
export USE_SNAPSHOT="${USE_SNAPSHOT:-0}"

# -----------------------------------------------------------------------------
# Results root — MUST be absolute.
# ultra_launch.sh defaults RESULTS_DIR to the relative "results/${EXP_NAME}".
# The host mkdir would land it in the repo, but TRAIN_CMD does `cd /opt/nemo-rl`
# inside the container, so the same relative string resolves to
# /opt/nemo-rl/results/... — the container's ephemeral overlay. Checkpoints
# would vanish at job end and the singleton auto-resume would never find them,
# so on a 4 h wall the run would restart from the SFT checkpoint forever.
# An absolute Lustre path fixes checkpoints, logs, ray_logs and slurm output.
# -----------------------------------------------------------------------------
export RESULTS_DIR="${RESULTS_DIR:-/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_llm/users/akamehra/runs/${EXP_NAME}}"

# -----------------------------------------------------------------------------
# SLURM
# Job shape: 8 train + 40 gen + 16 gym = 64 GB200 nodes (4 GPUs each), a 5:1
# generation-to-training split. GenRM runs in its own allocation (2 replicas x
# 2 nodes), so the campaign footprint is 68 nodes.
# SEGMENT_SIZE=2 is the nano value; ultra defaults to 16.
# Partition `batch` caps at 4 h (batch_long is 7 d), so WALLTIME is 4 h and
# CHECKPOINTING_SAVE_BY keeps the reference's 25-minute teardown margin.
# -----------------------------------------------------------------------------
export SLURM_ACCOUNT="${SLURM_ACCOUNT:-nemotron_sw_post}"
export SLURM_PARTITION="${SLURM_PARTITION:-batch}"
export WALLTIME="${WALLTIME:-4:00:00}"
# Async rollout collection can leave the training GPUs idle while judges work,
# and startup is idle end to end: job 5965796 needed 56 minutes to reach its
# first optimizer step and the reaper took it at 60. This is the name
# ultra_launch.sh actually reads -- JOB_REAPER_EXEMPT_IDLE_MINS was never
# consumed by anything, so the banner reported a number with no effect.
export JOB_REAPER_EXEMPT_MINS="${JOB_REAPER_EXEMPT_MINS:-120}"
export CHECKPOINTING_SAVE_BY="${CHECKPOINTING_SAVE_BY:-00:03:35:00}"
export NUM_TRAIN_NODES="${NUM_TRAIN_NODES:-8}"
export NUM_GEN_NODES="${NUM_GEN_NODES:-40}"
export NUM_GYM_NODES="${NUM_GYM_NODES:-${_DEFAULT_GYM_NODES}}"
export SEGMENT_SIZE="${SEGMENT_SIZE:-2}"

# -----------------------------------------------------------------------------
# W&B. WANDB_API_KEY must already be in the environment — ultra_launch.sh needs
# it. If it is exported only from ~/.zshrc, submit from zsh; a bash context
# will not see it.
# -----------------------------------------------------------------------------
export WANDB_PROJ="${WANDB_PROJ:-ultra-streaming}"
export WANDB_ENTITY="${WANDB_ENTITY:-joc}"

# MTP: head *training* is on via the config (5 repeated layers, loss 0.3,
# detached heads), matching the reference. MTP *speculative decoding* for vLLM
# is a separate, independent switch and is off, also matching the reference.
export ENABLE_MTP_INFERENCE="${ENABLE_MTP_INFERENCE:-0}"

echo "================================================================"
echo "  Nemotron 3.5 Nano — RLVR async-1 (honest-dolphin)"
echo "================================================================"
echo "  Experiment : ${EXP_NAME}"
echo "  Config     : ${CONFIG_PATH}"
echo "  Model      : ${MODEL_PATH}"
echo "  Blend      : ${TRAIN_PATH}"
echo "  Container  : ${CONTAINER}"
echo "  Cache      : ${PERSISTENT_CACHE}"
echo "  HF_HOME    : ${HF_HOME}"
if [[ "${EXTERNAL_JUDGES}" == "1" ]]; then
echo "  GenRM      : in-job hetgroup — ${GENRM_REPLICAS} x TP=${GENRM_TENSOR_PARALLEL_SIZE}"
echo "               ${GENRM_MODEL}"
echo "  NL2Bash    : in-job hetgroup — ${NL2BASH_REPLICAS} x TP=${NL2BASH_TENSOR_PARALLEL_SIZE}"
else
echo "  GenRM      : ${GENRM_BASE_URL} (external pool; served model name: model)"
echo "  NL2Bash    : served in the Gym pool"
fi
echo "  SLURM      : ${SLURM_ACCOUNT} / ${SLURM_PARTITION} / ${WALLTIME}"
echo "  Reaper     : ${JOB_REAPER_EXEMPT_MINS} min idle exemption"
echo "  Nodes      : ${NUM_TRAIN_NODES} train + ${NUM_GEN_NODES} gen + ${NUM_GYM_NODES} gym"
echo "  W&B        : ${WANDB_ENTITY}/${WANDB_PROJ}"
echo "================================================================"
echo ""

exec bash examples/nemo_gym/nemotron-3-ultra/ultra_launch.sh "$@"
