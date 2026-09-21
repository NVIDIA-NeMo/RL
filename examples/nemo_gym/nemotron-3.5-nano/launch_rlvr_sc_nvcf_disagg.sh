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
# launch_rlvr_sc_nvcf_disagg.sh
#
# Submits the all-features arm of the Nemotron 3.5 Nano Amplified Dolphin RLVR
# study: SingleController with streaming train, the ready_first sampler at a
# staleness window of 1, judges hosted on NVCF, ns_tools sandboxes on a remote
# OpenSandbox pool, rollouts over CPU RDMA, and nccl_reshard refit.
#
# The training semantics live in rlvr_sc_nvcf_disagg.yaml. This wrapper only
# switches the deployment: which judges are hosted, where sandboxes come from,
# and the node split those two choices imply. It sets no site paths -- pass
# MODEL_PATH, TRAIN_PATH, VAL_PATH, CONTAINER, PERSISTENT_CACHE, SLURM_ACCOUNT
# and SLURM_PARTITION as you would to nano35_launch.sh, which validates them.
#
# Usage:
#
#   set -a; source /path/to/creds.env; set +a   # see "Credentials" below
#   EXP_NAME=nano35-allfeatures \
#   MODEL_PATH=... TRAIN_PATH=... VAL_PATH=... \
#   CONTAINER=... PERSISTENT_CACHE=... \
#   SLURM_ACCOUNT=... SLURM_PARTITION=batch \
#   bash examples/nemo_gym/nemotron-3.5-nano/launch_rlvr_sc_nvcf_disagg.sh
#
# Hydra overrides are forwarded verbatim, so a cheap smoke run is:
#
#   EXP_NAME=allfeatures-smoke NUM_TRAIN_NODES=4 NUM_GEN_NODES=2 NUM_GYM_NODES=2 \
#   NS_SANDBOX_POOL_SIZE=8 NRL_MAX_STEPS=2 WALLTIME=02:00:00 \
#   bash .../launch_rlvr_sc_nvcf_disagg.sh \
#     grpo.num_prompts_per_step=8 policy.train_global_batch_size=128 \
#     async_rl.min_groups_for_streaming_train=2 async_rl.max_inflight_prompts=16 \
#     async_rl.max_buffered_rollouts=16
#
# Note the last three: at a reduced prompt count the inherited 1024-slot buffers
# are far past what admission can reach, and min_groups_for_streaming_train must
# come down or a step never accumulates enough groups to train on.
#
# Credentials (export before running; every one is read inside the job, not
# baked into the submission):
#
#   NVIDIA_API_KEY        NVCF key. The config carries ${oc.env:NVIDIA_API_KEY}
#                         for all three judges, resolved in-container.
#   OPENSANDBOX_BASE_URL  Sandbox control plane, read by Gym's ns_tools.yaml.
#   OPENSANDBOX_API_KEY   Its key, likewise.
#   NS_SANDBOX_IMAGE      Sandbox image. Required even with a pool_ref: pods
#                         claimed from the pool template use it, and selecting
#                         the sandbox_pool backend with an empty image is a hard
#                         startup error rather than a fallback.
#   WANDB_API_KEY         Optional; absent disables W&B.
#   HF_TOKEN              Optional.
#
# These reach the job the same way WANDB_API_KEY always has: exported at submit
# time, inherited by every srun ray.sub launches, so gym actors on remote nodes
# see them too. ray.sub redacts *API_KEY* / *TOKEN* from its env dumps.
#
# Pool contract (nothing here touches the pool -- run these yourself):
#
#   Before submitting, raise the warm pool to cover the claim and confirm it is
#   warm, because a cold pool with pool_fallback=false fails slots rather than
#   creating pods on demand:
#
#     kubectl -n opensandbox patch pools.sandbox.opensandbox.io ${NS_SANDBOX_POOL_REF:-<pool>} \
#       --type merge -p '{"spec":{"capacitySpec":{"poolMin":<>=size>}}}'
#     kubectl -n opensandbox get pools.sandbox.opensandbox.io ${NS_SANDBOX_POOL_REF:-<pool>} \
#       -o jsonpath='{.status}{"\n"}'
#
#   The pool is shared, so size NS_SANDBOX_POOL_SIZE to what the run needs.
#   Afterwards reap claims by run label and drop poolMin back:
#
#     kubectl -n opensandbox delete bsbx \
#       -l "nemo-gym.nvidia.com/run=<NEMO_GYM_RUN_ID>" --wait=false
#
# Optional knobs (all defaulted below): NS_SANDBOX_POOL_REF,
# NS_SANDBOX_POOL_SIZE, NS_SANDBOX_POOL_FALLBACK, NS_SANDBOX_TTL_S,
# NEMO_GYM_RUN_ID, plus everything nano35_launch.sh accepts.
#
#   DISAGG_SANDBOX=1      0 keeps the NVCF judges but returns ns_tools to the
#                         per-node sidecar. Use it when comparing throughput
#                         against the local-judge arm, so judge placement is the
#                         only variable; the shared warm pool is a second one,
#                         and not one this launcher can hold constant. At 0 the
#                         OPENSANDBOX_* and NS_SANDBOX_* requirements above are
#                         dropped and SANDBOX_CONTAINER becomes required.
# =============================================================================

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

# -----------------------------------------------------------------------------
# Fail at submit time, not twelve minutes in
# -----------------------------------------------------------------------------
# Both failures below surface late and expensively: a missing NVCF key reaches
# the judges only once rollouts start, and a missing sandbox setting kills the
# ns_tools server after the allocation is up. Check them here.
if [[ "${DRY_RUN:-0}" != "1" ]]; then
  : "${NVIDIA_API_KEY:?Export your own NVIDIA_API_KEY (or set CREDS_FILE): required for all three NVCF-hosted judges}"
fi

# DISAGG_SANDBOX=0 keeps the NVCF judges but puts ns_tools back on the per-node
# sidecar, leaving judge placement as the ONLY difference from the local-judge
# arm. That is the point of the switch: with the remote pool also in play, a
# throughput delta against nano35_launch.sh confounds two variables at once, and
# the pool is shared, so its warmth is not even ours to hold constant between
# arms. Default stays 1 so the all-features arm is unchanged.
DISAGG_SANDBOX="${DISAGG_SANDBOX:-1}"
if [[ "${DISAGG_SANDBOX}" == "1" ]]; then
  : "${OPENSANDBOX_BASE_URL:?OPENSANDBOX_BASE_URL is required for the remote sandbox pool}"
  : "${OPENSANDBOX_API_KEY:?OPENSANDBOX_API_KEY is required for the remote sandbox pool}"
  : "${NS_SANDBOX_IMAGE:?NS_SANDBOX_IMAGE is required: sandbox_pool rejects an empty image even when claiming from a pool}"
else
  # nano35_launch.sh guards this too, but its message tells you to set
  # NO_COLOCATED_SANDBOX=1, which is the wrong knob from inside this wrapper.
  : "${SANDBOX_CONTAINER:?SANDBOX_CONTAINER is required when DISAGG_SANDBOX=0: the nemo-skills sandbox image for the per-node sidecar}"
fi

# -----------------------------------------------------------------------------
# Judges off the allocation
# -----------------------------------------------------------------------------
# rlvr_sc_nvcf_disagg.yaml carries every judge endpoint, so the launcher must
# emit no judge override and raise no GenRM hetgroup. 32 train + 32 gen + 2 gym.
export HOSTED_JUDGES=1
if (( "${NUM_EXTERNAL_SERVICE_NODES:-0}" != 0 )); then
  echo "ERROR: this arm hosts every judge on NVCF, so it has nothing to run on the ${NUM_EXTERNAL_SERVICE_NODES} external service nodes requested. Drop NUM_EXTERNAL_SERVICE_NODES, or launch nano35_launch.sh directly to serve judges locally." >&2
  exit 1
fi
export NUM_EXTERNAL_SERVICE_NODES=0
export NUM_TRAIN_NODES="${NUM_TRAIN_NODES:-32}"
export NUM_GEN_NODES="${NUM_GEN_NODES:-32}"
export NUM_GYM_NODES="${NUM_GYM_NODES:-2}"

# -----------------------------------------------------------------------------
# Sandboxes off the allocation
# -----------------------------------------------------------------------------
# Gym's ns_tools.yaml reads all of this straight from the environment at the
# pinned commit, which is why none of it appears in the NeMo-RL config. That
# also means the DISAGG_SANDBOX=0 branch has to UNSET rather than merely skip:
# ray.sub inherits the submitting shell, and these are exactly the variables a
# previous disagg submission left exported in it. Skipping them would hand the
# colocated arm a live sandbox_pool backend and quietly reproduce the very
# configuration this switch exists to remove.
if [[ "${DISAGG_SANDBOX}" == "1" ]]; then
  export NO_COLOCATED_SANDBOX=1
  export NS_TOOLS_SANDBOX_TYPE=sandbox_pool
  export NS_SANDBOX_POOL_REF="${NS_SANDBOX_POOL_REF:-ns-tools-warm}"
  export NS_SANDBOX_POOL_SIZE="${NS_SANDBOX_POOL_SIZE:-256}"
  # false: fail the slot instead of creating a pod on demand when the pool is
  # short. A silent fallback would answer with cold pods and read as a sandbox
  # latency regression rather than as an under-provisioned pool.
  export NS_SANDBOX_POOL_FALLBACK="${NS_SANDBOX_POOL_FALLBACK:-false}"
  export NS_SANDBOX_TTL_S="${NS_SANDBOX_TTL_S:-21600}"
  # Lean verification goes to the same sandboxes over HTTP rather than a local
  # server, so it does not need the sidecar this arm no longer starts.
  export MATH_FORMAL_LEAN_BACKEND="${MATH_FORMAL_LEAN_BACKEND:-ns_http}"
else
  export NO_COLOCATED_SANDBOX=0
  unset NS_TOOLS_SANDBOX_TYPE NS_SANDBOX_POOL_REF NS_SANDBOX_POOL_SIZE \
        NS_SANDBOX_POOL_FALLBACK NS_SANDBOX_TTL_S NS_SANDBOX_IMAGE \
        OPENSANDBOX_BASE_URL OPENSANDBOX_API_KEY
  # Left unset so ns_tools and math_formal_lean take their in-config defaults,
  # which target the sidecar. Forcing ns_http here would send Lean over HTTP to
  # a pool that this branch has just torn out of the environment.
  unset MATH_FORMAL_LEAN_BACKEND
fi
# Labels every claimed pod, and is the only handle for reaping leaks afterwards.
export NEMO_GYM_RUN_ID="${NEMO_GYM_RUN_ID:-nano35-allfeatures-$(date +%m%d-%H%M%S)}"

# -----------------------------------------------------------------------------
# Driver dependency resolution
# -----------------------------------------------------------------------------
# UV_FROZEN=1 makes the driver's `uv run` use the committed lock rather than
# re-locking when it disagrees with the tree. Two reasons, either sufficient:
# this arm bumps the Gym submodule pin, which is exactly the kind of change that
# invalidates the lock; and a re-lock fetches packages, which the CMH compute
# nodes cannot do -- the driver then hangs for tens of minutes against a
# blackholed CDN rather than failing. It also stops a 66-node job from quietly
# resolving different dependencies than the run before it. This assumes the
# image's baked venv already satisfies the tree, which is what prebaking is for.
export UV_FROZEN="${UV_FROZEN:-1}"

# -----------------------------------------------------------------------------
# SingleController entrypoint and config
# -----------------------------------------------------------------------------
export CONFIG_PATH="${CONFIG_PATH:-examples/nemo_gym/nemotron-3.5-nano/rlvr_sc_nvcf_disagg.yaml}"
export TRAIN_ENTRYPOINT="${TRAIN_ENTRYPOINT:-./examples/run_grpo_single_controller.py}"

echo "[allfeatures] judges: NVCF (no GenRM hetgroup)"
if [[ "${DISAGG_SANDBOX}" == "1" ]]; then
  echo "[allfeatures] sandboxes: pool=${NS_SANDBOX_POOL_REF} size=${NS_SANDBOX_POOL_SIZE} fallback=${NS_SANDBOX_POOL_FALLBACK}"
  echo "[allfeatures] run id (sandbox claim label): ${NEMO_GYM_RUN_ID}"
  echo "[allfeatures] confirm the pool is warm before this job starts claiming"
else
  echo "[allfeatures] sandboxes: COLOCATED per-node sidecar (DISAGG_SANDBOX=0)"
  echo "[allfeatures] sandbox image: ${SANDBOX_CONTAINER}"
  echo "[allfeatures] judge placement is the only delta vs the local-judge arm"
fi

exec bash "${SCRIPT_DIR}/nano35_launch.sh" rlvr "$@"
