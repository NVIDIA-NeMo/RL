#!/bin/bash
set -euo pipefail

# =============================================================================
# nano35_dolphin_launch_sc.sh
#
# SingleController variant of nano35_dolphin_launch.sh: the same 64-node
# Nemotron 3.5 Nano RLVR pipeclean, driven by run_grpo_single_controller.py
# with streaming forward/backward, the TransferQueue data plane and
# shard-to-shard NCCL-reshard weight refit.
#
# Shape (GB200, 4 GPUs/node -> 256 GPUs), unchanged from the baseline so the
# two runs are comparable:
#   8 train + 40 generation + 16 gym = 64 nodes  (5:1 generation-to-training)
# GenRM adds 4 nodes from its own allocation, so the campaign footprint is 68.
#
# This is a thin wrapper over nano35_dolphin_launch.sh, which already carries
# every site default (model, blend, judges, container, mounts, caches, Slurm).
# We only swap the config and the driver, so the two runs differ solely in the
# SC wiring.
#
# NO CHECKPOINTING. The SC path raises if checkpointing.enabled is true, so the
# run cannot resume across the wall clock and the baseline's singleton
# auto-resume buys nothing here. Size NRL_MAX_STEPS to fit the 4 h allocation.
#
# Usage:
#   GENRM_BASE_URL=http://<lb-host>:9213/v1 \
#     bash examples/nemo_gym/nemotron-3.5-nano/nano35_dolphin_launch_sc.sh
#
#   DRY_RUN=1 GENRM_BASE_URL=... bash .../nano35_dolphin_launch_sc.sh
#
# GenRM must already be serving — see the GenRM section of the baseline script
# for how to stand up the external pool.
#
# Optional:
#   NRL_MAX_STEPS=10             # short pipeclean
#   STREAM_MIN_GROUPS=32         # async_rl.min_groups_for_streaming_train
#   MAX_LOOKAHEAD_VERSIONS=1     # async_rl.sampler.max_lookahead_versions
#   NUM_STORAGE_UNITS=8          # data_plane.num_storage_units
#   REFIT_TRANSPORT=null         # fall back to the full-tensor NCCL broadcast
#
# Extra positional args are forwarded as Hydra overrides, after ours, so they win.
# =============================================================================

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)

export CONFIG_PATH="${CONFIG_PATH:-examples/nemo_gym/nemotron-3.5-nano/rlvr_dolphin_sc.yaml}"

# The SC driver. data_plane.enabled=true (set in the config) is mandatory for it.
export TRAIN_ENTRYPOINT="${TRAIN_ENTRYPOINT:-./examples/run_grpo_single_controller.py}"

# Distinct from the baseline's EXP_NAME so this starts a new W&B run, run dir
# and singleton job name rather than colliding with the async-1 baseline.
export EXP_NAME="${EXP_NAME:-akamehra-nano35-honest-dolphin-v10-iter6000-rlvr-sc-tp4_cp4_ep16_pp1_gpp16_pps128_gbs2048}"

# The SC knobs worth sweeping without editing the config.
#
# STREAM_MIN_GROUPS starts the optimizer step earlier on partial cohorts.
# NUM_STORAGE_UNITS is the untuned one at this data volume.
# MAX_LOOKAHEAD_VERSIONS is the off-policyness ceiling; raising it also raises
# the two capacity floors below, which validate_sampler_buffer_capacity
# enforces (max_buffered_rollouts >= num_prompts_per_step * (lookahead + 1)).
STREAM_MIN_GROUPS="${STREAM_MIN_GROUPS:-32}"
NUM_STORAGE_UNITS="${NUM_STORAGE_UNITS:-8}"
MAX_LOOKAHEAD_VERSIONS="${MAX_LOOKAHEAD_VERSIONS:-1}"

# Shard-to-shard weight refit, on by default in this variant. It is still
# experimental, so keep the escape hatch one env var away: REFIT_TRANSPORT=null
# restores the full-tensor broadcast that rlvr_dolphin.yaml uses.
REFIT_TRANSPORT="${REFIT_TRANSPORT:-nccl_reshard}"

_NUM_PROMPTS_PER_STEP="${_NUM_PROMPTS_PER_STEP:-128}"
_BUFFER_CAPACITY=$(( _NUM_PROMPTS_PER_STEP * (MAX_LOOKAHEAD_VERSIONS + 1) ))

echo "================================================================"
echo "  Nemotron 3.5 Nano — RLVR SingleController (honest-dolphin)"
echo "================================================================"
echo "  Entrypoint : ${TRAIN_ENTRYPOINT}"
echo "  Config     : ${CONFIG_PATH}"
echo "  Refit      : ${REFIT_TRANSPORT}"
echo "  Streaming  : min ${STREAM_MIN_GROUPS} of ${_NUM_PROMPTS_PER_STEP} groups per dispatch"
echo "  Lookahead  : ${MAX_LOOKAHEAD_VERSIONS} (buffer ${_BUFFER_CAPACITY} groups)"
echo "  TQ units   : ${NUM_STORAGE_UNITS}"
echo "================================================================"
echo ""

exec bash "${SCRIPT_DIR}/nano35_dolphin_launch.sh" \
  "async_rl.min_groups_for_streaming_train=${STREAM_MIN_GROUPS}" \
  "async_rl.sampler.max_lookahead_versions=${MAX_LOOKAHEAD_VERSIONS}" \
  "async_rl.max_inflight_prompts=${_BUFFER_CAPACITY}" \
  "async_rl.max_buffered_rollouts=${_BUFFER_CAPACITY}" \
  "data_plane.num_storage_units=${NUM_STORAGE_UNITS}" \
  "policy.generation.refit_transport=${REFIT_TRANSPORT}" \
  "$@"
