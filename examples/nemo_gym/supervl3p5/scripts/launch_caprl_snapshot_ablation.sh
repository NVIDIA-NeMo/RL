#!/bin/bash
# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Submit independent fresh controls; DRY_RUN=1 validates both without submitting.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ABLATION_NAME="${ABLATION_NAME:-supervl35-caprl-snapshot-ablation-$(date +%Y%m%d-%H%M%S)}"
export SLURM_QOS="${SLURM_QOS:-normal}"
export WALLTIME="${WALLTIME:-3:00:00}"
export RESUME=0

# Keep both arms identical except for rollout snapshotting and its required
# recovery level. Trainer/TQ saves remain enabled in the no-rollout-snapshot arm.
COMMON=(
  grpo.max_num_steps=30
  ++async_rl.batch_attribution=true
  ++async_rl.log_full_train_data=false
  ++rollout_checkpointing.telemetry_interval_s=30
  ++policy.generation.vllm_cfg.enable_vllm_metrics_logger=true
  ++policy.generation.vllm_cfg.vllm_metrics_logger_interval=10
)

launch_arm() {
  local arm="$1" recovery="$2" interval="$3"
  EXP_NAME="${ABLATION_NAME}-${arm}" bash "${SCRIPT_DIR}/launch_caprl_prefix_smoke.sh" \
    "${COMMON[@]}" \
    "rollout_recovery.target_level=${recovery}" \
    "rollout_checkpointing.snapshot_attempt_interval_s=${interval}"
}

# Validate both configurations and all assets before allocating either arm.
if [[ "${DRY_RUN:-0}" != 1 ]]; then
  (export DRY_RUN=1; launch_arm baseline-nosnapshot sibling null)
  (export DRY_RUN=1; launch_arm prefix600 prefix 600)
fi
launch_arm baseline-nosnapshot sibling null
launch_arm prefix600 prefix 600
