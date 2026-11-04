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
# convert_all_checkpoints.sh
#
# Run this ON YOUR MACBOOK. SSHes into the login node, finds every step_*
# checkpoint under a run's checkpoints/ dir, and submits one
# convert_checkpoint_to_hf.sbatch job per checkpoint (each an independent
# 4-GPU job). Every export lands under results/<run>/hf/step_N/.
#
# Usage:
#   ./scripts/convert_all_checkpoints.sh oapl-20000seq-lr1e-5-beta-1-1e3-run4
#
#   # Only convert specific steps:
#   ./scripts/convert_all_checkpoints.sh oapl-20000seq-lr1e-5-beta-1-1e3-run4 step_100 step_464
# =============================================================================

# Defaults for the HSG cluster (aarch64, 4 GPUs per node) -- override by exporting
# before running if you need a different cluster/path. Keep all three in sync if
# you do: a LOGIN_HOST from one cluster combined with a REMOTE_REPO_DIR/CONTAINER
# default from another silently breaks. REMOTE_REPO_DIR is the training checkout
# that holds results/<run>/checkpoints (see scripts/local.env.rc4).
LOGIN_HOST="${LOGIN_HOST:-dmisra@oci-hsg-cs-001-login-01}"
REMOTE_REPO_DIR="${REMOTE_REPO_DIR:-/lustre/fsw/portfolios/nemotron/users/dmisra/nemo-rl-r080}"
CONTAINER="${CONTAINER:-/lustre/fsw/portfolios/nemotron/users/dmisra/containers/nemo-rl-v0.8.0.rc4-arm64.sqsh}"

RUN="${1:?Usage: $0 <run_name> [step_N ...]}"
shift || true
REQUESTED_STEPS=("$@")

CHECKPOINTS_REL="results/${RUN}/checkpoints"

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)

echo "Syncing scripts/convert_checkpoint_to_hf.sbatch -> ${LOGIN_HOST}:${REMOTE_REPO_DIR} ..."
ssh "${LOGIN_HOST}" "mkdir -p '${REMOTE_REPO_DIR}/results/slurm-logs'"
scp "${SCRIPT_DIR}/convert_checkpoint_to_hf.sbatch" "${LOGIN_HOST}:${REMOTE_REPO_DIR}/convert_checkpoint_to_hf.sbatch"

echo "Discovering checkpoints under ${CHECKPOINTS_REL} ..."
# Each line: "<step_dir> <iter_dir> <done|pending>", e.g. "step_464 iter_0000000 done"
# "done" means results/<run>/hf/<step>/config.json already exists -- the same
# file evaluate_bird.sbatch itself checks for, so it's a reliable completion
# marker for a prior successful conversion.
# Portable read loop instead of `mapfile`/`readarray`: macOS ships bash 3.2,
# which predates both builtins.
STEP_ITER_PAIRS=()
while IFS= read -r line; do
  [[ -n "${line}" ]] && STEP_ITER_PAIRS+=("${line}")
done < <(ssh "${LOGIN_HOST}" "cd '${REMOTE_REPO_DIR}' && for d in ${CHECKPOINTS_REL}/step_*/; do step=\$(basename \"\$d\"); iter=\$(ls \"\${d}policy/weights/\" 2>/dev/null | head -1); if [[ -n \"\$iter\" ]]; then if [[ -f \"results/${RUN}/hf/\$step/config.json\" ]]; then echo \"\$step \$iter done\"; else echo \"\$step \$iter pending\"; fi; fi; done")

if [[ "${#STEP_ITER_PAIRS[@]}" -eq 0 ]]; then
  echo "ERROR: no step_*/policy/weights/iter_* checkpoints found under ${REMOTE_REPO_DIR}/${CHECKPOINTS_REL}" >&2
  exit 1
fi

echo "Found:"
printf '  %s\n' "${STEP_ITER_PAIRS[@]}"

for pair in "${STEP_ITER_PAIRS[@]}"; do
  read -r step iter status <<< "${pair}"

  if [[ "${#REQUESTED_STEPS[@]}" -gt 0 ]]; then
    match=0
    for wanted in "${REQUESTED_STEPS[@]}"; do
      [[ "${wanted}" == "${step}" ]] && match=1
    done
    [[ "${match}" -eq 1 ]] || continue
  fi

  hf_ckpt_path="results/${RUN}/hf/${step}"

  if [[ "${status}" == "done" ]]; then
    echo "Skipping ${step} (${iter}): HF export already exists at ${hf_ckpt_path}/"
    continue
  fi

  megatron_ckpt_path="${CHECKPOINTS_REL}/${step}/policy/weights/${iter}"
  config_path="${CHECKPOINTS_REL}/${step}/config.yaml"

  echo "Submitting conversion for ${step} (${iter}) -> ${hf_ckpt_path} ..."
  ssh "${LOGIN_HOST}" "cd '${REMOTE_REPO_DIR}' && \
    MEGATRON_CKPT_PATH='${megatron_ckpt_path}' \
    CONFIG_PATH='${config_path}' \
    HF_CKPT_PATH='${hf_ckpt_path}' \
    CONTAINER='${CONTAINER}' \
    REMOTE_REPO_DIR='${REMOTE_REPO_DIR}' \
    sbatch --job-name=convert-${RUN}-${step} \
      --output='${REMOTE_REPO_DIR}/results/slurm-logs/convert-${RUN}-${step}-%j.log' \
      --export=ALL,MEGATRON_CKPT_PATH,CONFIG_PATH,HF_CKPT_PATH,CONTAINER,REMOTE_REPO_DIR \
      convert_checkpoint_to_hf.sbatch"
done

echo "All conversion jobs submitted. Check with: ssh ${LOGIN_HOST} squeue -u \$(whoami)"
