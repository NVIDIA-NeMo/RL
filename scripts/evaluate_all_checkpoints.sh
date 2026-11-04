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
# evaluate_all_checkpoints.sh
#
# Run this ON YOUR MACBOOK. Given a run name, finds every converted HF
# checkpoint under results/<run>/hf/step_N/ (i.e. every step already
# processed by convert_all_checkpoints.sh) and submits one
# evaluate_bird.sbatch job per checkpoint, overriding MODEL_ROOT/MODEL_NAME
# per job -- same convention as the current hardcoded values in
# evaluate_bird.sbatch (MODEL_ROOT resolved to the physical hf/step_N path,
# MODEL_NAME = nemotron-3.5-lightning-<run>-<step>).
#
# Usage:
#   ./scripts/evaluate_all_checkpoints.sh oapl-20000seq-lr1e-5-beta-1-1-datafilter-500maxsteps-run14-avg-filter
#
#   # Only evaluate specific steps:
#   ./scripts/evaluate_all_checkpoints.sh <run_name> step_100 step_464
#
#   # Different sampling settings (defaults: 24000 / 0.6 / 3 repeats). Results go to
#   # results/maxout-24000-temp-1-num-repeats-3/<model_name>/ so earlier evals are kept:
#   MAX_OUTPUT_TOKENS=24000 TEMPERATURE=1.0 NUM_REPEATS=3 ./scripts/evaluate_all_checkpoints.sh <run_name> step_100
# =============================================================================

# Defaults for the HSG cluster (aarch64, 4 GPUs per node) -- override by exporting
# before running if you need a different cluster/path. REMOTE_REPO_DIR is the
# training checkout holding results/<run>/hf (written by convert_all_checkpoints.sh);
# GYM_ROOT is the Gym checkout (dipendra/birdspark branch) that evals run from.
LOGIN_HOST="${LOGIN_HOST:-dmisra@oci-hsg-cs-001-login-01}"
REMOTE_REPO_DIR="${REMOTE_REPO_DIR:-/lustre/fsw/portfolios/nemotron/users/dmisra/nemo-rl-r080}"
GYM_ROOT="${GYM_ROOT:-/lustre/fsw/portfolios/nemotron/users/dmisra/Gym}"

# Must mirror the defaults and RESULT_VARIANT naming in evaluate_bird.sbatch.
MAX_OUTPUT_TOKENS="${MAX_OUTPUT_TOKENS:-24000}"
TEMPERATURE="${TEMPERATURE:-0.6}"
NUM_REPEATS="${NUM_REPEATS:-3}"
# lightning (default) or dory; auto-detected from the run name, or export
# MODEL_FAMILY to override. Selects the vLLM flags in evaluate_bird.sbatch.
if [[ -z "${MODEL_FAMILY:-}" ]]; then
  case "${1:-}" in
    *dory*) MODEL_FAMILY=dory ;;
    *) MODEL_FAMILY=lightning ;;
  esac
fi
RESULT_VARIANT="maxout-${MAX_OUTPUT_TOKENS}-temp-$(printf '%g' "${TEMPERATURE}" | tr . p)-num-repeats-${NUM_REPEATS}"
# Dory only: the recurrent loop count changes the model's compute/scores, so it is part
# of the variant (e.g. ...-num-repeats-1-loops-2). Lightning variants are unchanged.
if [[ "${MODEL_FAMILY}" == dory ]]; then
  N_RECURRENT_LOOPS="${N_RECURRENT_LOOPS:-2}"
  RESULT_VARIANT="${RESULT_VARIANT}-loops-${N_RECURRENT_LOOPS}"
fi

RUN="${1:?Usage: $0 <run_name> [step_N ...]}"
shift || true
REQUESTED_STEPS=("$@")

HF_REL="results/${RUN}/hf"

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)

echo "Syncing scripts/evaluate_bird.sbatch -> ${LOGIN_HOST}:${GYM_ROOT} ..."
# sbatch opens its --output file before the job starts, so the directory must exist.
ssh "${LOGIN_HOST}" "mkdir -p '${GYM_ROOT}/results/slurm-logs'"
scp "${SCRIPT_DIR}/evaluate_bird.sbatch" "${LOGIN_HOST}:${GYM_ROOT}/evaluate_bird.sbatch"
if [[ "${MODEL_FAMILY}" == dory ]]; then
  # Vendored vLLM Dory overlay, applied inside the container by evaluate_bird.sbatch.
  echo "Syncing scripts/dory_vllm/ -> ${LOGIN_HOST}:${GYM_ROOT}/dory_vllm ..."
  ssh "${LOGIN_HOST}" "mkdir -p '${GYM_ROOT}/dory_vllm'"
  scp "${SCRIPT_DIR}"/dory_vllm/*.py "${LOGIN_HOST}:${GYM_ROOT}/dory_vllm/"
fi

echo "Discovering converted checkpoints under ${REMOTE_REPO_DIR}/${HF_REL} ..."
# Each line: "<step_dir> <physical_model_root>", e.g. "step_100 /scratch/fsw/.../hf/step_100"
# Only steps with a completed HF export (config.json present) are listed.
# Portable read loop instead of `mapfile`/`readarray`: macOS ships bash 3.2,
# which predates both builtins.
STEPS=()
while IFS= read -r line; do
  [[ -n "${line}" ]] && STEPS+=("${line}")
done < <(ssh "${LOGIN_HOST}" "cd '${REMOTE_REPO_DIR}' && for d in ${HF_REL}/step_*/; do step=\$(basename \"\$d\"); if [[ -f \"\${d}config.json\" ]]; then physical=\$(cd \"\$d\" && pwd -P); echo \"\$step \$physical\"; fi; done")

if [[ "${#STEPS[@]}" -eq 0 ]]; then
  echo "ERROR: no converted checkpoints (step_*/config.json) found under ${REMOTE_REPO_DIR}/${HF_REL}" >&2
  echo "Run convert_all_checkpoints.sh first." >&2
  exit 1
fi

echo "Found:"
printf '  %s\n' "${STEPS[@]}"

for entry in "${STEPS[@]}"; do
  read -r step model_root <<< "${entry}"

  if [[ "${#REQUESTED_STEPS[@]}" -gt 0 ]]; then
    match=0
    for wanted in "${REQUESTED_STEPS[@]}"; do
      [[ "${wanted}" == "${step}" ]] && match=1
    done
    [[ "${match}" -eq 1 ]] || continue
  fi

  if [[ "${MODEL_FAMILY}" == dory ]]; then
    model_name="${RUN}-${step}"
  else
    model_name="nemotron-3.5-lightning-${RUN}-${step}"
  fi

  # results/<variant>/<model_name>/ existing (regardless of whether it's
  # complete, in progress, or stale) means an eval for this checkpoint with
  # these MAX_OUTPUT_TOKENS/TEMPERATURE was already submitted -- skip rather
  # than resubmit. Other variants don't count.
  if ssh "${LOGIN_HOST}" "test -d '${GYM_ROOT}/results/${RESULT_VARIANT}/${model_name}'"; then
    echo "Skipping ${step}: results folder already exists for ${RESULT_VARIANT}/${model_name}"
    continue
  fi

  echo "Submitting eval for ${step} -> MODEL_ROOT=${model_root} MODEL_NAME=${model_name} variant=${RESULT_VARIANT} ..."
  ssh "${LOGIN_HOST}" "cd '${GYM_ROOT}' && \
    MODEL_ROOT='${model_root}' \
    MODEL_NAME='${model_name}' \
    MAX_OUTPUT_TOKENS='${MAX_OUTPUT_TOKENS}' \
    TEMPERATURE='${TEMPERATURE}' \
    NUM_REPEATS='${NUM_REPEATS}' \
    MODEL_FAMILY='${MODEL_FAMILY}' \
    N_RECURRENT_LOOPS='${N_RECURRENT_LOOPS:-2}' \
    sbatch --job-name=eval-${RUN}-${step}-${RESULT_VARIANT} \
      --export=ALL,MODEL_ROOT,MODEL_NAME,MAX_OUTPUT_TOKENS,TEMPERATURE,NUM_REPEATS,MODEL_FAMILY,N_RECURRENT_LOOPS \
      evaluate_bird.sbatch"
done

echo "All eval jobs submitted. Check with: ssh ${LOGIN_HOST} squeue -u \$(whoami)"
