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
# Run on the OCI HSG login host after staging the private workspace.
# Math:       bash tools/launch_interactive.sh
# Synthetic:  FLINT_SYNTHETIC_SMOKE=1 bash tools/launch_interactive.sh
# Training overrides are forwarded unchanged to run_grpo.sh.
set -euo pipefail
set +x
umask 077

flint_task_root=${FLINT_TASK_ROOT:-/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/wenweng/Automodel/flint-grpo-49layer-20261019}
flint_model=${FLINT_MODEL_PATH:-/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/wenweng/Automodel/cedargrove-model_vv1}
flint_image=${FLINT_CONTAINER_IMAGE:-/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/wenweng/Automodel/flint-nemorl-20261019/images/flint-nemorl-terry-20261019-arm64-native-01.sqsh}
flint_secrets=${FLINT_SECRETS_FILE:-/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/wenweng/nemo-rl-internal/secrets.sh}
export FLINT_SYNTHETIC_SMOKE=${FLINT_SYNTHETIC_SMOKE:-0}

if [[ -n "${SLURM_JOB_ID:-}" ]]; then
  echo "Run this launcher on the OCI HSG login host outside an existing allocation." >&2
  exit 2
fi
if [[ ! -f "$flint_image" || ! -f "$flint_task_root/tools/run_grpo.sh" || ! -f "$flint_model/config.json" ]]; then
  echo "The container image, staged launcher, and full Flint checkpoint must exist before launching." >&2
  exit 2
fi
if [[ -e "$flint_task_root/RL/tools/generate_fingerprint.py" ]]; then
  echo "Stage the private runtime source with the stock fingerprint helper omitted, matching the custom image contract." >&2
  exit 2
fi
if [[ ! -r "$flint_secrets" ]]; then
  echo "The requested credentials file must be readable before launching." >&2
  exit 2
fi
# The file is user-owned credential configuration. Silence its output and never
# inspect or print the resulting environment. Export only variables it exports.
if ! source "$flint_secrets" >/dev/null 2>&1; then
  set +x
  echo "Could not source the requested credentials file." >&2
  exit 2
fi
set +x

flint_command=(/bin/bash /flint-work/tools/run_grpo.sh "$@")
if [[ "${FLINT_ALLOCATION_SHELL:-0}" == 1 ]]; then
  flint_command=(/bin/bash --noprofile --norc)
fi

# Fresh unnamed writable container. Original model files are mounted read-only.
exec srun \
  --partition=batch \
  --qos=interactive \
  --account=coreai_dlalgo_nemorl \
  --job-name=flint-grpo-49layer \
  --nodes=1 \
  --ntasks=1 \
  --gpus-per-node=4 \
  --cpus-per-task=64 \
  --mem=800G \
  --time=02:00:00 \
  --export=ALL \
  --container-image="$flint_image" \
  --container-mounts="$flint_task_root:/flint-work,$flint_model:/flint-model:ro" \
  --container-workdir=/flint-work/RL \
  --no-container-mount-home \
  --no-container-entrypoint \
  --container-writable \
  --pty \
  "${flint_command[@]}"
