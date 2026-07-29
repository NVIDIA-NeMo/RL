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

: "${CONTAINER:?Set CONTAINER to the derived Dynamo squashfs image}"
: "${SLURM_ACCOUNT:?Set SLURM_ACCOUNT}"
: "${SLURM_PARTITION:?Set SLURM_PARTITION}"

script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
repo_root=${REPO_ROOT:-$(cd "${script_dir}/../../.." && pwd)}
config=${CONFIG:-${repo_root}/examples/slurm/dynamo/grpo_math_1b_dynamo_ray.yaml}
ray_sub=${RAY_SUB:-${repo_root}/ray.sub}
results_dir=${RESULTS_DIR:-${repo_root}/results/dynamo-slurm-smoke}
log_dir=${LOG_DIR:-${results_dir}/logs}
dry_run=${DRY_RUN:-0}

if [[ ! -f "${config}" || ! -f "${ray_sub}" ]]; then
  echo "Missing config or ray.sub under ${repo_root}" >&2
  exit 1
fi
if [[ "${dry_run}" != "1" && ! -f "${CONTAINER}" ]]; then
  echo "Missing Dynamo container: ${CONTAINER}" >&2
  exit 1
fi

mkdir -p "${results_dir}" "${log_dir}"

export BASE_LOG_DIR="${log_dir}"
export RESULTS_DIR="${results_dir}"
export DYNAMO_PYTHON=/opt/dynamo_venv/bin/python
export GPUS_PER_NODE=2
export NEMO_RL_PY_EXECUTABLES_SYSTEM=1
export MOUNTS="${repo_root}:${repo_root}${EXTRA_MOUNTS:+,${EXTRA_MOUNTS}}"
printf -v COMMAND '%q ' \
  /opt/nemo_rl_venv/bin/python \
  -u \
  "${repo_root}/examples/run_grpo.py" \
  --config \
  "${config}"
export COMMAND

sbatch_args=(
  --nodes=1
  --account="${SLURM_ACCOUNT}"
  --job-name=nemo-rl-dynamo-smoke
  --partition="${SLURM_PARTITION}"
  --time="${SLURM_TIME_LIMIT:-01:00:00}"
  --gres=gpu:2
  --exclusive
  --output="${log_dir}/slurm-%j.out"
)

if [[ "${dry_run}" == "1" ]]; then
  printf 'COMMAND=%q\n' "${COMMAND}"
  printf 'sbatch'
  printf ' %q' "${sbatch_args[@]}"
  printf ' %q\n' "${ray_sub}"
  exit 0
fi

sbatch "${sbatch_args[@]}" "${ray_sub}"
