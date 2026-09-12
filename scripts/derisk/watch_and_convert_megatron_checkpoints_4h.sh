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
#
#SBATCH --account=nemotron_omni_vision
#SBATCH --partition=batch
#SBATCH --job-name=hf-malformed-v3
#SBATCH --nodes=8
#SBATCH --ntasks=8
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=64
#SBATCH --mem=0
#SBATCH --exclusive
#SBATCH --time=04:00:00

# Poll a live malformed-v3 run and convert every completed five-step MCore
# checkpoint to HF. MCore sources are never modified, moved, or deleted.
set -euo pipefail
umask 0022

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_path="$(cd "${script_dir}/../.." && pwd)"

run_root="${RUN_ROOT:?Set RUN_ROOT to the training result directory}"
hf_root="${HF_ROOT:-${run_root}/hf}"
container_path="${CONTAINER:-/scratch/fsw/portfolios/nemotron/projects/nemotron_omni_vision/users/ehosseiniasl/images/rl-gym.67009223-gym_ln_fix.sqsh}"
max_steps="${MAX_STEPS:-60}"
save_period="${SAVE_PERIOD:-5}"
poll_seconds="${POLL_SECONDS:-60}"
minimum_conversion_seconds="${MINIMUM_CONVERSION_SECONDS:-1200}"
watch_seconds="${WATCH_SECONDS:-13800}"
array_job_id="${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID:-0}}"
array_task_id="${SLURM_ARRAY_TASK_ID:-0}"

[[ -d "${run_root}" ]] || {
  echo "Waiting for run root to appear: ${run_root}"
}
[[ -f "${container_path}" ]] || {
  echo "ERROR: container not found: ${container_path}" >&2
  exit 1
}

mkdir -p "${hf_root}" "${repo_path}/.cache/hf_modules"
watch_deadline=$(( $(date +%s) + watch_seconds ))

finalize_hf_permissions() {
  local output_path="$1"
  if [[ -d "${output_path}" ]]; then
    chmod -R a+rX "${output_path}" || true
    find "${output_path}" -type d -exec chmod g+s {} + || true
  fi
}

hf_is_complete() {
  local output_path="$1"
  [[ -s "${output_path}/config.json" ]] &&
    [[ -s "${output_path}/model.safetensors.index.json" ]] &&
    find "${output_path}" -maxdepth 1 -name '*.safetensors' -type f -print -quit |
      grep -q .
}

convert_step() {
  local step="$1"
  local source_path="${run_root}/checkpoints/step_${step}"
  local export_root="${hf_root}/step_${step}"
  local output_path="${export_root}/mcore_to_hf"
  local partial_path="${export_root}/mcore_to_hf.partial.${array_job_id}.${array_task_id}"
  local modules_cache="${repo_path}/.cache/hf_modules/malformed-v3-step-${step}-${array_job_id}-${array_task_id}"

  if hf_is_complete "${output_path}"; then
    return 0
  fi

  test -s "${source_path}/config.yaml"
  test -s "${source_path}/policy/weights/iter_0000000/.metadata"
  test ! -e "${output_path}"
  test ! -e "${partial_path}"
  mkdir -p "${export_root}" "${modules_cache}"

  mapfile -t conversion_hosts < <(scontrol show hostnames "${SLURM_JOB_NODELIST}")
  export MASTER_ADDR="${conversion_hosts[0]}"
  export MASTER_PORT=$((18000 + (array_job_id + array_task_id + step) % 20000))
  export SOURCE_PATH="${source_path}"
  export OUTPUT_PATH="${partial_path}"
  export HF_MODULES_CACHE="${modules_cache}"

  echo "Converting step ${step}: ${source_path} -> ${output_path}"
  srun \
    --mpi=pmix \
    --nodes=8 \
    --ntasks=8 \
    --ntasks-per-node=1 \
    --gpus-per-task=4 \
    --no-container-mount-home \
    --container-image="${container_path}" \
    --container-mounts="/lustre:/lustre,/scratch:/scratch,${repo_path}/nemo_rl:/opt/nemo-rl/nemo_rl,${repo_path}/examples:/opt/nemo-rl/examples,${repo_path}/scripts:/opt/nemo-rl/scripts,${repo_path}/3rdparty/Megatron-Bridge-workspace/Megatron-Bridge:/opt/nemo-rl/3rdparty/Megatron-Bridge-workspace/Megatron-Bridge" \
    --container-workdir=/opt/nemo-rl \
    bash -lc '
      set -euo pipefail
      export CUDA_DEVICE_MAX_CONNECTIONS=1
      export OMP_NUM_THREADS=8
      export PYTHONPATH="/opt/nemo-rl:/opt/nemo-rl/3rdparty/Megatron-Bridge-workspace/Megatron-Bridge/src:/opt/nemo-rl/3rdparty/Megatron-Bridge-workspace/Megatron-Bridge/3rdparty/Megatron-LM:${PYTHONPATH:-}"
      policy_python=/opt/ray_venvs/nemo_rl.models.policy.workers.megatron_policy_worker.MegatronPolicyWorker/bin/python
      test -x "${policy_python}"
      "${policy_python}" -m torch.distributed.run \
        --nnodes="${SLURM_NNODES}" \
        --nproc_per_node=4 \
        --node_rank="${SLURM_PROCID}" \
        --master_addr="${MASTER_ADDR}" \
        --master_port="${MASTER_PORT}" \
        /opt/nemo-rl/scripts/derisk/convert_megatron_to_hf_gpu_no_mtp.py \
        --config "${SOURCE_PATH}/config.yaml" \
        --megatron-ckpt-path "${SOURCE_PATH}/policy/weights/iter_0000000" \
        --hf-ckpt-path "${OUTPUT_PATH}" \
        --tp 2 \
        --pp 1 \
        --ep 16 \
        --etp 1
    '

  hf_is_complete "${partial_path}"
  mv "${partial_path}" "${output_path}"
  touch "${output_path}/.conversion_complete"
  finalize_hf_permissions "${output_path}"
  du -sh "${output_path}"
}

while (( $(date +%s) < watch_deadline )); do
  all_complete=true
  converted_any=false

  for (( step=save_period; step<=max_steps; step+=save_period )); do
    output_path="${hf_root}/step_${step}/mcore_to_hf"
    source_path="${run_root}/checkpoints/step_${step}"

    if hf_is_complete "${output_path}"; then
      continue
    fi
    all_complete=false

    if [[ -d "${source_path}" ]] &&
      [[ -s "${source_path}/config.yaml" ]] &&
      [[ -s "${source_path}/policy/weights/iter_0000000/.metadata" ]]; then
      remaining_seconds=$(( watch_deadline - $(date +%s) ))
      if (( remaining_seconds < minimum_conversion_seconds )); then
        echo "Deferring step ${step}: only ${remaining_seconds}s remain in this four-hour watcher window."
        exit 0
      fi
      convert_step "${step}"
      converted_any=true
    fi
  done

  if [[ "${all_complete}" == "true" ]]; then
    echo "All checkpoints through step ${max_steps} have complete HF exports."
    exit 0
  fi
  if [[ "${converted_any}" == "false" ]]; then
    echo "No new completed checkpoint; polling again in ${poll_seconds}s."
    sleep "${poll_seconds}"
  fi
done

echo "Watcher window complete; the next serialized array task will continue."
