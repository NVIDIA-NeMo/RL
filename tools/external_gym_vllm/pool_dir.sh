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

# Helpers for persistent pool directories (EXTERNAL_VLLM_POOL_DIR), shared by
# run_in_allocation.sh in serve mode, attach_and_run.sh and pool_config.sh.
# Source this file; it only defines functions.

# Slurm job states in which a pool job holds, or is waiting to hold, its nodes.
_EXTERNAL_VLLM_LIVE_JOB_STATES=" PENDING CONFIGURING RUNNING COMPLETING REQUEUED REQUEUE_FED REQUEUE_HOLD RESIZING SIGNALING STAGE_OUT STOPPED SUSPENDED "

# external_vllm_job_state JOB: print JOB's Slurm state, or GONE once Slurm no
# longer knows the job. Returns 1 and prints nothing when squeue itself fails,
# so a transient slurmctld error is never mistaken for a finished job.
external_vllm_job_state() {
  local job="$1" output status=0
  output=$(squeue -h -j "${job}" -o %T 2>&1) || status=$?
  if (( status == 0 )); then
    # stderr is merged in, and squeue can print an error there (e.g. about
    # slurm.conf) yet succeed; the state is the line that is a bare state name.
    output=$(grep -m 1 -xE '[A-Z_]+' <<< "${output}" || true)
    printf '%s\n' "${output:-GONE}"
    return 0
  fi
  # squeue exits non-zero for a job id slurmctld has already purged.
  if [[ "${output}" == *"Invalid job id"* ]]; then
    printf 'GONE\n'
    return 0
  fi
  return 1
}

# external_vllm_job_is_live STATE: whether STATE (from external_vllm_job_state)
# means the job holds or will hold its nodes.
external_vllm_job_is_live() {
  [[ "${_EXTERNAL_VLLM_LIVE_JOB_STATES}" == *" $1 "* ]]
}

# external_vllm_pool_dir_live_job DIR: print the id of a live pool job that owns
# DIR or was submitted for it, or nothing. Returns 1 when Slurm cannot be queried.
external_vllm_pool_dir_live_job() {
  local dir="$1" candidate state
  local -a candidates=(
    "$(cat "${dir}/owner" 2>/dev/null || true)"
    "$(cat "${dir}/submitted_job" 2>/dev/null || true)"
  )
  for candidate in "${candidates[@]}"; do
    # owner holds a generation, <job-id>-<restart-count>; keep the job id.
    candidate="${candidate%%-*}"
    [[ -n "${candidate}" ]] || continue
    state=$(external_vllm_job_state "${candidate}") || return 1
    if external_vllm_job_is_live "${state}"; then
      printf '%s\n' "${candidate}"
      return 0
    fi
  done
  return 0
}

# external_vllm_file_age_s FILE: seconds since FILE was last modified, or
# nothing when FILE does not exist.
external_vllm_file_age_s() {
  local mtime
  mtime=$(stat -c %Y "$1" 2>/dev/null) || return 0
  printf '%s\n' "$(( $(date +%s) - mtime ))"
}

# external_vllm_slurm_end_epoch JOB: JOB's scheduled end as seconds since the
# epoch, or nothing when Slurm reports none (e.g. no time limit).
external_vllm_slurm_end_epoch() {
  local end_time
  end_time=$(scontrol show job "$1" 2>/dev/null | grep -oE 'EndTime=[^ ]+' | head -n 1 || true)
  end_time="${end_time#EndTime=}"
  case "${end_time}" in
    ""|Unknown|None) return 0 ;;
  esac
  date -d "${end_time}" +%s 2>/dev/null || true
}
