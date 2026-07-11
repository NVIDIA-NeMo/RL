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

if [[ $# -gt 1 || ( $# -eq 1 && "$1" != "--test-only" ) ]]; then
    echo "Usage: $0 [--test-only]" >&2
    exit 2
fi

REPLICATES="${CUTEDSL_BENCHMARK_REPLICATES:-3}"
WARMUP_UPDATES="${CUTEDSL_BENCHMARK_WARMUP_UPDATES:-3}"
MEASURED_UPDATES="${CUTEDSL_BENCHMARK_MEASURED_UPDATES:-20}"
PROFILE_ENABLED="${CUTEDSL_BENCHMARK_PROFILE:-1}"
readonly REPLICATES
if [[ ! "${REPLICATES}" =~ ^[0-9]+$ ]] || ((REPLICATES < 3)); then
    echo "[ERROR] CUTEDSL_BENCHMARK_REPLICATES must be an integer >= 3." >&2
    exit 1
fi

REPO_ROOT=$(git rev-parse --show-toplevel)
readonly REPO_ROOT
readonly EXPERIMENT_DIR="${REPO_ROOT}/experiments/cutedsl_qwen3_30ba3b_oci_1n4g"
readonly BENCHMARK_SCRIPT="${EXPERIMENT_DIR}/run_cutedsl_matrix.sbatch"
source "${EXPERIMENT_DIR}/lib/cluster_profile.sh"
load_cutedsl_cluster_profile
sbatch_args=()
while IFS= read -r argument; do
    sbatch_args+=("${argument}")
done <<< "${CUTEDSL_SBATCH_ARGS}"
sbatch_args+=("--time=${CUTEDSL_BENCHMARK_TIME}")
if [[ ${1-} == "--test-only" ]]; then
    sbatch_args+=("--test-only")
fi
SUBMISSION_GROUP="$(date -u +%Y%m%dT%H%M%SZ)-$$"
readonly SUBMISSION_GROUP
readonly SUBMISSION_DIR="${EXPERIMENT_DIR}/benchmark_submissions"
readonly SUBMISSION_RECORD="${SUBMISSION_DIR}/${SUBMISSION_GROUP}.jsonl"
EXPORT_PAYLOAD=$(mktemp "${TMPDIR:-/tmp}/cutedsl-benchmark-export.XXXXXX")
readonly EXPORT_PAYLOAD
trap 'rm -f "${EXPORT_PAYLOAD}"' EXIT
chmod 600 "${EXPORT_PAYLOAD}"
mkdir -p "${SUBMISSION_DIR}"

for ((replicate_index = 0; replicate_index < REPLICATES; replicate_index++)); do
    if ((replicate_index % 2 == 0)); then
        timing_order="on,off"
    else
        timing_order="off,on"
    fi

    env -0 \
        -u CUTEDSL_BENCHMARK_REPLICATES \
        -u CUTEDSL_BENCHMARK_REPLICATE \
        -u CUTEDSL_BENCHMARK_ORDER \
        -u CUTEDSL_BENCHMARK_SUBMISSION_GROUP \
        -u CUTEDSL_BENCHMARK_WARMUP_UPDATES \
        -u CUTEDSL_BENCHMARK_MEASURED_UPDATES \
        -u CUTEDSL_BENCHMARK_PROFILE \
        -u SLURM_EXPORT_ENV \
        "CUTEDSL_BENCHMARK_REPLICATE=${replicate_index}" \
        "CUTEDSL_BENCHMARK_ORDER=${timing_order}" \
        "CUTEDSL_BENCHMARK_SUBMISSION_GROUP=${SUBMISSION_GROUP}" \
        "CUTEDSL_BENCHMARK_WARMUP_UPDATES=${WARMUP_UPDATES}" \
        "CUTEDSL_BENCHMARK_MEASURED_UPDATES=${MEASURED_UPDATES}" \
        "CUTEDSL_BENCHMARK_PROFILE=${PROFILE_ENABLED}" \
        "SLURM_EXPORT_ENV=ALL" \
        > "${EXPORT_PAYLOAD}"

    job_id=$(sbatch --parsable "${sbatch_args[@]}" \
        --export-file="${EXPORT_PAYLOAD}" \
        "${BENCHMARK_SCRIPT}")
    printf '{"replicate_index":%d,"timing_order":"%s","job_id":"%s","submission_group":"%s"}\n' \
        "${replicate_index}" "${timing_order}" "${job_id}" "${SUBMISSION_GROUP}" \
        | tee -a "${SUBMISSION_RECORD}"
done

echo "[INFO] Submitted ${REPLICATES} matched paired jobs. Record: ${SUBMISSION_RECORD}"
