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

REPO_ROOT=$(git rev-parse --show-toplevel)
readonly REPO_ROOT
readonly EXPERIMENT_DIR="${REPO_ROOT}/experiments/cutedsl_qwen3_30ba3b_oci_1n4g"
source "${EXPERIMENT_DIR}/lib/cluster_profile.sh"
capture_cutedsl_submission_source "${REPO_ROOT}"
load_cutedsl_cluster_profile
sbatch_args=()
while IFS= read -r argument; do
    sbatch_args+=("${argument}")
done <<< "${CUTEDSL_SBATCH_ARGS}"
sbatch_args+=(
    "--job-name=${CUTEDSL_ACCOUNT}-cutedsl.func"
    "--time=${CUTEDSL_FUNCTIONAL_TIME}"
    "--export=ALL"
)
if [[ ${1-} == "--test-only" ]]; then
    sbatch_args+=("--test-only")
fi
sbatch "${sbatch_args[@]}" "${EXPERIMENT_DIR}/run_cutedsl_functional.sbatch"
