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

readonly CUTEDSL_EXPERIMENT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
readonly CUTEDSL_CLUSTER_PROFILE_DIR="${CUTEDSL_EXPERIMENT_DIR}/cluster_profiles"
readonly CUTEDSL_REQUIRED_GIT_BRANCH="sna/nemo-2606-cutedsl-20260710"

capture_cutedsl_submission_source() {
    local repo_root="${1:?capture_cutedsl_submission_source requires a repository root}"
    CUTEDSL_SUBMISSION_GIT_BRANCH=$(git -C "${repo_root}" branch --show-current)
    CUTEDSL_SUBMISSION_GIT_SHA=$(git -C "${repo_root}" rev-parse HEAD)
    if [[ "${CUTEDSL_SUBMISSION_GIT_BRANCH}" != "${CUTEDSL_REQUIRED_GIT_BRANCH}" ]]; then
        echo "[ERROR] Submission source branch ${CUTEDSL_SUBMISSION_GIT_BRANCH} is not required branch ${CUTEDSL_REQUIRED_GIT_BRANCH}." >&2
        return 1
    fi
    export CUTEDSL_SUBMISSION_GIT_BRANCH CUTEDSL_SUBMISSION_GIT_SHA
}

validate_cutedsl_runtime_source() {
    local runtime_branch="${1:?validate_cutedsl_runtime_source requires a runtime branch}"
    local runtime_sha="${2:?validate_cutedsl_runtime_source requires a runtime SHA}"
    local submission_branch="${CUTEDSL_SUBMISSION_GIT_BRANCH:?missing submission source branch}"
    local submission_sha="${CUTEDSL_SUBMISSION_GIT_SHA:?missing submission source SHA}"

    if [[ "${submission_branch}" != "${CUTEDSL_REQUIRED_GIT_BRANCH}" ]]; then
        echo "[ERROR] Submission source branch ${submission_branch} is not required branch ${CUTEDSL_REQUIRED_GIT_BRANCH}." >&2
        return 1
    fi
    if [[ "${runtime_branch}" != "${CUTEDSL_REQUIRED_GIT_BRANCH}" ]]; then
        echo "[ERROR] Runtime source branch ${runtime_branch} is not required branch ${CUTEDSL_REQUIRED_GIT_BRANCH}." >&2
        return 1
    fi
    if [[ "${runtime_branch}" != "${submission_branch}" || "${runtime_sha}" != "${submission_sha}" ]]; then
        echo "[ERROR] Runtime source ${runtime_branch}@${runtime_sha} differs from submission source ${submission_branch}@${submission_sha}." >&2
        return 1
    fi
}

load_cutedsl_cluster_profile() {
    local profile_name="${CUTEDSL_CLUSTER_PROFILE:?set CUTEDSL_CLUSTER_PROFILE to pre_tyche, aws_dfw, or lyris}"
    local profile_path
    case "${profile_name}" in
        pre_tyche|aws_dfw|lyris)
            profile_path="${CUTEDSL_CLUSTER_PROFILE_DIR}/${profile_name}.sh"
            ;;
        *)
            echo "[ERROR] Unknown CUTEDSL_CLUSTER_PROFILE: ${profile_name}" >&2
            return 1
            ;;
    esac

    unset CUTEDSL_PROFILE_NAME CUTEDSL_ACCOUNT CUTEDSL_PARTITION CUTEDSL_GRES
    unset CUTEDSL_SEGMENT CUTEDSL_COMMENT CUTEDSL_IMAGE CUTEDSL_IMAGE_SHA256
    unset CUTEDSL_FUNCTIONAL_TIME CUTEDSL_BENCHMARK_TIME CUTEDSL_SBATCH_ARGS
    source "${profile_path}"

    if [[ "${CUTEDSL_PROFILE_NAME-}" != "${profile_name}" ]]; then
        echo "[ERROR] Profile name does not match CUTEDSL_CLUSTER_PROFILE." >&2
        return 1
    fi
    if [[ -z "${CUTEDSL_ACCOUNT-}" || -z "${CUTEDSL_PARTITION-}" || -z "${CUTEDSL_COMMENT-}" ]]; then
        echo "[ERROR] Profile account, partition, and comment must be non-empty." >&2
        return 1
    fi
    if [[ "${CUTEDSL_IMAGE-}" != /* ]]; then
        echo "[ERROR] CUTEDSL_IMAGE must be an absolute path." >&2
        return 1
    fi
    if [[ ! "${CUTEDSL_IMAGE_SHA256-}" =~ ^[[:xdigit:]]{64}$ ]]; then
        echo "[ERROR] CUTEDSL_IMAGE_SHA256 must be a 64-character SHA256 value." >&2
        return 1
    fi
    if [[ ! "${CUTEDSL_FUNCTIONAL_TIME-}" =~ ^[0-9]{2}:[0-9]{2}:[0-9]{2}$ ]] || \
        [[ ! "${CUTEDSL_BENCHMARK_TIME-}" =~ ^[0-9]{2}:[0-9]{2}:[0-9]{2}$ ]]; then
        echo "[ERROR] Profile walltimes must use HH:MM:SS." >&2
        return 1
    fi
    if [[ -n "${CUTEDSL_SEGMENT-}" && "${CUTEDSL_SEGMENT}" != "1" ]]; then
        echo "[ERROR] Unsupported CUTEDSL segment: ${CUTEDSL_SEGMENT}" >&2
        return 1
    fi
    if [[ -n "${CUTEDSL_GRES-}" && "${CUTEDSL_GRES}" != "gpu:4" ]]; then
        echo "[ERROR] Unsupported CUTEDSL GRES: ${CUTEDSL_GRES}" >&2
        return 1
    fi
    if [[ -n "${CUTEDSL_GRES-}" && -n "${CUTEDSL_SEGMENT-}" ]]; then
        echo "[ERROR] A profile cannot configure both GRES and segment." >&2
        return 1
    fi
    if [[ -z "${CUTEDSL_GRES-}" && -z "${CUTEDSL_SEGMENT-}" ]]; then
        echo "[ERROR] A profile must configure either GRES or segment." >&2
        return 1
    fi

    local -a sbatch_args=(
        "--account=${CUTEDSL_ACCOUNT}"
        "--partition=${CUTEDSL_PARTITION}"
        "--comment=${CUTEDSL_COMMENT}"
    )
    if [[ -n "${CUTEDSL_GRES}" ]]; then
        sbatch_args+=("--gres=${CUTEDSL_GRES}")
    else
        sbatch_args+=("--segment=${CUTEDSL_SEGMENT}")
    fi
    CUTEDSL_SBATCH_ARGS=$(printf '%s\n' "${sbatch_args[@]}")
    export CUTEDSL_CLUSTER_PROFILE CUTEDSL_PROFILE_NAME CUTEDSL_ACCOUNT
    export CUTEDSL_PARTITION CUTEDSL_GRES CUTEDSL_SEGMENT CUTEDSL_COMMENT
    export CUTEDSL_IMAGE CUTEDSL_IMAGE_SHA256 CUTEDSL_FUNCTIONAL_TIME
    export CUTEDSL_BENCHMARK_TIME CUTEDSL_SBATCH_ARGS
}
