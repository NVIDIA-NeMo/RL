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

readonly SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
readonly SYSTEM_CONFIG="${SCRIPT_DIR}/config_GB300_64x4_t16g48_tp4pp2ep32gtp8.sh"

usage() {
    cat <<'EOF'
Usage: submit_rcp.sh --gbs {256|512|768|1024} [options]

Submit independent Qwen 3.5 RCP replicas with the qualified baked recipe.

Required:
  --gbs N                 Global batch size (256, 512, 768, or 1024).

Study options:
  --val-start N           First training step at which validation runs.
  --max-steps N           Maximum number of training steps.
  --lr VALUE              Adam learning rate and minimum learning rate.
  --clip VALUE            Maximum gradient norm.
  --target VALUE          MLPerf pass@4 target. Pass "" to disable target
                          stopping and run until max steps or wall time.
  --replicas N            Number of independent Slurm jobs (default: 1).
  --seed-base N           Seed for replica 1; later replicas increment it.

Submission options:
  --name PREFIX           Job/experiment prefix.
  --time MINUTES          Slurm and driver wall time.
  --partition NAME        Slurm partition.
  --account NAME          Slurm account.
  --data-config PATH      External site data config. Required unless supplied
                          through RCP_DATA_CONFIG.
  --dry-run               Print submissions without calling sbatch.
  -h, --help              Show this help.

CONT must name the image or SquashFS. LOGDIR and Slurm SBATCH_* variables may
be supplied through the environment. EXTRA_ARGS remains available as an
explicit development/testing escape hatch; production studies should use the
named options above.
EOF
}

die() {
    echo "ERROR: $*" >&2
    exit 2
}

require_option_value() {
    local option="$1"
    local count="$2"
    ((count >= 2)) || die "${option} requires a value"
}

is_positive_integer() {
    [[ "$1" =~ ^[1-9][0-9]*$ ]]
}

is_nonnegative_integer() {
    [[ "$1" =~ ^[0-9]+$ ]]
}

is_number() {
    [[ "$1" =~ ^[+]?[0-9]+([.][0-9]*)?([eE][-+]?[0-9]+)?$ ]]
}

gbs=""
replicas=1
seed_base="${SEED_BASE:-}"
val_start=""
max_steps=""
learning_rate=""
max_grad_norm=""
target=""
target_is_set=0
name_prefix=""
walltime=""
partition=""
account=""
data_config="${RCP_DATA_CONFIG:-}"
dry_run=0

while (($# > 0)); do
    case "$1" in
        --gbs)
            require_option_value "$1" "$#"
            gbs="$2"
            shift 2
            ;;
        --gbs=*)
            gbs="${1#*=}"
            shift
            ;;
        --val-start)
            require_option_value "$1" "$#"
            val_start="$2"
            shift 2
            ;;
        --val-start=*)
            val_start="${1#*=}"
            shift
            ;;
        --max-steps)
            require_option_value "$1" "$#"
            max_steps="$2"
            shift 2
            ;;
        --max-steps=*)
            max_steps="${1#*=}"
            shift
            ;;
        --lr)
            require_option_value "$1" "$#"
            learning_rate="$2"
            shift 2
            ;;
        --lr=*)
            learning_rate="${1#*=}"
            shift
            ;;
        --clip)
            require_option_value "$1" "$#"
            max_grad_norm="$2"
            shift 2
            ;;
        --clip=*)
            max_grad_norm="${1#*=}"
            shift
            ;;
        --target)
            require_option_value "$1" "$#"
            target="$2"
            target_is_set=1
            shift 2
            ;;
        --target=*)
            target="${1#*=}"
            target_is_set=1
            shift
            ;;
        --replicas)
            require_option_value "$1" "$#"
            replicas="$2"
            shift 2
            ;;
        --replicas=*)
            replicas="${1#*=}"
            shift
            ;;
        --seed-base)
            require_option_value "$1" "$#"
            seed_base="$2"
            shift 2
            ;;
        --seed-base=*)
            seed_base="${1#*=}"
            shift
            ;;
        --name)
            require_option_value "$1" "$#"
            name_prefix="$2"
            shift 2
            ;;
        --name=*)
            name_prefix="${1#*=}"
            shift
            ;;
        --time)
            require_option_value "$1" "$#"
            walltime="$2"
            shift 2
            ;;
        --time=*)
            walltime="${1#*=}"
            shift
            ;;
        --partition)
            require_option_value "$1" "$#"
            partition="$2"
            shift 2
            ;;
        --partition=*)
            partition="${1#*=}"
            shift
            ;;
        --account)
            require_option_value "$1" "$#"
            account="$2"
            shift 2
            ;;
        --account=*)
            account="${1#*=}"
            shift
            ;;
        --data-config)
            require_option_value "$1" "$#"
            data_config="$2"
            shift 2
            ;;
        --data-config=*)
            data_config="${1#*=}"
            shift
            ;;
        --dry-run)
            dry_run=1
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            die "unknown option: $1"
            ;;
    esac
done

case "${gbs}" in
    256|512|768|1024) ;;
    "") die "--gbs is required" ;;
    *) die "unsupported --gbs ${gbs}; choose 256, 512, 768, or 1024" ;;
esac
is_positive_integer "${replicas}" || die "--replicas must be a positive integer"
[[ -z "${val_start}" ]] || is_nonnegative_integer "${val_start}" || die "--val-start must be a nonnegative integer"
[[ -z "${max_steps}" ]] || is_positive_integer "${max_steps}" || die "--max-steps must be a positive integer"
[[ -z "${learning_rate}" ]] || is_number "${learning_rate}" || die "--lr must be a nonnegative number"
[[ -z "${max_grad_norm}" ]] || is_number "${max_grad_norm}" || die "--clip must be a nonnegative number"
if ((target_is_set)) && [[ -n "${target}" ]]; then
    is_number "${target}" || die "--target must be a number or an empty string"
fi
[[ -z "${walltime}" ]] || is_positive_integer "${walltime}" || die "--time must be a positive integer"
[[ -z "${seed_base}" ]] || is_nonnegative_integer "${seed_base}" || die "--seed-base must be a nonnegative integer"

[[ -n "${data_config}" ]] || die "pass --data-config PATH or set RCP_DATA_CONFIG"
[[ -f "${data_config}" ]] || die "data config does not exist: ${data_config}"

# shellcheck source=/dev/null
source "${data_config}"

export RECIPE="qwen_35/configs/grpo_qwen35_397b_swe_openhands_async_gbs${gbs}.yaml"
[[ -z "${val_start}" ]] || export VAL_START_AT="${val_start}"
[[ -z "${max_steps}" ]] || export MAX_STEPS="${max_steps}"
[[ -z "${learning_rate}" ]] || export LEARNING_RATE="${learning_rate}"
[[ -z "${max_grad_norm}" ]] || export MAX_GRAD_NORM="${max_grad_norm}"
if ((target_is_set)); then
    export MLPERF_TARGET_ACCURACY="${target:-null}"
fi
if [[ -n "${walltime}" ]]; then
    export WALLTIME="${walltime}"
    export WALLTIME_RUNANDTIME="${walltime}"
fi
[[ -z "${partition}" ]] || export SBATCH_PARTITION="${partition}"
[[ -z "${account}" ]] || export SBATCH_ACCOUNT="${account}"

# The production profile disables source/runtime overlays and supplies the
# fixed 16-policy/48-generation topology.
# shellcheck source=/dev/null
source "${SYSTEM_CONFIG}"

: "${CONT:?CONT must name the image or SquashFS to run}"
export LOGDIR="${LOGDIR:-${PWD}/results}"
export NEXP=1

if [[ -z "${seed_base}" ]]; then
    seed_base="$(od -An -N4 -tu4 /dev/urandom | tr -d ' ')"
fi
if [[ -z "${name_prefix}" ]]; then
    name_prefix="${SBATCH_ACCOUNT:+${SBATCH_ACCOUNT}-}grpo.rcp-gbs${gbs}"
fi

declare -a common_sbatch_args
common_sbatch_args=(--export=ALL -N "${DGXNNODES}" --time="${WALLTIME}")
[[ -z "${SBATCH_SEGMENT:-}" ]] || common_sbatch_args+=(--segment="${SBATCH_SEGMENT}")
common_sbatch_args+=(--gres="${SBATCH_GRES:-gpu:${DGXNGPU}}")
[[ -z "${SBATCH_PARTITION:-}" ]] || common_sbatch_args+=(--partition="${SBATCH_PARTITION}")
[[ -z "${SBATCH_ACCOUNT:-}" ]] || common_sbatch_args+=(--account="${SBATCH_ACCOUNT}")

echo "RCP recipe: ${RECIPE}"
echo "Target: ${MLPERF_TARGET_ACCURACY} (null means disabled)"
echo "Replica seed base: ${seed_base}"
if [[ -n "${EXTRA_ARGS:-}" ]]; then
    echo "Development EXTRA_ARGS: ${EXTRA_ARGS}"
fi

for ((replica = 1; replica <= replicas; replica++)); do
    replica_seed=$((seed_base + replica - 1))
    job_name="${name_prefix}-r${replica}"
    declare -a sbatch_args=("${common_sbatch_args[@]}" --job-name="${job_name}")

    if ((dry_run)); then
        printf 'SEED_BASE=%q EXP_NAME=%q sbatch ' "${replica_seed}" "${job_name}"
        printf '%q ' "${sbatch_args[@]}" "${SCRIPT_DIR}/run.sub"
        printf '\n'
        continue
    fi

    (
        export SEED_BASE="${replica_seed}"
        export EXP_NAME="${job_name}"
        sbatch "${sbatch_args[@]}" "${SCRIPT_DIR}/run.sub"
    )
done
