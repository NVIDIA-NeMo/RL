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
# Submit a portable four-node prefix-cut proxy; see prefix_checkpoint_benchmark.md.
set -euo pipefail

BENCH_DRY_RUN=0
if [[ "${1:-}" == --dry-run ]]; then BENCH_DRY_RUN=1; shift; fi
BENCH_SOURCE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
export BENCH_SOURCE
: "${BENCH_IMAGE:?Set the container squashfs path}"
: "${BENCH_MODEL:?Set a shared Hugging Face model directory}"
: "${BENCH_RESULTS:?Set a shared results directory}"
: "${BENCH_ACCOUNT:?Set the Slurm account}"
: "${BENCH_MOUNTS:?Set container mounts covering source, model, image and results}"
export BENCH_IMAGE BENCH_MODEL BENCH_RESULTS BENCH_MOUNTS
export BENCH_PYTHON=${BENCH_PYTHON:-/opt/ray_venvs/nemo_rl.models.generation.vllm.vllm_worker_async.VllmAsyncGenerationWorker/bin/python}
test -f "${BENCH_IMAGE}"
test -f "${BENCH_MODEL}/config.json"
test -f "${BENCH_SOURCE}/3rdparty/Gym-workspace/Gym/nemo_gym/__init__.py"
if (( BENCH_DRY_RUN )); then
    BENCH_RUN="${BENCH_RESULTS}/pr5176-prefix-4n.DRYRUN"
else
    mkdir -p -- "${BENCH_RESULTS}"
    BENCH_RUN=$(mktemp -d "${BENCH_RESULTS}/pr5176-prefix-4n.XXXXXX")
fi
export BENCH_RUN
BENCH_SUBMIT=(sbatch --parsable --account="${BENCH_ACCOUNT}"
    --partition="${BENCH_PARTITION:-batch}" --nodes=4 --gpus-per-node=4 --exclusive
    --time="${BENCH_WALLTIME:-01:00:00}" --job-name=pr5176-prefix-benchmark-4n
    --chdir="${BENCH_SOURCE}" --output="${BENCH_RUN}/slurm-%j.out"
    --error="${BENCH_RUN}/slurm-%j.err")
if [[ -n "${BENCH_QOS:-}" ]]; then BENCH_SUBMIT+=(--qos="${BENCH_QOS}"); fi
BENCH_SUBMIT+=("${BENCH_SOURCE}/tools/model_diagnostics/prefix_checkpoint_benchmark.sbatch" "$@")
if (( BENCH_DRY_RUN )); then
    printf '%q ' "${BENCH_SUBMIT[@]}"
    printf '\n'
else
    "${BENCH_SUBMIT[@]}"
    printf 'Results: %s\n' "${BENCH_RUN}"
fi
