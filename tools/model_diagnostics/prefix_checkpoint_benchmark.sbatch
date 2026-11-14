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
ulimit -c 0
export CONTAINER="${BENCH_IMAGE}"
export MOUNTS="${BENCH_MOUNTS}"
export GPUS_PER_NODE=4 BASE_LOG_DIR="${BENCH_RUN}"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
# Deliberately reuse the image venvs; do not rebuild them for the benchmark branch.
export NRL_IGNORE_VERSION_MISMATCH=1 RAY_ENABLE_UV_RUN_RUNTIME_ENV=0
export NEMO_RL_PY_EXECUTABLES_SYSTEM=1 NEMO_RL_VENV_DIR=/opt/ray_venvs
export NRL_FORCE_REBUILD_VENVS=false PYTHONUNBUFFERED=1 TQ_LOGGING_LEVEL=WARNING
export VLLM_V1_OUTPUT_PROC_CHUNK_SIZE=32
export HF_HOME=/tmp/pr5176-benchmark-hf HF_MODULES_CACHE=/tmp/pr5176-benchmark-hf-modules
export PYTHONPATH="${BENCH_SOURCE}:${BENCH_SOURCE}/3rdparty/Gym-workspace/Gym"
export PATH="$(dirname -- "${BENCH_PYTHON}"):/usr/local/bin:/usr/bin:/bin:${PATH}"
printf -v COMMAND '%q ' bash "${BENCH_SOURCE}/tools/model_diagnostics/run_prefix_checkpoint_benchmark.sh" "$@"
export COMMAND
bash "${BENCH_SOURCE}/ray.sub"
