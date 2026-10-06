#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Build a NeMo-RL container matching THIS checkout (RL PR #4200 head + Gym PR #3368
# head) on a CPU node with Enroot, then export it to lustre.
#   CODE_DIR=$PWD sbatch --export=ALL,CODE_DIR=$PWD --account=nemotron_n4_omni --partition=cpu \
#     --cpus-per-task=64 --mem=700G --time=12:00:00 scripts/agentic_vbench/container/build_container.sbatch
set -euo pipefail
# Slurm copies the batch script into its spool directory, so the checkout must be
# passed explicitly (CODE_DIR) or taken from the submission directory.
code_dir=$(cd "${CODE_DIR:-$SLURM_SUBMIT_DIR}" && pwd)
[[ -f $code_dir/pyproject.toml && -d $code_dir/nemo_rl ]] || { echo "CODE_DIR is not a NeMo-RL checkout: $code_dir" >&2; exit 2; }
base_image=${BASE_IMAGE:-nvcr.io#nvidia/cuda-dl-base:26.05-cuda13.2-devel-ubuntu24.04}
output=${OUTPUT_SQSH:-/lustre/fsw/portfolios/nemotron/users/arushig/images/nemo-rl-cc-pr4200.sqsh}
name=nrlcc-$$
# The node-local /raid enroot data path is quota-limited (the dependency build
# writes ~100 GB). Keep the rootfs, cache and temp files on the node's tmpfs; the
# job's memory request must cover them (--mem=700G or more).
scratch=${BUILD_SCRATCH:-/tmp/$USER/nrlcc-$$}
export ENROOT_DATA_PATH=$scratch/data
export ENROOT_CACHE_PATH=$scratch/cache
export ENROOT_TEMP_PATH=$scratch/tmp
export ENROOT_RUNTIME_PATH=$scratch/runtime
# With root remapping Enroot would mount the (quota-limited) user home at /root,
# which is where uv keeps its cache and the image keeps its hardlinked wheels.
export ENROOT_MOUNT_HOME=no
work=$scratch/work
mkdir -p "$ENROOT_DATA_PATH" "$ENROOT_CACHE_PATH" "$ENROOT_TEMP_PATH" "$ENROOT_RUNTIME_PATH" "$work" "$(dirname "$output")"
trap 'enroot remove -f "$name" 2>/dev/null || true; rm -rf "$scratch"' EXIT
df -h "$scratch" | tail -1
echo "host=$(hostname) cpus=$(nproc) code_dir=$code_dir output=$output"
echo "rl_commit=$(git -C "$code_dir" rev-parse HEAD) gym_commit=$(git -C "$code_dir/3rdparty/Gym-workspace/Gym" rev-parse HEAD)"
enroot import --output "$work/base.sqsh" "docker://$base_image"
enroot create --name "$name" "$work/base.sqsh"
rm -f "$work/base.sqsh"
enroot start --root --rw \
  --mount "$code_dir:/mnt/nemo-rl-src:rbind,ro,x-create=dir" \
  --env SRC_DIR=/mnt/nemo-rl-src \
  --env MAX_JOBS="${MAX_JOBS:-24}" --env NVTE_BUILD_MAX_JOBS="${NVTE_BUILD_MAX_JOBS:-16}" \
  "$name" bash /mnt/nemo-rl-src/scripts/agentic_vbench/container/build_inside.sh
tmp_output="$output.partial"
rm -f "$tmp_output"
ENROOT_SQUASH_OPTIONS='-noI -noD -noF -noX -no-duplicates -mem 2G' ENROOT_MAX_PROCESSORS=16 \
  enroot export --output "$tmp_output" "$name"
mv "$tmp_output" "$output"
echo "verifying venvs (CPU node: CUDA-bound packages are checked by import spec only)"
enroot start --root --rw --env NRL_IGNORE_VERSION_MISMATCH=1 "$name" bash -lc '
  set -e; cd /opt/nemo-rl
  ls /opt/ray_venvs
  /opt/nemo_rl_venv/bin/python -c "import ray, nemo_rl; print(\"driver venv ok\", ray.__version__)"
  /opt/ray_venvs/nemo_rl.models.generation.vllm.vllm_worker_async.VllmAsyncGenerationWorker/bin/python -c "import importlib.util as u, importlib.metadata as m; assert u.find_spec(\"vllm\"); print(\"vllm venv ok\", m.version(\"vllm\"))"
  /opt/ray_venvs/nemo_rl.models.policy.workers.megatron_policy_worker.MegatronPolicyWorker/bin/python -c "import importlib.util as u, importlib.metadata as m; assert u.find_spec(\"megatron.core\") and u.find_spec(\"transformer_engine\") and u.find_spec(\"megatron.bridge\"); print(\"mcore venv ok TE\", m.version(\"transformer-engine\"))"
  /opt/ray_venvs/nemo_rl.environments.nemo_gym.NemoGym/bin/python -c "import nemo_gym; print(\"gym venv ok\")"
  cat /opt/nemo_rl_container_fingerprint' || echo "WARNING: post-export verification reported a problem; the image was exported anyway"
ls -la "$output"
echo "CONTAINER_BUILD_OK $output"
