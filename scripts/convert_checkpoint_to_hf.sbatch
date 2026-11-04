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
#SBATCH --account=nemotron_n4_post
#SBATCH --partition=batch
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --exclusive
#SBATCH --job-name=convert-oapl-hf
# --output is passed on the sbatch command line by convert_all_checkpoints.sh (SBATCH directives can't expand variables).

set -euo pipefail

# Resolve the physical path -- /lustre/fsw/portfolios/nemotron/users/dmisra is
# a symlink into /lustre/fs1/...; the container only bind-mounts the literal paths
# given below, so using the physical path keeps everything consistent inside the
# container. REMOTE_REPO_DIR is the training checkout holding results/<run>/.
REMOTE_REPO_DIR="${REMOTE_REPO_DIR:?Set REMOTE_REPO_DIR to the checkout that holds results/<run>/checkpoints}"
PROJECT_ROOT=$(cd "$REMOTE_REPO_DIR" && pwd -P)
MEGATRON_CKPT_PATH="${MEGATRON_CKPT_PATH:?Set MEGATRON_CKPT_PATH, e.g. results/<run>/checkpoints/step_464/policy/weights/iter_0000000}"
CONFIG_PATH="${CONFIG_PATH:?Set CONFIG_PATH, e.g. results/<run>/checkpoints/step_464/config.yaml}"
HF_CKPT_PATH="${HF_CKPT_PATH:?Set HF_CKPT_PATH, e.g. results/<run>/checkpoints/step_464/hf_export}"
CONTAINER="${CONTAINER:?Set CONTAINER to the same training container image (see scripts/local.env.rc4)}"
# Megatron's distributed checkpoint format supports resharding to a different
# parallel layout at load time, so the layout here does not need to match the
# training config. It only has to fit this job's 4 GPUs (one HSG node):
# TP*PP must divide 4, and EP*ETP*PP must divide 4 too.
NPROC="${NPROC:-4}"
TP="${TP:-4}"
PP="${PP:-1}"
EP="${EP:-4}"
ETP="${ETP:-1}"
# The image's prebuilt Megatron actor venv (exactly the mcore extra, CUDA extensions
# already compiled). The converter imports only torch/megatron.bridge/megatron.core,
# so it runs straight from this venv: no `uv sync`, no checkout mount, and parallel
# conversion jobs can't race on a shared checkout. The script itself is the image's
# baked copy under /opt/nemo-rl (present since v0.7.0).
CONVERT_PYTHON="${CONVERT_PYTHON:-/opt/ray_venvs/nemo_rl.models.value.workers.megatron_value_worker.MegatronValueWorker/bin/python}"
CONVERT_SCRIPT="${CONVERT_SCRIPT:-/opt/nemo-rl/examples/converters/convert_megatron_to_hf_gpu.py}"

mkdir -p "$(dirname "$PROJECT_ROOT/$HF_CKPT_PATH")" "$PROJECT_ROOT/results/slurm-logs"

# Mount the logical (/lustre/fsw) and physical (/lustre/fs1) trees, plus /scratch/fsw
# where it exists. Mounting a path that doesn't exist on the host makes enroot-mount
# fail before the job even starts, so each one is only requested when present on the
# node running this script.
CONTAINER_MOUNTS="/lustre/fsw:/lustre/fsw"
for extra in /lustre/fs1 /scratch/fsw; do
    if [ -d "$extra" ]; then
        CONTAINER_MOUNTS="${CONTAINER_MOUNTS},${extra}:${extra}"
    fi
done

echo "started_at=$(date --iso-8601=seconds)"
echo "megatron_ckpt_path=$MEGATRON_CKPT_PATH"
echo "config_path=$CONFIG_PATH"
echo "hf_ckpt_path=$HF_CKPT_PATH"
echo "nproc=$NPROC tp=$TP pp=$PP ep=$EP etp=$ETP"
echo "project_root=$PROJECT_ROOT"

srun \
    --nodes=1 \
    --ntasks=1 \
    --gpus=$NPROC \
    --container-image="$CONTAINER" \
    --container-mounts="$CONTAINER_MOUNTS" \
    --container-workdir="$PROJECT_ROOT" \
    --no-container-mount-home \
    bash -lc "
        set -euo pipefail
        '$CONVERT_PYTHON' -m torch.distributed.run --nproc_per_node=$NPROC --nnodes=1 \
            '$CONVERT_SCRIPT' \
            --config '$CONFIG_PATH' \
            --megatron-ckpt-path '$MEGATRON_CKPT_PATH' \
            --hf-ckpt-path '$HF_CKPT_PATH' \
            --tp $TP --pp $PP --ep $EP --etp $ETP
    "

echo "finished_at=$(date --iso-8601=seconds)"
