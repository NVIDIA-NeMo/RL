#!/usr/bin/env bash

# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
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

#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --time=01:00:00

set -euo pipefail

: "${SLURM_JOB_ID:?SLURM_JOB_ID is required}"

BUILDER=/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/sna/nemo-rl-hybridep-q30-triplet-20260729/scripts/experiments/x86/hybridep/build_deepep_wheel.sbatch
BUILDER_SHA256=1e7479c1b1252eb35735627e0980b0edbae228f05e86a6e6d65a50af8bb72ee9
CONTAINER=/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/sna/containers/nemo-rl-nightly-20260805/nemo_rl_nightly_20260805_15171871.sqsh
CONTAINER_SHA256=6623720fedcc82b31ab1f09f385590a5cf07751c35e5f6bf740a8b79c691b680
NCCL_WHEEL=/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/sna/experiments/hybridep-x86-b200-h100/cw-dfw/q30-triplet-main-d3f3eb5-20260729/artifacts/nccl/nvidia-nccl-cu13-2.30.4-14633389/nvidia_nccl_cu13-2.30.4-py3-none-manylinux_2_18_x86_64.whl
NCCL_WHEEL_SHA256=534dbf3058cadb625f08ab0d17f1dffad3b961a2bfa360d66633fcf21be53f57
OUTPUT_DIR=/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/sna/experiments/hybridep-padding-ab-q30/cw-h100/artifacts/deepep
DEEPEP_COMMIT=f725d29699f5bda9ba789456bb9579af69844685
DEEPEP_VARIANT=hybridep
GPU_ARCH=9.0
HYBRID_EP_TRANSPORT=doca
PYTHON_BIN=/opt/nemo_rl_venv/bin/python

[[ $(sha256sum "$BUILDER" | cut -d' ' -f1) == "$BUILDER_SHA256" ]]
[[ $(sha256sum "$CONTAINER" | cut -d' ' -f1) == "$CONTAINER_SHA256" ]]

export BUILDER CONTAINER CONTAINER_SHA256 NCCL_WHEEL NCCL_WHEEL_SHA256
export OUTPUT_DIR DEEPEP_COMMIT DEEPEP_VARIANT GPU_ARCH HYBRID_EP_TRANSPORT
export PYTHON_BIN HYBRID_EP_SCRIPT_PATH="$BUILDER"

bash "$BUILDER"

ARTIFACT_DIR="$OUTPUT_DIR/hybridep-$DEEPEP_COMMIT-sm90-$SLURM_JOB_ID"
mapfile -t WHEELS < <(find "$ARTIFACT_DIR" -maxdepth 1 -type f -name '*.whl')
[[ ${#WHEELS[@]} -eq 1 ]]
WHEEL=${WHEELS[0]}
WHEEL_SHA256=$(sha256sum "$WHEEL" | cut -d' ' -f1)

python3 - "$ARTIFACT_DIR/build-metadata.json" "$DEEPEP_COMMIT" "$WHEEL" "$WHEEL_SHA256" <<'PY'
import json
import pathlib
import sys

output, commit, wheel, wheel_sha256 = sys.argv[1:]
payload = {
    "architecture": "x86_64",
    "commit": commit,
    "platform": "linux",
    "sha256": wheel_sha256,
    "wheel": wheel,
}
pathlib.Path(output).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
PY

srun --overlap \
  --container-image="$CONTAINER" \
  --no-container-mount-home \
  --container-mounts=/lustre:/lustre \
  bash -lc '"$PYTHON_BIN" - <<"PY"
import platform
import torch

print(f"python_version={platform.python_version()}")
print(f"torch_version={torch.__version__}")
print(f"runtime_nccl_version={torch.cuda.nccl.version()}")
print(f"gpu_name={torch.cuda.get_device_name(0)}")
print(f"gpu_capability={torch.cuda.get_device_capability(0)}")
PY' > "$ARTIFACT_DIR/runtime.env"

printf 'wheel=%s\nmetadata=%s\nruntime=%s\n' \
  "$WHEEL" "$ARTIFACT_DIR/build-metadata.json" "$ARTIFACT_DIR/runtime.env"
