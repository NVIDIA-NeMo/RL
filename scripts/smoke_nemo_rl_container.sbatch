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

#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=8
#SBATCH --time=00:10:00
#SBATCH --job-name=smoke_nemo_rl_container
#SBATCH --output=smoke_nemo_rl_container_%j.log

set -euo pipefail

: "${CONTAINER_IMAGE:?Set CONTAINER_IMAGE to an immutable squashfs path}"
: "${CONTAINER_SMOKE_DIR:?Set CONTAINER_SMOKE_DIR to an absolute host result directory}"
: "${SLURM_SUBMIT_DIR:?Submit this job from the NeMo-RL checkout root}"
if [[ "${CONTAINER_SMOKE_DIR}" != /* ]]; then
    echo "[ERROR] CONTAINER_SMOKE_DIR must be an absolute path: ${CONTAINER_SMOKE_DIR}" >&2
    exit 1
fi
mkdir -p "${CONTAINER_SMOKE_DIR}"

repo_root=$(git -C "${SLURM_SUBMIT_DIR}" rev-parse --show-toplevel)
readonly repo_root

readonly -a container_run=(
    srun
    --nodes=1
    --ntasks=1
    --gres=gpu:4
    --container-image="${CONTAINER_IMAGE}"
    --container-mounts="${repo_root}:/workspace/nemo-rl,${CONTAINER_SMOKE_DIR}:/results"
    --container-workdir=/workspace/nemo-rl
)

"${container_run[@]}" bash -s 2>&1 <<'BASH' | tee "${CONTAINER_SMOKE_DIR}/smoke.log"
set -euo pipefail

readonly python_bin="/opt/nemo_rl_venv/bin/python"
[[ "$("${python_bin}" --version)" == "Python 3.13.13" ]]

"${python_bin}" - <<'PY'
import cutlass
import torch
import transformer_engine.pytorch as te
from cutlass import cute

import nemo_rl

assert torch.cuda.is_available()
actual_devices = torch.cuda.device_count()
assert actual_devices == 4, f"expected 4 visible CUDA devices, found {actual_devices}"
device_names = [torch.cuda.get_device_name(index) for index in range(actual_devices)]
assert all("GB200" in device_name for device_name in device_names), device_names

print("IMPORT_PASS", cutlass.__file__, cute.__file__, nemo_rl.__file__, te.__file__)
for device_index in range(actual_devices):
    with torch.cuda.device(device_index):
        model = te.Linear(64, 64).cuda(device_index)
        inputs = torch.randn(
            8, 64, device=f"cuda:{device_index}", requires_grad=True
        )
        outputs = model(inputs)
        outputs.sum().backward()
        torch.cuda.synchronize()

        assert torch.isfinite(outputs).all()
        assert inputs.grad is not None and torch.isfinite(inputs.grad).all()
        for name, parameter in model.named_parameters():
            assert parameter.grad is not None and torch.isfinite(parameter.grad).all(), (
                f"missing or non-finite TE gradient {name} on CUDA device {device_index}"
            )
        print(
            "GPU_SMOKE_PASS",
            device_index,
            torch.__version__,
            torch.version.cuda,
            device_names[device_index],
        )
PY
BASH
