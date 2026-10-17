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

"""CUDA device visibility for one-GPU-per-worker Ray actors.

By default the Megatron workers set ``RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES=1``,
so every rank sees every GPU on its node and selects its own with
``torch.cuda.set_device(ray.get_gpu_ids()[0])``.

Set ``NRL_NCCLEP_ONE_GPU_PER_RANK=1`` to leave that variable unset instead, so
Ray narrows ``CUDA_VISIBLE_DEVICES`` to the worker's single GPU before the actor
constructor runs. NCCL EP buffers come from ``ncclMemAlloc``, which maps them
into every GPU visible to the process; on Grace+Blackwell each extra mapped GPU
adds ~11 us to every ``munmap``, a host-side cost paid on every training step.

Do not narrow ``CUDA_VISIBLE_DEVICES`` by setting it directly: Ray treats the
value already in the worker env as the device list its assigned ids index
into, so a pre-narrowed value makes every rank whose GPU is not index 0 fail in
``get_accelerator_ids_for_accelerator_resource`` with ``IndexError``.
"""

import os

ONE_GPU_PER_RANK_ENV = "NRL_NCCLEP_ONE_GPU_PER_RANK"


def one_gpu_per_rank_enabled() -> bool:
    """Return True if ``NRL_NCCLEP_ONE_GPU_PER_RANK=1`` in the driver env."""
    return os.environ.get(ONE_GPU_PER_RANK_ENV, "0") == "1"


def single_gpu_worker_env_vars() -> dict[str, str]:
    """Env vars a one-GPU-per-worker actor returns from ``configure_worker``."""
    if one_gpu_per_rank_enabled():
        return {}
    return {"RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES": "1"}


def resolve_local_device_index(physical_gpu_id: int) -> int:
    """Map this worker's Ray-assigned physical GPU id to a CUDA device index.

    When ``CUDA_VISIBLE_DEVICES`` lists the whole node, the physical id is
    also the device index. When Ray has narrowed it to one device, that device
    is index 0, and it must be the GPU Ray assigned to this worker.

    Args:
        physical_gpu_id: Node-global GPU id, i.e. ``int(ray.get_gpu_ids()[0])``.

    Raises:
        RuntimeError: ``CUDA_VISIBLE_DEVICES`` holds a single device that is
            not ``physical_gpu_id``.
    """
    visible = [
        d.strip()
        for d in os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",")
        if d.strip()
    ]
    if len(visible) != 1:
        return physical_gpu_id
    if visible[0] != str(physical_gpu_id):
        raise RuntimeError(
            f"CUDA_VISIBLE_DEVICES={visible[0]} is not the GPU Ray assigned "
            f"this worker ({physical_gpu_id})"
        )
    return 0
