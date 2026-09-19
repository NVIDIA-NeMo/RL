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
"""Compatibility shim for CUDA IPC storage handles crossing torch versions.

NeMo-RL hands CUDA tensors from the training venv to the inference venvs
(sglang, TRT-LLM, vLLM) through ``torch.multiprocessing.reductions`` IPC
handles. The storage handle torch produces is a small byte string written by
``c10::cuda::CUDACachingAllocator::shareIpcHandle``::

    [SHAREABLE_HANDLE_VERSION][type byte][payload]

where ``type`` is ``'c'`` (a plain ``cudaMalloc`` block, followed by the
64-byte ``cudaIpcMemHandle_t``) or ``'e'`` (an expandable segment). The
consumer rejects any handle whose version byte is newer than its own with
``received sharable handle from a future version of torch that this version
does not know how to handle``.

torch 2.13 bumped the version byte from 2 to 3 without changing the ``'c'``
payload, so when the training venv runs torch 2.13 and an inference venv runs
torch 2.11 (sglang / TRT-LLM, whose kernels are built against 2.11) every
colocated refit dies at the first handle. Rewriting the version byte of
``'c'`` handles to the legacy value makes them readable by both versions; the
expandable-segment format did change, so ``'e'`` handles are left alone and
fail loudly as before.
"""

CUDA_IPC_MEM_HANDLE_SIZE = 64
"""``sizeof(cudaIpcMemHandle_t)``; a raw handle of exactly this size predates versioning."""

SHAREABLE_CUDA_MALLOC = ord("c")
"""Type byte for a plain ``cudaMalloc`` block (``SHAREABLE_CUDA_MALLOC`` in c10)."""

LEGACY_SHAREABLE_HANDLE_VERSION = 2
"""The version byte torch <= 2.12 writes. torch 2.13 writes 3 with an identical ``'c'`` payload."""


def normalize_cuda_ipc_handle(handle):
    """Rewrite a newer ``'c'``-type storage handle so older torch can open it.

    Args:
        handle: The ``storage_handle`` element of a ``rebuild_cuda_tensor``
            argument tuple. Anything that is not a versioned ``'c'`` handle
            from a newer torch is returned unchanged.

    Returns:
        The handle, with its version byte lowered to
        :data:`LEGACY_SHAREABLE_HANDLE_VERSION` when it is a ``'c'`` handle
        carrying a newer version byte.
    """
    if not isinstance(handle, (bytes, bytearray)):
        return handle
    if len(handle) <= CUDA_IPC_MEM_HANDLE_SIZE:
        # A bare 64-byte cudaIpcMemHandle_t (pre-versioning torch) or something
        # too short to carry a header; torch handles both itself.
        return handle
    if handle[1] != SHAREABLE_CUDA_MALLOC:
        return handle
    if handle[0] <= LEGACY_SHAREABLE_HANDLE_VERSION:
        return handle
    return bytes([LEGACY_SHAREABLE_HANDLE_VERSION]) + bytes(handle[1:])
