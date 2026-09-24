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
"""CUDA IPC storage handles produced by a newer torch must open on an older one."""

import pytest

from nemo_rl.utils.cuda_ipc import (
    CUDA_IPC_MEM_HANDLE_SIZE,
    CURRENT_SHAREABLE_HANDLE_VERSION,
    LEGACY_SHAREABLE_HANDLE_VERSION,
    normalize_cuda_ipc_handle,
)

_MEM_HANDLE = bytes(range(CUDA_IPC_MEM_HANDLE_SIZE))


def _versioned(version: int, kind: bytes, payload: bytes = _MEM_HANDLE) -> bytes:
    return bytes([version]) + kind + payload


def test_newer_cudamalloc_handle_is_downgraded_to_the_legacy_version():
    handle = _versioned(CURRENT_SHAREABLE_HANDLE_VERSION, b"c")
    out = normalize_cuda_ipc_handle(handle)
    assert out[0] == LEGACY_SHAREABLE_HANDLE_VERSION
    # Only the version byte changes; type byte and cudaIpcMemHandle_t are intact.
    assert out[1:] == handle[1:]
    assert isinstance(out, bytes)


@pytest.mark.parametrize("version", [1, LEGACY_SHAREABLE_HANDLE_VERSION])
def test_legacy_or_older_handles_pass_through(version: int):
    handle = _versioned(version, b"c")
    assert normalize_cuda_ipc_handle(handle) is handle


@pytest.mark.parametrize("version", [CURRENT_SHAREABLE_HANDLE_VERSION + 1, 255])
def test_unverified_newer_cudamalloc_version_raises(version: int):
    # Only the 2/3 'c' payloads are known to be identical; relabelling an
    # unknown version as legacy would hand the consumer a handle it misparses.
    with pytest.raises(ValueError, match=f"handle version {version}"):
        normalize_cuda_ipc_handle(_versioned(version, b"c"))


def test_unverified_newer_expandable_segment_version_still_passes_through():
    # The strictness applies to 'c' handles only; 'e' handles are never touched.
    handle = _versioned(
        CURRENT_SHAREABLE_HANDLE_VERSION + 1, b"e", payload=b"\x00" * 200
    )
    assert normalize_cuda_ipc_handle(handle) is handle


def test_expandable_segment_handles_are_left_alone():
    # The 'e' payload format did change between torch versions, so rewriting
    # its version byte would hide a real incompatibility.
    handle = _versioned(3, b"e", payload=b"\x00" * 200)
    assert normalize_cuda_ipc_handle(handle) is handle


def test_raw_pre_versioning_handle_passes_through():
    # Exactly sizeof(cudaIpcMemHandle_t): torch treats it as an unversioned handle.
    assert normalize_cuda_ipc_handle(_MEM_HANDLE) is _MEM_HANDLE


@pytest.mark.parametrize("value", [None, 7, "not-a-handle", b"", b"\x03"])
def test_non_handle_values_pass_through(value):
    assert normalize_cuda_ipc_handle(value) is value


def test_bytearray_input_is_returned_as_bytes():
    handle = bytearray(_versioned(CURRENT_SHAREABLE_HANDLE_VERSION, b"c"))
    out = normalize_cuda_ipc_handle(handle)
    assert isinstance(out, bytes)
    assert out[0] == LEGACY_SHAREABLE_HANDLE_VERSION


def test_rebuild_cuda_tensor_from_ipc_normalizes_the_storage_handle(monkeypatch):
    import nemo_rl.models.policy.utils as policy_utils

    seen = {}

    def fake_rebuild(*args):
        seen["args"] = args
        return "tensor"

    monkeypatch.setattr(policy_utils, "rebuild_cuda_tensor", fake_rebuild)
    newer = _versioned(3, b"c")
    # Mirrors what reduce_tensor(...)[1:] yields: a 1-tuple wrapping the
    # rebuild_cuda_tensor argument tuple (storage_device at 6, handle at 7).
    args = ("cls", (2,), (1,), 0, "storage_cls", "dtype", 0, newer, 128, 0, False)
    args += ("refcnt", 0, "event", False)

    assert policy_utils.rebuild_cuda_tensor_from_ipc((args,), device_id=3) == "tensor"
    assert seen["args"][6] == 3
    assert seen["args"][7][0] == LEGACY_SHAREABLE_HANDLE_VERSION
    assert seen["args"][7][1:] == newer[1:]
    # Everything else is passed through untouched.
    assert seen["args"][:6] == args[:6]
    assert seen["args"][8:] == args[8:]


def test_sglang_rebuild_patch_normalizes_the_storage_handle(monkeypatch):
    from torch.multiprocessing import reductions

    from nemo_rl.models.generation.sglang.utils import train_utils

    seen = {}

    def fake_original(*args):
        seen["args"] = args
        return "tensor"

    monkeypatch.setattr(
        reductions, "_rebuild_cuda_tensor_original", fake_original, raising=False
    )
    monkeypatch.setattr(train_utils, "_device_from_maybe_uuid", lambda d: 5)
    newer = _versioned(3, b"c")
    args = ("cls", (2,), (1,), 0, "storage_cls", "dtype", "gpu-uuid", newer, 128, 0)
    args += (False, "refcnt", 0, "event", False)

    assert train_utils._rebuild_cuda_tensor_modified(*args) == "tensor"
    assert seen["args"][6] == 5
    assert seen["args"][7][0] == LEGACY_SHAREABLE_HANDLE_VERSION
    assert seen["args"][7][1:] == newer[1:]
