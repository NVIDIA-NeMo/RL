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
"""Why can't a torch tensor be fabric-exported? Ask CUDA, don't infer.

Register mode over NVLink failed under both torch allocators. The default
caching allocator's failure has an observed mechanism -- mooncake logs
``Memory region ... is not allocated by cuMemCreate`` and publishes nothing.
Under ``expandable_segments`` it failed too, but nothing was logged: no
warning, no export failure. The reason was *inferred* ("torch never requests a
FABRIC handle type") and never checked.

This runs mooncake's own registration sequence directly against a torch
tensor, reading each CUresult:

    cuMemRetainAllocationHandle(&h, ptr)
    cuMemGetAddressRange(&base, &size, ptr)
    cuMemExportToShareableHandle(&out, h, CU_MEM_HANDLE_TYPE_FABRIC, 0)

and compares against a control allocated the way mooncake allocates its own
buffers (``cuMemCreate`` with ``requestedHandleTypes = FABRIC``). Whichever
call returns non-zero is the answer, and the CUresult names it.
"""

from __future__ import annotations

import ctypes
import os

# CUresult values that actually come up here.
_CUDA_ERRORS = {
    0: "CUDA_SUCCESS",
    1: "CUDA_ERROR_INVALID_VALUE",
    200: "CUDA_ERROR_INVALID_IMAGE",
    201: "CUDA_ERROR_INVALID_CONTEXT",
    400: "CUDA_ERROR_INVALID_HANDLE",
    401: "CUDA_ERROR_ILLEGAL_STATE",
    500: "CUDA_ERROR_NOT_FOUND",
    801: "CUDA_ERROR_NOT_SUPPORTED",
}
_CU_MEM_HANDLE_TYPE_POSIX_FD = 1
_CU_MEM_HANDLE_TYPE_FABRIC = 8
_CU_MEM_ALLOCATION_TYPE_PINNED = 1
_CU_MEM_LOCATION_TYPE_DEVICE = 1


def _rc(code: int) -> str:
    return f"{code} ({_CUDA_ERRORS.get(code, 'unknown')})"


class _CUmemAllocationProp(ctypes.Structure):
    """Matches CUmemAllocationProp_v1; win/allocFlags padded to the real size."""

    _fields_ = [
        ("type", ctypes.c_int),
        ("requestedHandleTypes", ctypes.c_int),
        ("location_type", ctypes.c_int),
        ("location_id", ctypes.c_int),
        ("win32HandleMetaData", ctypes.c_void_p),
        ("allocFlags", ctypes.c_ubyte * 32),
    ]


def _fabric_handle_buf() -> ctypes.Array:
    return (ctypes.c_ubyte * 64)()  # CUmemFabricHandle is 64 bytes


def probe_pointer(cuda, label: str, ptr: int, nbytes: int) -> None:
    """Run mooncake's fabric-registration sequence against one allocation."""
    print(f"\n--- {label} (ptr=0x{ptr:x}, {nbytes} bytes) ---")

    handle = ctypes.c_ulonglong(0)
    rc = cuda.cuMemRetainAllocationHandle(ctypes.byref(handle), ctypes.c_void_p(ptr))
    print(f"  cuMemRetainAllocationHandle -> {_rc(rc)}")
    if rc != 0:
        print("    => not cuMemCreate memory. Mooncake logs its warning here and")
        print("       returns 0 WITHOUT publishing, which is the proven failure.")
        return

    # cuMemGetAddressRange is a VERSIONED symbol: the CUDA headers macro it to
    # cuMemGetAddressRange_v2, so that is what mooncake actually calls. Looking
    # up the bare name via ctypes gets the legacy entry point, which returns
    # CUDA_ERROR_INVALID_CONTEXT no matter how the context is set up -- that
    # cost one run's worth of a misread measurement.
    base = ctypes.c_ulonglong(0)
    size = ctypes.c_size_t(0)
    fn = getattr(cuda, "cuMemGetAddressRange_v2", None) or cuda.cuMemGetAddressRange
    rc = fn(ctypes.byref(base), ctypes.byref(size), ctypes.c_ulonglong(ptr))
    print(f"  (symbol: {'cuMemGetAddressRange_v2' if hasattr(cuda, 'cuMemGetAddressRange_v2') else 'cuMemGetAddressRange'})")
    print(f"  cuMemGetAddressRange        -> {_rc(rc)}  "
          f"base=0x{base.value:x} size={size.value}")
    if rc == 0:
        pub_end = base.value + size.value
        covers = base.value <= ptr and ptr + nbytes <= pub_end
        print(f"    mooncake would publish [0x{base.value:x}, 0x{pub_end:x}) "
              f"= {size.value} bytes")
        print(f"    tensor spans           [0x{ptr:x}, 0x{ptr + nbytes:x}) "
              f"= {nbytes} bytes")
        print(f"    published range covers the whole tensor: {covers}")
        if not covers:
            print("    => THIS is the failure: the reader requires")
            print("       entry.addr <= dest && dest+len <= entry.addr+entry.length,")
            print("       so any offset past the published chunk is 'not found!'.")

    # The call whose result decides everything: mooncake publishes the handle
    # it returns, and publishes nothing useful if it fails.
    out = _fabric_handle_buf()
    rc_fab = cuda.cuMemExportToShareableHandle(
        ctypes.byref(out), handle, ctypes.c_int(_CU_MEM_HANDLE_TYPE_FABRIC),
        ctypes.c_ulonglong(0),
    )
    print(f"  export(FABRIC)              -> {_rc(rc_fab)}")

    # Every allocation probed here now terminates in one of two states, so the
    # POSIX_FD contrast and its branches were unreachable -- and one of them
    # printed "Confirms the inference" for an inference this tool disproved.
    print("    => fabric-exportable." if rc_fab == 0 else "    => NOT fabric-exportable.")


def mooncake_style_alloc(cuda, nbytes: int):
    """Allocate the way NvlinkTransport::allocatePinnedLocalMemory does."""
    dev = ctypes.c_int(0)
    cuda.cuDeviceGet(ctypes.byref(dev), 0)
    prop = _CUmemAllocationProp()
    prop.type = _CU_MEM_ALLOCATION_TYPE_PINNED
    prop.requestedHandleTypes = _CU_MEM_HANDLE_TYPE_FABRIC
    prop.location_type = _CU_MEM_LOCATION_TYPE_DEVICE
    prop.location_id = dev.value

    gran = ctypes.c_size_t(0)
    rc = cuda.cuMemGetAllocationGranularity(
        ctypes.byref(gran), ctypes.byref(prop), ctypes.c_int(1)
    )
    if rc != 0 or not gran.value:
        print(f"  granularity failed -> {_rc(rc)}")
        return None, 0
    size = ((nbytes + gran.value - 1) // gran.value) * gran.value

    handle = ctypes.c_ulonglong(0)
    rc = cuda.cuMemCreate(ctypes.byref(handle), ctypes.c_size_t(size),
                          ctypes.byref(prop), ctypes.c_ulonglong(0))
    if rc != 0:
        print(f"  cuMemCreate(FABRIC) failed -> {_rc(rc)}")
        return None, 0
    ptr = ctypes.c_void_p(0)
    cuda.cuMemAddressReserve(ctypes.byref(ptr), ctypes.c_size_t(size),
                             ctypes.c_size_t(0), ctypes.c_void_p(0),
                             ctypes.c_ulonglong(0))
    cuda.cuMemMap(ptr, ctypes.c_size_t(size), ctypes.c_size_t(0), handle,
                  ctypes.c_ulonglong(0))
    return ptr.value, size


def main() -> None:
    import torch

    cuda = ctypes.CDLL("libcuda.so.1")
    cuda.cuInit(0)
    torch.cuda.init()
    dev = ctypes.c_int(0)
    cuda.cuDeviceGet(ctypes.byref(dev), 0)
    ctx = ctypes.c_void_p(0)
    rc = cuda.cuDevicePrimaryCtxRetain(ctypes.byref(ctx), dev)
    rc |= cuda.cuCtxSetCurrent(ctx)
    print(f"primary context current: rc={rc}")
    conf = os.environ.get("PYTORCH_CUDA_ALLOC_CONF", "(unset)")
    print(f"PYTORCH_CUDA_ALLOC_CONF={conf}")
    print(f"device={torch.cuda.get_device_name(0)}")

    t = torch.empty(8 * 1024 * 1024, dtype=torch.uint8, device="cuda")
    storage = t.untyped_storage()
    probe_pointer(cuda, f"torch tensor [{conf}]", storage.data_ptr(), storage.nbytes())

    ptr, size = mooncake_style_alloc(cuda, 8 * 1024 * 1024)
    if ptr:
        probe_pointer(cuda, "mooncake-style cuMemCreate(FABRIC)", ptr, size)


if __name__ == "__main__":
    main()
