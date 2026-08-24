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
"""Probe whether this Mooncake build can serve register mode over NVLink IPC.

Register mode publishes a buffer address at put and reads it at get. Mooncake's
``nvlink`` transport is the same shape with a different handle: it exports a
``cudaIpcMemHandle_t`` (or a ``CUmemFabricHandle`` on MNNVL fabric) instead of
an RDMA MR, and the read becomes a device copy rather than an ibv read. That
would only be a protocol change for us -- ``initialize`` already takes the
protocol from config -- but only if this wheel was built with the transport in.

Reports four facts, none of which are inferable from source:
  1. whether ``nvlink`` / ``nvlink_intra`` are compiled in (``-DUSE_MNNVL=ON``,
     ``-DUSE_INTRA_NVLINK=ON``),
  2. whether this platform reports fabric-memory support, which decides which
     of the two export paths ``registerLocalMemory`` takes,
  3. whether a plain torch CUDA tensor can be IPC-exported at all, and
  4. whether it can be exported under ``expandable_segments``, which routes
     torch through the cuMem VMM APIs that the fabric path requires.

(3) and (4) matter because the fabric branch calls
``cuMemRetainAllocationHandle`` and, when the memory did not come from
``cuMemCreate``, logs a warning and returns 0 -- success, with nothing
published. A register-mode put over that path would look fine and leave the
key unreadable.
"""

from __future__ import annotations

import ctypes
import os
import subprocess
import sys


def section(title: str) -> None:
    print(f"\n=== {title} ===", flush=True)


def probe_symbols() -> None:
    section("compiled-in transports")
    try:
        import mooncake.engine as engine_mod

        so = engine_mod.__file__
    except Exception as exc:  # noqa: BLE001
        print(f"  cannot import mooncake.engine: {exc}")
        return
    print(f"  module: {so}")
    # One scan, then substring tests. This used to re-run `strings` per needle,
    # five full passes over a large .so. USE_MNNVL is not in the needle list:
    # it is a compile-time macro, never a string in the binary, so it always
    # reported ABSENT -- a permanent false negative. probe_init() answers that
    # question unambiguously instead.
    try:
        hit = subprocess.run(["strings", "-a", so], capture_output=True,
                             text=True, timeout=120).stdout
    except Exception as exc:  # noqa: BLE001
        print(f"  strings failed: {exc}")
        return
    for needle in ("NvlinkTransport", "IntraNodeNvlinkTransport",
                   "cudaIpcGetMemHandle", "cuMemExportToShareableHandle",
                   "selectTransport route"):
        print(f"  {needle:32} {'present' if needle in hit else 'ABSENT'}")


def probe_init(protocol: str) -> None:
    """Ask the binding directly. It logs a specific error when the build lacks
    the transport, which is the only unambiguous answer."""
    section(f"initialize(protocol={protocol!r})")
    try:
        from mooncake.engine import TransferEngine
    except Exception as exc:  # noqa: BLE001
        print(f"  import failed: {exc}")
        return
    eng = TransferEngine()
    try:
        # P2PHANDSHAKE: no master, no metadata server, as register mode uses.
        rc = eng.initialize("127.0.0.1:0", "P2PHANDSHAKE", protocol, "")
        print(f"  rc={rc}  ({'OK' if rc == 0 else 'FAILED'})")
    except Exception as exc:  # noqa: BLE001
        print(f"  raised: {type(exc).__name__}: {exc}")


def probe_fabric() -> None:
    section("fabric-memory support (decides the export path)")
    try:
        cuda = ctypes.CDLL("libcuda.so.1")
    except OSError as exc:
        print(f"  libcuda not loadable: {exc}")
        return
    cuda.cuInit(0)
    dev = ctypes.c_int(0)
    if cuda.cuDeviceGet(ctypes.byref(dev), 0) != 0:
        print("  cuDeviceGet failed")
        return
    # CU_DEVICE_ATTRIBUTE_HANDLE_TYPE_FABRIC_SUPPORTED
    val = ctypes.c_int(0)
    rc = cuda.cuDeviceGetAttribute(ctypes.byref(val), 132, dev)
    print(f"  HANDLE_TYPE_FABRIC_SUPPORTED = {val.value} (rc={rc})")


def probe_torch_ipc() -> None:
    section("torch CUDA tensor -> IPC handle")
    import torch

    if not torch.cuda.is_available():
        print("  no CUDA")
        return
    conf = os.environ.get("PYTORCH_CUDA_ALLOC_CONF", "(unset)")
    print(f"  PYTORCH_CUDA_ALLOC_CONF={conf}")
    t = torch.arange(1024, device="cuda", dtype=torch.float32)
    storage = t.untyped_storage()
    try:
        # What torch itself uses to share a tensor across processes.
        handle = storage._share_cuda_()
        print(f"  _share_cuda_ OK: {len(handle)} fields, "
              f"offset_bytes={handle[3] if len(handle) > 3 else '?'}")
    except Exception as exc:  # noqa: BLE001
        print(f"  _share_cuda_ FAILED: {type(exc).__name__}: {exc}")
    print(f"  storage base=0x{storage.data_ptr():x} nbytes={storage.nbytes()}")


if __name__ == "__main__":
    print(f"python={sys.version.split()[0]}")
    probe_symbols()
    probe_fabric()
    probe_torch_ipc()
    for proto in ("nvlink", "nvlink_intra"):
        probe_init(proto)
