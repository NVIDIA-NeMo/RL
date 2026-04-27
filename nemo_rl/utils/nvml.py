# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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
import contextlib
import os
from typing import Any, Generator

import pynvml


@contextlib.contextmanager
def nvml_context() -> Generator[None, None, None]:
    """Context manager for NVML initialization and shutdown.

    Raises:
        RuntimeError: If NVML initialization fails
    """
    try:
        pynvml.nvmlInit()
        yield
    except pynvml.NVMLError as e:
        raise RuntimeError(f"Failed to initialize NVML: {e}")
    finally:
        try:
            pynvml.nvmlShutdown()
        except:
            pass


def device_id_to_physical_device_id(device_id: int) -> int:
    """Convert a logical device ID to a physical device ID considering CUDA_VISIBLE_DEVICES."""
    if "CUDA_VISIBLE_DEVICES" in os.environ:
        device_ids = os.environ["CUDA_VISIBLE_DEVICES"].split(",")
        try:
            physical_device_id = int(device_ids[device_id])
            return physical_device_id
        except ValueError:
            raise RuntimeError(
                f"Failed to convert logical device ID {device_id} to physical device ID. Available devices are: {device_ids}."
            )
    else:
        return device_id


def get_device_uuid(device_idx: int) -> str:
    """Get the UUID of a CUDA device using NVML."""
    # Convert logical device index to physical device index
    global_device_idx = device_id_to_physical_device_id(device_idx)

    # Get the device handle and UUID
    with nvml_context():
        try:
            handle = pynvml.nvmlDeviceGetHandleByIndex(global_device_idx)
            uuid = pynvml.nvmlDeviceGetUUID(handle)
            # Ensure the UUID is returned as a string, not bytes
            if isinstance(uuid, bytes):
                return uuid.decode("utf-8")
            elif isinstance(uuid, str):
                return uuid
            else:
                raise RuntimeError(
                    f"Unexpected UUID type: {type(uuid)} for device {device_idx} (global index: {global_device_idx})"
                )
        except pynvml.NVMLError as e:
            raise RuntimeError(
                f"Failed to get device UUID for device {device_idx} (global index: {global_device_idx}): {e}"
            )


def get_free_memory_bytes(device_idx: int) -> float:
    """Get the free memory of a CUDA device in bytes using NVML."""
    global_device_idx = device_id_to_physical_device_id(device_idx)
    with nvml_context():
        try:
            handle = pynvml.nvmlDeviceGetHandleByIndex(global_device_idx)
            return pynvml.nvmlDeviceGetMemoryInfo(handle).free
        except pynvml.NVMLError as e:
            raise RuntimeError(
                f"Failed to get free memory for device {device_idx} (global index: {global_device_idx}): {e}"
            )


def get_full_nvml_snapshot() -> dict[str, Any]:
    """Return per-GPU memory and per-process usage for the local node."""
    with nvml_context():
        device_count = pynvml.nvmlDeviceGetCount()
        devices: list[dict[str, Any]] = []
        for i in range(device_count):
            handle = pynvml.nvmlDeviceGetHandleByIndex(i)
            mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
            name = pynvml.nvmlDeviceGetName(handle)
            if isinstance(name, bytes):
                name = name.decode("utf-8", errors="ignore")
            procs: list[dict[str, Any]] = []
            try:
                proc_info = pynvml.nvmlDeviceGetComputeRunningProcesses_v2(handle)
            except Exception:
                proc_info = []
            for p in proc_info:
                used_bytes = getattr(p, "usedGpuMemory", 0) or 0
                try:
                    pname = pynvml.nvmlSystemGetProcessName(p.pid)
                    if isinstance(pname, bytes):
                        pname = pname.decode("utf-8", errors="ignore")
                except Exception:
                    pname = ""
                procs.append(
                    {
                        "pid": int(p.pid),
                        "usedMiB": int(used_bytes / (1024**2)),
                        "name": pname,
                    }
                )
            devices.append(
                {
                    "index": i,
                    "name": name,
                    "totalMiB": int(mem.total / (1024**2)),
                    "usedMiB": int(mem.used / (1024**2)),
                    "freeMiB": int(mem.free / (1024**2)),
                    "processes": procs,
                }
            )
        return {
            "node": os.uname().nodename,
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "devices": devices,
        }
