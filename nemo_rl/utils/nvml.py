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
from typing import Generator

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
