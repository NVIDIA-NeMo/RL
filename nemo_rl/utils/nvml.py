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
import logging
import os
import socket
from typing import Generator

import pynvml

logger = logging.getLogger(__name__)


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


def get_device_compute_capability(device_idx: int = 0) -> tuple[int, int] | None:
    """Get the CUDA compute capability (major, minor) of a CUDA device using NVML.

    Returns:
        The (major, minor) compute capability, e.g. (8, 0) for an A100, or
        None when it cannot be determined (no GPU, NVML unavailable, or an
        unexpected value). This function never raises.
    """
    try:
        physical_device_idx = device_id_to_physical_device_id(device_idx)
        with nvml_context():
            handle = pynvml.nvmlDeviceGetHandleByIndex(physical_device_idx)
            major, minor = pynvml.nvmlDeviceGetCudaComputeCapability(handle)
        major, minor = int(major), int(minor)
        if major <= 0 or minor < 0:
            return None
        return major, minor
    except Exception:
        return None


def ensure_torch_cuda_arch_list() -> None:
    """Default TORCH_CUDA_ARCH_LIST from the local GPU when it is unset.

    PyTorch builds CUDA extensions (used by the Megatron and Automodel
    backends, e.g. TransformerEngine, DeepEP, and flash-attention) for the
    architectures in TORCH_CUDA_ARCH_LIST. When the variable is unset, a
    run can fail with errors such as "RuntimeError: No CUDA GPUs are
    available" even though GPUs are visible. This sets the variable from
    the local GPU's compute capability, e.g. "8.0" for an A100, so a
    single-node run works without manual setup.

    A value the user already set is never overridden. On multi-node runs
    with different GPUs per node, set the variable explicitly on every
    node (or list every architecture, e.g. "8.0 9.0"), since only the
    driver node's GPU can be detected here. This function never raises:
    when the GPU architecture cannot be detected, it warns and leaves the
    variable unset.
    """
    if os.environ.get("TORCH_CUDA_ARCH_LIST"):
        return
    capability = get_device_compute_capability()
    if capability is None:
        msg = (
            "TORCH_CUDA_ARCH_LIST is not set and the local GPU architecture "
            "could not be detected. Set it to match your GPU before running, "
            "e.g. export TORCH_CUDA_ARCH_LIST='8.0' for A100, '9.0' for H100, "
            "or '10.0' for B200. See docs/local-workstation.md."
        )
        logger.warning(msg)
        print(f"WARNING: {msg}", flush=True)
        return
    major, minor = capability
    arch = f"{major}.{minor}"
    os.environ["TORCH_CUDA_ARCH_LIST"] = arch
    msg = (
        f"TORCH_CUDA_ARCH_LIST was not set; defaulted to '{arch}' from the "
        "local GPU. Set it explicitly when your workers run on different "
        "GPUs, e.g. export TORCH_CUDA_ARCH_LIST='8.0 9.0'."
    )
    logger.info(msg)
    print(msg, flush=True)


def _resolve_device_id(device_id=None):
    """Resolve the logical CUDA device ID.

    Priority: explicit argument > torch.cuda.current_device() > LOCAL_RANK env > 0.
    """
    if device_id is not None:
        return int(device_id)
    try:
        import torch

        if torch.cuda.is_initialized():
            return torch.cuda.current_device()
    except Exception:
        pass
    local_rank = os.environ.get("LOCAL_RANK")
    if local_rank is not None:
        try:
            return int(local_rank)
        except ValueError:
            pass
    return 0


def log_gpu_memory_diagnostics(
    *,
    label: str,
    worker_type: str,
    device_id=None,
    extra_context: str = "",
):
    """Log detailed GPU memory diagnostics with a greppable [GPU_DIAG] prefix.

    This function is designed to never crash -- every NVML/PyTorch call is
    individually guarded. It is safe to call before CUDA is initialized.
    """
    logical_dev = _resolve_device_id(device_id)
    hostname = socket.gethostname()
    cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES", "unset")
    rank = os.environ.get("RANK", "?")

    # --- Physical device + NVML info ---
    physical_dev = "?"
    uuid_str = "?"
    nvml_total = "?"
    nvml_used = "?"
    nvml_free = "?"
    gpu_procs_str = "none detected"
    nvml_error = None

    try:
        physical_dev = device_id_to_physical_device_id(logical_dev)
    except Exception:
        pass

    try:
        with nvml_context():
            handle = pynvml.nvmlDeviceGetHandleByIndex(
                physical_dev if physical_dev != "?" else 0
            )

            try:
                raw_uuid = pynvml.nvmlDeviceGetUUID(handle)
                uuid_str = (
                    raw_uuid.decode("utf-8")
                    if isinstance(raw_uuid, bytes)
                    else str(raw_uuid)
                )
            except Exception:
                uuid_str = "error"

            try:
                mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                nvml_total = f"{mem_info.total / (1024**2):.0f}MB"
                nvml_used = f"{mem_info.used / (1024**2):.0f}MB"
                nvml_free = f"{mem_info.free / (1024**2):.0f}MB"
            except Exception:
                pass

            try:
                procs = pynvml.nvmlDeviceGetComputeRunningProcesses(handle)
                my_pid = os.getpid()
                if procs:
                    parts = []
                    for p in procs:
                        mem_mb = p.usedGpuMemory / (1024**2) if p.usedGpuMemory else 0
                        tag = " (self)" if p.pid == my_pid else ""
                        parts.append(f"PID={p.pid}{tag}: {mem_mb:.1f} MB")
                    gpu_procs_str = "; ".join(parts)
            except Exception:
                pass
    except Exception as e:
        nvml_error = str(e)

    # --- PyTorch allocator info ---
    pt_allocated = "N/A"
    pt_reserved = "N/A"
    try:
        import torch

        if torch.cuda.is_initialized():
            pt_allocated = (
                f"{torch.cuda.memory_allocated(logical_dev) / (1024**2):.1f}MB"
            )
            pt_reserved = f"{torch.cuda.memory_reserved(logical_dev) / (1024**2):.1f}MB"
    except Exception:
        pass

    # --- Build log line ---
    parts = [
        f"[GPU_DIAG] [{worker_type}] [{label}]",
        f"host={hostname} rank={rank} logical_dev={logical_dev} physical_dev={physical_dev}",
        f"CUDA_VISIBLE_DEVICES={cuda_visible} uuid={uuid_str}",
    ]
    if nvml_error:
        parts.append(f"| nvml_error: {nvml_error}")
    parts.append(f"| NVML: total={nvml_total} used={nvml_used} free={nvml_free}")
    parts.append(f"| PyTorch: allocated={pt_allocated} reserved={pt_reserved}")
    parts.append(f"| GPU procs: [{gpu_procs_str}]")
    if extra_context:
        parts.append(f"| {extra_context}")

    msg = " ".join(parts)
    logger.info(msg)
    print(msg, flush=True)
