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
import logging
import os
from unittest.mock import MagicMock, patch

from nemo_rl.utils.nvml import (
    _resolve_device_id,
    device_id_to_physical_device_id,
    get_device_uuid,
    log_gpu_memory_diagnostics,
    nvml_context,
)


def test_resolve_device_id_explicit_arg():
    """Explicit device_id argument takes highest priority."""
    assert _resolve_device_id(device_id=3) == 3
    assert _resolve_device_id(device_id="2") == 2


def test_resolve_device_id_cuda_initialized():
    with (
        patch("torch.cuda.is_initialized", return_value=True),
        patch("torch.cuda.current_device", return_value=1),
    ):
        assert _resolve_device_id() == 1


def test_resolve_device_id_local_rank_env():
    with (
        patch("torch.cuda.is_initialized", return_value=False),
        patch.dict(os.environ, {"LOCAL_RANK": "2"}),
    ):
        assert _resolve_device_id() == 2


def test_resolve_device_id_default_zero():
    with (
        patch("torch.cuda.is_initialized", return_value=False),
        patch.dict(os.environ, {}, clear=True),
    ):
        assert _resolve_device_id() == 0


def test_resolve_device_id_invalid_local_rank():
    with (
        patch("torch.cuda.is_initialized", return_value=False),
        patch.dict(os.environ, {"LOCAL_RANK": "not-a-number"}),
    ):
        assert _resolve_device_id() == 0


@patch("nemo_rl.utils.nvml.pynvml")
def test_log_gpu_memory_diagnostics_emits_prefix(mock_pynvml, capfd):
    mock_pynvml.nvmlDeviceGetHandleByIndex.return_value = MagicMock()
    mock_pynvml.nvmlDeviceGetUUID.return_value = b"GPU-FAKE-UUID"
    mem_info = MagicMock()
    mem_info.total, mem_info.used, mem_info.free = 24 * 1024**3, 4 * 1024**3, 20 * 1024**3
    mock_pynvml.nvmlDeviceGetMemoryInfo.return_value = mem_info
    mock_pynvml.nvmlDeviceGetComputeRunningProcesses.return_value = []

    log_gpu_memory_diagnostics(label="test-label", worker_type="TestWorker", device_id=0)

    out = capfd.readouterr().out
    assert "[GPU_DIAG]" in out and "TestWorker" in out and "test-label" in out


@patch("nemo_rl.utils.nvml.pynvml")
def test_log_gpu_memory_diagnostics_never_raises_on_nvml_failure(mock_pynvml, capfd):
    mock_pynvml.nvmlInit.side_effect = Exception("NVML not available")
    log_gpu_memory_diagnostics(label="fail-label", worker_type="TestWorker")
    out = capfd.readouterr().out
    assert "[GPU_DIAG]" in out and "nvml_error" in out


@patch("nemo_rl.utils.nvml.pynvml")
def test_log_gpu_memory_diagnostics_extra_context(mock_pynvml, capfd):
    mock_pynvml.nvmlInit.side_effect = Exception("skip")
    log_gpu_memory_diagnostics(
        label="ctx-label", worker_type="TestWorker", extra_context="my_custom_info=42"
    )
    assert "my_custom_info=42" in capfd.readouterr().out
