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
import os
from unittest.mock import patch

from nemo_rl.utils.nvml import (
    device_id_to_physical_device_id,
    get_device_uuid,
    nvml_context,
)


@patch("nemo_rl.utils.nvml.pynvml")
def test_nvml_context(mock_pynvml):
    """Test that nvml_context initializes and shuts down NVML."""
    with nvml_context():
        pass

    # Verify init and shutdown were called
    mock_pynvml.nvmlInit.assert_called_once()
    mock_pynvml.nvmlShutdown.assert_called_once()


def test_device_id_conversion():
    """Test device ID conversion with and without CUDA_VISIBLE_DEVICES."""
    with patch.dict(os.environ, {}, clear=True):
        assert device_id_to_physical_device_id(0) == 0

    with patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "2,3"}):
        assert device_id_to_physical_device_id(0) == 2
        assert device_id_to_physical_device_id(1) == 3


@patch("nemo_rl.utils.nvml.device_id_to_physical_device_id")
@patch("nemo_rl.utils.nvml.pynvml")
def test_get_device_uuid(mock_pynvml, mock_convert_id):
    """Test that get_device_uuid correctly retrieves a UUID."""

    # Setup
    mock_convert_id.return_value = 1
    mock_handle = mock_pynvml.nvmlDeviceGetHandleByIndex.return_value
    mock_pynvml.nvmlDeviceGetUUID.return_value = b"GPU-12345"

    # Call function
    uuid = get_device_uuid(0)

    # Verify
    assert uuid == "GPU-12345"
    mock_convert_id.assert_called_once_with(0)
    mock_pynvml.nvmlDeviceGetHandleByIndex.assert_called_once_with(1)
