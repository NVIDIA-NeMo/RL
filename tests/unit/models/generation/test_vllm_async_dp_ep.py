"""Unit tests for vLLM async DP+EP patches."""

import os
from unittest.mock import MagicMock, patch


class TestVllmDeviceAllocationPatch:
    """Test device allocation patch for DP+EP."""

    def test_single_device(self):
        """Single device should return string value."""
        from nemo_rl.models.generation.vllm.vllm_worker_async import (
            VllmAsyncGenerationWorker,
        )

        worker = VllmAsyncGenerationWorker.__new__(VllmAsyncGenerationWorker)
        with patch("vllm.v1.engine.utils") as mock_utils:
            mock_utils.get_device_indices = MagicMock(
                side_effect=ValueError("parse error")
            )
            worker._patch_vllm_device_allocation()

            os.environ["CUDA_VISIBLE_DEVICES"] = "1"
            result = mock_utils.get_device_indices("CUDA_VISIBLE_DEVICES", 0, 1)
            assert result == "1"

    def test_no_env(self):
        """No env var should use sequential allocation."""
        from nemo_rl.models.generation.vllm.vllm_worker_async import (
            VllmAsyncGenerationWorker,
        )

        worker = VllmAsyncGenerationWorker.__new__(VllmAsyncGenerationWorker)
        with patch("vllm.v1.engine.utils") as mock_utils:
            mock_utils.get_device_indices = MagicMock(
                side_effect=ValueError("parse error")
            )
            worker._patch_vllm_device_allocation()

            os.environ.pop("CUDA_VISIBLE_DEVICES", None)
            result = mock_utils.get_device_indices("CUDA_VISIBLE_DEVICES", 0, 2)
            assert result == [0, 1]


class TestVllmStatsAddressPatch:
    """Test stats address patch conditional behavior."""

    def test_skips_patch_when_dp_size_is_one(self):
        """Should skip patch when VLLM_DP_SIZE=1."""
        from nemo_rl.models.generation.vllm.vllm_worker_async import (
            VllmAsyncGenerationWorker,
        )

        worker = VllmAsyncGenerationWorker.__new__(VllmAsyncGenerationWorker)
        os.environ["VLLM_DP_SIZE"] = "1"

        with patch("vllm.v1.engine.core_client") as mock_client:
            original_fn = MagicMock(name="original_ensure")
            mock_dp_client = MagicMock()
            mock_dp_client._ensure_stats_update_task = original_fn
            mock_client.DPLBAsyncMPClient = mock_dp_client

            worker._patch_vllm_stats_address()

            assert mock_dp_client._ensure_stats_update_task is original_fn

    def test_applies_patch_when_dp_size_greater_than_one(self):
        """Should apply patch when VLLM_DP_SIZE>1."""
        from nemo_rl.models.generation.vllm.vllm_worker_async import (
            VllmAsyncGenerationWorker,
        )

        worker = VllmAsyncGenerationWorker.__new__(VllmAsyncGenerationWorker)
        os.environ["VLLM_DP_SIZE"] = "2"

        with patch("vllm.v1.engine.core_client") as mock_client:
            original_fn = MagicMock(name="original_ensure")
            mock_dp_client = MagicMock()
            mock_dp_client._ensure_stats_update_task = original_fn
            mock_client.DPLBAsyncMPClient = mock_dp_client

            worker._patch_vllm_stats_address()

            patched_fn = mock_dp_client._ensure_stats_update_task
            assert patched_fn is not original_fn
            assert callable(patched_fn)
