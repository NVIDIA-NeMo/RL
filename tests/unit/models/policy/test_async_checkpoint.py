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
"""Unit tests for async checkpointing config flow and validation.

These tests exercise the config pipeline and validation guards without
requiring megatron dependencies or Ray actors.
"""

from unittest.mock import MagicMock

import pytest


class TestAsyncSaveConfigPipeline:
    """Test that async_save flows correctly through the config pipeline."""

    def test_create_checkpoint_config_async_save_true(self):
        from nemo_rl.models.megatron.setup import _create_checkpoint_config

        cfg = _create_checkpoint_config(
            "/fake/pretrained", "/fake/weights", async_save=True
        )
        assert cfg.async_save is True

    def test_create_checkpoint_config_async_save_false(self):
        from nemo_rl.models.megatron.setup import _create_checkpoint_config

        cfg = _create_checkpoint_config(
            "/fake/pretrained", "/fake/weights", async_save=False
        )
        assert cfg.async_save is False

    def test_create_checkpoint_config_default_is_false(self):
        from nemo_rl.models.megatron.setup import _create_checkpoint_config

        cfg = _create_checkpoint_config("/fake/pretrained", "/fake/weights")
        assert cfg.async_save is False


class TestColocatedValidation:
    """Test that async_save + colocated is rejected at config validation time."""

    def test_async_save_rejected_in_colocated_mode(self):
        from nemo_rl.models.megatron.setup import validate_and_set_config

        config = {
            "megatron_cfg": {"async_save": True},
            "generation": {"colocated": {"enabled": True}},
        }

        with pytest.raises(ValueError, match="colocated"):
            validate_and_set_config(
                config,
                rank=0,
                hf_model_name="test",
                pretrained_path="/fake",
                weights_path=None,
                tokenizer=None,
            )

    def test_async_save_allowed_in_non_colocated_mode(self):
        """async_save + non-colocated should pass the colocated guard."""
        from nemo_rl.models.megatron.setup import validate_and_set_config

        config = {
            "megatron_cfg": {"async_save": True},
            "generation": {"colocated": {"enabled": False}},
            "precision": "bfloat16",
        }

        try:
            validate_and_set_config(
                config,
                rank=0,
                hf_model_name="test",
                pretrained_path="/fake",
                weights_path=None,
                tokenizer=None,
            )
        except ValueError as e:
            if "colocated" in str(e).lower():
                pytest.fail("Should not reject async_save in non-colocated mode")
        except (KeyError, TypeError, FileNotFoundError):
            pass  # Expected -- other config fields missing, but colocated check passed

    def test_no_async_save_passes_in_colocated_mode(self):
        """async_save=False + colocated should not raise on the async guard."""
        from nemo_rl.models.megatron.setup import validate_and_set_config

        config = {
            "megatron_cfg": {"async_save": False},
            "generation": {"colocated": {"enabled": True}},
            "precision": "bfloat16",
        }

        try:
            validate_and_set_config(
                config,
                rank=0,
                hf_model_name="test",
                pretrained_path="/fake",
                weights_path=None,
                tokenizer=None,
            )
        except ValueError as e:
            if "async_save" in str(e).lower():
                pytest.fail("Should not reject async_save=False")
        except (KeyError, TypeError, FileNotFoundError):
            pass


class TestBaseWorkerStubs:
    """Verify that AbstractPolicyWorker has no-op stubs so non-Megatron backends don't crash."""

    def test_base_worker_finalize_is_noop(self):
        from nemo_rl.models.policy.workers.base_policy_worker import (
            AbstractPolicyWorker,
        )

        worker = MagicMock(spec=AbstractPolicyWorker)
        AbstractPolicyWorker.finalize_pending_checkpoint(worker)

    def test_base_worker_shutdown_async_is_noop(self):
        from nemo_rl.models.policy.workers.base_policy_worker import (
            AbstractPolicyWorker,
        )

        worker = MagicMock(spec=AbstractPolicyWorker)
        AbstractPolicyWorker.shutdown_async_checkpoint_worker(worker)
