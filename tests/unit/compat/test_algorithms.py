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
"""Tests for backward-compatible algorithm wrappers."""

import warnings
import pytest

from nemo_rl.compat.algorithms import GRPO, SFT, DPO


class TestGRPOCompat:
    """Tests for GRPO backward compatibility class."""

    def test_grpo_init_emits_warning(self):
        """Test that GRPO initialization emits deprecation warning."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            try:
                grpo = GRPO(model="test-model")
            except Exception:
                # May fail due to missing dependencies, but warning should still emit
                pass
            
            # Check that at least one deprecation warning was emitted
            deprecation_warnings = [x for x in w if issubclass(x.category, DeprecationWarning)]
            assert len(deprecation_warnings) >= 1
            assert "GRPO" in str(deprecation_warnings[0].message)
            assert "GRPOTrainer" in str(deprecation_warnings[0].message)

    def test_grpo_from_config_emits_warning(self):
        """Test that GRPO.from_config() emits deprecation warning."""
        # Create a mock config-like object
        class MockConfig:
            def __init__(self):
                self.policy = type("Policy", (), {"model_name": "test-model"})()
            
            def to_container(self, resolve=True):
                return {"policy": {"model_name": "test-model"}}
        
        mock_cfg = MockConfig()
        
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            try:
                grpo = GRPO.from_config(mock_cfg)
            except Exception:
                # May fail, but warning should still emit
                pass
            
            deprecation_warnings = [x for x in w if issubclass(x.category, DeprecationWarning)]
            assert len(deprecation_warnings) >= 1
            # Check for from_config deprecation
            from_config_warnings = [x for x in deprecation_warnings if "from_config" in str(x.message)]
            assert len(from_config_warnings) >= 1


class TestSFTCompat:
    """Tests for SFT backward compatibility class."""

    def test_sft_init_emits_warning(self):
        """Test that SFT initialization emits deprecation warning."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            try:
                sft = SFT(model="test-model")
            except Exception:
                pass
            
            deprecation_warnings = [x for x in w if issubclass(x.category, DeprecationWarning)]
            assert len(deprecation_warnings) >= 1
            assert "SFT" in str(deprecation_warnings[0].message)
            assert "SFTTrainer" in str(deprecation_warnings[0].message)


class TestDPOCompat:
    """Tests for DPO backward compatibility class."""

    def test_dpo_init_emits_warning(self):
        """Test that DPO initialization emits deprecation warning."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            try:
                dpo = DPO(model="test-model")
            except Exception:
                pass
            
            deprecation_warnings = [x for x in w if issubclass(x.category, DeprecationWarning)]
            assert len(deprecation_warnings) >= 1
            assert "DPO" in str(deprecation_warnings[0].message)
            assert "DPOTrainer" in str(deprecation_warnings[0].message)


class TestDeprecatedFunctions:
    """Tests for deprecated training functions."""

    def test_grpo_train_emits_warning(self):
        """Test that grpo_train() emits deprecation warning."""
        from nemo_rl.compat.algorithms import grpo_train
        
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            try:
                grpo_train(
                    model="test",
                    dataset="test",
                    reward_fn=lambda p, r: 1.0,
                )
            except Exception:
                pass
            
            deprecation_warnings = [x for x in w if issubclass(x.category, DeprecationWarning)]
            assert len(deprecation_warnings) >= 1
            assert "grpo_train" in str(deprecation_warnings[0].message)
            assert "nemo_rl.train" in str(deprecation_warnings[0].message)

    def test_sft_train_emits_warning(self):
        """Test that sft_train() emits deprecation warning."""
        from nemo_rl.compat.algorithms import sft_train
        
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            try:
                sft_train(model="test", dataset="test")
            except Exception:
                pass
            
            deprecation_warnings = [x for x in w if issubclass(x.category, DeprecationWarning)]
            assert len(deprecation_warnings) >= 1
            assert "sft_train" in str(deprecation_warnings[0].message)

    def test_dpo_train_emits_warning(self):
        """Test that dpo_train() emits deprecation warning."""
        from nemo_rl.compat.algorithms import dpo_train
        
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            try:
                dpo_train(model="test", dataset="test")
            except Exception:
                pass
            
            deprecation_warnings = [x for x in w if issubclass(x.category, DeprecationWarning)]
            assert len(deprecation_warnings) >= 1
            assert "dpo_train" in str(deprecation_warnings[0].message)
