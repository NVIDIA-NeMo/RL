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
"""Unit tests for Policy class separation into TrainingPolicy and GenerationPolicy.

Tests verify:
- TrainingPolicy and GenerationPolicy can be imported separately
- Each class has distinct responsibilities with no overlap
- Backward compatibility is maintained through the unified Policy class
- Protocol interfaces are correctly defined
"""

import pytest


class TestPolicyImports:
    """Test that policy classes can be imported correctly."""

    def test_import_training_policy(self):
        """Verify TrainingPolicy can be imported."""
        from nemo_rl.models.policy.training_policy import TrainingPolicy

        assert TrainingPolicy is not None

    def test_import_generation_policy(self):
        """Verify GenerationPolicy can be imported."""
        from nemo_rl.models.policy.generation_policy import GenerationPolicy

        assert GenerationPolicy is not None

    def test_import_protocols(self):
        """Verify protocol classes can be imported."""
        from nemo_rl.models.policy.training_policy import TrainingPolicyProtocol
        from nemo_rl.models.policy.generation_policy import GenerationPolicyProtocol

        assert TrainingPolicyProtocol is not None
        assert GenerationPolicyProtocol is not None

    def test_import_from_package(self):
        """Verify classes can be imported from the package __init__."""
        from nemo_rl.models.policy import (
            TrainingPolicy,
            GenerationPolicy,
            TrainingPolicyProtocol,
            GenerationPolicyProtocol,
        )

        assert TrainingPolicy is not None
        assert GenerationPolicy is not None
        assert TrainingPolicyProtocol is not None
        assert GenerationPolicyProtocol is not None

    def test_import_unified_policy(self):
        """Verify unified Policy class can still be imported."""
        from nemo_rl.models.policy import Policy
        from nemo_rl.models.policy.lm_policy import Policy as LMPolicy

        assert Policy is not None
        assert LMPolicy is not None
        assert Policy is LMPolicy


class TestTrainingPolicyInterface:
    """Test TrainingPolicy has required training methods."""

    def test_training_policy_has_train_method(self):
        """Verify TrainingPolicy has train() method."""
        from nemo_rl.models.policy.training_policy import TrainingPolicy

        assert hasattr(TrainingPolicy, "train")
        assert callable(getattr(TrainingPolicy, "train", None))

    def test_training_policy_has_get_logprobs(self):
        """Verify TrainingPolicy has get_logprobs() method."""
        from nemo_rl.models.policy.training_policy import TrainingPolicy

        assert hasattr(TrainingPolicy, "get_logprobs")
        assert callable(getattr(TrainingPolicy, "get_logprobs", None))

    def test_training_policy_has_save_checkpoint(self):
        """Verify TrainingPolicy has save_checkpoint() method."""
        from nemo_rl.models.policy.training_policy import TrainingPolicy

        assert hasattr(TrainingPolicy, "save_checkpoint")
        assert callable(getattr(TrainingPolicy, "save_checkpoint", None))

    def test_training_policy_has_prepare_for_training(self):
        """Verify TrainingPolicy has prepare_for_training() method."""
        from nemo_rl.models.policy.training_policy import TrainingPolicy

        assert hasattr(TrainingPolicy, "prepare_for_training")
        assert callable(getattr(TrainingPolicy, "prepare_for_training", None))

    def test_training_policy_has_get_reference_policy_logprobs(self):
        """Verify TrainingPolicy has get_reference_policy_logprobs() method."""
        from nemo_rl.models.policy.training_policy import TrainingPolicy

        assert hasattr(TrainingPolicy, "get_reference_policy_logprobs")
        assert callable(getattr(TrainingPolicy, "get_reference_policy_logprobs", None))


class TestGenerationPolicyInterface:
    """Test GenerationPolicy has required generation methods."""

    def test_generation_policy_has_generate(self):
        """Verify GenerationPolicy has generate() method."""
        from nemo_rl.models.policy.generation_policy import GenerationPolicy

        assert hasattr(GenerationPolicy, "generate")
        assert callable(getattr(GenerationPolicy, "generate", None))

    def test_generation_policy_has_score(self):
        """Verify GenerationPolicy has score() method."""
        from nemo_rl.models.policy.generation_policy import GenerationPolicy

        assert hasattr(GenerationPolicy, "score")
        assert callable(getattr(GenerationPolicy, "score", None))

    def test_generation_policy_has_prepare_for_generation(self):
        """Verify GenerationPolicy has prepare_for_generation() method."""
        from nemo_rl.models.policy.generation_policy import GenerationPolicy

        assert hasattr(GenerationPolicy, "prepare_for_generation")
        assert callable(getattr(GenerationPolicy, "prepare_for_generation", None))

    def test_generation_policy_has_update_weights(self):
        """Verify GenerationPolicy has update_weights() method."""
        from nemo_rl.models.policy.generation_policy import GenerationPolicy

        assert hasattr(GenerationPolicy, "update_weights")
        assert callable(getattr(GenerationPolicy, "update_weights", None))


class TestResponsibilitySeparation:
    """Test that responsibilities are correctly separated between classes."""

    def test_training_policy_no_generate(self):
        """Verify TrainingPolicy does NOT have generate() method."""
        from nemo_rl.models.policy.training_policy import TrainingPolicy

        # TrainingPolicy should NOT have generate - that's GenerationPolicy's responsibility
        assert not hasattr(TrainingPolicy, "generate")

    def test_generation_policy_no_train(self):
        """Verify GenerationPolicy does NOT have train() method."""
        from nemo_rl.models.policy.generation_policy import GenerationPolicy

        # GenerationPolicy should NOT have train - that's TrainingPolicy's responsibility
        assert not hasattr(GenerationPolicy, "train")

    def test_generation_policy_no_save_checkpoint(self):
        """Verify GenerationPolicy does NOT have save_checkpoint() method."""
        from nemo_rl.models.policy.generation_policy import GenerationPolicy

        # save_checkpoint is a training responsibility
        assert not hasattr(GenerationPolicy, "save_checkpoint")

    def test_training_policy_no_score(self):
        """Verify TrainingPolicy does NOT have score() method."""
        from nemo_rl.models.policy.training_policy import TrainingPolicy

        # score is a generation responsibility
        assert not hasattr(TrainingPolicy, "score")


class TestUnifiedPolicyBackwardCompatibility:
    """Test that unified Policy class maintains backward compatibility."""

    def test_unified_policy_has_training_methods(self):
        """Verify unified Policy has all training methods."""
        from nemo_rl.models.policy.lm_policy import Policy

        training_methods = [
            "train",
            "get_logprobs",
            "get_reference_policy_logprobs",
            "save_checkpoint",
            "prepare_for_training",
            "finish_training",
        ]
        for method in training_methods:
            assert hasattr(Policy, method), f"Policy missing training method: {method}"
            assert callable(getattr(Policy, method, None))

    def test_unified_policy_has_generation_methods(self):
        """Verify unified Policy has all generation methods."""
        from nemo_rl.models.policy.lm_policy import Policy

        generation_methods = [
            "generate",
            "score",
            "prepare_for_generation",
            "finish_generation",
            "invalidate_kv_cache",
        ]
        for method in generation_methods:
            assert hasattr(Policy, method), f"Policy missing generation method: {method}"
            assert callable(getattr(Policy, method, None))

    def test_unified_policy_has_infrastructure_methods(self):
        """Verify unified Policy has infrastructure methods."""
        from nemo_rl.models.policy.lm_policy import Policy

        infra_methods = [
            "init_collective",
            "offload_before_refit",
            "offload_after_refit",
            "shutdown",
            "broadcast_weights_for_collective",
        ]
        for method in infra_methods:
            assert hasattr(Policy, method), f"Policy missing infra method: {method}"
            assert callable(getattr(Policy, method, None))


class TestProtocolDefinitions:
    """Test protocol class definitions."""

    def test_training_protocol_methods(self):
        """Verify TrainingPolicyProtocol has correct method signatures."""
        from nemo_rl.models.policy.training_policy import TrainingPolicyProtocol
        import inspect

        # Get abstract methods from the protocol
        protocol_methods = [
            name for name, _ in inspect.getmembers(TrainingPolicyProtocol, predicate=inspect.isfunction)
            if not name.startswith("_")
        ]

        # These are the key methods that should be defined
        expected_methods = ["train", "get_logprobs", "save_checkpoint"]
        for method in expected_methods:
            assert method in protocol_methods, f"Protocol missing method: {method}"

    def test_generation_protocol_methods(self):
        """Verify GenerationPolicyProtocol has correct method signatures."""
        from nemo_rl.models.policy.generation_policy import GenerationPolicyProtocol
        import inspect

        protocol_methods = [
            name for name, _ in inspect.getmembers(GenerationPolicyProtocol, predicate=inspect.isfunction)
            if not name.startswith("_")
        ]

        expected_methods = ["generate", "update_weights"]
        for method in expected_methods:
            assert method in protocol_methods, f"Protocol missing method: {method}"


class TestLineCountCompliance:
    """Test that files meet the line count requirements."""

    def test_training_policy_under_400_lines(self):
        """Verify training_policy.py is under 400 lines."""
        import os
        filepath = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))),
            "nemo_rl", "models", "policy", "training_policy.py"
        )
        with open(filepath, "r") as f:
            line_count = len(f.readlines())
        assert line_count < 400, f"training_policy.py has {line_count} lines, should be <400"

    def test_generation_policy_under_400_lines(self):
        """Verify generation_policy.py is under 400 lines."""
        import os
        filepath = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))),
            "nemo_rl", "models", "policy", "generation_policy.py"
        )
        with open(filepath, "r") as f:
            line_count = len(f.readlines())
        assert line_count < 400, f"generation_policy.py has {line_count} lines, should be <400"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
