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
"""Tests for callable reward function support (TASK-027)."""

import pytest
from typing import Dict


class TestCallableRewardBasics:
    """Test basic callable reward functionality."""

    def test_simple_callable_works(self):
        """AC1: Any callable(prompt, response) -> float works as reward."""
        from nemo_rl.environments.functional_reward import FunctionalRewardWrapper

        def my_reward(prompt: str, response: str) -> float:
            return len(response) / 100.0

        wrapper = FunctionalRewardWrapper(my_reward)
        assert wrapper.reward_fn == my_reward

    def test_lambda_works(self):
        """Test that lambdas work as reward functions."""
        from nemo_rl.environments.functional_reward import FunctionalRewardWrapper

        wrapper = FunctionalRewardWrapper(lambda p, r: 1.0)
        assert wrapper.reward_fn is not None

    def test_class_method_works(self):
        """Test that class methods work as reward functions."""
        from nemo_rl.environments.functional_reward import FunctionalRewardWrapper

        class RewardComputer:
            def compute(self, prompt: str, response: str) -> float:
                return 1.0

        computer = RewardComputer()
        wrapper = FunctionalRewardWrapper(computer.compute)
        assert wrapper.reward_fn is not None


class TestCallableWrapping:
    """Test auto-wrapping of callables."""

    def test_wrap_reward_callable_with_function(self):
        """AC2: Auto-wrap callable in appropriate environment interface."""
        from nemo_rl.environments import wrap_reward_callable, FunctionalRewardWrapper

        def my_reward(prompt: str, response: str) -> float:
            return 1.0

        wrapped = wrap_reward_callable(my_reward)
        assert isinstance(wrapped, FunctionalRewardWrapper)

    def test_wrap_reward_callable_passthrough(self):
        """Test that existing wrappers pass through."""
        from nemo_rl.environments import wrap_reward_callable, FunctionalRewardWrapper

        def my_reward(prompt: str, response: str) -> float:
            return 1.0

        original = FunctionalRewardWrapper(my_reward)
        wrapped = wrap_reward_callable(original)
        assert wrapped is original

    def test_wrap_reward_callable_none(self):
        """Test that None returns None."""
        from nemo_rl.environments import wrap_reward_callable

        result = wrap_reward_callable(None)
        assert result is None

    def test_wrap_reward_callable_invalid(self):
        """Test that invalid inputs raise TypeError."""
        from nemo_rl.environments import wrap_reward_callable

        with pytest.raises(TypeError, match="must be callable"):
            wrap_reward_callable("not a function")


class TestDictRewards:
    """Test callables returning dict of rewards."""

    def test_dict_reward_accepted(self):
        """AC3: Support for callables returning dict of rewards."""
        from nemo_rl.environments.functional_reward import FunctionalRewardWrapper

        def multi_reward(prompt: str, response: str) -> Dict[str, float]:
            return {
                "length": len(response) / 100.0,
                "correctness": 1.0 if "correct" in response else 0.0,
            }

        wrapper = FunctionalRewardWrapper(multi_reward)
        assert wrapper.reward_fn == multi_reward


class TestSignatureValidation:
    """Test signature validation and error messages."""

    def test_not_callable_raises_error(self):
        """AC4: Clear error messages for invalid callable signatures."""
        from nemo_rl.environments.functional_reward import FunctionalRewardWrapper

        with pytest.raises(TypeError, match="must be callable"):
            FunctionalRewardWrapper("not a function")

    def test_too_few_params_raises_error(self):
        """Test that functions with < 2 params raise error."""
        from nemo_rl.environments.functional_reward import FunctionalRewardWrapper

        def bad_reward(x: str) -> float:
            return 1.0

        with pytest.raises(ValueError, match="at least 2 parameters"):
            FunctionalRewardWrapper(bad_reward)

    def test_validate_reward_callable_success(self):
        """Test validate_reward_callable with valid function."""
        from nemo_rl.environments import validate_reward_callable

        def good_reward(prompt: str, response: str) -> float:
            return 1.0

        is_valid, error = validate_reward_callable(good_reward)
        assert is_valid
        assert error == ""

    def test_validate_reward_callable_failure(self):
        """Test validate_reward_callable with invalid function."""
        from nemo_rl.environments import validate_reward_callable

        def bad_reward(x: str) -> float:
            return 1.0

        is_valid, error = validate_reward_callable(bad_reward)
        assert not is_valid
        assert "at least 2 parameters" in error

    def test_validate_reward_callable_not_callable(self):
        """Test validate_reward_callable with non-callable."""
        from nemo_rl.environments import validate_reward_callable

        is_valid, error = validate_reward_callable("not callable")
        assert not is_valid
        assert "Expected callable" in error


class TestRewardComputation:
    """Test actual reward computation."""

    def test_compute_single_reward(self):
        """Test computing a single reward."""
        from nemo_rl.environments.functional_reward import FunctionalRewardWrapper

        def length_reward(prompt: str, response: str) -> float:
            return len(response) / 10.0

        wrapper = FunctionalRewardWrapper(length_reward)
        result = wrapper._compute_single_reward("Hello", "World")
        assert result == 0.5  # len("World") / 10 = 0.5

    def test_call_directly(self):
        """Test calling wrapper directly."""
        from nemo_rl.environments.functional_reward import FunctionalRewardWrapper

        def simple_reward(prompt: str, response: str) -> float:
            return 1.0

        wrapper = FunctionalRewardWrapper(simple_reward)
        # Direct call with lists
        rewards = wrapper(["p1", "p2"], ["r1", "r2"])
        assert len(rewards) == 2


class TestDecorators:
    """Test reward function decorators."""

    def test_reward_function_decorator(self):
        """Test @reward_function decorator."""
        from nemo_rl.environments import reward_function, FunctionalRewardWrapper

        @reward_function("test_reward")
        def my_reward(prompt: str, response: str) -> float:
            return 1.0

        assert isinstance(my_reward, FunctionalRewardWrapper)
        assert my_reward.name == "test_reward"

    def test_batched_reward_decorator(self):
        """Test @batched_reward decorator."""
        from nemo_rl.environments import batched_reward, FunctionalRewardWrapper
        from typing import List

        @batched_reward("batch_reward")
        def batch_score(prompts: List[str], responses: List[str]) -> List[float]:
            return [len(r) / 100 for r in responses]

        assert isinstance(batch_score, FunctionalRewardWrapper)
        assert batch_score._is_batched


class TestVerifyCriterion:
    """VERIFY: Pass function def my_reward(p, r): return len(r) to trainer."""

    def test_verify_simple_reward_function(self):
        """Test the VERIFY criterion from acceptance criteria."""
        from nemo_rl.environments.functional_reward import FunctionalRewardWrapper

        # The exact function from VERIFY criterion
        def my_reward(p, r):
            return len(r)

        # This should work without errors
        wrapper = FunctionalRewardWrapper(my_reward, validate_signature=False)
        
        # Should be able to compute rewards
        result = wrapper._compute_single_reward("prompt", "response")
        assert result == 8  # len("response")

    def test_verify_with_type_hints(self):
        """Test with proper type hints."""
        from nemo_rl.environments.functional_reward import FunctionalRewardWrapper

        def my_reward(p: str, r: str) -> float:
            return float(len(r))

        wrapper = FunctionalRewardWrapper(my_reward)
        result = wrapper._compute_single_reward("prompt", "response")
        assert result == 8.0


class TestCreateRewardWrapper:
    """Test create_reward_wrapper factory function."""

    def test_create_reward_wrapper_basic(self):
        """Test basic factory function usage."""
        from nemo_rl.environments import create_reward_wrapper

        def my_reward(prompt: str, response: str) -> float:
            return 1.0

        wrapper = create_reward_wrapper(my_reward, name="test")
        assert wrapper.name == "test"

    def test_create_reward_wrapper_with_config(self):
        """Test factory with config kwargs."""
        from nemo_rl.environments import create_reward_wrapper

        def my_reward(prompt: str, response: str) -> float:
            return 1.0

        wrapper = create_reward_wrapper(
            my_reward,
            name="test",
            max_workers=4,
            batch_size=16,
        )
        assert wrapper.config.max_workers == 4
        assert wrapper.config.batch_size == 16
