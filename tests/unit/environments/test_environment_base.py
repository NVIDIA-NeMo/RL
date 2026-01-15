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
"""Unit tests for Environment base class."""

import asyncio
from typing import Any, Dict, List

import pytest
import torch

from nemo_rl.environments import (
    Environment,
    EnvironmentConfig,
    SimpleEnvironment,
    StatefulEnvironment,
)
from nemo_rl.environments.interfaces import EnvironmentReturn


class TestEnvironmentConfig:
    """Tests for EnvironmentConfig dataclass."""

    def test_default_config(self):
        """Test default configuration values."""
        config = EnvironmentConfig()

        assert config.name == "environment"
        assert config.max_workers == 8
        assert config.timeout == 60.0
        assert config.default_reward == 0.0
        assert config.enable_metrics is True

    def test_custom_config(self):
        """Test custom configuration values."""
        config = EnvironmentConfig(
            name="my_env",
            max_workers=4,
            timeout=30.0,
            default_reward=-1.0,
            enable_metrics=False,
        )

        assert config.name == "my_env"
        assert config.max_workers == 4
        assert config.timeout == 30.0
        assert config.default_reward == -1.0
        assert config.enable_metrics is False


class TestEnvironmentBase:
    """Tests for Environment base class."""

    def test_must_implement_score_or_score_batch(self):
        """Test that subclass must implement score() or score_batch()."""

        class EmptyEnvironment(Environment):
            pass

        env = EmptyEnvironment()

        with pytest.raises(NotImplementedError) as exc_info:
            env.score("prompt", "response")

        assert "must implement either score() or score_batch()" in str(exc_info.value)

    def test_simple_score_implementation(self):
        """Test simple score() implementation."""

        class LengthReward(Environment):
            def score(self, prompt: str, response: str) -> float:
                return len(response) / 100.0

        env = LengthReward()
        reward = env.score("Test prompt", "This is a response")

        assert reward == pytest.approx(0.18)

    def test_score_batch_uses_score(self):
        """Test that score_batch() default implementation uses score()."""

        class LengthReward(Environment):
            def score(self, prompt: str, response: str) -> float:
                return len(response) / 100.0

        env = LengthReward()
        prompts = ["prompt1", "prompt2", "prompt3"]
        responses = ["short", "longer response", "very very long response"]

        rewards = env.score_batch(prompts, responses)

        assert len(rewards) == 3
        assert rewards[0] == pytest.approx(0.05)  # len("short") = 5
        assert rewards[1] == pytest.approx(0.15)  # len("longer response") = 15
        assert rewards[2] == pytest.approx(0.23)  # len("very very long response") = 23

    def test_custom_score_batch(self):
        """Test custom score_batch() implementation."""

        class BatchedReward(Environment):
            def score_batch(
                self, prompts: List[str], responses: List[str]
            ) -> List[float]:
                # Optimized batch processing
                return [len(r) / 100.0 for r in responses]

        env = BatchedReward()
        rewards = env.score_batch(
            ["p1", "p2"], ["response one", "response two here"]
        )

        assert len(rewards) == 2
        assert rewards[0] == pytest.approx(0.12)  # len("response one") = 12
        assert rewards[1] == pytest.approx(0.17)  # len("response two here") = 17

    def test_step_returns_environment_return(self):
        """Test that step() returns EnvironmentReturn."""

        class SimpleReward(SimpleEnvironment):
            def score(self, prompt: str, response: str) -> float:
                return 1.0 if "good" in response else 0.0

        env = SimpleReward()

        message_logs = [
            [
                {"role": "user", "content": "What is 2+2?"},
                {"role": "assistant", "content": "The answer is 4, good job!"},
            ],
            [
                {"role": "user", "content": "What is 3+3?"},
                {"role": "assistant", "content": "The answer is 6"},
            ],
        ]

        result = env.step(message_logs, [None, None])

        assert isinstance(result, EnvironmentReturn)
        assert len(result.rewards) == 2
        assert result.rewards[0].item() == 1.0  # Contains "good"
        assert result.rewards[1].item() == 0.0  # Does not contain "good"
        assert all(result.terminateds)

    def test_validate_response(self):
        """Test validate_response() filtering."""

        class ValidatedReward(SimpleEnvironment):
            def validate_response(self, prompt: str, response: str) -> bool:
                # Reject empty responses
                return len(response.strip()) > 0

            def score(self, prompt: str, response: str) -> float:
                return 1.0

        env = ValidatedReward()

        # Valid response
        rewards = env.score_batch(["prompt"], ["valid response"])
        assert rewards[0] == 1.0

        # Invalid response (empty)
        rewards = env.score_batch(["prompt"], ["   "])
        assert rewards[0] == 0.0  # Default reward

    def test_get_metrics(self):
        """Test custom metrics via get_metrics()."""

        class MetricReward(SimpleEnvironment):
            def score(self, prompt: str, response: str) -> float:
                return 1.0 if "correct" in response else 0.0

            def get_metrics(self, batch: Any) -> Dict[str, float]:
                return {
                    "custom_metric": 42.0,
                    "another_metric": 3.14,
                }

        env = MetricReward()
        batch = {"rewards": torch.tensor([1.0, 0.0, 1.0])}

        _, metrics = env.global_post_process_and_metrics(batch)

        assert metrics["environment_name"] == "environment"
        assert metrics["custom_metric"] == 42.0
        assert metrics["another_metric"] == 3.14
        assert "mean_reward" in metrics

    def test_setup_and_teardown(self):
        """Test setup() and teardown() lifecycle methods."""

        class LifecycleEnv(SimpleEnvironment):
            def __init__(self):
                super().__init__()
                self.setup_called = False
                self.teardown_called = False

            def setup(self):
                self.setup_called = True

            def teardown(self):
                self.teardown_called = True

            def score(self, prompt: str, response: str) -> float:
                return 1.0

        env = LifecycleEnv()

        # setup not called yet
        assert not env.setup_called

        # step() should trigger setup
        env.step(
            [[{"role": "user", "content": "test"}, {"role": "assistant", "content": "resp"}]],
            [None],
        )
        assert env.setup_called

        # teardown called on shutdown
        assert not env.teardown_called
        env.shutdown()
        assert env.teardown_called

    def test_name_override(self):
        """Test name can be overridden via constructor."""

        class NamedEnv(SimpleEnvironment):
            def score(self, prompt: str, response: str) -> float:
                return 1.0

        # Default name
        env1 = NamedEnv()
        assert env1.name == "environment"

        # Name via config
        env2 = NamedEnv(config=EnvironmentConfig(name="config_name"))
        assert env2.name == "config_name"

        # Name override takes precedence
        env3 = NamedEnv(
            config=EnvironmentConfig(name="config_name"), name="override_name"
        )
        assert env3.name == "override_name"

    def test_repr(self):
        """Test string representation."""

        class CustomEnv(SimpleEnvironment):
            def score(self, prompt: str, response: str) -> float:
                return 1.0

        env = CustomEnv(name="test_env")
        repr_str = repr(env)

        assert "CustomEnv" in repr_str
        assert "test_env" in repr_str

    def test_error_handling_in_score(self):
        """Test that errors in score() are handled gracefully."""

        class ErrorProneEnv(SimpleEnvironment):
            def score(self, prompt: str, response: str) -> float:
                if "error" in response:
                    raise ValueError("Intentional error")
                return 1.0

        env = ErrorProneEnv()

        rewards = env.score_batch(
            ["p1", "p2"], ["normal response", "error response"]
        )

        assert rewards[0] == 1.0
        assert rewards[1] == 0.0  # Default reward on error


class TestSimpleEnvironment:
    """Tests for SimpleEnvironment convenience class."""

    def test_simple_environment_uses_none_metadata(self):
        """Test that SimpleEnvironment uses None for metadata."""

        class MyEnv(SimpleEnvironment):
            def score(self, prompt: str, response: str) -> float:
                return 1.0

        env = MyEnv()
        result = env.step(
            [[{"role": "user", "content": "test"}, {"role": "assistant", "content": "resp"}]],
            [None],
        )

        assert result.metadata == [None]


class TestStatefulEnvironment:
    """Tests for StatefulEnvironment convenience class."""

    def test_stateful_environment_uses_dict_metadata(self):
        """Test that StatefulEnvironment uses dict for metadata."""

        class GameEnv(StatefulEnvironment):
            def score(self, prompt: str, response: str) -> float:
                return 1.0

        env = GameEnv()
        metadata = [{"turn": 1, "score": 0}]
        result = env.step(
            [[{"role": "user", "content": "test"}, {"role": "assistant", "content": "resp"}]],
            metadata,
        )

        # Metadata is passed through
        assert result.metadata == metadata


class TestExtractPromptResponse:
    """Tests for prompt/response extraction from message logs."""

    def test_extract_single_turn(self):
        """Test extraction from single-turn conversation."""

        class TestEnv(SimpleEnvironment):
            def score(self, prompt: str, response: str) -> float:
                return 1.0

        env = TestEnv()
        message_log = [
            {"role": "user", "content": "What is 2+2?"},
            {"role": "assistant", "content": "The answer is 4"},
        ]

        prompt, response = env._extract_prompt_response(message_log)

        assert prompt == "What is 2+2?"
        assert response == "The answer is 4"

    def test_extract_multi_turn(self):
        """Test extraction from multi-turn conversation."""

        class TestEnv(SimpleEnvironment):
            def score(self, prompt: str, response: str) -> float:
                return 1.0

        env = TestEnv()
        message_log = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there!"},
            {"role": "user", "content": "How are you?"},
            {"role": "assistant", "content": "I'm doing well"},
        ]

        prompt, response = env._extract_prompt_response(message_log)

        assert prompt == "Hello\nHow are you?"
        assert response == "Hi there!\nI'm doing well"

    def test_extract_with_system_message(self):
        """Test extraction ignores system messages."""

        class TestEnv(SimpleEnvironment):
            def score(self, prompt: str, response: str) -> float:
                return 1.0

        env = TestEnv()
        message_log = [
            {"role": "system", "content": "You are a helpful assistant"},
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there!"},
        ]

        prompt, response = env._extract_prompt_response(message_log)

        # System message is ignored
        assert prompt == "Hello"
        assert response == "Hi there!"


class TestAsyncSupport:
    """Tests for async reward computation."""

    def test_async_score_default_calls_sync(self):
        """Test that async score defaults to sync version."""

        class SyncEnv(SimpleEnvironment):
            def score(self, prompt: str, response: str) -> float:
                return len(response) / 100.0

        env = SyncEnv()

        # Run async version
        loop = asyncio.new_event_loop()
        try:
            result = loop.run_until_complete(
                env.score_async("prompt", "test response")
            )
        finally:
            loop.close()

        assert result == pytest.approx(0.13)


class TestGlobalPostProcessAndMetrics:
    """Tests for global_post_process_and_metrics()."""

    def test_computes_reward_statistics(self):
        """Test that reward statistics are computed."""

        class TestEnv(SimpleEnvironment):
            def score(self, prompt: str, response: str) -> float:
                return 1.0

        env = TestEnv()
        batch = {"rewards": torch.tensor([0.0, 0.5, 1.0])}

        processed_batch, metrics = env.global_post_process_and_metrics(batch)

        assert processed_batch is batch
        assert metrics["mean_reward"] == pytest.approx(0.5)
        assert metrics["min_reward"] == pytest.approx(0.0)
        assert metrics["max_reward"] == pytest.approx(1.0)
        assert "std_reward" in metrics

    def test_handles_empty_batch(self):
        """Test handling of batch without rewards."""

        class TestEnv(SimpleEnvironment):
            def score(self, prompt: str, response: str) -> float:
                return 1.0

        env = TestEnv()
        batch = {}

        _, metrics = env.global_post_process_and_metrics(batch)

        assert "environment_name" in metrics
        assert "mean_reward" not in metrics


class TestEnvironmentIntegration:
    """Integration tests for Environment with trainers."""

    def test_environment_implements_interface(self):
        """Test that Environment properly implements EnvironmentInterface."""
        from nemo_rl.environments.interfaces import EnvironmentInterface

        class TestEnv(SimpleEnvironment):
            def score(self, prompt: str, response: str) -> float:
                return 1.0

        env = TestEnv()

        # Check it implements the interface
        assert isinstance(env, EnvironmentInterface)
        assert hasattr(env, "step")
        assert hasattr(env, "global_post_process_and_metrics")

    def test_environment_used_as_reward_wrapper_replacement(self):
        """Test Environment can be used where FunctionalRewardWrapper was used."""

        class CorrectnessEnv(SimpleEnvironment):
            def __init__(self, correct_answers: Dict[str, str]):
                super().__init__(name="correctness")
                self.correct_answers = correct_answers

            def score(self, prompt: str, response: str) -> float:
                # Simple check if response contains expected answer
                for key, answer in self.correct_answers.items():
                    if key in prompt and answer.lower() in response.lower():
                        return 1.0
                return 0.0

        env = CorrectnessEnv({"q1": "4", "q2": "paris"})

        message_logs = [
            [
                {"role": "user", "content": "q1: What is 2+2?"},
                {"role": "assistant", "content": "The answer is 4"},
            ],
            [
                {"role": "user", "content": "q2: Capital of France?"},
                {"role": "assistant", "content": "It's London"},
            ],
        ]

        result = env.step(message_logs, [None, None])

        assert result.rewards[0].item() == 1.0  # Correct
        assert result.rewards[1].item() == 0.0  # Incorrect
