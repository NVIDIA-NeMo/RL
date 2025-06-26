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
import time

import pytest
import ray

from nemo_rl.distributed.ray_actor_environment_registry import (
    get_actor_python_env,
)
from nemo_rl.environments.math_environment import MathEnvironment


@pytest.fixture(scope="module")
def math_env():
    """Create a MathEnvironment actor for testing."""
    env = MathEnvironment.options(
        runtime_env={
            "py_executable": get_actor_python_env(
                "nemo_rl.environments.math_environment.MathEnvironment"
            ),
            "env_vars": dict(os.environ),
        }
    ).remote({"num_workers": 2})
    yield env
    # Clean up the actor and wait for it to be killed
    env.shutdown.remote()
    ray.kill(env)
    # Give some time for cleanup
    time.sleep(0.1)


@pytest.fixture
def basic_test_data():
    """Common test data for basic math problems."""
    return {
        "message_log_batch": [
            [
                {"role": "user", "content": "What is 2 + 2?"},
                {"role": "assistant", "content": "2 + 2 = \\boxed{4}"},
            ],
            [
                {"role": "user", "content": "What is 3 * 4?"},
                {"role": "assistant", "content": "3 * 4 = \\boxed{12}"},
            ],
            [
                {"role": "user", "content": "What is 10 - 5?"},
                {"role": "assistant", "content": "10 - 5 = \\boxed{5}"},
            ],
        ],
        "metadata": [
            {"ground_truth": "4"},
            {"ground_truth": "\\boxed{12}"},
            {"ground_truth": "\\boxed{5}"},
        ],
    }


@pytest.fixture
def mixed_test_data():
    """Test data with mix of correct and incorrect responses."""
    return {
        "message_log_batch": [
            [
                {"role": "user", "content": "What is 2 + 2?"},
                {"role": "assistant", "content": "2 + 2 = \\boxed{\\frac{8}{2}}"},
            ],
            [
                {"role": "user", "content": "What is 3 * 4?"},
                {"role": "assistant", "content": "3 * 4 = 13"},
            ],
            [
                {"role": "user", "content": "What is 10 - 5?"},
                {"role": "assistant", "content": "10 - 5 = \\boxed{5}"},
            ],
        ],
        "metadata": [
            {"ground_truth": "4.0"},
            {"ground_truth": "\\boxed{12}"},
            {"ground_truth": "\\boxed{5}"},
        ],
    }


@pytest.fixture
def multiple_assistant_test_data():
    """Test data with multiple assistant messages in conversations."""
    return {
        "message_log_batch": [
            [
                {"role": "user", "content": "What is 2 + 2?"},
                {"role": "assistant", "content": "Let me think..."},
                {"role": "assistant", "content": "2 + 2 = \\boxed{4}"},
            ],
            [
                {"role": "user", "content": "What is 3 * 4?"},
                {"role": "assistant", "content": "I'll calculate that..."},
                {"role": "assistant", "content": "3 * 4 = \\boxed{12}"},
            ],
        ],
        "metadata": [{"ground_truth": "4"}, {"ground_truth": "\\boxed{12}"}],
    }


def test_math_env_step_basic(math_env, basic_test_data):
    """Test basic functionality of MathEnvironment step with simple messages."""
    result = ray.get(
        math_env.step.remote(
            basic_test_data["message_log_batch"], basic_test_data["metadata"]
        )
    )

    # Check observations using field access
    assert len(result.observations) == 3, (
        "Should return observations for all 3 messages"
    )
    assert all(obs["role"] == "environment" for obs in result.observations), (
        "All observations should be from environment"
    )
    assert all(
        obs["content"] == "Environment: correct" for obs in result.observations
    ), "All responses should be correct"

    # Check metadata
    assert len(result.metadata) == 3, "Should return metadata for all 3 messages"
    assert result.metadata == basic_test_data["metadata"], (
        "Metadata should be unchanged"
    )

    # Check rewards and done flags
    assert result.rewards.shape == (3,), "Rewards should be a tensor of shape (3,)"
    assert all(result.rewards == 1.0), "All rewards should be 1.0 for correct answers"
    assert result.terminateds.shape == (3,), (
        "Terminated flags should be a tensor of shape (3,)"
    )
    assert all(result.terminateds == 1.0), "All terminated flags should be 1.0"


def test_math_env_step_mixed(math_env, mixed_test_data):
    """Test MathEnvironment step with a mix of correct and incorrect responses."""
    result = ray.get(
        math_env.step.remote(
            mixed_test_data["message_log_batch"], mixed_test_data["metadata"]
        )
    )

    # Check observations and rewards
    assert len(result.observations) == 3, (
        "Should return observations for all 3 messages"
    )
    assert result.observations[0]["content"] == "Environment: correct", (
        "First response should be correct"
    )
    assert result.observations[1]["content"] == "Environment: incorrect", (
        "Second response should be incorrect"
    )
    assert result.observations[2]["content"] == "Environment: correct", (
        "Third response should be correct"
    )

    assert result.rewards.shape == (3,), "Rewards should be a tensor of shape (3,)"
    assert result.rewards[0] == 1.0, "First reward should be 1.0"
    assert result.rewards[1] == 0.0, "Second reward should be 0.0"
    assert result.rewards[2] == 1.0, "Third reward should be 1.0"


def test_math_env_step_empty(math_env):
    """Test MathEnvironment step with empty input."""
    result = ray.get(math_env.step.remote([], []))

    # Check all outputs are empty
    assert len(result.observations) == 0, "Should return empty observations list"
    assert len(result.metadata) == 0, "Should return empty metadata list"
    assert result.rewards.shape == (0,), "Should return empty rewards tensor"
    assert result.terminateds.shape == (0,), "Should return empty terminateds tensor"


def test_math_env_step_multiple_assistant_messages(
    math_env, multiple_assistant_test_data
):
    """Test MathEnvironment step with multiple assistant messages in a conversation."""
    result = ray.get(
        math_env.step.remote(
            multiple_assistant_test_data["message_log_batch"],
            multiple_assistant_test_data["metadata"],
        )
    )

    # Check that only the last assistant message is used
    assert len(result.observations) == 2, (
        "Should return observations for both conversations"
    )
    assert all(
        obs["content"] == "Environment: correct" for obs in result.observations
    ), "All responses should be correct"
    assert all(result.rewards == 1.0), "All rewards should be 1.0"


@pytest.mark.parametrize("batch_size", [1, 2, 10, 25, 101])
def test_math_env_various_batches(math_env, batch_size):
    """Test MathEnvironment step with different batch sizes."""
    message_log_batch = [
        [
            {"role": "user", "content": "What is 2 + 2?"},
            {"role": "assistant", "content": "2 + 1.333 = \\boxed{\\frac{10}{3}}"},
        ]
    ] * batch_size
    metadata = [{"ground_truth": "3.33333333"}] * batch_size

    result = ray.get(math_env.step.remote(message_log_batch, metadata))

    # Check outputs
    assert len(result.observations) == batch_size, (
        f"Should return observations for all {batch_size} messages"
    )
    assert all(
        obs["content"] == "Environment: correct" for obs in result.observations
    ), "All responses should be correct"
    assert result.rewards.shape == (batch_size,), (
        "Rewards should be a tensor of shape (batch_size,)"
    )
    assert all(result.rewards == 1.0), "All rewards should be 1.0"
    assert result.terminateds.shape == (batch_size,), (
        "Terminated flags should be a tensor of shape (batch_size,)"
    )
    assert all(result.terminateds == 1.0), "All terminated flags should be 1.0"


def test_math_exception_handling(math_env):
    """Test MathEnvironment step with an exception in the verify function."""
    message_log_batch = [
        [
            {"role": "user", "content": "Question"},
            {"role": "assistant", "content": "\\boxed{Eq(x**2/16 + y**2/12, 1)}"},
        ]
    ]
    metadata = [{"ground_truth": "Eq(x**2/4 + y**2/3, 1)"}]

    result = ray.get(math_env.step.remote(message_log_batch, metadata))

    # Program should not crash
    assert result.rewards.shape == (1,), "Rewards should be a tensor of shape (1,)"
    assert result.rewards[0] == 0.0, "Reward should be 0.0"


def test_math_timeout_handling(math_env):
    """Test MathEnvironment step with content that causes TimeoutException."""
    # This content contains complex symbolic math that causes sympy to timeout
    timeout_content = r"""We are given that \(x = (2 + \sqrt{3})^{1000}\), so \(n = \lfloor x \rfloor\), and \(f = x - n\). We want to find the value of \(x(1 - f)\).

First, let's examine the binomial expansion of \(x = (2 + \sqrt{3})^{1000}\). Notice that:
\[
(2 + \sqrt{3}) = 2 + \sqrt{3} = \left(2 - (\sqrt{3})\right)^{-1}
\]

Using the binomial theorem, we can expand \(x\):
\[
x = \sum_{k=0}^{1000} \binom{1000}{k} (2)^{1000-k} (\sqrt{3})^k
\]

Notice that \((2 - \sqrt{3}) = -1\), so we can use this to our advantage. We can rewrite \(x - 1\) as:
\[
x - 1 = (2 + \sqrt{3})^{1000} - 1 = \sum_{k=0}^{1000} \binom{1000}{k} (2)^{1000-k} (\sqrt{3})^k - 1
\]

Now, let's consider \(f = x - n = (2 + \sqrt{3})^{1000} - \lfloor (2 + \sqrt{3})^{10"""

    message_log_batch = [
        [
            {"role": "user", "content": "Solve this complex problem"},
            {"role": "assistant", "content": timeout_content},
        ]
    ]
    metadata = [{"ground_truth": "13"}]

    # This should complete without hanging, even though it contains timeout-inducing content
    result = ray.get(math_env.step.remote(message_log_batch, metadata))

    # Program should not crash and should handle timeout gracefully
    assert result.rewards.shape == (1,), "Rewards should be a tensor of shape (1,)"
    assert result.rewards[0] == 0.0, "Reward should be 0.0 due to timeout"
    assert len(result.observations) == 1, "Should return one observation"
    assert result.observations[0]["role"] == "environment", (
        "Observation should be from environment"
    )
    assert result.observations[0]["content"] == "Environment: incorrect", (
        "Should be marked as incorrect due to timeout"
    )
    assert result.terminateds.shape == (1,), (
        "Terminated flags should be a tensor of shape (1,)"
    )
    assert result.terminateds[0] == 1.0, "Terminated flag should be 1.0"
