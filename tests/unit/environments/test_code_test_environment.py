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

import pytest
import ray

from nemo_rl.environments.code_test_environment import (
    CodeTestCaseEnvironment,
    CodeTestEnvConfig,
    CodeTestMetadata,
    extract_code,
    run_single_test,
)


cfg: CodeTestEnvConfig = {
    "num_workers": 2,
    "timeout_per_test": 5,
}


@pytest.fixture(scope="function")
def code_test_env():
    env_actor = CodeTestCaseEnvironment.remote(cfg)
    yield env_actor
    ray.kill(env_actor)


class TestExtractCode:
    def test_extract_from_markdown_python(self):
        response = "Here is the solution:\n```python\nprint('hello')\n```"
        assert extract_code(response) == "print('hello')"

    def test_extract_from_markdown_plain(self):
        response = "Solution:\n```\nx = 1\nprint(x)\n```"
        assert extract_code(response) == "x = 1\nprint(x)"

    def test_fallback_to_full_response(self):
        response = "print('hello')"
        assert extract_code(response) == "print('hello')"

    def test_empty_response(self):
        assert extract_code("") == ""

    def test_multiple_code_blocks_takes_first(self):
        response = "```python\nfirst()\n```\ntext\n```python\nsecond()\n```"
        assert extract_code(response) == "first()"


class TestRunSingleTest:
    def test_passing_test(self):
        code = "x = input()\nprint(x)"
        assert run_single_test(code, "hello", "hello", 5) is True

    def test_failing_test(self):
        code = "print('wrong')"
        assert run_single_test(code, "hello", "hello", 5) is False

    def test_timeout(self):
        code = "import time\ntime.sleep(100)"
        assert run_single_test(code, "", "", 1) is False

    def test_runtime_error(self):
        code = "raise ValueError('boom')"
        assert run_single_test(code, "", "anything", 5) is False

    def test_multiline_output(self):
        code = "n = int(input())\nfor i in range(n):\n    print(i)"
        expected = "0\n1\n2"
        assert run_single_test(code, "3", expected, 5) is True

    def test_computation(self):
        code = "a, b = map(int, input().split())\nprint(a + b)"
        assert run_single_test(code, "3 5", "8", 5) is True


class TestCodeTestCaseEnvironment:
    def test_all_tests_pass(self, code_test_env):
        code = "a, b = map(int, input().split())\nprint(a + b)"
        response = f"```python\n{code}\n```"

        message_log_batch = [
            [{"role": "assistant", "content": response}]
        ]
        metadata: list[CodeTestMetadata] = [
            {
                "test_cases": [
                    {"input": "1 2", "expected_output": "3"},
                    {"input": "10 20", "expected_output": "30"},
                ],
                "ground_truth": "",
            }
        ]

        result = ray.get(code_test_env.step.remote(message_log_batch, metadata))
        assert result.rewards[0].item() == 1.0
        assert result.terminateds[0].item() == 1
        assert "passed" in result.observations[0]["content"]

    def test_some_tests_fail(self, code_test_env):
        code = "print(42)"
        response = f"```python\n{code}\n```"

        message_log_batch = [
            [{"role": "assistant", "content": response}]
        ]
        metadata: list[CodeTestMetadata] = [
            {
                "test_cases": [
                    {"input": "", "expected_output": "42"},
                    {"input": "", "expected_output": "99"},
                ],
                "ground_truth": "",
            }
        ]

        result = ray.get(code_test_env.step.remote(message_log_batch, metadata))
        assert result.rewards[0].item() == 0.0
        assert "failed" in result.observations[0]["content"]

    def test_empty_code(self, code_test_env):
        message_log_batch = [
            [{"role": "assistant", "content": ""}]
        ]
        metadata: list[CodeTestMetadata] = [
            {
                "test_cases": [{"input": "1", "expected_output": "1"}],
                "ground_truth": "",
            }
        ]

        result = ray.get(code_test_env.step.remote(message_log_batch, metadata))
        assert result.rewards[0].item() == 0.0

    def test_batch_processing(self, code_test_env):
        good_code = "```python\nprint(int(input()) * 2)\n```"
        bad_code = "```python\nprint('wrong')\n```"

        message_log_batch = [
            [{"role": "assistant", "content": good_code}],
            [{"role": "assistant", "content": bad_code}],
        ]
        metadata: list[CodeTestMetadata] = [
            {
                "test_cases": [{"input": "5", "expected_output": "10"}],
                "ground_truth": "",
            },
            {
                "test_cases": [{"input": "5", "expected_output": "10"}],
                "ground_truth": "",
            },
        ]

        result = ray.get(code_test_env.step.remote(message_log_batch, metadata))
        assert result.rewards[0].item() == 1.0
        assert result.rewards[1].item() == 0.0

    def test_extracted_answer(self, code_test_env):
        code = "print(int(input()) + 1)"
        response = f"```python\n{code}\n```"

        message_log_batch = [
            [{"role": "assistant", "content": response}]
        ]
        metadata: list[CodeTestMetadata] = [
            {
                "test_cases": [{"input": "1", "expected_output": "2"}],
                "ground_truth": "",
            }
        ]

        result = ray.get(
            code_test_env.step.remote(message_log_batch, metadata, True)
        )
        assert result.rewards[0].item() == 1.0
        assert result.answers is not None
        assert result.answers[0] == code

    def test_json_string_test_cases(self, code_test_env):
        """Test that test_cases provided as JSON string are handled correctly."""
        code = "print('hello')"
        response = f"```python\n{code}\n```"

        message_log_batch = [
            [{"role": "assistant", "content": response}]
        ]
        metadata: list[CodeTestMetadata] = [
            {
                "test_cases": '[{"input": "", "expected_output": "hello"}]',
                "ground_truth": "",
            }
        ]

        result = ray.get(code_test_env.step.remote(message_log_batch, metadata))
        assert result.rewards[0].item() == 1.0
