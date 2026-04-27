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

import importlib.util
import json
import sys
import types
import unittest.mock as mock

import pytest
import torch


# ---------------------------------------------------------------------------
# Module-level mocks: tau_bench is an optional dependency not available in CI
# ---------------------------------------------------------------------------

def _make_module(name):
    mod = types.ModuleType(name)
    mod.__spec__ = importlib.util.spec_from_loader(name, loader=None)
    return mod


class MockAction:
    def __init__(self, name, kwargs):
        self.name = name
        self.kwargs = kwargs

    def __repr__(self):
        return f"Action(name={self.name!r}, kwargs={self.kwargs!r})"


# Patch tau_bench before any nemo_rl import that would transitively pull it.
_tau_bench = _make_module("tau_bench")
_tau_bench_types = _make_module("tau_bench.types")
_tau_bench_types.Action = MockAction
_tau_bench.types = _tau_bench_types

for _name, _mod in [
    ("tau_bench", _tau_bench),
    ("tau_bench.types", _tau_bench_types),
    ("tau_bench.envs", _make_module("tau_bench.envs")),
]:
    sys.modules.setdefault(_name, _mod)

# decord is an optional multimedia dependency; stub it out so transformers
# can discover it via importlib.util.find_spec without crashing.
if "decord" not in sys.modules:
    _decord = _make_module("decord")
    sys.modules["decord"] = _decord

from nemo_rl.distributed.batched_data_dict import BatchedDataDict
from nemo_rl.environments.tau_bench_environment import (
    TauBenchEnvironment,
    TauBenchWorker,
    _TOOL_CALL_RE,
)

# ---------------------------------------------------------------------------
# Helpers: instantiate the underlying (non-Ray) classes for unit testing
# ---------------------------------------------------------------------------

def _make_worker(
    judge_model=None,
    judge_base_url=None,
    judge_api_key="dummy",
    max_steps=30,
):
    """Return a bare TauBenchWorker instance without spawning a Ray actor."""
    cls = TauBenchWorker.__ray_metadata__.modified_class
    worker = cls.__new__(cls)
    worker._env_name = "retail"
    worker._task_split = "test"
    worker._user_strategy = "llm"
    worker._user_model = "dummy"
    worker._max_steps = max_steps
    worker._judge_model = judge_model
    worker._judge_base_url = judge_base_url
    worker._judge_api_key = judge_api_key
    worker._active_envs = {}
    return worker


def _make_env(judge_weight=0.0):
    """Return a bare TauBenchEnvironment instance without spawning Ray actors."""
    cls = TauBenchEnvironment.__ray_metadata__.modified_class
    env = cls.__new__(cls)
    env.cfg = {}
    env._num_workers = 1
    env._judge_weight = judge_weight
    env._workers = []
    return env


# ===========================================================================
# Tests: _TOOL_CALL_RE regex
# ===========================================================================


class TestToolCallRegex:
    def test_matches_simple_tool_call(self):
        text = '<tool_call>{"name": "foo"}</tool_call>'
        m = _TOOL_CALL_RE.search(text)
        assert m is not None
        assert m.group(1).strip() == '{"name": "foo"}'

    def test_matches_multiline_tool_call(self):
        text = "<tool_call>\n  {\"name\": \"bar\"}\n</tool_call>"
        m = _TOOL_CALL_RE.search(text)
        assert m is not None

    def test_no_match_without_tags(self):
        assert _TOOL_CALL_RE.search("plain text response") is None

    def test_matches_first_tag_when_multiple_present(self):
        text = '<tool_call>{"name": "first"}</tool_call> some text <tool_call>{"name": "second"}</tool_call>'
        matches = _TOOL_CALL_RE.findall(text)
        assert len(matches) == 2
        assert "first" in matches[0]


# ===========================================================================
# Tests: TauBenchWorker._parse_action
# ===========================================================================


class TestParseAction:
    @pytest.fixture
    def worker(self):
        return _make_worker()

    def test_valid_tool_call_with_arguments_key(self, worker):
        text = '<tool_call>{"name": "cancel_order", "arguments": {"order_id": "O123"}}</tool_call>'
        action = worker._parse_action(text)
        assert action.name == "cancel_order"
        assert action.kwargs == {"order_id": "O123"}

    def test_valid_tool_call_with_kwargs_key(self, worker):
        text = '<tool_call>{"name": "get_flight", "kwargs": {"flight_id": "F42"}}</tool_call>'
        action = worker._parse_action(text)
        assert action.name == "get_flight"
        assert action.kwargs == {"flight_id": "F42"}

    def test_valid_tool_call_no_args_key(self, worker):
        # Neither 'arguments' nor 'kwargs' — should default to empty dict
        text = '<tool_call>{"name": "list_orders"}</tool_call>'
        action = worker._parse_action(text)
        assert action.name == "list_orders"
        assert action.kwargs == {}

    def test_malformed_json_falls_back_to_respond(self, worker):
        text = "<tool_call>not valid json{{{</tool_call>"
        action = worker._parse_action(text)
        assert action.name == "respond"

    def test_no_tool_call_returns_respond(self, worker):
        text = "I'm sorry, I cannot help with that request."
        action = worker._parse_action(text)
        assert action.name == "respond"
        assert action.kwargs == {"content": text}

    def test_empty_string_returns_respond(self, worker):
        action = worker._parse_action("")
        assert action.name == "respond"
        assert action.kwargs["content"] == ""

    def test_tool_call_with_leading_trailing_text(self, worker):
        text = "Okay, I will cancel it now. <tool_call>{'name': 'cancel_order', 'arguments': {}}</tool_call> Done."
        # Single-quoted JSON is invalid; should fall back to respond
        action = worker._parse_action(text)
        assert action.name == "respond"

    def test_tool_call_whitespace_stripped(self, worker):
        text = "<tool_call>  \n  {\"name\": \"find_user\", \"arguments\": {\"id\": 1}}  \n  </tool_call>"
        action = worker._parse_action(text)
        assert action.name == "find_user"
        assert action.kwargs == {"id": 1}


# ===========================================================================
# Tests: TauBenchWorker._call_judge
# ===========================================================================


class TestCallJudge:
    @pytest.fixture
    def worker(self):
        return _make_worker(
            judge_model="gpt-4o",
            judge_base_url="https://api.example.com",
            judge_api_key="test-key",
        )

    def _make_response(self, score):
        resp = mock.MagicMock()
        resp.json.return_value = {
            "choices": [
                {"message": {"content": json.dumps({"score": score, "reasoning": "good"})}}
            ]
        }
        resp.raise_for_status = mock.MagicMock()
        return resp

    def test_successful_judge_call_returns_score(self, worker):
        with mock.patch("requests.post", return_value=self._make_response(0.9)) as mock_post:
            score = worker._call_judge(
                [{"role": "user", "content": "hello"}],
                domain_rules="Be helpful.",
                task_instruction="Cancel my order.",
            )
        assert score == pytest.approx(0.9)
        mock_post.assert_called_once()

    def test_judge_uses_correct_url(self, worker):
        with mock.patch("requests.post", return_value=self._make_response(0.5)) as mock_post:
            worker._call_judge([], domain_rules="", task_instruction="")
        url = mock_post.call_args[0][0]
        assert url == "https://api.example.com/v1/chat/completions"

    def test_judge_default_url_when_no_base_url(self):
        worker = _make_worker(judge_model="gpt-4o", judge_base_url=None)
        with mock.patch("requests.post", return_value=self._make_response(0.7)) as mock_post:
            score = worker._call_judge([], domain_rules="", task_instruction="")
        url = mock_post.call_args[0][0]
        assert "api.openai.com" in url
        assert score == pytest.approx(0.7)

    def test_http_error_returns_zero(self, worker):
        resp = mock.MagicMock()
        resp.raise_for_status.side_effect = Exception("HTTP 500")
        with mock.patch("requests.post", return_value=resp):
            score = worker._call_judge([], domain_rules="", task_instruction="")
        assert score == 0.0

    def test_network_exception_returns_zero(self, worker):
        with mock.patch("requests.post", side_effect=ConnectionError("timeout")):
            score = worker._call_judge([], domain_rules="", task_instruction="")
        assert score == 0.0

    def test_invalid_json_in_response_returns_zero(self, worker):
        resp = mock.MagicMock()
        resp.raise_for_status = mock.MagicMock()
        resp.json.return_value = {
            "choices": [{"message": {"content": "not json at all"}}]
        }
        with mock.patch("requests.post", return_value=resp):
            score = worker._call_judge([], domain_rules="", task_instruction="")
        assert score == 0.0

    def test_missing_score_key_returns_zero(self, worker):
        resp = mock.MagicMock()
        resp.raise_for_status = mock.MagicMock()
        resp.json.return_value = {
            "choices": [{"message": {"content": json.dumps({"reasoning": "looks good"})}}]
        }
        with mock.patch("requests.post", return_value=resp):
            score = worker._call_judge([], domain_rules="", task_instruction="")
        assert score == 0.0

    def test_score_boundary_values(self, worker):
        for expected in (0.0, 1.0):
            with mock.patch("requests.post", return_value=self._make_response(expected)):
                score = worker._call_judge([], domain_rules="", task_instruction="")
            assert score == pytest.approx(expected)

    def test_conversation_formatted_in_request(self, worker):
        convo = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there"},
        ]
        with mock.patch("requests.post", return_value=self._make_response(0.8)) as mock_post:
            worker._call_judge(convo, domain_rules="Rules.", task_instruction="Task.")
        body = mock_post.call_args[1]["json"]
        user_prompt = body["messages"][1]["content"]
        assert "USER: Hello" in user_prompt
        assert "ASSISTANT: Hi there" in user_prompt
        assert "Rules." in user_prompt
        assert "Task." in user_prompt


# ===========================================================================
# Tests: TauBenchEnvironment.global_post_process_and_metrics
# ===========================================================================


class TestGlobalPostProcessAndMetrics:
    def _batch(self, rewards, is_end, extra_env_info, text=None):
        n = len(rewards)
        if text is None:
            text = torch.zeros(n, 5, dtype=torch.long)
        return BatchedDataDict(
            {
                "rewards": torch.tensor(rewards, dtype=torch.float32),
                "is_end": torch.tensor(is_end, dtype=torch.float32),
                "text": text,
                "extra_env_info": extra_env_info,
            }
        )

    def test_basic_metrics_computed(self):
        env = _make_env()
        batch = self._batch(
            rewards=[1.0, 0.0],
            is_end=[1, 1],
            extra_env_info=[
                {"tau_reward": 1.0, "judge_score": None},
                {"tau_reward": 0.0, "judge_score": None},
            ],
        )
        _, metrics = env.global_post_process_and_metrics(batch)
        assert "tau_bench/task_completion_rate" in metrics
        assert "tau_bench/pass_at_k" in metrics
        assert "tau_bench/fraction_properly_ended" in metrics

    def test_task_completion_rate_is_reward_mean(self):
        env = _make_env()
        batch = self._batch(
            rewards=[1.0, 0.0, 1.0],
            is_end=[1, 1, 1],
            extra_env_info=[{"tau_reward": 1.0}, {"tau_reward": 0.0}, {"tau_reward": 1.0}],
        )
        _, metrics = env.global_post_process_and_metrics(batch)
        assert metrics["tau_bench/task_completion_rate"] == pytest.approx(2 / 3)

    def test_fraction_properly_ended(self):
        env = _make_env()
        batch = self._batch(
            rewards=[1.0, 1.0, 1.0],
            is_end=[1, 0, 1],
            extra_env_info=[{"tau_reward": 1.0}, {"tau_reward": 1.0}, {"tau_reward": 1.0}],
        )
        _, metrics = env.global_post_process_and_metrics(batch)
        assert metrics["tau_bench/fraction_properly_ended"] == pytest.approx(2 / 3)

    def test_mean_tau_reward_reported_when_present(self):
        env = _make_env()
        batch = self._batch(
            rewards=[0.8, 0.2],
            is_end=[1, 1],
            extra_env_info=[
                {"tau_reward": 0.8, "judge_score": None},
                {"tau_reward": 0.2, "judge_score": None},
            ],
        )
        _, metrics = env.global_post_process_and_metrics(batch)
        assert "tau_bench/mean_tau_reward" in metrics
        assert metrics["tau_bench/mean_tau_reward"] == pytest.approx(0.5)

    def test_mean_judge_score_included_when_present(self):
        env = _make_env(judge_weight=0.3)
        batch = self._batch(
            rewards=[0.7, 0.3],
            is_end=[1, 1],
            extra_env_info=[
                {"tau_reward": 0.7, "judge_score": 0.9},
                {"tau_reward": 0.3, "judge_score": 0.5},
            ],
        )
        _, metrics = env.global_post_process_and_metrics(batch)
        assert "tau_bench/mean_judge_score" in metrics
        assert metrics["tau_bench/mean_judge_score"] == pytest.approx(0.7)

    def test_mean_judge_score_excluded_when_none(self):
        env = _make_env()
        batch = self._batch(
            rewards=[1.0],
            is_end=[1],
            extra_env_info=[{"tau_reward": 1.0, "judge_score": None}],
        )
        _, metrics = env.global_post_process_and_metrics(batch)
        assert "tau_bench/mean_judge_score" not in metrics

    def test_mean_tau_reward_excluded_when_missing(self):
        env = _make_env()
        # extra_env_info entries with no 'tau_reward' key
        batch = self._batch(
            rewards=[1.0],
            is_end=[1],
            extra_env_info=[{"task_index": 0}],
        )
        _, metrics = env.global_post_process_and_metrics(batch)
        assert "tau_bench/mean_tau_reward" not in metrics

    def test_rewards_masked_by_is_end(self):
        env = _make_env()
        # Only the episode that has ended should contribute to completion rate
        batch = self._batch(
            rewards=[1.0, 1.0],
            is_end=[1, 0],
            extra_env_info=[{"tau_reward": 1.0}, {"tau_reward": 1.0}],
        )
        _, metrics = env.global_post_process_and_metrics(batch)
        # rewards * is_end = [1.0, 0.0] → mean = 0.5
        assert metrics["tau_bench/task_completion_rate"] == pytest.approx(0.5)

    def test_original_batch_returned_unchanged(self):
        env = _make_env()
        rewards = torch.tensor([0.5], dtype=torch.float32)
        batch = self._batch(
            rewards=[0.5],
            is_end=[1],
            extra_env_info=[{"tau_reward": 0.5}],
        )
        result_batch, _ = env.global_post_process_and_metrics(batch)
        assert result_batch is batch

    def test_2d_rewards_squeezed(self):
        env = _make_env()
        n = 2
        batch = BatchedDataDict(
            {
                "rewards": torch.tensor([[1.0], [0.0]]),
                "is_end": torch.tensor([1.0, 1.0]),
                "text": torch.zeros(n, 5, dtype=torch.long),
                "extra_env_info": [{"tau_reward": 1.0}, {"tau_reward": 0.0}],
            }
        )
        _, metrics = env.global_post_process_and_metrics(batch)
        assert metrics["tau_bench/task_completion_rate"] == pytest.approx(0.5)

    def test_empty_extra_env_info_skipped(self):
        env = _make_env()
        batch = self._batch(
            rewards=[0.0],
            is_end=[1],
            extra_env_info=[None],
        )
        _, metrics = env.global_post_process_and_metrics(batch)
        assert "tau_bench/mean_tau_reward" not in metrics
        assert "tau_bench/mean_judge_score" not in metrics
