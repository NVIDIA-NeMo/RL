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

"""async_rl.gym_results_log: config, writer, and the RL-side outcome records."""

import asyncio
import json
import os
import pickle
from pathlib import Path
from types import SimpleNamespace

import pytest

from nemo_rl.algorithms.single_controller_utils.config import GymResultsLogConfig
from nemo_rl.algorithms.single_controller_utils.setup import resolve_gym_results_log
from nemo_rl.environments.nemo_gym import NemoGym
from nemo_rl.utils.gym_results_log import (
    GYM_RESULTS_PREFIX,
    REQUEST_INPUT_REF,
    RESPONSE_OUTPUT_REF,
    RL_OUTCOMES_PREFIX,
    TOOLS_REF,
    JsonlAppender,
    compact_gym_result,
    gym_results_dir,
)


def _lines(appender: JsonlAppender) -> list[dict]:
    appender.flush()
    return [json.loads(line) for line in Path(appender.path).read_text().splitlines()]


def test_appender_snapshots_the_record_when_appended(tmp_path):
    appender = JsonlAppender(str(tmp_path / "gym_results"), GYM_RESULTS_PREFIX)
    record = {"full_result": {"response": {"output": [{"token_ids": [1, 2]}]}}}

    appender.append(record)
    # RL post-processing pops token arrays in place right after logging.
    record["full_result"]["response"]["output"][0].pop("token_ids")

    assert _lines(appender) == [
        {"full_result": {"response": {"output": [{"token_ids": [1, 2]}]}}}
    ]
    assert appender.path == str(
        tmp_path / "gym_results" / f"{GYM_RESULTS_PREFIX}.{os.getpid()}.jsonl"
    )
    appender.close()


def test_appender_survives_being_shipped_to_another_process(tmp_path):
    # The replay buffer holding the outcome appender is built in the driver and
    # pickled into the controller actor's args.
    appender = JsonlAppender(str(tmp_path), RL_OUTCOMES_PREFIX)
    appender.append({"outcome": "buffered"})
    appender.flush()

    shipped = pickle.loads(pickle.dumps(appender))
    shipped.append({"outcome": "evicted"})

    assert [line["outcome"] for line in _lines(shipped)] == ["buffered", "evicted"]
    appender.close()
    shipped.close()


def test_config_rejects_deleting_the_only_copy_of_the_bodies():
    with pytest.raises(ValueError, match="only copy"):
        GymResultsLogConfig(include_payloads=False, delete_capture_after_merge=True)
    GymResultsLogConfig(include_payloads=False, delete_capture_after_merge=False)


def _master_config(tmp_path, *, use_gym=True, **gym_env):
    return SimpleNamespace(
        env={"should_use_nemo_gym": use_gym, "nemo_gym": dict(gym_env)},
        policy={
            "generation": {
                "backend": "vllm",
                "vllm_cfg": {"async_engine": True, "expose_http_server": True},
            }
        },
        async_rl=SimpleNamespace(gym_results_log=GymResultsLogConfig(enabled=True)),
        logger=SimpleNamespace(log_dir=str(tmp_path / "logs")),
    )


def test_resolve_is_none_when_disabled(tmp_path):
    cfg = _master_config(tmp_path)
    cfg.async_rl.gym_results_log = GymResultsLogConfig()
    assert resolve_gym_results_log(cfg) is None


def test_resolve_requires_a_gym_run(tmp_path):
    with pytest.raises(ValueError, match="does not use NeMo-Gym"):
        resolve_gym_results_log(_master_config(tmp_path, use_gym=False))


@pytest.mark.parametrize(
    "gym_env",
    [
        {},
        {"observability_enabled": True},
        {"observability_enabled": True, "model_call_capture_dir": "relative/dir"},
        {"model_call_capture_dir": "/abs/dir"},
    ],
)
def test_resolve_requires_gym_model_call_capture(tmp_path, gym_env):
    with pytest.raises(ValueError, match="model_call_capture_dir"):
        resolve_gym_results_log(_master_config(tmp_path, **gym_env))


def test_resolve_returns_the_shared_directory_and_settings(tmp_path):
    cfg = _master_config(
        tmp_path, observability_enabled=True, model_call_capture_dir="/abs/calls"
    )
    assert resolve_gym_results_log(cfg) == {
        "dir": gym_results_dir(str(tmp_path / "logs")),
        "enabled": True,
        "include_model_calls": True,
        "include_payloads": True,
        "delete_capture_after_merge": True,
        "compact": True,
    }
    # Model calls off: no Gym capture needed.
    cfg = _master_config(tmp_path)
    cfg.async_rl.gym_results_log = GymResultsLogConfig(
        enabled=True, include_model_calls=False
    )
    assert resolve_gym_results_log(cfg)["include_model_calls"] is False


class _StreamingSelf:
    """Just enough NemoGym for ``_stream_rollouts`` with the log on."""

    _stream_rollouts = NemoGym.__ray_metadata__.modified_class._stream_rollouts
    _log_gym_result = NemoGym.__ray_metadata__.modified_class._log_gym_result
    _log_gym_failure = NemoGym.__ray_metadata__.modified_class._log_gym_failure
    _resolve_model_call_capture = (
        NemoGym.__ray_metadata__.modified_class._resolve_model_call_capture
    )
    _clear_model_call_captures = (
        NemoGym.__ray_metadata__.modified_class._clear_model_call_captures
    )

    def __init__(self, tmp_path, run_examples, postprocess):
        self.cfg = {}
        self.head_server_config = object()
        self._token_capture_enabled = False
        self._tokenizer = object()
        self._gym_results_cfg = {
            "dir": str(tmp_path),
            "include_model_calls": False,
            "include_payloads": True,
            "delete_capture_after_merge": True,
            "compact": True,
        }
        self._gym_results_log = JsonlAppender(str(tmp_path), GYM_RESULTS_PREFIX)
        self._gym_capture_dirs = None
        self._gym_observability_enabled = False
        self.rch = SimpleNamespace(run_examples=run_examples)
        self._postprocess = postprocess

    def _require_spinup(self):
        pass

    def _postprocess_nemo_gym_to_nemo_rl_result(
        self, row, result, tokenizer, *, include_initial_multimodal_data
    ):
        return self._postprocess(row, result)


def _row(rowidx: int) -> dict:
    return {
        "_rowidx": rowidx,
        "agent_ref": {"name": "math_agent"},
        "responses_create_params": {"input": []},
        "_ng_group_id": "group",
        "_ng_group_attempt": 1,
        "_ng_rollout_index": rowidx,
        "_ng_rollout_id": f"group_g{rowidx}_a1",
        "_ng_dispatch_weight_version": 4,
    }


async def _drain(gym, rows):
    return [item async for item in gym._stream_rollouts(rows, "timing/test")]


def test_results_are_logged_raw_and_failures_name_their_rollouts(tmp_path):
    rows = [_row(0), _row(1)]
    raw = {
        "reward": 1.0,
        "response": {"output": [{"generation_token_ids": [5, 6]}]},
    }

    def run_examples(examples, head_server_config):
        async def ok():
            return examples[0], raw

        async def failed():
            raise RuntimeError("agent server crashed")

        return [ok(), failed()]

    def postprocess(row, result):
        # The legacy token walk pops the arrays it consumes.
        result["response"]["output"][0].pop("generation_token_ids")
        return {"message_log": []}

    gym = _StreamingSelf(tmp_path, run_examples, postprocess)
    with pytest.raises(RuntimeError, match="agent server crashed"):
        asyncio.run(_drain(gym, rows))

    result_line, error_line = _lines(gym._gym_results_log)
    assert result_line["event"] == "result"
    assert result_line["rollout_id"] == "group_g0_a1"
    assert result_line["group_id"] == "group"
    assert result_line["group_attempt"] == 1
    assert result_line["rollout_index"] == 0
    assert result_line["agent"] == "math_agent"
    assert result_line["dispatch_weight_version"] == 4
    assert result_line["token_capture"] is False
    # Logged before post-processing removed the arrays.
    assert result_line["full_result"]["response"]["output"] == [
        {"generation_token_ids": [5, 6]}
    ]
    assert error_line["event"] == "error"
    assert error_line["stage"] == "run"
    assert [r["rollout_id"] for r in error_line["rollouts"]] == ["group_g1_a1"]


def test_a_result_rl_cannot_postprocess_is_logged_then_raised(tmp_path):
    def run_examples(examples, head_server_config):
        async def ok():
            return examples[0], {"reward": 0.0, "response": {"output": []}}

        return [ok()]

    def postprocess(row, result):
        raise ValueError("NeMo Gym returned a result with no generation data")

    gym = _StreamingSelf(tmp_path, run_examples, postprocess)
    with pytest.raises(ValueError, match="no generation data"):
        asyncio.run(_drain(gym, [_row(0)]))

    result_line, error_line = _lines(gym._gym_results_log)
    assert result_line["event"] == "result"
    assert error_line["stage"] == "postprocess"
    assert error_line["error_type"] == "ValueError"
    assert [r["rollout_id"] for r in error_line["rollouts"]] == ["group_g0_a1"]


def test_merged_model_calls_stay_out_of_the_result_rl_receives(tmp_path):
    rollout_collection = pytest.importorskip("nemo_gym.rollout_collection")
    if not hasattr(rollout_collection, "enrich_run_result"):
        pytest.skip("this NeMo-Gym predates enrich_run_result")
    from nemo_gym.base_responses_api_model import CaptureStore

    capture_dir = tmp_path / "calls"
    # Left over from an earlier dispatch of the same rollout id: cleared first.
    CaptureStore(capture_dir).record(
        "group_g0_a1", {"model_call_id": "stale", "dialect": "responses"}
    )
    raw = {"reward": 1.0, "response": {"output": []}}
    received = []

    def run_examples(examples, head_server_config):
        async def ok():
            # The model server records the call while the rollout runs.
            CaptureStore(capture_dir).record(
                "group_g0_a1",
                {
                    "model_call_id": "c1",
                    "dialect": "responses",
                    "request": {"input": [{"role": "user", "content": "q"}]},
                    "response": {"id": "r1", "output": [], "usage": {}},
                    "status_code": 200,
                },
            )
            return examples[0], raw

        return [ok()]

    def postprocess(row, result):
        received.append(dict(result))
        return {"message_log": []}

    gym = _StreamingSelf(tmp_path / "logs", run_examples, postprocess)
    gym._gym_results_cfg["include_model_calls"] = True
    gym.cfg = {
        "initial_global_config_dict": {
            "observability_enabled": True,
            "model_call_capture_dir": str(capture_dir),
        }
    }
    asyncio.run(_drain(gym, [_row(0)]))

    [line] = _lines(gym._gym_results_log)
    [call] = line["full_result"]["ng_trajectory"]["model_calls"]
    assert call["model_call_id"] == "c1"
    assert call["request"]["input"][0]["content"] == "q"
    # The trainer-bound result is untouched, and the merged file is gone.
    assert received == [{"reward": 1.0, "response": {"output": []}}]
    assert not CaptureStore(capture_dir).path_for("group_g0_a1").exists()


def test_outcome_appender_names_its_file_per_process(tmp_path):
    appender = JsonlAppender(gym_results_dir(str(tmp_path)), RL_OUTCOMES_PREFIX)
    appender.append({"outcome": "trained"})
    appender.flush()
    assert Path(appender.path).name == f"{RL_OUTCOMES_PREFIX}.{os.getpid()}.jsonl"
    appender.close()


def _enriched_three_step_result() -> dict:
    tools = [{"type": "function", "name": "search", "parameters": {}}]
    system = {"role": "system", "content": "be brief"}
    user = {"role": "user", "content": "find it"}
    call_1 = {
        "type": "function_call",
        "call_id": "t1",
        "name": "search",
        "arguments": "{}",
    }
    out_1 = {"type": "function_call_output", "call_id": "t1", "output": "found"}
    answer = {
        "type": "message",
        "role": "assistant",
        "content": [{"type": "output_text", "text": "done"}],
    }
    return {
        "_ng_rollout_id": "g_g0_a0",
        "_ng_task_index": 3,
        "responses_create_params": {
            "input": [
                {**system, "type": "message", "phase": None},
                {**user, "type": "message"},
            ],
            "tools": tools,
            "temperature": 1.0,
            "metadata": None,
        },
        "response": {
            "id": "r2",
            "output": [call_1, out_1, answer],
            "tools": tools,
            "usage": None,
        },
        "reward": 1.0,
        "ng_trajectory": {
            "model_calls": [
                {
                    "model_call_id": "c1",
                    "duration_ms": 10.0,
                    "token_stats": {"prompt_tokens": 5, "cached_tokens": None},
                    # Gym's request models omit the default message type and nulls.
                    "request": {"input": [system, user], "tools": tools},
                    "response": {"id": "r1", "output": [call_1], "tools": tools},
                },
                {
                    "model_call_id": "c2",
                    "duration_ms": 12.0,
                    "request": {"input": [system, user, call_1, out_1], "tools": tools},
                    "response": {"id": "r2", "output": [answer], "tools": tools},
                },
                {
                    "model_call_id": "judge",
                    "request": {"input": [{"role": "user", "content": "grade it"}]},
                    "response": {"output": [{"type": "message", "content": "A"}]},
                },
            ]
        },
        "ng_model_call_capture": {
            "rollout_id": "g_g0_a0",
            "metrics": {"num_calls": 3},
            "calls": [
                {"model_call_id": "c1"},
                {"model_call_id": "c2"},
                {"model_call_id": "judge"},
            ],
        },
    }


def test_compaction_keeps_every_call_and_only_references_identical_fields():
    result = _enriched_three_step_result()
    before = json.dumps(result, sort_keys=True)

    compact = compact_gym_result(
        result,
        added_keys={
            "_ng_rollout_id",
            "_ng_task_index",
            "ng_trajectory",
            "ng_model_call_capture",
        },
    )

    # The input is shared with what RL post-processes: never mutated.
    assert json.dumps(result, sort_keys=True) == before
    assert "_ng_rollout_id" not in compact and "_ng_task_index" not in compact
    assert compact["ng_model_call_capture"] == {
        "rollout_id": "g_g0_a0",
        "metrics": {"num_calls": 3},
    }
    assert "metadata" not in compact["responses_create_params"]
    assert "usage" not in compact["response"]
    assert compact["response"]["tools"] == TOOLS_REF
    c1, c2, judge = compact["ng_trajectory"]["model_calls"]
    assert [c["model_call_id"] for c in (c1, c2, judge)] == ["c1", "c2", "judge"]
    # The first request is the agent's prompt; every request repeats the tools.
    assert c1["request"] == {"input": REQUEST_INPUT_REF, "tools": TOOLS_REF}
    assert c2["request"]["tools"] == TOOLS_REF
    assert len(c2["request"]["input"]) == 4
    # No single call's output is the whole multi-turn response: bodies stay.
    assert c1["response"]["output"][0]["call_id"] == "t1"
    assert c2["response"]["output"][0]["type"] == "message"
    assert judge["request"]["input"][0]["content"] == "grade it"
    # Metadata stays; only null fields go.
    assert c1["duration_ms"] == 10.0 and c1["token_stats"] == {"prompt_tokens": 5}


def test_compaction_references_a_single_call_that_is_the_whole_response():
    result = _enriched_three_step_result()
    result["response"]["output"] = result["ng_trajectory"]["model_calls"][1][
        "response"
    ]["output"]

    compact = compact_gym_result(result, added_keys=())

    assert (
        compact["ng_trajectory"]["model_calls"][1]["response"]["output"]
        == RESPONSE_OUTPUT_REF
    )
    assert (
        compact["ng_trajectory"]["model_calls"][0]["response"]["output"][0]["call_id"]
        == "t1"
    )
