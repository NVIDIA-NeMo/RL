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

"""Tests for RolloutManager.

Two groups:

* TestGenerateAndPushFlow — lightweight unit tests for the reserve→run→commit
  flow in generate_and_push (no Ray/vLLM; fakes for impl + tq_buffer).
* AsyncRollout / AsyncNemoGymRollout tests — vLLM/Ray-backed end-to-end checks
  for the underlying run_rollout paths (AsyncRolloutImpl / AsyncNemoGymRolloutImpl).
"""

from __future__ import annotations

import asyncio
import json
import tempfile
import uuid
from copy import deepcopy

import pytest
import torch

from nemo_rl.algorithms.async_utils.replay_buffer import DataPlaneCheckpointBarrier
from nemo_rl.data.collate_fn import rl_collate_fn
from nemo_rl.data.datasets.response_datasets import NemoGymDataset
from nemo_rl.data.interfaces import DatumSpec
from nemo_rl.data.processors import nemo_gym_data_processor
from nemo_rl.distributed.batched_data_dict import BatchedDataDict
from nemo_rl.experience.interfaces import Completion, PromptGroupRecord
from nemo_rl.experience.rollout_manager import AsyncNemoGymRolloutImpl, RolloutManager
from nemo_rl.experience.rollout_recovery import (
    RolloutAttemptStatus,
    RolloutRecoveryLedger,
)
from nemo_rl.experience.rollouts import (
    run_async_multi_turn_rollout,
    run_async_nemo_gym_rollout,
)
from nemo_rl.utils.timer import Timer

# Fixtures shared with the heavyweight rollout tests.
from tests.unit.environments.test_nemo_gym import (
    cluster,  # noqa: F401
    nemo_gym,  # noqa: F401
    nemo_gym_sanity_test_data,  # noqa: F401
    nemo_gym_tokenizer,  # noqa: F401
    nemo_gym_vllm_generation,  # noqa: F401
)
from tests.unit.experience.test_rollouts import (
    initial_multi_step_calculator_batch,  # noqa: F401
    multi_step_calculator_environment,  # noqa: F401
    multi_step_setup_vllm_async,  # noqa: F401
    rollout_cluster,  # noqa: F401
    rollout_tokenizer,  # noqa: F401
)
from tests.unit.test_envs import MultiStepCalcMetadata


def _run(coro):
    return asyncio.run(coro)


class _StreamingGymEnv:
    """Ray-style streaming actor stub that yields results out of order."""

    def __init__(self, results):
        self._results = results
        self.inputs = None
        self.run_rollouts = self

    def options(self, *, num_returns):
        assert num_returns == "streaming"
        return self

    def remote(self, inputs, _tokenizer, _timer_prefix):
        self.inputs = inputs

        async def _stream():
            for result in self._results:

                async def _result_ref(value=result):
                    return value

                yield _result_ref()

        return _stream()


def test_nemo_gym_stream_callback_runs_before_group_completion() -> None:
    def _result(rowidx: int) -> tuple[int, dict, None]:
        rollout_id = f"rollout-{rowidx}"
        return (
            rowidx,
            {
                "receipt": {"rollout_id": rollout_id, "manifest": []},
                "rollout_id": rollout_id,
                "full_result": {"reward": float(rowidx)},
                "message_log": [],
                "input_message_log": [],
            },
            None,
        )

    impl = object.__new__(AsyncNemoGymRolloutImpl)
    impl._task_to_env = {"nemo_gym": _StreamingGymEnv([_result(1), _result(0)])}
    impl._tokenizer = None
    impl._compute_rollout_metrics = lambda _completions, _agent_name: {}
    callback_order: list[int] = []

    async def _record(rowidx: int, _completion: Completion) -> None:
        callback_order.append(rowidx)
        if rowidx == 1:
            assert len(callback_order) == 1

    completions, _, _ = _run(
        impl._run_rollouts(
            [
                {"agent_ref": {"name": "agent"}},
                {"agent_ref": {"name": "agent"}},
            ],
            Timer(),
            "timing/test",
            on_completion=_record,
        )
    )

    assert callback_order == [1, 0]
    assert [completion.env_extras["ng_rollout_id"] for completion in completions] == [
        "rollout-0",
        "rollout-1",
    ]


def test_nemo_gym_subset_maps_local_rows_to_logical_siblings() -> None:
    def _result(rowidx: int, rollout_id: str) -> tuple[int, dict, None]:
        return (
            rowidx,
            {
                "receipt": {"rollout_id": rollout_id, "manifest": []},
                "rollout_id": rollout_id,
                "full_result": {"reward": float(rowidx)},
                "message_log": [],
                "input_message_log": [],
            },
            None,
        )

    env = _StreamingGymEnv([_result(1, "retry-4"), _result(0, "retry-1")])
    impl = object.__new__(AsyncNemoGymRolloutImpl)
    impl._task_to_env = {"nemo_gym": env}
    impl._tokenizer = None
    impl._num_generations_per_prompt = 5
    impl._generation_config = {
        "temperature": 1.0,
        "top_p": 1.0,
        "max_new_tokens": 32,
    }
    impl._compute_rollout_metrics = lambda _completions, _agent_name: {}
    callback_order: list[int] = []

    async def _record(generation_index: int, _completion: Completion) -> None:
        callback_order.append(generation_index)

    record = _run(
        impl.run_rollout(
            {
                "idx": 9,
                "message_log": [],
                "extra_env_info": {
                    "agent_ref": {"name": "agent"},
                    "responses_create_params": {},
                },
                "task_name": "nemo_gym",
            },
            rollout_ids=["retry-1", "retry-4"],
            generation_indices=[1, 4],
            on_completion=_record,
        )
    )

    assert callback_order == [4, 1]
    assert [row["_rowidx"] for row in env.inputs] == [0, 1]
    assert [row["_ng_rollout_id"] for row in env.inputs] == ["retry-1", "retry-4"]
    assert len(record.completions) == 2


class _FakeBuffer:
    """Minimal TQReplayBuffer stand-in that records reserve/commit calls."""

    def __init__(self) -> None:
        self.reserve_calls: list[int] = []  # weight_versions passed to reserve
        self.commit_calls: list[tuple[str, object, int, int]] = []
        self.remove_calls: list[str] = []
        self.abort_calls: list[str] = []
        # reserve(weight_version=X) -> group_id; commit fills the slot.
        self._slots: list[str] = []

    def reserve(
        self,
        *,
        weight_version: int,
        target_step: int | None = None,
        group_id: str | None = None,
        rollout_ids: list[str] | None = None,
    ) -> str:
        del target_step, rollout_ids
        if group_id is None:
            group_id = str(uuid.uuid4())
        self.reserve_calls.append(weight_version)
        self._slots.append(group_id)
        return group_id

    def abort(self, group_id: str) -> bool:
        self.abort_calls.append(group_id)
        if group_id in self._slots:
            self._slots.remove(group_id)
            return True
        return False

    async def commit(
        self,
        group_id: str,
        record,
        start_weight_version: int,
        end_weight_version: int,
    ):
        self.commit_calls.append(
            (group_id, record, start_weight_version, end_weight_version)
        )
        return record

    async def remove_group(self, group_id: str, *, remove_in_dp: bool = False) -> int:
        del remove_in_dp
        self.remove_calls.append(group_id)
        self._slots.remove(group_id)
        return 1


class _FakeImpl:
    """Stand-in for AsyncRolloutImpl that returns a typed prompt-group record."""

    def __init__(self, record="sentinel-record", on_run=None) -> None:
        self._record = PromptGroupRecord(
            prompt_idx=0,
            prompt=[],
            extra_env_info={},
            metadata={"label": record},
            completions=[],
            rollout_metrics={},
        )
        self._on_run = on_run

    async def run_rollout(self, input_sample):
        if self._on_run is not None:
            await self._on_run(input_sample)
        return self._record


def _make_manager(buffer: _FakeBuffer, impl: _FakeImpl) -> RolloutManager:
    """Build a RolloutManager without firing the real __init__."""
    mgr = object.__new__(RolloutManager)
    mgr._impl = impl
    mgr._tokenizer = None
    mgr._num_generations_per_prompt = 1
    mgr._tq_buffer = buffer
    mgr._finalizer = None
    mgr._recovery_ledger = None
    mgr._env_handles = {}
    mgr._weight_version = 0
    mgr._canonical_groups_finalized = 0
    mgr._canonical_output_tokens = 0
    mgr._recovery_siblings_reused = 0
    mgr._recovery_siblings_redispatched = 0
    return mgr


class TestGenerateAndPushFlow:
    def test_rollout_failure_removes_reserved_group(self):
        async def _fail_rollout(_sample):
            raise RuntimeError("injected rollout failure")

        buf = _FakeBuffer()
        mgr = _make_manager(buf, _FakeImpl(on_run=_fail_rollout))

        with pytest.raises(RuntimeError, match="injected rollout failure"):
            _run(mgr.generate_and_push({"prompt": "p"}))

        assert len(buf.reserve_calls) == 1
        assert len(buf.remove_calls) == 1
        assert buf._slots == []
        assert buf.commit_calls == []

    def test_reserves_then_runs_then_commits(self):
        events: list[str] = []
        buf = _FakeBuffer()

        async def _track_run(_sample):
            events.append("run")

        impl = _FakeImpl(record="r0", on_run=_track_run)
        mgr = _make_manager(buf, impl)

        # Wrap reserve/commit to log ordering.
        original_reserve = buf.reserve
        original_commit = buf.commit

        def _logged_reserve(**kwargs):
            events.append("reserve")
            return original_reserve(**kwargs)

        async def _logged_commit(*args, **kwargs):
            events.append("commit")
            return await original_commit(*args, **kwargs)

        buf.reserve = _logged_reserve  # type: ignore[method-assign]
        buf.commit = _logged_commit  # type: ignore[method-assign]

        _run(mgr.generate_and_push({"prompt": "p"}))

        assert events == ["reserve", "run", "commit"]
        assert buf.reserve_calls == [0]
        assert len(buf.commit_calls) == 1
        gid, record, start_v, end_v = buf.commit_calls[0]
        assert gid in buf._slots
        assert record.metadata["label"] == "r0"
        assert start_v == 0
        assert end_v == 0
        assert mgr.telemetry_snapshot()["canonical_groups_finalized"] == 1

    def test_start_weight_version_pinned_at_reserve_time(self):
        """If set_weight_version is called mid-rollout, start != end."""
        buf = _FakeBuffer()

        async def _bump_weight_mid_rollout(_sample):
            # Simulate a sync_weights bump during the rollout.
            mgr.set_weight_version(5)

        impl = _FakeImpl(record="r0", on_run=_bump_weight_mid_rollout)
        mgr = _make_manager(buf, impl)
        mgr.set_weight_version(3)

        _run(mgr.generate_and_push({"prompt": "p"}))

        # reserve happened before run_rollout → captured weight 3.
        assert buf.reserve_calls == [3]
        # commit's start is the same dispatch-time value; end reflects the post-rollout weight.
        _, _, start_v, end_v = buf.commit_calls[0]
        assert start_v == 3
        assert end_v == 5

    def test_no_weight_change_means_start_equals_end(self):
        buf = _FakeBuffer()
        impl = _FakeImpl(record="r0")
        mgr = _make_manager(buf, impl)
        mgr.set_weight_version(7)

        _run(mgr.generate_and_push({"prompt": "p"}))

        _, _, start_v, end_v = buf.commit_calls[0]
        assert start_v == 7
        assert end_v == 7

    def test_concurrent_dispatch_preserves_reserve_order(self):
        """Two concurrent generate_and_push calls must reserve before either commits.

        The contract: reserve order == dispatch order, even if rollouts finish
        out of order. Slot order in the buffer reflects the order reserve was
        called (not the order run_rollout completed).
        """
        buf = _FakeBuffer()

        # First call's rollout blocks until second call has reserved.
        first_reserved = asyncio.Event()
        second_reserved = asyncio.Event()

        async def _first_run(_sample):
            first_reserved.set()
            await second_reserved.wait()

        async def _second_run(_sample):
            # Second is dispatched only after first reserves, so by the time
            # second's reserve fires, slots[0] == first's gid.
            second_reserved.set()

        first_impl = _FakeImpl(record="r0", on_run=_first_run)
        second_impl = _FakeImpl(record="r1", on_run=_second_run)

        first_mgr = _make_manager(buf, first_impl)
        # Share buffer across two managers (mimics two dispatches from one pump).
        second_mgr = _make_manager(buf, second_impl)

        async def _drive():
            t1 = asyncio.create_task(first_mgr.generate_and_push({"prompt": "p1"}))
            # Wait until first has reserved before kicking off second so the
            # reserve ordering is deterministic.
            await first_reserved.wait()
            t2 = asyncio.create_task(second_mgr.generate_and_push({"prompt": "p2"}))
            await asyncio.gather(t1, t2)

        _run(_drive())

        # Slots in buffer == reserve order.
        first_gid, second_gid = buf._slots
        # Commit recorded both, in either order, but each maps to its own gid.
        commit_gids = [c[0] for c in buf.commit_calls]
        assert set(commit_gids) == {first_gid, second_gid}
        assert buf.reserve_calls == [0, 0]

    def test_requires_tq_buffer(self):
        mgr = _make_manager(_FakeBuffer(), _FakeImpl())
        mgr._tq_buffer = None
        with pytest.raises(AssertionError, match="tq_buffer"):
            _run(mgr.generate_and_push({"prompt": "p"}))

    def test_failed_rollout_aborts_reserved_slot(self):
        """A dispatch that raises must not leave a phantom unready slot."""

        async def _boom(_input_sample):
            raise RuntimeError("rollout exploded")

        buf = _FakeBuffer()
        mgr = _make_manager(buf, _FakeImpl(on_run=_boom))

        with pytest.raises(RuntimeError, match="rollout exploded"):
            _run(mgr.generate_and_push({"prompt": "p"}))

        assert len(buf.reserve_calls) == 1
        assert buf.commit_calls == []
        assert len(buf.remove_calls) == 1
        assert buf._slots == []  # the reserved slot was dropped

    def test_failed_commit_aborts_reserved_slot(self):
        """Commit failures (e.g. evicted slot) also abort the reservation."""

        class _CommitBoomBuffer(_FakeBuffer):
            async def commit(
                self, group_id, record, start_weight_version, end_weight_version
            ):
                raise ValueError("no live slot")

        buf = _CommitBoomBuffer()
        mgr = _make_manager(buf, _FakeImpl())

        with pytest.raises(ValueError, match="no live slot"):
            _run(mgr.generate_and_push({"prompt": "p"}))
        assert len(buf.remove_calls) == 1


# ---------------------------------------------------------------------------
# Tests for RolloutManager
# ---------------------------------------------------------------------------


def test_rollout_manager_raises_without_impl_params():
    """RolloutManager raises AssertionError when required params are missing."""
    common = {
        "tokenizer": None,
        "task_to_env": {},
        "num_generations_per_prompt": 1,
        "max_seq_len": 1,
    }

    with pytest.raises(AssertionError, match="num_generations_per_prompt must be >= 1"):
        updated_common = common.copy()
        updated_common["num_generations_per_prompt"] = 0
        RolloutManager(**updated_common, use_nemo_gym=False)

    with pytest.raises(AssertionError, match="policy_generation is required"):
        RolloutManager(**common, use_nemo_gym=False)

    with pytest.raises(AssertionError, match="generation_config is required"):
        RolloutManager(**common, use_nemo_gym=True)


# ---------------------------------------------------------------------------
# Tests for AsyncRolloutManager (native async path)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="function")
def single_multi_step_calculator_input_sample(rollout_tokenizer):  # noqa: F811
    """Returns a single DatumSpec prompt dict (problem 0) for AsyncRolloutManager tests."""
    problem_text = "(5 + 3) * 2"
    expected_answer = 16.0
    max_steps = 5

    tool_instructions = (
        "You have a calculator tool. To use it, respond with:\n"
        "'[operand1, operand2, operation_name]<call: calculator>'\n"
        "The valid 'operation_name' values are exactly: 'sum', 'diff', 'prod', 'div'.\n"
        "Example: [5, 3, sum]<call: calculator>\n"
        "You will receive the result of your calculation as <result>...</result>\n"
        "Use this result to make the next calculation if needed.\n"
        "IMPORTANT: Only perform one calculation step (one tool call) before waiting for a result and making a new tool call.\n"
        "IMPORTANT: Do not perform any other calculations or operations aside from the tool call and result. Doing so will result in failure.\n"
        "To give the final answer, just output the number. numbers inside of <result> don't count, so output just the final number yourself outside of this.\n"
        "Example full output: [2, 4, sum]<call: calculator>\n<result>6.0</result>\n[6, 6, diff]<call: calculator>\n<result>0.0</result> 0\n(note how you have to output the final 0 outside of the tags)"
        "------\n"
        f"Solve: {problem_text}"
    )

    initial_prompt_content = rollout_tokenizer.apply_chat_template(
        [{"role": "user", "content": tool_instructions}],
        tokenize=False,
        add_system_prompt=False,
        add_generation_prompt=True,
        add_special_tokens=False,
    )
    tokenized_prompt = rollout_tokenizer(
        initial_prompt_content, return_tensors="pt", add_special_tokens=False
    )["input_ids"][0]
    message_log = [
        {
            "role": "user",
            "content": initial_prompt_content,
            "token_ids": tokenized_prompt,
        }
    ]
    metadata = MultiStepCalcMetadata(
        problem=problem_text,
        expected_final_answer=expected_answer,
        max_steps=max_steps,
        current_step=0,
    )
    return {
        "message_log": message_log,
        "extra_env_info": metadata,
        "task_name": "multi_step_calculator_game",
        "stop_strings": ["<call: calculator>"],
        "idx": 0,
    }


@pytest.mark.vllm
def test_async_rollout_manager(
    multi_step_setup_vllm_async,  # noqa: F811
    single_multi_step_calculator_input_sample,
):
    """Standalone test for AsyncRolloutManager.

    Given 1 prompt with num_generations_per_prompt=N, asserts:
    - output is a PromptGroupRecord with N Completion objects
    - each Completion has a reward (float) and a non-empty message_log
    - rollout_metrics has the expected keys with correct types
    - completions hold independent (not aliased) message_log objects
    """
    vllm_generation, tokenizer, task_to_env, _, _ = multi_step_setup_vllm_async
    input_sample = single_multi_step_calculator_input_sample
    num_generations = 2
    max_seq_len = 1024
    max_rollout_turns = input_sample["extra_env_info"]["max_steps"] + 1

    manager = RolloutManager(
        use_nemo_gym=False,
        tokenizer=tokenizer,
        task_to_env=task_to_env,
        num_generations_per_prompt=num_generations,
        max_seq_len=max_seq_len,
        max_rollout_turns=max_rollout_turns,
        policy_generation=vllm_generation,
    )

    vllm_generation.prepare_for_generation()
    record = asyncio.run(manager.run_rollout(input_sample))
    vllm_generation.finish_generation()

    assert isinstance(record, PromptGroupRecord)
    assert len(record.completions) == num_generations, (
        f"Expected {num_generations} completions, got {len(record.completions)}"
    )
    assert record.prompt_idx == input_sample["idx"]

    for i, completion in enumerate(record.completions):
        assert isinstance(completion, Completion)

        # 1. message_log length
        assert len(completion.message_log) >= 4, (
            f"Completion {i}: expected >= 4 messages, got {len(completion.message_log)}"
        )

        # 2. last assistant content
        last_assistant = next(
            (m for m in reversed(completion.message_log) if m["role"] == "assistant"),
            None,
        )
        assert last_assistant is not None, f"Completion {i}: no assistant message found"
        assert last_assistant["content"].strip() == "16", (
            f"Completion {i}: last assistant content {last_assistant['content']!r} != '16'"
        )

        # 3. reward
        assert completion.reward == 1.0, (
            f"Completion {i}: reward {completion.reward} != 1.0"
        )

    # completions must be independent objects
    assert record.completions[0].message_log is not record.completions[1].message_log


@pytest.mark.vllm
def test_async_rollout_manager_truncation(
    multi_step_setup_vllm_async,  # noqa: F811
    single_multi_step_calculator_input_sample,
):
    """Small max_seq_len forces truncation and truncation_rate=1.0."""
    vllm_generation, tokenizer, task_to_env, _, _ = multi_step_setup_vllm_async
    input_sample = single_multi_step_calculator_input_sample
    num_generations = 2
    max_seq_len = 290
    max_rollout_turns = input_sample["extra_env_info"]["max_steps"] + 1

    manager = RolloutManager(
        use_nemo_gym=False,
        tokenizer=tokenizer,
        task_to_env=task_to_env,
        num_generations_per_prompt=num_generations,
        max_seq_len=max_seq_len,
        max_rollout_turns=max_rollout_turns,
        policy_generation=vllm_generation,
    )
    vllm_generation.prepare_for_generation()
    record = asyncio.run(manager.run_rollout(input_sample))
    vllm_generation.finish_generation()

    assert len(record.completions) == num_generations
    assert all(c.truncated for c in record.completions)
    assert record.rollout_metrics["truncation_rate"] == 1.0
    assert record.rollout_metrics["natural_termination_rate"] == 0.0


@pytest.mark.vllm
def test_async_rollout_manager_matches_original(
    multi_step_setup_vllm_async,  # noqa: F811
    single_multi_step_calculator_input_sample,
):
    """Comparison test: AsyncRolloutManager output is structurally equivalent to the original.

    Calls run_async_multi_turn_rollout with a batch of N identical prompts,
    then calls AsyncRolloutManager with 1 prompt and N generations.
    Asserts that both produce N results with matching message-log depth, rewards,
    and rollout_metrics numeric values.

    TODO: remove this test together with run_async_multi_turn_rollout when the legacy path is deleted.
    """
    vllm_generation, tokenizer, task_to_env, _, _ = multi_step_setup_vllm_async
    input_sample = single_multi_step_calculator_input_sample
    num_generations = 2
    max_seq_len = 1024
    max_rollout_turns = input_sample["extra_env_info"]["max_steps"] + 1

    # Build a batch of N identical prompts for the original function
    batch = BatchedDataDict(
        {
            "message_log": [
                deepcopy(input_sample["message_log"]) for _ in range(num_generations)
            ],
            "extra_env_info": [
                deepcopy(input_sample["extra_env_info"]) for _ in range(num_generations)
            ],
            "task_name": [input_sample["task_name"]] * num_generations,
            "stop_strings": [input_sample["stop_strings"]] * num_generations,
            "idx": list(range(num_generations)),
            "loss_multiplier": [1.0] * num_generations,
        }
    )

    vllm_generation.prepare_for_generation()
    original_batch, original_metrics = run_async_multi_turn_rollout(
        policy_generation=vllm_generation,
        input_batch=batch,
        tokenizer=tokenizer,
        task_to_env=task_to_env,
        max_seq_len=max_seq_len,
        max_rollout_turns=max_rollout_turns,
    )

    manager = RolloutManager(
        use_nemo_gym=False,
        tokenizer=tokenizer,
        task_to_env=task_to_env,
        num_generations_per_prompt=num_generations,
        max_seq_len=max_seq_len,
        max_rollout_turns=max_rollout_turns,
        policy_generation=vllm_generation,
    )
    record = asyncio.run(manager.run_rollout(input_sample))
    vllm_generation.finish_generation()

    # Both should produce N results
    assert len(original_batch["message_log"]) == num_generations
    assert len(record.completions) == num_generations

    for i in range(num_generations):
        orig_msg_log = original_batch["message_log"][i]
        new_msg_log = record.completions[i].message_log

        # 1. message_log length matches
        assert len(orig_msg_log) == len(new_msg_log), (
            f"Completion {i}: message_log length {len(new_msg_log)} != original {len(orig_msg_log)}"
        )

        # 2. last assistant content matches
        def _last_assistant_content(msg_log):
            for m in reversed(msg_log):
                if m["role"] == "assistant":
                    return m.get("content", "")
            return ""

        orig_last = _last_assistant_content(orig_msg_log)
        new_last = _last_assistant_content(new_msg_log)
        assert orig_last == new_last, (
            f"Completion {i}: last assistant content mismatch\n"
            f"  original:  {orig_last!r}\n"
            f"  manager:   {new_last!r}"
        )

        # 3. reward matches
        orig_reward = original_batch["total_reward"][i].item()
        new_reward = record.completions[i].reward
        assert orig_reward == new_reward, (
            f"Completion {i}: reward mismatch — original {orig_reward}, manager {new_reward}"
        )

    # 4. rollout_metrics numeric values match (timing and histogram fields are excluded).
    # The new impl emits slash-style keys (X/mean, X/max, X/min) via calculate_single_metric;
    # translate the legacy prefix-style keys before comparing.
    def _translate_legacy_key(key: str) -> str:
        if key == "avg_turns_per_sample":
            return "turns_per_sample/mean"
        if key == "max_turns_reached_rate":
            return key
        # Keys already in slash-style (e.g. turns_per_sample/p95, max_gen_tokens_per_turn/max)
        # are new-style and should not be re-translated by the prefix-strip logic.
        if "/" in key:
            return key
        for prefix, suffix in (("mean_", "/mean"), ("max_", "/max"), ("min_", "/min")):
            if key.startswith(prefix):
                return f"{key[len(prefix) :]}{suffix}"
        return key

    new_metrics = record.rollout_metrics
    for key in original_metrics.keys():
        if key.startswith("timing/") or key.startswith("histogram/"):
            continue

        new_key = _translate_legacy_key(key)
        assert new_key in new_metrics, (
            f"rollout_metrics[{new_key!r}] missing from manager"
        )

        orig_val = original_metrics[key]
        new_val = new_metrics[new_key]

        assert type(orig_val) == type(new_val), (
            f"rollout_metrics[{key!r}] type mismatch: {type(orig_val)} != {type(new_val)}"
        )
        if not isinstance(orig_val, (bool, int, float)):
            continue

        assert orig_val == pytest.approx(new_val), (
            f"rollout_metrics[{key!r}] mismatch — original {orig_val}, manager {new_val}"
        )


# ---------------------------------------------------------------------------
# Tests for AsyncNemoGymRolloutManager
# ---------------------------------------------------------------------------


@pytest.mark.nemo_gym
def test_async_nemo_gym_rollout_manager(
    nemo_gym,  # noqa: F811
    nemo_gym_vllm_generation,  # noqa: F811
    nemo_gym_sanity_test_data,  # noqa: F811
    nemo_gym_tokenizer,  # noqa: F811
):
    """Standalone test for AsyncNemoGymRolloutManager.

    Given 1 prompt with num_generations_per_prompt=N, asserts:
    - output is a PromptGroupRecord with N Completion objects
    - each Completion has a reward (float) and a non-empty message_log
    - completions hold independent message_log objects

    If the result here does not match, please check the following:
    1. Test data changed: re-run test_nemo_gym_sanity (tests/unit/environments/test_nemo_gym.py)
       and use _write_actual_test_data output to refresh test_nemo_gym_sanity.json.
    2. Logic changed: inspect recent changes to AsyncNemoGymRolloutManager or the gym env.
    """
    with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
        for data in nemo_gym_sanity_test_data["input"]:
            f.write(json.dumps(data) + "\n")
        data_path = f.name

    dataset = NemoGymDataset(data_path)
    examples = [
        nemo_gym_data_processor(dataset.dataset[idx], None, None, None, idx)
        for idx in range(len(dataset.dataset))
    ]
    input_batch: BatchedDataDict[DatumSpec] = rl_collate_fn(examples)

    # Use only the first prompt
    single_prompt = {
        "message_log": input_batch["message_log"][0],
        "extra_env_info": input_batch["extra_env_info"][0],
        "task_name": "nemo_gym",
        "idx": 0,
        "loss_multiplier": float(input_batch["loss_multiplier"][0]),
    }
    num_generations = 2

    manager = RolloutManager(
        use_nemo_gym=True,
        tokenizer=nemo_gym_tokenizer,
        task_to_env={"nemo_gym": nemo_gym},
        num_generations_per_prompt=num_generations,
        max_seq_len=nemo_gym_vllm_generation.cfg["vllm_cfg"]["max_model_len"],
        generation_config=nemo_gym_vllm_generation.cfg,
    )
    record = asyncio.run(manager.run_rollout(single_prompt))

    assert isinstance(record, PromptGroupRecord)
    assert len(record.completions) == num_generations, (
        f"Expected {num_generations} completions, got {len(record.completions)}"
    )
    assert record.prompt_idx == 0

    for i, completion in enumerate(record.completions):
        assert isinstance(completion, Completion)

        # 1. message_log length
        assert len(completion.message_log) == 2, (
            f"Completion {i}: expected 2 messages, got {len(completion.message_log)}"
        )

        # 2. last assistant token_ids
        last_assistant = next(
            (m for m in reversed(completion.message_log) if m["role"] == "assistant"),
            None,
        )
        assert last_assistant is not None, f"Completion {i}: no assistant message found"
        assert torch.equal(
            last_assistant["token_ids"],
            torch.tensor([151667, 198, 32313, 11, 1077]),
        ), (
            f"Completion {i}: last assistant token_ids {last_assistant['token_ids'].tolist()} "
            f"!= [151667, 198, 32313, 11, 1077]"
        )

        # 3. reward
        assert completion.reward == 0.0, (
            f"Completion {i}: reward {completion.reward} != 0.0"
        )

    # completions must be independent objects
    assert record.completions[0].message_log is not record.completions[1].message_log


@pytest.mark.nemo_gym
def test_async_nemo_gym_rollout_manager_matches_original(
    nemo_gym,  # noqa: F811
    nemo_gym_vllm_generation,  # noqa: F811
    nemo_gym_sanity_test_data,  # noqa: F811
    nemo_gym_tokenizer,  # noqa: F811
):
    """Comparison test: AsyncNemoGymRolloutManager output is structurally equivalent to the original.

    Calls run_async_nemo_gym_rollout with a batch of N identical rows,
    then calls AsyncNemoGymRolloutManager with 1 prompt, N generations.
    Asserts that both produce N results and rewards are in the same numeric domain.

    TODO: remove this test together with run_async_nemo_gym_rollout when the legacy path is deleted.
    """
    with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
        for data in nemo_gym_sanity_test_data["input"]:
            f.write(json.dumps(data) + "\n")
        data_path = f.name

    dataset = NemoGymDataset(data_path)
    examples = [
        nemo_gym_data_processor(dataset.dataset[idx], None, None, None, idx)
        for idx in range(len(dataset.dataset))
    ]
    input_batch: BatchedDataDict[DatumSpec] = rl_collate_fn(examples)

    num_generations = 2
    single_prompt = {
        "message_log": input_batch["message_log"][0],
        "extra_env_info": input_batch["extra_env_info"][0],
        "task_name": "nemo_gym",
        "idx": 0,
        "loss_multiplier": float(input_batch["loss_multiplier"][0]),
    }

    # Build a batch of N identical rows for the original function
    repeated_batch = BatchedDataDict(
        {
            "message_log": [
                deepcopy(input_batch["message_log"][0]) for _ in range(num_generations)
            ],
            "extra_env_info": [
                deepcopy(input_batch["extra_env_info"][0])
                for _ in range(num_generations)
            ],
            "loss_multiplier": input_batch["loss_multiplier"][0:1].repeat(
                num_generations
            ),
            "idx": list(range(num_generations)),
            "task_name": ["nemo_gym"] * num_generations,
        }
    )

    async def _collect_original_results():
        return [
            result
            async for result in run_async_nemo_gym_rollout(
                policy_generation=nemo_gym_vllm_generation,
                input_batch=repeated_batch,
                tokenizer=nemo_gym_tokenizer,
                task_to_env={"nemo_gym": nemo_gym},
                generation_config=nemo_gym_vllm_generation.cfg,
                num_generations=num_generations,
                log_full_result_tables=False,
                max_seq_len=nemo_gym_vllm_generation.cfg["vllm_cfg"]["max_model_len"],
                max_rollout_turns=None,
            )
        ]

    original_results = asyncio.run(_collect_original_results())
    assert len(original_results) == 1
    original_result = original_results[0]

    manager = RolloutManager(
        use_nemo_gym=True,
        tokenizer=nemo_gym_tokenizer,
        task_to_env={"nemo_gym": nemo_gym},
        num_generations_per_prompt=num_generations,
        max_seq_len=nemo_gym_vllm_generation.cfg["vllm_cfg"]["max_model_len"],
        generation_config=nemo_gym_vllm_generation.cfg,
    )
    record = asyncio.run(manager.run_rollout(single_prompt))

    # Both should produce N completions
    assert len(original_result.final_batch["message_log"]) == num_generations
    assert len(record.completions) == num_generations

    for i in range(num_generations):
        orig_msg_log = original_result.final_batch["message_log"][i]
        new_msg_log = record.completions[i].message_log

        # 1. message_log length matches
        assert len(orig_msg_log) == len(new_msg_log), (
            f"Completion {i}: message_log length {len(new_msg_log)} != original {len(orig_msg_log)}"
        )

        # 2. last assistant token_ids match
        def _last_assistant_token_ids(msg_log):
            for m in reversed(msg_log):
                if m["role"] == "assistant":
                    return m.get("token_ids")
            return None

        orig_token_ids = _last_assistant_token_ids(orig_msg_log)
        new_token_ids = _last_assistant_token_ids(new_msg_log)
        assert orig_token_ids is not None, (
            f"Completion {i}: no assistant message in original"
        )
        assert new_token_ids is not None, (
            f"Completion {i}: no assistant message in manager"
        )
        assert torch.equal(orig_token_ids, new_token_ids), (
            f"Completion {i}: last assistant token_ids mismatch\n"
            f"  original:  {orig_token_ids.tolist()}\n"
            f"  manager:   {new_token_ids.tolist()}"
        )

        # 3. reward matches
        orig_reward = original_result.final_batch["total_reward"][i].item()
        new_reward = record.completions[i].reward
        assert orig_reward == new_reward, (
            f"Completion {i}: reward mismatch — original {orig_reward}, manager {new_reward}"
        )

    # 4. rollout_metrics numeric values match (timing and Table fields are excluded)
    orig_metrics = original_result.rollout_metrics
    new_metrics = record.rollout_metrics
    for key in orig_metrics.keys():
        # Skip timing and full_result fields
        if key.startswith("timing/") or key.endswith("/full_result"):
            continue

        # Check that the key is present in the new metrics
        assert key in new_metrics, f"rollout_metrics[{key!r}] missing from manager"

        orig_val = orig_metrics[key]
        new_val = new_metrics[key]

        # Skip non-numeric fields
        assert type(orig_val) == type(new_val), (
            f"rollout_metrics[{key!r}] type mismatch: {type(orig_val)} != {type(new_val)}"
        )
        if not isinstance(orig_val, (bool, int, float)):
            continue

        # Check equal
        assert orig_val == pytest.approx(new_val), (
            f"rollout_metrics[{key!r}] mismatch — original {orig_val}, manager {new_val}"
        )


class _FakeFinalizedGroup:
    def __init__(self, *, dropped=False, canonical_output_tokens=0):
        self.meta = None if dropped else "meta-sentinel"
        self.fields = None if dropped else "fields-sentinel"
        self.group_min_wv = 3
        self.group_max_wv = 4
        self.staging_keys = ["stage-0"]
        self.canonical_output_tokens = canonical_output_tokens
        self.metrics = {"finalize/invalid_row_rate": 0.0}
        self.dropped = dropped


class _FakeFinalizer:
    def __init__(self, *, dropped=False):
        self.calls: list[tuple] = []
        self._dropped = dropped

    def finalize_group(
        self,
        group_id,
        rollout_ids,
        receipts,
        rewards,
        *,
        fallback_weight_version,
        canonical_sample_ids=None,
    ):
        self.calls.append(
            (
                group_id,
                rollout_ids,
                receipts,
                rewards,
                fallback_weight_version,
                canonical_sample_ids,
            )
        )
        # Model a finalizer-owned count that is deliberately independent of
        # receipt delta_len, which includes prompt-side tokens in Gym.
        canonical_output_tokens = sum(range(1, len(rollout_ids) + 1))
        return _FakeFinalizedGroup(
            dropped=self._dropped,
            canonical_output_tokens=canonical_output_tokens,
        )


class _FakeCaptureBuffer(_FakeBuffer):
    def __init__(self):
        super().__init__()
        self.reserve_rollout_ids: list[list[str] | None] = []
        self.commit_finalized_calls: list[tuple] = []
        self.abort_finalized_calls: list[tuple[str, list[str]]] = []

    def reserve(
        self, *, weight_version, target_step=None, group_id=None, rollout_ids=None
    ):
        self.reserve_rollout_ids.append(rollout_ids)
        return super().reserve(
            weight_version=weight_version,
            target_step=target_step,
            group_id=group_id,
            rollout_ids=rollout_ids,
        )

    async def commit_finalized(
        self,
        group_id,
        meta,
        fields,
        group_min_wv,
        group_max_wv,
        *,
        staging_keys=None,
    ):
        self.commit_finalized_calls.append(
            (group_id, meta, fields, group_min_wv, group_max_wv, staging_keys)
        )
        return meta

    async def abort_finalized(self, group_id, *, staging_keys):
        self.abort_finalized_calls.append((group_id, list(staging_keys)))
        return self.abort(group_id)


class _FakeGymEnvHandle:
    """NemoGym actor stand-in exposing fail_rollouts.remote."""

    def __init__(self):
        self.failed: list[tuple[list[str], str]] = []
        outer = self

        class _FailRollouts:
            def remote(self, rollout_ids, reason):
                outer.failed.append((list(rollout_ids), reason))

                async def _done():
                    return None

                return _done()

        self.fail_rollouts = _FailRollouts()


def _receipt_record(rollout_ids, receipts):
    completions = [
        Completion(
            message_log=[],
            env_extras={"reward": 0.5, "ng_receipt": receipt, "ng_rollout_id": rid},
            truncated=False,
            reward=0.5,
        )
        for rid, receipt in zip(rollout_ids, receipts)
    ]
    return PromptGroupRecord(
        prompt_idx=0,
        prompt=[],
        extra_env_info={},
        metadata={"task_name": "nemo_gym"},
        completions=completions,
        rollout_metrics={},
    )


def _make_capture_manager(
    buf,
    finalizer,
    *,
    on_run=None,
    num_generations=2,
    fail_after_completions=None,
):
    mgr = object.__new__(RolloutManager)
    mgr._tokenizer = None
    mgr._num_generations_per_prompt = num_generations
    mgr._tq_buffer = buf
    mgr._finalizer = finalizer
    mgr._recovery_ledger = RolloutRecoveryLedger()
    mgr._data_plane_checkpoint_barrier = DataPlaneCheckpointBarrier()
    mgr._env_handles = {"nemo_gym": _FakeGymEnvHandle()}
    mgr._weight_version = 7
    mgr._canonical_groups_finalized = 0
    mgr._canonical_output_tokens = 0
    mgr._recovery_siblings_reused = 0
    mgr._recovery_siblings_redispatched = 0

    class _CaptureImpl(AsyncNemoGymRolloutImpl):
        def __init__(self):
            self.seen_rollout_ids = None
            self.seen_generation_indices = None
            self.streamed_recovery_groups = []

        async def run_rollout(
            self,
            _sample,
            *,
            rollout_ids=None,
            generation_indices=None,
            on_completion=None,
        ):
            self.seen_rollout_ids = rollout_ids
            self.seen_generation_indices = generation_indices
            if on_run is not None:
                await on_run(_sample)
            logical_indices = generation_indices or list(range(len(rollout_ids)))
            receipts = [
                {
                    "rollout_id": rollout_id,
                    "manifest": [
                        {
                            "staging_key": f"stage-{logical_index}",
                            "delta_len": 1_000,
                        }
                    ],
                }
                for logical_index, rollout_id in zip(logical_indices, rollout_ids)
            ]
            record = _receipt_record(rollout_ids, receipts)
            for local_index, completion in enumerate(record.completions):
                if on_completion is not None:
                    await on_completion(logical_indices[local_index], completion)
                    self.streamed_recovery_groups.append(
                        mgr.recovery_ledger.get_group(buf._slots[0])
                    )
                if fail_after_completions == local_index + 1:
                    raise RuntimeError("stream interrupted")
            return record

    mgr._impl = _CaptureImpl()
    return mgr


def _capture_sample():
    return {
        "idx": 42,
        "message_log": [],
        "extra_env_info": {},
        "task_name": "nemo_gym",
    }


class TestGenerateAndFinalizeFlow:
    def test_dispatch_uses_pre_reserved_dataloader_lineage(self):
        buf = _FakeCaptureBuffer()
        mgr = _make_capture_manager(buf, _FakeFinalizer())
        group_id = mgr.reserve_prompt_group(_capture_sample(), target_step=5)

        assert group_id is not None
        _run(
            mgr.generate_and_push(
                _capture_sample(),
                target_step=5,
                recovery_group_id=group_id,
            )
        )

        assert buf._slots == [group_id]
        assert len(mgr.recovery_ledger) == 0

    def test_recover_sealed_group_replays_finalizer_without_generation(self):
        buf = _FakeCaptureBuffer()
        finalizer = _FakeFinalizer()
        mgr = _make_capture_manager(buf, finalizer)
        group = mgr.recovery_ledger.reserve_group(
            group_id="recovery-g0",
            prompt_id="42",
            prompt_payload=_capture_sample(),
            expected_generations=2,
            target_step=None,
            start_weight_version=7,
        )
        mgr.recovery_ledger.mark_group_dispatched(group.group_id)
        for sibling in group.siblings:
            rollout_id = sibling.current_attempt.gate_rollout_id
            mgr.recovery_ledger.mark_sibling_sealed(
                group.group_id,
                generation_index=sibling.generation_index,
                gate_rollout_id=rollout_id,
                receipt={
                    "rollout_id": rollout_id,
                    "manifest": [{"staging_key": f"stage-{sibling.generation_index}"}],
                },
                reward=0.5,
            )

        committed = _run(
            mgr.ensure_rollout_group(
                _capture_sample(),
                group_id=group.group_id,
                target_step=None,
            )
        )

        assert committed is True
        assert mgr._impl.seen_rollout_ids is None
        assert len(finalizer.calls) == 1
        assert buf.commit_finalized_calls == [
            (
                group.group_id,
                "meta-sentinel",
                "fields-sentinel",
                3,
                4,
                ["stage-0"],
            )
        ]
        assert len(mgr.recovery_ledger) == 0

    def test_recover_dropped_group_cleans_staging_and_releases_lineage(self):
        buf = _FakeCaptureBuffer()
        finalizer = _FakeFinalizer(dropped=True)
        mgr = _make_capture_manager(buf, finalizer)
        group = mgr.recovery_ledger.reserve_group(
            group_id="recovery-g0",
            prompt_id="42",
            prompt_payload=_capture_sample(),
            expected_generations=2,
            target_step=None,
            start_weight_version=7,
        )
        mgr.recovery_ledger.mark_group_dispatched(group.group_id)
        for sibling in group.siblings:
            rollout_id = sibling.current_attempt.gate_rollout_id
            mgr.recovery_ledger.mark_sibling_sealed(
                group.group_id,
                generation_index=sibling.generation_index,
                gate_rollout_id=rollout_id,
                receipt={
                    "rollout_id": rollout_id,
                    "manifest": [{"staging_key": f"stage-{sibling.generation_index}"}],
                },
                reward=0.5,
            )

        committed = _run(mgr.recover_group(group.group_id))

        assert committed is False
        assert buf.commit_finalized_calls == []
        assert buf.abort_finalized_calls == [(group.group_id, ["stage-0"])]
        assert buf.abort_calls == [group.group_id]
        assert len(mgr.recovery_ledger) == 0
        assert mgr._env_handles["nemo_gym"].failed == [
            (group.gate_rollout_ids, "finalizer_dropped")
        ]

    def test_recover_partial_group_redispatches_only_missing_siblings(self):
        buf = _FakeCaptureBuffer()
        finalizer = _FakeFinalizer()
        mgr = _make_capture_manager(buf, finalizer, num_generations=3)
        group = mgr.recovery_ledger.reserve_group(
            group_id="recovery-g0",
            prompt_id="42",
            prompt_payload=_capture_sample(),
            expected_generations=3,
            target_step=None,
            start_weight_version=7,
        )
        mgr.recovery_ledger.mark_group_dispatched(group.group_id)
        for generation_index in (0, 2):
            attempt = group.siblings[generation_index].current_attempt
            mgr.recovery_ledger.mark_sibling_sealed(
                group.group_id,
                generation_index=generation_index,
                gate_rollout_id=attempt.gate_rollout_id,
                receipt={
                    "rollout_id": attempt.gate_rollout_id,
                    "manifest": [
                        {
                            "staging_key": f"stage-{generation_index}",
                            "delta_len": 1_000,
                        }
                    ],
                },
                reward=float(generation_index),
            )
        mgr.recovery_ledger.prepare_for_restart()
        old_missing_attempt = group.siblings[1].current_attempt

        committed = _run(mgr.recover_group(group.group_id))

        assert committed is True
        assert mgr._impl.seen_generation_indices == [1]
        assert len(mgr._impl.seen_rollout_ids) == 1
        assert mgr._impl.seen_rollout_ids[0] != old_missing_attempt.gate_rollout_id
        (_, finalizer_ids, receipts, rewards, _, canonical_ids) = finalizer.calls[0]
        assert canonical_ids == [
            "recovery-g0_g0",
            "recovery-g0_g1",
            "recovery-g0_g2",
        ]
        assert finalizer_ids[0] == group.siblings[0].current_attempt.gate_rollout_id
        assert finalizer_ids[2] == group.siblings[2].current_attempt.gate_rollout_id
        assert finalizer_ids[1] == mgr._impl.seen_rollout_ids[0]
        assert [receipt["rollout_id"] for receipt in receipts] == finalizer_ids
        assert rewards == [0.0, 0.5, 2.0]
        assert len(mgr.recovery_ledger) == 0
        assert mgr.telemetry_snapshot() == {
            "canonical_groups_finalized": 1,
            "canonical_output_tokens": 6,
            "recovery_siblings_reused": 2,
            "recovery_siblings_redispatched": 1,
        }

    def test_recovery_failure_preserves_old_and_newly_sealed_siblings(self):
        buf = _FakeCaptureBuffer()
        mgr = _make_capture_manager(
            buf,
            _FakeFinalizer(),
            num_generations=3,
            fail_after_completions=1,
        )
        group = mgr.recovery_ledger.reserve_group(
            group_id="recovery-g0",
            prompt_id="42",
            prompt_payload=_capture_sample(),
            expected_generations=3,
            target_step=None,
            start_weight_version=7,
        )
        mgr.recovery_ledger.mark_group_dispatched(group.group_id)
        attempt = group.siblings[0].current_attempt
        mgr.recovery_ledger.mark_sibling_sealed(
            group.group_id,
            generation_index=0,
            gate_rollout_id=attempt.gate_rollout_id,
            receipt={
                "rollout_id": attempt.gate_rollout_id,
                "manifest": [{"staging_key": "stage-0"}],
            },
            reward=0.0,
        )
        mgr.recovery_ledger.prepare_for_restart()

        with pytest.raises(RuntimeError, match="stream interrupted"):
            _run(mgr.recover_group(group.group_id))

        recovered = mgr.recovery_ledger.get_group(group.group_id)
        assert [sibling.current_attempt.status for sibling in recovered.siblings] == [
            RolloutAttemptStatus.SEALED,
            RolloutAttemptStatus.SEALED,
            RolloutAttemptStatus.ABANDONED,
        ]
        assert mgr._impl.seen_generation_indices == [1, 2]
        (failed_ids, reason) = mgr._env_handles["nemo_gym"].failed[-1]
        assert failed_ids == [recovered.siblings[2].current_attempt.gate_rollout_id]
        assert reason == "recovery_failed"

    def test_mints_ids_finalizes_and_commits(self):
        buf = _FakeCaptureBuffer()
        finalizer = _FakeFinalizer()
        observed_recovery_groups = []

        async def _observe_dispatched_group(_sample):
            observed_recovery_groups.append(
                mgr.recovery_ledger.get_group(buf._slots[0])
            )

        mgr = _make_capture_manager(buf, finalizer, on_run=_observe_dispatched_group)

        _run(mgr.generate_and_push(_capture_sample(), target_step=5))

        # Canonical IDs are stable logical siblings. Physical gate IDs carry
        # an attempt suffix and are threaded reserve -> impl -> finalizer.
        (group_id,) = buf._slots
        logical_ids = [f"{group_id}_g0", f"{group_id}_g1"]
        (gate_ids,) = buf.reserve_rollout_ids
        assert gate_ids is not None
        assert all(
            gate_id.startswith(f"{logical_id}_a")
            for gate_id, logical_id in zip(gate_ids, logical_ids)
        )
        assert len(set(gate_ids)) == 2
        assert mgr._impl.seen_rollout_ids == gate_ids
        (
            fin_group_id,
            fin_ids,
            receipts,
            rewards,
            fallback_wv,
            canonical_ids,
        ) = finalizer.calls[0]
        assert fin_group_id == group_id
        assert fin_ids == gate_ids
        assert canonical_ids == logical_ids
        assert [r["rollout_id"] for r in receipts] == gate_ids
        assert rewards == [0.5, 0.5]
        assert fallback_wv == 7
        # commit_finalized carried the group's min/max call versions.
        assert buf.commit_finalized_calls == [
            (
                group_id,
                "meta-sentinel",
                "fields-sentinel",
                3,
                4,
                ["stage-0"],
            )
        ]
        # The legacy commit path was not used and nothing failed at the gate.
        assert buf.commit_calls == []
        assert mgr._env_handles["nemo_gym"].failed == []
        assert mgr.telemetry_snapshot() == {
            "canonical_groups_finalized": 1,
            "canonical_output_tokens": 3,
            "recovery_siblings_reused": 0,
            "recovery_siblings_redispatched": 0,
        }
        (recovery_group,) = observed_recovery_groups
        assert recovery_group.prompt_id == "42"
        assert recovery_group.target_step == 5
        assert recovery_group.logical_rollout_ids == logical_ids
        assert all(
            sibling.current_attempt.status == RolloutAttemptStatus.DISPATCHED
            for sibling in recovery_group.siblings
        )
        streamed_group = mgr._impl.streamed_recovery_groups[-1]
        assert all(
            sibling.current_attempt.status == RolloutAttemptStatus.SEALED
            for sibling in streamed_group.siblings
        )
        assert [
            sibling.current_attempt.staging_keys for sibling in streamed_group.siblings
        ] == [["stage-0"], ["stage-1"]]
        assert len(mgr.recovery_ledger) == 0

    def test_dropped_group_aborts_slot(self):
        buf = _FakeCaptureBuffer()
        mgr = _make_capture_manager(buf, _FakeFinalizer(dropped=True))
        with pytest.raises(RuntimeError, match="min_valid_fraction"):
            _run(mgr.generate_and_push(_capture_sample()))
        assert buf.commit_finalized_calls == []
        assert len(buf.abort_finalized_calls) == 1
        assert len(buf.abort_calls) >= 1
        assert len(mgr.recovery_ledger) == 0

    def test_dropped_group_preserves_sealed_receipts_when_cleanup_fails(self):
        class _FailingCleanupBuffer(_FakeCaptureBuffer):
            async def abort_finalized(self, group_id, *, staging_keys):
                self.abort_finalized_calls.append((group_id, list(staging_keys)))
                raise RuntimeError("staging cleanup failed")

        buf = _FailingCleanupBuffer()
        mgr = _make_capture_manager(buf, _FakeFinalizer(dropped=True))

        with pytest.raises(RuntimeError, match="staging cleanup failed"):
            _run(mgr.generate_and_push(_capture_sample()))

        (group_id,) = buf.abort_calls
        recovery_group = mgr.recovery_ledger.get_group(group_id)
        assert all(
            sibling.current_attempt.status == RolloutAttemptStatus.SEALED
            for sibling in recovery_group.siblings
        )
        assert mgr._env_handles["nemo_gym"].failed == []

    def test_failed_dispatch_aborts_and_fails_gate_rollouts(self):
        buf = _FakeCaptureBuffer()

        async def _boom(_sample):
            raise RuntimeError("rollout exploded")

        mgr = _make_capture_manager(buf, _FakeFinalizer(), on_run=_boom)
        with pytest.raises(RuntimeError, match="rollout exploded"):
            _run(mgr.generate_and_push(_capture_sample()))
        assert buf.commit_finalized_calls == []
        assert len(buf.abort_calls) == 1
        (failed_ids, reason) = mgr._env_handles["nemo_gym"].failed[0]
        (group_id,) = [buf.abort_calls[0]]
        assert all(
            gate_id.startswith(f"{group_id}_g{generation_index}_a")
            for generation_index, gate_id in enumerate(failed_ids)
        )
        assert reason == "dispatch_failed"
        recovery_group = mgr.recovery_ledger.get_group(group_id)
        assert all(
            sibling.current_attempt.status == RolloutAttemptStatus.ABANDONED
            for sibling in recovery_group.siblings
        )

    def test_stream_failure_preserves_completed_siblings(self):
        buf = _FakeCaptureBuffer()
        mgr = _make_capture_manager(
            buf,
            _FakeFinalizer(),
            num_generations=3,
            fail_after_completions=2,
        )

        with pytest.raises(RuntimeError, match="stream interrupted"):
            _run(mgr.generate_and_push(_capture_sample()))

        (group_id,) = buf.abort_calls
        recovery_group = mgr.recovery_ledger.get_group(group_id)
        assert [
            sibling.current_attempt.status for sibling in recovery_group.siblings
        ] == [
            RolloutAttemptStatus.SEALED,
            RolloutAttemptStatus.SEALED,
            RolloutAttemptStatus.ABANDONED,
        ]
        assert [
            sibling.current_attempt.staging_keys
            for sibling in recovery_group.siblings[:2]
        ] == [["stage-0"], ["stage-1"]]
        (failed_ids, reason) = mgr._env_handles["nemo_gym"].failed[0]
        assert failed_ids == [
            recovery_group.siblings[2].current_attempt.gate_rollout_id
        ]
        assert reason == "dispatch_failed"
        assert buf.commit_finalized_calls == []
