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
from dataclasses import replace
from pathlib import Path

import pytest
import torch

from nemo_rl.data.collate_fn import rl_collate_fn
from nemo_rl.data.datasets.response_datasets import NemoGymDataset
from nemo_rl.data.interfaces import DatumSpec
from nemo_rl.data.processors import nemo_gym_data_processor
from nemo_rl.distributed.batched_data_dict import BatchedDataDict
from nemo_rl.experience.interfaces import Completion, PromptGroupRecord
from nemo_rl.experience.rollout_checkpoint import (
    ROLLOUT_CHECKPOINT_SCHEMA_VERSION,
    CompletedSiblingRecord,
    IncompatibleCheckpointError,
    PersistAck,
    RolloutWorkItem,
    StorageUnavailableError,
)
from nemo_rl.experience.rollout_manager import (
    AsyncNemoGymRolloutImpl,
    RolloutCheckpointIOPolicy,
    RolloutManager,
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


class _FakeBuffer:
    """Minimal TQReplayBuffer stand-in that records reserve/commit calls."""

    def __init__(self) -> None:
        self.reserve_calls: list[int] = []  # weight_versions passed to reserve
        self.commit_calls: list[tuple[str, object, int, int]] = []
        # reserve(weight_version=X) -> group_id; commit fills the slot.
        self._slots: list[str] = []

    def reserve(
        self,
        *,
        weight_version: int,
        group_id: str | None = None,
        target_step: int | None = None,
    ) -> str:
        del target_step
        if group_id is None:
            group_id = str(uuid.uuid4())
        self.reserve_calls.append(weight_version)
        self._slots.append(group_id)
        return group_id

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


class _FakeImpl:
    """Stand-in for AsyncRolloutImpl that returns a sentinel record."""

    def __init__(self, record="sentinel-record", on_run=None) -> None:
        self._record = record
        self._on_run = on_run
        self.calls: list[tuple[object, dict, int]] = []

    async def run_rollout(
        self,
        input_sample,
        *,
        env_handles,
        num_generations_per_prompt,
    ):
        self.calls.append((input_sample, env_handles, num_generations_per_prompt))
        if self._on_run is not None:
            await self._on_run(input_sample)
        return self._record


def _make_manager(buffer: _FakeBuffer, impl: _FakeImpl) -> RolloutManager:
    """Build a RolloutManager without firing the real __init__."""
    mgr = object.__new__(RolloutManager)
    mgr._impl = impl
    mgr._tokenizer = None
    mgr._env_handles = {"train": object()}
    mgr._val_env_handles = {"validation": object()}
    mgr._tq_buffer = buffer
    mgr._weight_version = 0
    return mgr


class TestGenerateAndPushFlow:
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

        _run(mgr.generate_and_push({"prompt": "p"}, num_generations_per_prompt=2))

        assert events == ["reserve", "run", "commit"]
        assert buf.reserve_calls == [0]
        assert len(buf.commit_calls) == 1
        gid, record, start_v, end_v = buf.commit_calls[0]
        assert gid in buf._slots
        assert record == "r0"
        assert start_v == 0
        assert end_v == 0
        assert impl.calls == [({"prompt": "p"}, mgr._env_handles, 2)]

    def test_start_weight_version_pinned_at_reserve_time(self):
        """If set_weight_version is called mid-rollout, start != end."""
        buf = _FakeBuffer()

        async def _bump_weight_mid_rollout(_sample):
            # Simulate a sync_weights bump during the rollout.
            mgr.set_weight_version(5)

        impl = _FakeImpl(record="r0", on_run=_bump_weight_mid_rollout)
        mgr = _make_manager(buf, impl)
        mgr.set_weight_version(3)

        _run(mgr.generate_and_push({"prompt": "p"}, num_generations_per_prompt=2))

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

        _run(mgr.generate_and_push({"prompt": "p"}, num_generations_per_prompt=2))

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
        second_mgr = object.__new__(RolloutManager)
        second_mgr._impl = second_impl
        second_mgr._tokenizer = None
        second_mgr._env_handles = {"train": object()}
        second_mgr._val_env_handles = {"validation": object()}
        second_mgr._tq_buffer = buf
        second_mgr._weight_version = 0

        async def _drive():
            t1 = asyncio.create_task(
                first_mgr.generate_and_push(
                    {"prompt": "p1"}, num_generations_per_prompt=2
                )
            )
            # Wait until first has reserved before kicking off second so the
            # reserve ordering is deterministic.
            await first_reserved.wait()
            t2 = asyncio.create_task(
                second_mgr.generate_and_push(
                    {"prompt": "p2"}, num_generations_per_prompt=2
                )
            )
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
            _run(mgr.generate_and_push({"prompt": "p"}, num_generations_per_prompt=2))


class _ReadyRef:
    def __init__(self, value):
        self._value = value

    def __await__(self):
        async def _resolve():
            return self._value

        return _resolve().__await__()


class _ResultStream:
    def __init__(self, values):
        self._values = iter(values)

    def __aiter__(self):
        return self

    async def __anext__(self):
        try:
            return next(self._values)
        except StopIteration as error:
            raise StopAsyncIteration from error


class _StreamingRunRollouts:
    def options(self, *, num_returns):
        assert num_returns == "streaming"
        return self

    def remote(self, inputs, tokenizer, timer_prefix):
        del inputs, tokenizer, timer_prefix
        return _ResultStream(
            [
                _ReadyRef((1, _gym_result(1), None)),
                _ReadyRef((0, _gym_result(0), {"gym/total": 1.0})),
            ]
        )


def _gym_result(generation_index: int) -> dict:
    return {
        "input_message_log": [
            {
                "role": "user",
                "content": "question",
                "token_ids": torch.tensor([10]),
            }
        ],
        "message_log": [
            {
                "role": "assistant",
                "content": f"answer-{generation_index}",
                "token_ids": torch.tensor([generation_index]),
            }
        ],
        "full_result": {"reward": float(generation_index)},
    }


def _checkpoint_work(num_generations: int = 1) -> RolloutWorkItem:
    return RolloutWorkItem(
        run_id="run-1",
        group_id="group-1",
        prompt_id="prompt-1",
        dispatch_sequence=0,
        target_step=None,
        attempt_id=0,
        policy_version=0,
        prompt_fingerprint="prompt-fingerprint",
        sampling_fingerprint="sampling-fingerprint",
        tokenizer_fingerprint="tokenizer-fingerprint",
        num_generations=num_generations,
        prompt_ref={"idx": 1},
    )


def _checkpoint_io_policy(
    *,
    max_pending_writes: int = 1,
    write_timeout_s: float = 1.0,
    max_retries: int = 0,
    retry_backoff_s: float = 0.0,
    load_timeout_s: float = 1.0,
    max_load_retries: int = 0,
    load_retry_backoff_s: float = 0.0,
) -> RolloutCheckpointIOPolicy:
    return RolloutCheckpointIOPolicy(
        max_pending_writes=max_pending_writes,
        write_timeout_s=write_timeout_s,
        max_retries=max_retries,
        retry_backoff_s=retry_backoff_s,
        load_timeout_s=load_timeout_s,
        max_load_retries=max_load_retries,
        load_retry_backoff_s=load_retry_backoff_s,
    )


def _persist_ack(
    record: CompletedSiblingRecord,
    *,
    logical_key: tuple[str, str, int] | None = None,
) -> PersistAck:
    return PersistAck(
        logical_key=logical_key or record.logical_key,
        record_checksum="0" * 64,
        path=Path("/checkpoint") / f"g{record.generation_index:05d}.pt",
        already_existed=False,
    )


def _completed_sibling_record(generation_index: int) -> CompletedSiblingRecord:
    work = _checkpoint_work(num_generations=max(generation_index + 1, 1))
    return CompletedSiblingRecord(
        schema_version=ROLLOUT_CHECKPOINT_SCHEMA_VERSION,
        run_id=work.run_id,
        group_id=work.group_id,
        prompt_id=work.prompt_id,
        generation_index=generation_index,
        attempt_id=work.attempt_id,
        policy_version=work.policy_version,
        prompt_fingerprint=work.prompt_fingerprint,
        sampling_fingerprint=work.sampling_fingerprint,
        tokenizer_fingerprint=work.tokenizer_fingerprint,
        phase="SIBLING_COMPLETE",
        completion=Completion(
            message_log=[
                {
                    "role": "user",
                    "content": "question",
                    "token_ids": torch.tensor([10]),
                },
                {
                    "role": "assistant",
                    "content": f"restored-{generation_index}",
                    "token_ids": torch.tensor([generation_index]),
                },
            ],
            env_extras={"reward": float(generation_index)},
            truncated=False,
            reward=float(generation_index),
        ),
        sample_metrics={},
    )


def test_nemo_gym_stream_persists_in_completion_order_and_waits_for_acks():
    impl = object.__new__(AsyncNemoGymRolloutImpl)
    env_handles = {
        "nemo_gym": type(
            "_Environment",
            (),
            {"stream_rollouts": _StreamingRunRollouts()},
        )()
    }
    impl._tokenizer = None
    impl._max_seq_len = 32
    impl._result_to_completion = lambda result: Completion(
        message_log=result["message_log"],
        env_extras=result["full_result"],
        truncated=False,
        reward=float(result["full_result"]["reward"]),
    )
    impl._compute_rollout_metrics = lambda completions, agent: {
        "completion_count": len(completions),
        "agent": agent,
    }
    release_acks = asyncio.Event()
    callback_order: list[int] = []

    async def _persist(generation_index: int, completion: Completion) -> None:
        callback_order.append(generation_index)
        assert completion.reward == float(generation_index)
        await release_acks.wait()

    async def _drive():
        task = asyncio.create_task(
            impl._run_rollouts(
                inputs=[
                    {"_rowidx": 0, "agent_ref": {"name": "agent"}},
                    {"_rowidx": 1, "agent_ref": {"name": "agent"}},
                ],
                timer=Timer(),
                timer_prefix="timing/test",
                env_handles=env_handles,
                on_sibling_complete=_persist,
            )
        )
        while len(callback_order) < 2:
            await asyncio.sleep(0)
        assert not task.done()
        release_acks.set()
        return await task

    completions, _, metrics = _run(_drive())

    assert callback_order == [1, 0]
    assert [completion.reward for completion in completions] == [0.0, 1.0]
    assert metrics["gym/total"] == 1.0


def test_nemo_gym_stream_awaits_completed_sibling_persistence_before_reraising():
    class _FailingStream:
        def __init__(self) -> None:
            self._first = True

        def __aiter__(self):
            return self

        async def __anext__(self):
            if self._first:
                self._first = False
                return _ReadyRef((0, _gym_result(0), None))
            raise RuntimeError("injected NeMo-Gym stream failure")

    class _FailingRunRollouts(_StreamingRunRollouts):
        def remote(self, inputs, tokenizer, timer_prefix):
            del inputs, tokenizer, timer_prefix
            return _FailingStream()

    impl = object.__new__(AsyncNemoGymRolloutImpl)
    env_handles = {
        "nemo_gym": type(
            "_Environment",
            (),
            {"stream_rollouts": _FailingRunRollouts()},
        )()
    }
    impl._tokenizer = None
    impl._result_to_completion = lambda result: Completion(
        message_log=result["message_log"],
        env_extras=result["full_result"],
        truncated=False,
        reward=float(result["full_result"]["reward"]),
    )
    persisted_indices: list[int] = []

    async def _persist(generation_index: int, _completion: Completion) -> None:
        await asyncio.sleep(0)
        persisted_indices.append(generation_index)

    with pytest.raises(RuntimeError, match="injected NeMo-Gym stream failure"):
        _run(
            impl._run_rollouts(
                inputs=[
                    {"_rowidx": 0, "agent_ref": {"name": "agent"}},
                    {"_rowidx": 1, "agent_ref": {"name": "agent"}},
                ],
                timer=Timer(),
                timer_prefix="timing/test",
                env_handles=env_handles,
                on_sibling_complete=_persist,
            )
        )

    assert persisted_indices == [0]


def test_nemo_gym_stream_drains_all_persistence_tasks_before_raising():
    impl = object.__new__(AsyncNemoGymRolloutImpl)
    env_handles = {
        "nemo_gym": type(
            "_Environment",
            (),
            {"stream_rollouts": _StreamingRunRollouts()},
        )()
    }
    impl._tokenizer = None
    impl._max_seq_len = 32
    impl._result_to_completion = lambda result: Completion(
        message_log=result["message_log"],
        env_extras=result["full_result"],
        truncated=False,
        reward=float(result["full_result"]["reward"]),
    )
    impl._compute_rollout_metrics = lambda completions, agent: {}
    pending_write_started = asyncio.Event()
    release_pending_write = asyncio.Event()

    async def _persist(generation_index: int, _completion: Completion) -> None:
        if generation_index == 1:
            raise StorageUnavailableError("injected persistence failure")
        pending_write_started.set()
        await release_pending_write.wait()

    async def _drive() -> None:
        task = asyncio.create_task(
            impl._run_rollouts(
                inputs=[
                    {"_rowidx": 0, "agent_ref": {"name": "agent"}},
                    {"_rowidx": 1, "agent_ref": {"name": "agent"}},
                ],
                timer=Timer(),
                timer_prefix="timing/test",
                env_handles=env_handles,
                on_sibling_complete=_persist,
            )
        )
        await pending_write_started.wait()
        await asyncio.sleep(0)
        assert not task.done()
        release_pending_write.set()
        with pytest.raises(
            StorageUnavailableError, match="injected persistence failure"
        ):
            await task

    _run(_drive())


def test_rollout_manager_builds_completed_sibling_record_and_gates_group_return():
    completion = Completion(
        message_log=[
            {
                "role": "assistant",
                "content": "answer",
                "token_ids": torch.tensor([7]),
            }
        ],
        env_extras={"reward": 1.0},
        truncated=False,
        reward=1.0,
    )

    class _GymImpl(AsyncNemoGymRolloutImpl):
        async def run_rollout(
            self,
            input_sample,
            *,
            env_handles,
            num_generations_per_prompt,
            on_sibling_complete=None,
            restored_completions=None,
        ) -> PromptGroupRecord:
            assert env_handles == {"nemo_gym": "train"}
            assert num_generations_per_prompt == 1
            assert on_sibling_complete is not None
            assert restored_completions == {}
            await on_sibling_complete(0, completion)
            return PromptGroupRecord(
                prompt_idx=input_sample["idx"],
                prompt=[],
                extra_env_info={},
                metadata={"task_name": "nemo_gym"},
                completions=[completion],
                rollout_metrics={},
            )

    class _Writer:
        def __init__(self) -> None:
            self.started = asyncio.Event()
            self.release = asyncio.Event()
            self.records = []

        async def load_completed(self, _work):
            return {}

        async def persist_completed(self, record):
            self.records.append(record)
            self.started.set()
            await self.release.wait()
            return _persist_ack(record)

    manager = object.__new__(RolloutManager)
    manager._impl = object.__new__(_GymImpl)
    manager._env_handles = {"nemo_gym": "train"}
    manager._val_env_handles = {"nemo_gym": "validation"}
    manager._checkpoint_io_policy = _checkpoint_io_policy()
    manager._checkpoint_io_semaphore = None
    writer = _Writer()

    async def _drive():
        task = asyncio.create_task(
            manager.run_rollout(
                {"idx": 1},
                num_generations_per_prompt=1,
                checkpoint_work=_checkpoint_work(),
                checkpoint_writer=writer,
            )
        )
        await writer.started.wait()
        assert not task.done()
        writer.release.set()
        return await task

    record = _run(_drive())

    assert len(record.completions) == 1
    assert len(writer.records) == 1
    persisted = writer.records[0]
    assert persisted.logical_key == ("run-1", "group-1", 0)
    assert persisted.policy_version == 0
    assert persisted.completion.message_log[0]["token_ids"].device.type == "cpu"


def test_rollout_manager_retries_transient_checkpoint_storage_failure():
    completion = Completion(
        message_log=[
            {
                "role": "assistant",
                "content": "answer",
                "token_ids": torch.tensor([7]),
            }
        ],
        env_extras={"reward": 1.0},
        truncated=False,
        reward=1.0,
    )

    class _Writer:
        def __init__(self) -> None:
            self.records = []

        async def persist_completed(self, record):
            self.records.append(record)
            if len(self.records) == 1:
                raise StorageUnavailableError("injected transient failure")
            return _persist_ack(record)

    manager = object.__new__(RolloutManager)
    manager._checkpoint_io_policy = _checkpoint_io_policy(max_retries=1)
    manager._checkpoint_io_semaphore = None
    writer = _Writer()

    _run(
        manager._persist_completed_sibling(
            _checkpoint_work(),
            writer,
            0,
            completion,
        )
    )

    assert len(writer.records) == 2
    assert writer.records[0] is writer.records[1]


def test_rollout_manager_bounds_unacknowledged_checkpoint_wait():
    completion = Completion(
        message_log=[],
        env_extras={},
        truncated=False,
        reward=1.0,
    )

    class _Writer:
        def __init__(self) -> None:
            self.calls = 0

        async def persist_completed(self, _record):
            self.calls += 1
            await asyncio.Event().wait()

    manager = object.__new__(RolloutManager)
    manager._checkpoint_io_policy = _checkpoint_io_policy(
        write_timeout_s=0.01,
        max_retries=1,
    )
    manager._checkpoint_io_semaphore = None
    writer = _Writer()

    with pytest.raises(
        StorageUnavailableError, match="did not durably acknowledge.*after 2 attempts"
    ):
        _run(
            manager._persist_completed_sibling(
                _checkpoint_work(),
                writer,
                0,
                completion,
            )
        )

    assert writer.calls == 2


def test_rollout_manager_rejects_mismatched_checkpoint_acknowledgement():
    completion = Completion(
        message_log=[],
        env_extras={},
        truncated=False,
        reward=1.0,
    )

    class _Writer:
        async def persist_completed(self, record):
            return _persist_ack(record, logical_key=("other-run", "group-1", 0))

    manager = object.__new__(RolloutManager)
    manager._checkpoint_io_policy = _checkpoint_io_policy()
    manager._checkpoint_io_semaphore = None

    with pytest.raises(ValueError, match="acknowledgement key does not match"):
        _run(
            manager._persist_completed_sibling(
                _checkpoint_work(),
                _Writer(),
                0,
                completion,
            )
        )


def test_nemo_gym_reuses_restored_siblings_and_generates_only_missing_indices():
    requested_row_indices: list[int] = []

    class _MissingOnlyRunRollouts(_StreamingRunRollouts):
        def remote(self, inputs, tokenizer, timer_prefix):
            del tokenizer, timer_prefix
            requested_row_indices.extend(row["_rowidx"] for row in inputs)
            return _ResultStream([_ReadyRef((1, _gym_result(1), {"gym/total": 1.0}))])

    impl = object.__new__(AsyncNemoGymRolloutImpl)
    env_handles = {
        "nemo_gym": type(
            "_Environment",
            (),
            {"stream_rollouts": _MissingOnlyRunRollouts()},
        )()
    }
    impl._tokenizer = None
    impl._max_seq_len = 32
    impl._result_to_completion = lambda result: Completion(
        message_log=result["message_log"],
        env_extras=result["full_result"],
        truncated=False,
        reward=float(result["full_result"]["reward"]),
    )
    impl._compute_rollout_metrics = lambda completions, agent: {
        "completion_count": len(completions),
        "agent": agent,
    }
    persisted_indices: list[int] = []

    async def _persist(generation_index: int, _completion: Completion) -> None:
        persisted_indices.append(generation_index)

    completions, _, metrics = _run(
        impl._run_rollouts(
            inputs=[
                {"_rowidx": 0, "agent_ref": {"name": "agent"}},
                {"_rowidx": 1, "agent_ref": {"name": "agent"}},
            ],
            timer=Timer(),
            timer_prefix="timing/test",
            env_handles=env_handles,
            on_sibling_complete=_persist,
            restored_completions={
                0: _completed_sibling_record(0).completion,
            },
        )
    )

    assert requested_row_indices == [1]
    assert persisted_indices == [1]
    assert [completion.reward for completion in completions] == [0.0, 1.0]
    assert metrics["completion_count"] == 2
    assert metrics["gym/total"] == 1.0


def test_nemo_gym_fully_restored_group_skips_generation():
    impl = object.__new__(AsyncNemoGymRolloutImpl)
    env_handles = {"nemo_gym": object()}
    impl._tokenizer = None
    impl._compute_rollout_metrics = lambda completions, agent: {
        "completion_count": len(completions),
        "agent": agent,
    }

    async def _unexpected_persist(
        _generation_index: int, _completion: Completion
    ) -> None:
        raise AssertionError("restored siblings must not be persisted again")

    completions, prompt, metrics = _run(
        impl._run_rollouts(
            inputs=[
                {"_rowidx": 0, "agent_ref": {"name": "agent"}},
                {"_rowidx": 1, "agent_ref": {"name": "agent"}},
            ],
            timer=Timer(),
            timer_prefix="timing/test",
            env_handles=env_handles,
            on_sibling_complete=_unexpected_persist,
            restored_completions={
                0: _completed_sibling_record(0).completion,
                1: _completed_sibling_record(1).completion,
            },
        )
    )

    assert [completion.reward for completion in completions] == [0.0, 1.0]
    assert prompt[0]["role"] == "user"
    assert metrics["completion_count"] == 2


def test_rollout_manager_loads_checkpoint_before_starting_gym_rollout():
    events: list[str] = []
    loaded_record = _completed_sibling_record(0)

    class _GymImpl(AsyncNemoGymRolloutImpl):
        async def run_rollout(
            self,
            input_sample,
            *,
            env_handles,
            num_generations_per_prompt,
            on_sibling_complete=None,
            restored_completions=None,
        ) -> PromptGroupRecord:
            events.append("run")
            assert env_handles == {"nemo_gym": "train"}
            assert num_generations_per_prompt == 1
            assert on_sibling_complete is not None
            assert restored_completions is not None
            assert set(restored_completions) == {0}
            return PromptGroupRecord(
                prompt_idx=input_sample["idx"],
                prompt=restored_completions[0].message_log[:1],
                extra_env_info={},
                metadata={"task_name": "nemo_gym"},
                completions=[restored_completions[0]],
                rollout_metrics={},
            )

    class _Writer:
        async def load_completed(self, work):
            events.append("load")
            assert work == _checkpoint_work()
            return {0: loaded_record}

        async def persist_completed(self, _record):
            raise AssertionError("restored sibling must not be persisted again")

    manager = object.__new__(RolloutManager)
    manager._impl = object.__new__(_GymImpl)
    manager._env_handles = {"nemo_gym": "train"}
    manager._val_env_handles = {"nemo_gym": "validation"}
    manager._checkpoint_io_policy = _checkpoint_io_policy()
    manager._checkpoint_io_semaphore = None

    result = _run(
        manager.run_rollout(
            {"idx": 1},
            num_generations_per_prompt=1,
            checkpoint_work=_checkpoint_work(),
            checkpoint_writer=_Writer(),
        )
    )

    assert events == ["load", "run"]
    assert result.completions[0].reward == 0.0


def test_rollout_manager_retries_transient_checkpoint_load_failure():
    loaded_record = _completed_sibling_record(0)
    replacement_work = replace(_checkpoint_work(), attempt_id=1)

    class _Writer:
        def __init__(self) -> None:
            self.calls = 0

        async def load_completed(self, _work):
            self.calls += 1
            if self.calls == 1:
                raise StorageUnavailableError("injected transient read failure")
            return {0: loaded_record}

    manager = object.__new__(RolloutManager)
    manager._checkpoint_io_policy = _checkpoint_io_policy(max_load_retries=1)
    manager._checkpoint_io_semaphore = None
    writer = _Writer()

    restored = _run(
        manager._load_completed_siblings(
            replacement_work,
            writer,
        )
    )

    assert writer.calls == 2
    assert restored[0].reward == 0.0


def test_rollout_manager_bounds_unacknowledged_checkpoint_load():
    class _Writer:
        def __init__(self) -> None:
            self.calls = 0

        async def load_completed(self, _work):
            self.calls += 1
            await asyncio.Event().wait()

    manager = object.__new__(RolloutManager)
    manager._checkpoint_io_policy = _checkpoint_io_policy(
        load_timeout_s=0.01,
        max_load_retries=1,
    )
    manager._checkpoint_io_semaphore = None
    writer = _Writer()

    with pytest.raises(
        StorageUnavailableError,
        match="did not return completed siblings.*after 2 attempts",
    ):
        _run(
            manager._load_completed_siblings(
                _checkpoint_work(),
                writer,
            )
        )

    assert writer.calls == 2


def test_rollout_manager_bounds_concurrent_checkpoint_loads():
    class _Writer:
        def __init__(self) -> None:
            self.calls = 0
            self.active_calls = 0
            self.max_active_calls = 0
            self.first_call_started = asyncio.Event()
            self.release = asyncio.Event()

        async def load_completed(self, _work):
            self.calls += 1
            self.active_calls += 1
            self.max_active_calls = max(self.max_active_calls, self.active_calls)
            self.first_call_started.set()
            await self.release.wait()
            self.active_calls -= 1
            return {}

    manager = object.__new__(RolloutManager)
    manager._checkpoint_io_policy = _checkpoint_io_policy(max_pending_writes=1)
    manager._checkpoint_io_semaphore = None
    writer = _Writer()

    async def _drive() -> None:
        first = asyncio.create_task(
            manager._load_completed_siblings(_checkpoint_work(), writer)
        )
        second = asyncio.create_task(
            manager._load_completed_siblings(_checkpoint_work(), writer)
        )
        await writer.first_call_started.wait()
        await asyncio.sleep(0)
        assert writer.calls == 1
        writer.release.set()
        await asyncio.gather(first, second)

    _run(_drive())

    assert writer.calls == 2
    assert writer.max_active_calls == 1


@pytest.mark.parametrize(
    "loaded_record, error_match",
    [
        (
            replace(_completed_sibling_record(0), prompt_id="other-prompt"),
            "incompatible with rollout work",
        ),
        (
            replace(_completed_sibling_record(0), attempt_id=1),
            "newer than dispatched attempt",
        ),
    ],
)
def test_rollout_manager_rejects_incompatible_loaded_checkpoint(
    loaded_record: CompletedSiblingRecord,
    error_match: str,
) -> None:
    class _Writer:
        async def load_completed(self, _work):
            return {0: loaded_record}

    manager = object.__new__(RolloutManager)
    manager._checkpoint_io_policy = _checkpoint_io_policy()
    manager._checkpoint_io_semaphore = None

    with pytest.raises(IncompatibleCheckpointError, match=error_match):
        _run(
            manager._load_completed_siblings(
                _checkpoint_work(),
                _Writer(),
            )
        )


# ---------------------------------------------------------------------------
# Tests for RolloutManager
# ---------------------------------------------------------------------------


def test_run_rollout_routes_per_call_count_and_environment():
    impl = _FakeImpl()
    manager = _make_manager(_FakeBuffer(), impl)
    train_sample = {"prompt": "train"}
    val_sample = {"prompt": "validation"}

    _run(
        manager.run_rollout(
            train_sample,
            num_generations_per_prompt=4,
        )
    )
    _run(
        manager.run_rollout(
            val_sample,
            num_generations_per_prompt=1,
            is_validation=True,
        )
    )
    _run(
        manager.run_rollout(
            train_sample,
            num_generations_per_prompt=4,
        )
    )

    assert impl.calls == [
        (train_sample, manager._env_handles, 4),
        (val_sample, manager._val_env_handles, 1),
        (train_sample, manager._env_handles, 4),
    ]


def test_rollout_manager_raises_without_impl_params():
    """RolloutManager raises AssertionError when required params are missing."""
    common = {
        "tokenizer": None,
        "env_handles": {},
        "val_env_handles": {},
        "max_seq_len": 1,
    }

    with pytest.raises(AssertionError, match="num_generations_per_prompt must be >= 1"):
        manager = RolloutManager(
            **common,
            policy_generation=object(),
            use_nemo_gym=False,
        )
        _run(manager.run_rollout({}, num_generations_per_prompt=0))

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
    vllm_generation, tokenizer, env_handles, _, _ = multi_step_setup_vllm_async
    input_sample = single_multi_step_calculator_input_sample
    num_generations = 2
    max_seq_len = 1024
    max_rollout_turns = input_sample["extra_env_info"]["max_steps"] + 1

    manager = RolloutManager(
        use_nemo_gym=False,
        tokenizer=tokenizer,
        env_handles=env_handles,
        val_env_handles=env_handles,
        max_seq_len=max_seq_len,
        max_rollout_turns=max_rollout_turns,
        policy_generation=vllm_generation,
    )

    vllm_generation.prepare_for_generation()
    record = asyncio.run(
        manager.run_rollout(
            input_sample,
            num_generations_per_prompt=num_generations,
        )
    )
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
    vllm_generation, tokenizer, env_handles, _, _ = multi_step_setup_vllm_async
    input_sample = single_multi_step_calculator_input_sample
    num_generations = 2
    max_seq_len = 290
    max_rollout_turns = input_sample["extra_env_info"]["max_steps"] + 1

    manager = RolloutManager(
        use_nemo_gym=False,
        tokenizer=tokenizer,
        env_handles=env_handles,
        val_env_handles=env_handles,
        max_seq_len=max_seq_len,
        max_rollout_turns=max_rollout_turns,
        policy_generation=vllm_generation,
    )
    vllm_generation.prepare_for_generation()
    record = asyncio.run(
        manager.run_rollout(
            input_sample,
            num_generations_per_prompt=num_generations,
        )
    )
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
    vllm_generation, tokenizer, env_handles, _, _ = multi_step_setup_vllm_async
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
        task_to_env=env_handles,
        max_seq_len=max_seq_len,
        max_rollout_turns=max_rollout_turns,
    )

    manager = RolloutManager(
        use_nemo_gym=False,
        tokenizer=tokenizer,
        env_handles=env_handles,
        val_env_handles=env_handles,
        max_seq_len=max_seq_len,
        max_rollout_turns=max_rollout_turns,
        policy_generation=vllm_generation,
    )
    record = asyncio.run(
        manager.run_rollout(
            input_sample,
            num_generations_per_prompt=num_generations,
        )
    )
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
        env_handles={"nemo_gym": nemo_gym},
        val_env_handles={"nemo_gym": nemo_gym},
        max_seq_len=nemo_gym_vllm_generation.cfg["vllm_cfg"]["max_model_len"],
        generation_config=nemo_gym_vllm_generation.cfg,
    )
    record = asyncio.run(
        manager.run_rollout(
            single_prompt,
            num_generations_per_prompt=num_generations,
        )
    )

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

    original_result = run_async_nemo_gym_rollout(
        policy_generation=nemo_gym_vllm_generation,
        input_batch=repeated_batch,
        tokenizer=nemo_gym_tokenizer,
        task_to_env={"nemo_gym": nemo_gym},
        generation_config=nemo_gym_vllm_generation.cfg,
        max_seq_len=nemo_gym_vllm_generation.cfg["vllm_cfg"]["max_model_len"],
        max_rollout_turns=None,
    )

    manager = RolloutManager(
        use_nemo_gym=True,
        tokenizer=nemo_gym_tokenizer,
        env_handles={"nemo_gym": nemo_gym},
        val_env_handles={"nemo_gym": nemo_gym},
        max_seq_len=nemo_gym_vllm_generation.cfg["vllm_cfg"]["max_model_len"],
        generation_config=nemo_gym_vllm_generation.cfg,
    )
    record = asyncio.run(
        manager.run_rollout(
            single_prompt,
            num_generations_per_prompt=num_generations,
        )
    )

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
