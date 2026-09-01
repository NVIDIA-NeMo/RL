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

"""Unit tests for TQReplayBuffer (plain SC-process buffer + TQ proxy)."""

from __future__ import annotations

import asyncio
import threading
from typing import Any

import pytest
import torch

import nemo_rl.algorithms.async_utils.replay_buffer as _replay_buffer_module
from nemo_rl.algorithms.async_utils.replay_buffer import (
    REPLAY_BUFFER_METADATA_SCHEMA_VERSION,
    REPLAY_BUFFER_METADATA_STORAGE,
    DataPlaneCheckpointBarrier,
    PostWriteEnrichmentError,
    TQReplayBuffer,
    replay_manifest_digest,
)
from nemo_rl.data_plane import KVBatchMeta
from nemo_rl.distributed.batched_data_dict import BatchedDataDict
from nemo_rl.experience.interfaces import PromptGroupRecord

# Each record yields _N_GENS training rows.
_N_GENS = 2


def _stub_record_to_train_batch(
    record: PromptGroupRecord,
    *,
    pad_value_dict: Any,
    include_message_violation_fields: bool,
) -> BatchedDataDict[Any]:
    del record, pad_value_dict, include_message_violation_fields
    return BatchedDataDict[Any](
        {
            "input_ids": torch.ones((_N_GENS, 3), dtype=torch.long),
            "input_lengths": torch.full((_N_GENS,), 3, dtype=torch.long),
            "total_reward": torch.zeros(_N_GENS, dtype=torch.float32),
        }
    )


@pytest.fixture(autouse=True)
def _patch_converter(monkeypatch):
    """Bypass the real ``record_to_train_batch`` so tests can use empty records."""
    monkeypatch.setattr(
        _replay_buffer_module,
        "record_to_train_batch",
        _stub_record_to_train_batch,
    )


class FakeDataPlaneClient:
    """Sync in-memory DataPlaneClient stub used by TQReplayBuffer tests."""

    def __init__(self, partition_id: str = "rollout_data") -> None:
        self._partition_id = partition_id
        self._rows: dict[str, dict[str, Any]] = {}
        self.put_calls: list[dict[str, Any]] = []
        self.clear_calls: list[list[str]] = []
        self.clear_thread_ids: list[int] = []
        self.get_calls: list[dict[str, Any]] = []

    def put_samples(
        self,
        sample_ids: list[str],
        partition_id: str,
        fields: Any = None,
        tags: list[dict[str, Any]] | None = None,
    ) -> KVBatchMeta:
        assert partition_id == self._partition_id
        self.put_calls.append(
            {
                "sample_ids": list(sample_ids),
                "fields": fields,
                "tags": [dict(t) for t in tags] if tags is not None else None,
            }
        )
        for i, sid in enumerate(sample_ids):
            self._rows[sid] = {
                "tag": dict(tags[i]) if tags is not None else {},
            }
        return KVBatchMeta(
            partition_id=partition_id,
            task_name=None,
            sample_ids=list(sample_ids),
            fields=None,
            tags=[dict(t) for t in tags] if tags is not None else None,
        )

    def clear_samples(self, sample_ids: list[str] | None, partition_id: str) -> None:
        assert partition_id == self._partition_id
        self.clear_thread_ids.append(threading.get_ident())
        ids = list(sample_ids) if sample_ids is not None else list(self._rows)
        self.clear_calls.append(list(ids))
        for sid in ids:
            self._rows.pop(sid, None)

    def list_sample_ids(self, partition_id: str) -> list[str]:
        assert partition_id == self._partition_id
        return sorted(self._rows)

    def get_samples(
        self,
        sample_ids: list[str],
        partition_id: str,
        select_fields: list[str] | None = None,
    ) -> dict[str, Any]:
        assert partition_id == self._partition_id
        self.get_calls.append(
            {
                "sample_ids": list(sample_ids),
                "select_fields": (
                    list(select_fields) if select_fields is not None else None
                ),
            }
        )
        # Opaque payload used by tests that inspect direct DataPlane reads.
        return {"payload_for": list(sample_ids)}

    def depth(self) -> int:
        return len(self._rows)


class FailAfterPutDataPlaneClient(FakeDataPlaneClient):
    """Write all rows, then fail to simulate a partial-success RPC."""

    def put_samples(
        self,
        sample_ids: list[str],
        partition_id: str,
        fields: Any = None,
        tags: list[dict[str, Any]] | None = None,
    ) -> KVBatchMeta:
        super().put_samples(sample_ids, partition_id, fields, tags)
        raise RuntimeError("injected put failure")


class FailAfterPutAndClearDataPlaneClient(FailAfterPutDataPlaneClient):
    """Fail both the canonical write and its deterministic-ID rollback."""

    def clear_samples(self, sample_ids: list[str] | None, partition_id: str) -> None:
        del sample_ids, partition_id
        raise OSError("injected rollback failure")


def _run(coro):
    return asyncio.run(coro)


def _make_record(*, prompt_idx: int = 0) -> PromptGroupRecord:
    """Opaque PromptGroupRecord — converter is stubbed, so contents are unused."""
    return PromptGroupRecord(
        prompt_idx=prompt_idx,
        prompt=[],
        extra_env_info=None,
        metadata={},
        completions=[],
        rollout_metrics={},
    )


def _make_buffer(
    dp: FakeDataPlaneClient,
    *,
    require_routed_experts: bool = False,
    checkpoint_barrier: DataPlaneCheckpointBarrier | None = None,
) -> TQReplayBuffer:
    buffer = TQReplayBuffer(
        dp,
        partition_id="rollout_data",
        pad_value_dict={"token_ids": 0},
        include_message_violation_fields=False,
        require_routed_experts=require_routed_experts,
    )
    buffer.set_data_plane_checkpoint_barrier(
        checkpoint_barrier or DataPlaneCheckpointBarrier()
    )
    return buffer


def _add_group(
    buf: TQReplayBuffer,
    weight: int,
    end_weight: int | None = None,
    target_step: int | None = None,
) -> KVBatchMeta:
    if end_weight is None:
        end_weight = weight
    group_id = buf.reserve(weight_version=weight, target_step=target_step)
    return _run(
        buf.commit(
            group_id,
            _make_record(),
            start_weight_version=weight,
            end_weight_version=end_weight,
        )
    )


class TestDataPlaneCheckpointBarrier:
    def test_mutation_and_checkpoint_cuts_expire_on_context_exit(self):
        async def exercise() -> None:
            barrier = DataPlaneCheckpointBarrier()

            async with barrier.mutation() as mutation_cut:
                mutation_cut.require_live()
            with pytest.raises(RuntimeError, match="no longer active"):
                mutation_cut.require_live()

            async with barrier.checkpoint() as checkpoint_cut:
                checkpoint_cut.require_live()
            with pytest.raises(RuntimeError, match="no longer active"):
                checkpoint_cut.require_live()

        asyncio.run(exercise())

    def test_mutations_run_concurrently_without_checkpoint(self):
        async def exercise() -> None:
            barrier = DataPlaneCheckpointBarrier()
            both_entered = asyncio.Event()
            release = asyncio.Event()
            active = 0

            async def mutate() -> None:
                nonlocal active
                async with barrier.mutation():
                    active += 1
                    if active == 2:
                        both_entered.set()
                    await release.wait()
                    active -= 1

            tasks = [asyncio.create_task(mutate()) for _ in range(2)]
            await asyncio.wait_for(both_entered.wait(), timeout=5.0)
            assert active == 2
            release.set()
            await asyncio.gather(*tasks)

        asyncio.run(exercise())

    def test_checkpoint_waits_for_active_mutation(self):
        async def exercise() -> None:
            barrier = DataPlaneCheckpointBarrier()
            mutation_entered = asyncio.Event()
            release_mutation = asyncio.Event()
            checkpoint_entered = asyncio.Event()

            async def mutate() -> None:
                async with barrier.mutation():
                    mutation_entered.set()
                    await release_mutation.wait()

            async def checkpoint() -> None:
                async with barrier.checkpoint():
                    checkpoint_entered.set()

            mutation_task = asyncio.create_task(mutate())
            await mutation_entered.wait()
            checkpoint_task = asyncio.create_task(checkpoint())
            await asyncio.sleep(0)
            assert not checkpoint_entered.is_set()

            release_mutation.set()
            await asyncio.gather(mutation_task, checkpoint_task)
            assert checkpoint_entered.is_set()

        asyncio.run(exercise())

    def test_two_checkpoints_serialize_without_deadlock(self):
        async def exercise() -> None:
            barrier = DataPlaneCheckpointBarrier()
            release = asyncio.Event()
            entered: list[str] = []

            async def checkpoint(tag: str) -> None:
                async with barrier.checkpoint():
                    entered.append(f"{tag}-enter")
                    await release.wait()
                    entered.append(f"{tag}-exit")

            first = asyncio.create_task(checkpoint("first"))
            await asyncio.sleep(0)
            second = asyncio.create_task(checkpoint("second"))
            await asyncio.sleep(0)
            assert entered == ["first-enter"]

            release.set()
            await asyncio.wait_for(asyncio.gather(first, second), timeout=5.0)
            assert entered == [
                "first-enter",
                "first-exit",
                "second-enter",
                "second-exit",
            ]

        asyncio.run(exercise())


class TestTQReplayBufferReserveCommit:
    def test_commit_waits_for_active_checkpoint(self):
        async def exercise() -> None:
            dp = FakeDataPlaneClient()
            checkpoint_barrier = DataPlaneCheckpointBarrier()
            buf = _make_buffer(dp, checkpoint_barrier=checkpoint_barrier)
            group_id = buf.reserve(weight_version=3)

            async with checkpoint_barrier.checkpoint():
                commit_task = asyncio.create_task(
                    buf.commit(
                        group_id,
                        _make_record(),
                        start_weight_version=3,
                        end_weight_version=3,
                    )
                )
                await asyncio.sleep(0)
                assert dp.put_calls == []
                assert buf.ready_list == [False]

            await commit_task
            assert len(dp.put_calls) == 1
            assert buf.ready_list == [True]

        asyncio.run(exercise())

    def test_commit_enriches_after_put_before_slot_becomes_ready(self):
        dp = FakeDataPlaneClient()
        buf = _make_buffer(dp)
        observations = []

        async def enrich(meta, record):
            del record
            observations.append((dp.depth(), list(buf.ready_list)))
            return meta.with_fields(["teacher_reference_logprobs"])

        buf.set_post_write_enricher(enrich)
        meta = _add_group(buf, weight=3)

        assert observations == [(_N_GENS, [False])]
        assert "teacher_reference_logprobs" in meta.fields
        assert buf.ready_list == [True]

    def test_commit_rolls_back_when_post_write_enrichment_fails(self):
        dp = FakeDataPlaneClient()
        buf = _make_buffer(dp)

        async def fail_enrichment(meta, record):
            del meta, record
            raise RuntimeError("teacher unavailable")

        buf.set_post_write_enricher(fail_enrichment)
        group_id = buf.reserve(weight_version=3)

        with pytest.raises(PostWriteEnrichmentError, match="post-write enrichment"):
            _run(
                buf.commit(
                    group_id,
                    _make_record(),
                    start_weight_version=3,
                    end_weight_version=3,
                )
            )

        assert dp.depth() == 0
        assert buf.ready_list == [False]
        assert buf.meta_list == [None]

    def test_commit_clears_rows_when_put_raises_after_writing(self):
        dp = FailAfterPutDataPlaneClient()
        buf = _make_buffer(dp)
        group_id = buf.reserve(weight_version=3)

        with pytest.raises(RuntimeError, match="injected put failure"):
            _run(
                buf.commit(
                    group_id,
                    _make_record(),
                    start_weight_version=3,
                    end_weight_version=3,
                )
            )

        assert dp.depth() == 0
        assert dp.clear_calls == [dp.put_calls[0]["sample_ids"]]
        # commit() rolls back DataPlane rows; generate_and_push() owns removal
        # of the reserved buffer slot.
        assert buf.size() == 1
        assert buf.ready_list == [False]
        assert buf.meta_list == [None]

    def test_commit_reports_both_write_and_rollback_failures(self):
        dp = FailAfterPutAndClearDataPlaneClient()
        buf = _make_buffer(dp)
        group_id = buf.reserve(weight_version=3)

        with pytest.raises(BaseExceptionGroup) as exc_info:
            _run(
                buf.commit(
                    group_id,
                    _make_record(),
                    start_weight_version=3,
                    end_weight_version=3,
                )
            )

        assert exc_info.value.subgroup(RuntimeError) is not None
        assert exc_info.value.subgroup(OSError) is not None
        # The failed rollback leaves uncertain external rows and an unready local
        # slot. Both failures must remain visible so callers abort instead of retrying
        # the same stable group ID over potentially orphaned data.
        assert dp.depth() == _N_GENS
        assert buf.ready_list == [False]

    def test_reserve_appends_placeholder_unready(self):
        dp = FakeDataPlaneClient()
        buf = _make_buffer(dp)

        group_id = buf.reserve(weight_version=3)

        assert isinstance(group_id, str) and group_id
        assert buf.size() == 1
        assert buf.start_weight_list == [3]
        assert buf.end_weight_list == [-1]
        assert buf.ready_list == [False]
        assert buf.meta_list == [None]
        assert dp.depth() == 0
        assert dp.put_calls == []

    def test_commit_writes_tq_then_fills_meta(self, monkeypatch):
        dp = FakeDataPlaneClient()
        buf = _make_buffer(dp)
        trace_calls = []
        monkeypatch.setattr(
            _replay_buffer_module,
            "trace_rollout_payload",
            lambda **kwargs: trace_calls.append(kwargs),
        )

        group_id = buf.reserve(weight_version=3)
        meta = _run(
            buf.commit(
                group_id,
                _make_record(prompt_idx=418),
                start_weight_version=3,
                end_weight_version=4,
            )
        )

        # pack_payload stamps sample_ids as ``{group_uuid}_g{i}``.
        assert len(meta.sample_ids) == _N_GENS
        head, _, idx = meta.sample_ids[0].rpartition("_g")
        assert head == group_id and idx == "0"
        assert all(sid.startswith(group_id + "_g") for sid in meta.sample_ids)
        assert dp.depth() == _N_GENS
        assert buf.size() == 1
        assert buf.start_weight_list == [3]
        assert buf.end_weight_list == [4]
        assert buf.ready_list == [True]
        assert buf.meta_list[0].sample_ids == meta.sample_ids
        # TQ tags preserve both dispatch-time weight and dataset identity.
        assert meta.tags == [{"weight_version": 3, "prompt_idx": 418}] * _N_GENS
        assert len(dp.put_calls) == 1
        assert len(trace_calls) == 1
        assert trace_calls[0]["keys"] == meta.sample_ids
        assert trace_calls[0]["data"]["input_lengths"].tolist() == [3, 3]

    def test_commit_requires_routed_experts_before_tq_write(self):
        dp = FakeDataPlaneClient()
        buf = _make_buffer(dp, require_routed_experts=True)
        group_id = buf.reserve(weight_version=3)

        with pytest.raises(
            RuntimeError,
            match="router_replay.enabled=true requires routed_experts",
        ):
            _run(
                buf.commit(
                    group_id,
                    _make_record(),
                    start_weight_version=3,
                    end_weight_version=3,
                )
            )

        assert dp.put_calls == []
        assert dp.depth() == 0
        assert buf.ready_list == [False]

    def test_commit_raises_for_unknown_group_id(self):
        dp = FakeDataPlaneClient()
        buf = _make_buffer(dp)
        buf.reserve(weight_version=3)

        with pytest.raises(ValueError):
            _run(
                buf.commit(
                    "not-a-real-id",
                    _make_record(),
                    start_weight_version=3,
                    end_weight_version=3,
                )
            )

        # No orphan rows in DataPlane: commit must validate group_id before writing.
        assert dp.depth() == 0
        assert dp.put_calls == []

    def test_reserve_then_commit_preserves_dispatch_order(self):
        """Reserve in dispatch order, commit out of order; insertion order holds."""
        dp = FakeDataPlaneClient()
        buf = _make_buffer(dp)

        weights = (1, 2, 3)
        gids = [buf.reserve(weight_version=w) for w in weights]
        # Commit out of order: 2, 0, 1 — buffer order must still match reserve order.
        for i in (2, 0, 1):
            _run(
                buf.commit(
                    gids[i],
                    _make_record(),
                    start_weight_version=weights[i],
                    end_weight_version=weights[i],
                )
            )

        assert buf.size() == 3
        assert buf.start_weight_list == [1, 2, 3]
        assert buf.end_weight_list == [1, 2, 3]
        assert buf.ready_list == [True, True, True]
        # sample_id head equals reserved group_id at each slot.
        for i, gid in enumerate(gids):
            assert buf.meta_list[i] is not None
            assert buf.meta_list[i].sample_ids[0].startswith(gid + "_g")

    def test_commit_appends_multiple_records_in_order(self):
        dp = FakeDataPlaneClient()
        buf = _make_buffer(dp)

        metas = [_add_group(buf, weight=w) for w in (1, 2, 3)]

        assert buf.size() == 3
        assert buf.start_weight_list == [1, 2, 3]
        assert buf.end_weight_list == [1, 2, 3]
        assert [m.sample_ids for m in buf.meta_list] == [
            list(metas[0].sample_ids),
            list(metas[1].sample_ids),
            list(metas[2].sample_ids),
        ]


class TestTQReplayBufferRemove:
    def test_remove_with_dp_clear_fails_without_bound_checkpoint_barrier(self):
        dp = FakeDataPlaneClient()
        buf = TQReplayBuffer(
            dp,
            partition_id="rollout_data",
            pad_value_dict={"token_ids": 0},
            include_message_violation_fields=False,
        )

        with pytest.raises(RuntimeError, match="must be bound"):
            _run(buf.remove([0], remove_in_dp=True))

        assert dp.clear_calls == []

    def test_dp_clear_waits_for_active_checkpoint(self):
        async def exercise() -> None:
            dp = FakeDataPlaneClient()
            checkpoint_barrier = DataPlaneCheckpointBarrier()
            buf = _make_buffer(dp, checkpoint_barrier=checkpoint_barrier)
            group_id = buf.reserve(weight_version=0)
            await buf.commit(
                group_id,
                _make_record(),
                start_weight_version=0,
                end_weight_version=0,
            )

            async with checkpoint_barrier.checkpoint():
                remove_task = asyncio.create_task(buf.remove([0], remove_in_dp=True))
                await asyncio.sleep(0)
                assert dp.clear_calls == []

            await remove_task
            assert dp.clear_calls == [dp.put_calls[0]["sample_ids"]]

        asyncio.run(exercise())

    def test_dp_clear_does_not_block_actor_event_loop(self):
        async def exercise() -> tuple[FakeDataPlaneClient, int]:
            dp = FakeDataPlaneClient()
            buf = _make_buffer(dp)
            group_id = buf.reserve(weight_version=0)
            await buf.commit(
                group_id,
                _make_record(),
                start_weight_version=0,
                end_weight_version=0,
            )
            event_loop_thread_id = threading.get_ident()
            await buf.remove([0], remove_in_dp=True)
            return dp, event_loop_thread_id

        dp, event_loop_thread_id = asyncio.run(exercise())
        assert dp.clear_thread_ids
        assert dp.clear_thread_ids[0] != event_loop_thread_id

    def test_remove_drops_indices_and_clears_dp_when_requested(self):
        dp = FakeDataPlaneClient()
        buf = _make_buffer(dp)
        metas = [_add_group(buf, weight=g) for g in range(3)]

        n = _run(buf.remove([0, 2], remove_in_dp=True))

        assert n == 2
        assert buf.size() == 1
        assert buf.start_weight_list == [1]
        assert buf.end_weight_list == [1]
        assert buf.meta_list[0].sample_ids == list(metas[1].sample_ids)
        assert dp.depth() == _N_GENS
        assert set(dp._rows) == set(metas[1].sample_ids)

    def test_remove_without_dp_keeps_rows(self):
        dp = FakeDataPlaneClient()
        buf = _make_buffer(dp)
        metas = [_add_group(buf, weight=g) for g in range(2)]

        n = _run(buf.remove([0], remove_in_dp=False))

        assert n == 1
        assert buf.size() == 1
        assert buf.start_weight_list == [1]
        assert buf.end_weight_list == [1]
        assert buf.meta_list[0].sample_ids == list(metas[1].sample_ids)
        assert dp.clear_calls == []
        assert dp.depth() == 2 * _N_GENS

    def test_remove_rejects_out_of_range_before_mutating(self):
        dp = FakeDataPlaneClient()
        buf = _make_buffer(dp)
        metas = [_add_group(buf, weight=g) for g in range(2)]

        with pytest.raises(IndexError, match=r"out of range: 5; size=2"):
            _run(buf.remove([0, 5], remove_in_dp=True))

        assert buf.size() == 2
        assert [m.sample_ids for m in buf.meta_list] == [
            list(metas[0].sample_ids),
            list(metas[1].sample_ids),
        ]
        assert dp.depth() == 2 * _N_GENS
        assert dp.clear_calls == []

    def test_remove_empty_is_noop(self):
        dp = FakeDataPlaneClient()
        buf = _make_buffer(dp)
        _add_group(buf, weight=0)
        _add_group(buf, weight=0)

        n = _run(buf.remove([], remove_in_dp=True))

        assert n == 0
        assert buf.size() == 2
        assert dp.depth() == 2 * _N_GENS
        assert dp.clear_calls == []


class TestTQReplayBufferSize:
    def test_size_and_len(self):
        dp = FakeDataPlaneClient()
        buf = _make_buffer(dp)
        assert buf.size() == 0
        assert len(buf) == 0

        _add_group(buf, weight=0)
        assert buf.size() == 1
        assert len(buf) == 1

        _add_group(buf, weight=0)
        assert buf.size() == 2
        assert len(buf) == 2

        _run(buf.remove([0], remove_in_dp=True))
        assert buf.size() == 1
        assert len(buf) == 1

    def test_count_for_target_step_includes_reserved_slots(self):
        # The rollout pump uses this to size the top-up of a restored target
        # step, so in-flight (reserved, not yet committed) slots must count.
        dp = FakeDataPlaneClient()
        buf = _make_buffer(dp)
        _add_group(buf, weight=0, target_step=5)
        _add_group(buf, weight=0, target_step=6)
        buf.reserve(weight_version=0, target_step=5)

        assert buf.count_for_target_step(5) == 2
        assert buf.count_for_target_step(6) == 1
        assert buf.count_for_target_step(7) == 0


class TestTQReplayBufferPromote:
    """Borrowing inside on_dropped_prompt="replace".

    A step fills a hole from a later step's already-finished work. Not a mode of its
    own: "promote" is not a value on_dropped_prompt accepts.
    """

    def test_the_furthest_step_lends_because_it_has_the_most_slack(self):
        # Both could lend, but 7 is due two steps later than 6, so it has the longer
        # window to receive the repayment rollout before it is needed.
        dp = FakeDataPlaneClient()
        buf = _make_buffer(dp)
        _add_group(buf, weight=0, target_step=6)
        _add_group(buf, weight=0, target_step=7)

        assert buf.promote_ready_group(to_target_step=5) == 7
        assert buf.target_step_list == [6, 5]

    def test_an_unready_group_is_not_borrowed(self):
        # Its rollout is still running, so moving the stamp would hand the dropped step
        # exactly the wait that borrowing exists to avoid.
        dp = FakeDataPlaneClient()
        buf = _make_buffer(dp)
        buf.reserve(weight_version=0, target_step=6)

        assert buf.promote_ready_group(to_target_step=5) is None
        assert buf.target_step_list == [6]

    def test_only_later_steps_lend(self):
        # A group stamped for this step is already counted toward it, and one stamped
        # earlier belongs to a step the trainer has passed.
        dp = FakeDataPlaneClient()
        buf = _make_buffer(dp)
        _add_group(buf, weight=0, target_step=4)
        _add_group(buf, weight=0, target_step=5)

        assert buf.promote_ready_group(to_target_step=5) is None
        assert buf.target_step_list == [4, 5]

    def test_unstamped_groups_are_never_borrowed(self):
        # A sampler that does not stamp cannot strand a step, so its groups belong to
        # no step in particular and there is nothing to move them out of.
        dp = FakeDataPlaneClient()
        buf = _make_buffer(dp)
        _add_group(buf, weight=0, target_step=None)

        assert buf.promote_ready_group(to_target_step=5) is None
        assert buf.target_step_list == [None]

    def test_an_empty_buffer_has_nothing_to_lend(self):
        dp = FakeDataPlaneClient()
        buf = _make_buffer(dp)

        assert buf.promote_ready_group(to_target_step=5) is None

    def test_borrowing_twice_takes_from_two_different_groups(self):
        # Each borrow must move a distinct group: re-stamping the same one twice would
        # report two groups gained where only one exists.
        dp = FakeDataPlaneClient()
        buf = _make_buffer(dp)
        _add_group(buf, weight=0, target_step=6)
        _add_group(buf, weight=0, target_step=6)

        assert buf.promote_ready_group(to_target_step=5) == 6
        assert buf.promote_ready_group(to_target_step=5) == 6
        assert buf.promote_ready_group(to_target_step=5) is None
        assert buf.count_for_target_step(5) == 2


# ── state_dict / load_state_dict (checkpointing) ─────────────────────────────


def _group_id_of(meta: KVBatchMeta) -> str:
    head, _, _ = meta.sample_ids[0].rpartition("_g")
    return head


def _make_group_entry(
    group_id: str,
    weight: int,
    *,
    n: int = _N_GENS,
    target_step: int | None = None,
    sample_ids: list[str] | None = None,
    sequence_lengths: list[int] | None = None,
    partition_id: str = "rollout_data",
) -> dict[str, Any]:
    """Hand-built envelope group (bypasses commit) for preflight tests."""
    sids = (
        list(sample_ids)
        if sample_ids is not None
        else [f"{group_id}_g{i}" for i in range(n)]
    )
    meta = KVBatchMeta(
        partition_id=partition_id,
        task_name="train",
        sample_ids=sids,
        fields=["input_ids", "input_lengths", "total_reward"],
        sequence_lengths=(
            sequence_lengths if sequence_lengths is not None else [3] * len(sids)
        ),
        tags=[{"weight_version": weight}] * len(sids),
    )
    return {
        "meta": meta,
        "start_weight": weight,
        "end_weight": weight,
        "target_step": target_step,
        "group_id": group_id,
    }


def _make_metadata_envelope(
    groups: list[dict[str, Any]],
    *,
    partition_id: str = "rollout_data",
    saved_capacity: int = 8,
) -> dict[str, Any]:
    metadata_groups = [dict(group) for group in groups]
    return {
        "schema_version": REPLAY_BUFFER_METADATA_SCHEMA_VERSION,
        "storage": REPLAY_BUFFER_METADATA_STORAGE,
        "partition_id": partition_id,
        "saved_capacity": saved_capacity,
        "manifest_digest": replay_manifest_digest(metadata_groups),
        "groups": metadata_groups,
    }


def _load(
    buf: TQReplayBuffer,
    state: dict[str, Any],
    *,
    max_groups: int = 8,
    expected_partition_id: str = "rollout_data",
    expected_group_size: int = _N_GENS,
    expected_manifest_digest: str | None = None,
) -> int:
    if expected_manifest_digest is None:
        expected_manifest_digest = str(state.get("manifest_digest", ""))
    return _run(
        buf.load_state_dict(
            state,
            max_groups=max_groups,
            expected_partition_id=expected_partition_id,
            expected_group_size=expected_group_size,
            expected_manifest_digest=expected_manifest_digest,
        )
    )


class TestReplayManifestDigest:
    def test_rejects_non_json_metadata_with_field_path(self):
        group = _make_group_entry("group-1", weight=1)
        assert group["meta"].tags is not None
        group["meta"].tags[0]["unsupported"] = torch.tensor(1)

        with pytest.raises(
            TypeError,
            match=r"groups\[0\]\.meta\.tags\[0\]\.unsupported",
        ):
            replay_manifest_digest([group])

    def test_mapping_order_does_not_change_digest(self):
        first = _make_group_entry("group-1", weight=1)
        second = _make_group_entry("group-1", weight=1)
        first["meta"].extra_info = {"a": 1, "b": [2, 3]}
        second["meta"].extra_info = {"b": [2, 3], "a": 1}

        assert replay_manifest_digest([first]) == replay_manifest_digest([second])


class TestTQReplayBufferStateDict:
    def test_metadata_state_dict_omits_tensors_and_data_plane_reads(self):
        dp = FakeDataPlaneClient()
        buf = _make_buffer(dp)
        metas = [_add_group(buf, weight=w) for w in (1, 2)]
        buf.reserve(weight_version=3)

        state = buf.metadata_state_dict(saved_capacity=8)

        assert state["schema_version"] == REPLAY_BUFFER_METADATA_SCHEMA_VERSION
        assert state["storage"] == REPLAY_BUFFER_METADATA_STORAGE
        assert len(state["groups"]) == 2
        assert all("fields_data" not in group for group in state["groups"])
        assert [group["meta"].sample_ids for group in state["groups"]] == [
            list(meta.sample_ids) for meta in metas
        ]
        assert state["manifest_digest"] == replay_manifest_digest(state["groups"])
        assert dp.get_calls == []

    def test_native_tq_round_trip_restores_index_without_reputting_rows(self):
        dp = FakeDataPlaneClient()
        buf = _make_buffer(dp)
        metas = [_add_group(buf, weight=w) for w in (1, 2)]
        state = buf.metadata_state_dict(saved_capacity=8)

        restored_dp = FakeDataPlaneClient()
        restored_buf = _make_buffer(restored_dp)
        restored = _load(
            restored_buf,
            state,
            expected_manifest_digest=state["manifest_digest"],
        )

        assert restored == 2
        assert restored_buf.start_weight_list == [1, 2]
        assert restored_buf.ready_list == [True, True]
        assert [meta.sample_ids for meta in restored_buf.meta_list] == [
            list(meta.sample_ids) for meta in metas
        ]
        assert restored_dp.put_calls == []

    def test_round_trip_preserves_end_weight_and_target_step(self):
        # start != end and a non-None target_step must survive the round-trip:
        # a load that swapped start/end or dropped target_step (the
        # InOrderSampler's selection key) would corrupt resume silently.
        dp = FakeDataPlaneClient()
        buf = _make_buffer(dp)
        _add_group(buf, weight=1, end_weight=2)
        _add_group(buf, weight=5, target_step=7)
        state = buf.metadata_state_dict(saved_capacity=8)

        buf2 = _make_buffer(FakeDataPlaneClient())
        assert _load(buf2, state) == 2

        assert buf2.start_weight_list == [1, 5]
        assert buf2.end_weight_list == [2, 5]
        assert buf2.target_step_list == [None, 7]

    def test_round_trip_empty_buffer(self):
        # Common resume shape: no group committed before the checkpoint.
        buf = _make_buffer(FakeDataPlaneClient())
        state = buf.metadata_state_dict(saved_capacity=8)
        assert state["groups"] == []

        dp2 = FakeDataPlaneClient()
        buf2 = _make_buffer(dp2)
        assert _load(buf2, state) == 0
        assert buf2.size() == 0
        assert dp2.put_calls == []

    def test_state_dict_skips_middle_unready(self):
        # An unready slot between two ready ones: the by-index skip must not
        # shift the neighbouring groups' fields.
        dp = FakeDataPlaneClient()
        buf = _make_buffer(dp)
        first = _add_group(buf, weight=1)
        buf.reserve(weight_version=2)  # in-flight, sandwiched
        third = _add_group(buf, weight=3)

        state = buf.metadata_state_dict(saved_capacity=8)

        assert [g["start_weight"] for g in state["groups"]] == [1, 3]
        assert [g["group_id"] for g in state["groups"]] == [
            _group_id_of(first),
            _group_id_of(third),
        ]
        assert [g["meta"].sample_ids for g in state["groups"]] == [
            list(first.sample_ids),
            list(third.sample_ids),
        ]

    def test_state_dict_skips_older_long_tail_unready_group(self):
        # Model a long-running rollout reserved on an older weight while newer
        # rollouts finish. Completed-rollout recovery must checkpoint only the
        # ready group; recovering the unfinished group belongs to partial-
        # rollout checkpointing.
        dp = FakeDataPlaneClient()
        buf = _make_buffer(dp)
        unfinished_group_id = buf.reserve(weight_version=1)
        completed = _add_group(buf, weight=7)

        state = buf.metadata_state_dict(saved_capacity=8)

        assert [g["start_weight"] for g in state["groups"]] == [7]
        assert [g["group_id"] for g in state["groups"]] == [_group_id_of(completed)]
        assert unfinished_group_id not in {
            group["group_id"] for group in state["groups"]
        }


class TestTQReplayBufferLoadPreflight:
    """Malformed envelopes raise ValueError before any DataPlane write."""

    def _assert_rejected(self, state: dict[str, Any], match: str, **load_kwargs):
        dp = FakeDataPlaneClient()
        buf = _make_buffer(dp)
        with pytest.raises(ValueError, match=match):
            _load(buf, state, **load_kwargs)
        assert dp.put_calls == []
        assert buf.size() == 0

    def test_missing_envelope_keys(self):
        self._assert_rejected({"groups": []}, match="missing required keys")

    def test_partition_id_mismatch(self):
        state = _make_metadata_envelope([], partition_id="other_partition")
        self._assert_rejected(state, match="partition_id mismatch")

    def test_group_missing_keys(self):
        state = _make_metadata_envelope([_make_group_entry("g0", weight=1)])
        del state["groups"][0]["group_id"]
        self._assert_rejected(state, match="group missing keys")

    def test_group_with_tensor_payload_is_rejected(self):
        group = _make_group_entry("g0", weight=1)
        group["fields_data"] = {"input_ids": torch.ones(2, 3)}
        state = _make_metadata_envelope([group])
        self._assert_rejected(state, match="must not contain fields_data")

    def test_group_misaligned_sequence_lengths(self):
        group = _make_group_entry("g0", weight=1, sequence_lengths=[3])
        self._assert_rejected(_make_metadata_envelope([group]), match="misaligned")

    def test_group_size_mismatch(self):
        state = _make_metadata_envelope([_make_group_entry("g0", weight=1, n=2)])
        self._assert_rejected(state, match="misaligned", expected_group_size=3)

    def test_duplicate_sample_ids_across_groups(self):
        g0 = _make_group_entry("g0", weight=1)
        g1 = _make_group_entry(
            "g1", weight=2, sample_ids=["g0_g0", "g1_g1"]
        )  # g0_g0 collides
        self._assert_rejected(
            _make_metadata_envelope([g0, g1]), match="duplicate sample_id"
        )

    def test_metadata_only_restore_rejects_tq_digest_mismatch(self):
        state = _make_metadata_envelope([_make_group_entry("g0", weight=1)])
        self._assert_rejected(
            state,
            match="does not match the loaded TQ checkpoint",
            expected_manifest_digest="wrong-digest",
        )

    def test_metadata_only_restore_rejects_capacity_truncation(self):
        state = _make_metadata_envelope(
            [_make_group_entry(f"g{w}", weight=w) for w in (1, 2, 3)]
        )
        dp = FakeDataPlaneClient()
        buf = _make_buffer(dp)

        with pytest.raises(ValueError) as exc_info:
            _load(
                buf,
                state,
                max_groups=2,
                expected_manifest_digest=state["manifest_digest"],
            )

        message = str(exc_info.value)
        assert "checkpoint=3, current=2" in message
        assert "async_rl.max_buffered_rollouts >= 3" in message
        assert "Deleting replay_buffer_metadata.pt" in message
        assert "skips loading the matching TQ checkpoint" in message
        assert "dataloader has already moved past them" in message
        assert dp.put_calls == []
        assert buf.size() == 0
