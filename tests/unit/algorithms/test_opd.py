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
import torch

from nemo_rl.data.multimodal_utils import PackedTensor
from nemo_rl.distributed.batched_data_dict import BatchedDataDict

# ---------------------------------------------------------------------------
# Mock teacher worker group for _compute_teacher_logprobs tests
# ---------------------------------------------------------------------------


class _MockShardingAnnotations:
    def __init__(self, dp_size):
        self._dp_size = dp_size

    def get_axis_size(self, name):
        if name == "data_parallel":
            return self._dp_size
        return 1


class _MockTeacherWorkerGroup:
    """Returns logprobs filled with a constant; validates DP-divisible batch."""

    def __init__(self, fill_value=1.0, dp_size=4):
        self._fill_value = fill_value
        self.sharding_annotations = _MockShardingAnnotations(dp_size)
        self.use_sequence_packing = False
        self.sequence_length_pad_multiple = 1

    def get_logprobs(self, data):
        input_ids = data["input_ids"]
        B, S = input_ids.shape
        # Verify the caller already padded to dp_size
        dp_size = self.sharding_annotations.get_axis_size("data_parallel")
        assert B % dp_size == 0, (
            f"get_logprobs received batch_size={B} not divisible by dp_size={dp_size}"
        )
        return BatchedDataDict(
            {"reference_logprobs": torch.full((B, S), self._fill_value)}
        )


def _make_collector(**overrides):
    """Build a bare AsyncTrajectoryCollector (bypass Ray) for unit testing."""
    import threading

    from nemo_rl.algorithms.async_utils import AsyncTrajectoryCollector

    # AsyncTrajectoryCollector is @ray.remote-decorated; unwrap to the real class.
    real_cls = AsyncTrajectoryCollector.__ray_metadata__.modified_class
    defaults = {
        "teacher_worker_groups": {},
        "alias_to_group_alias": {},
        "on_policy_distillation_cfg": {},
        "_has_distillation_teachers": False,
    }
    defaults.update(overrides)
    obj = object.__new__(real_cls)
    for k, v in defaults.items():
        setattr(obj, k, v)
    obj._teacher_locks = {k: threading.Lock() for k in obj.teacher_worker_groups}
    return obj


# ---------------------------------------------------------------------------
# DP padding tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "batch_size,dp_size",
    [
        (1, 4),  # the exact bug: 1 sample, dp=4
        (2, 4),  # 2 samples, dp=4
        (3, 4),  # 3 samples, dp=4
        (4, 4),  # already aligned
        (1, 8),  # extreme: 1 sample, dp=8
        (5, 4),  # 5 samples → pad to 8
    ],
)
def test_compute_teacher_logprobs_dp_padding(batch_size, dp_size):
    """Teacher logprob computation must pad batch to dp_size multiple."""
    twg = _MockTeacherWorkerGroup(fill_value=2.0, dp_size=dp_size)
    collector = _make_collector(
        teacher_worker_groups={"math": twg},
        alias_to_group_alias={"math_agent": "math"},
        on_policy_distillation_cfg={
            "teacher_model_by_agent_name": {"math_agent": "/ckpt/math"},
        },
        _has_distillation_teachers=True,
    )

    S = 16
    input_ids = torch.randint(0, 100, (batch_size, S))
    agent_refs = [{"name": "math_agent"}] * batch_size

    result, _ = collector._compute_teacher_logprobs(input_ids, agent_refs)

    assert result.shape == (batch_size, S)
    assert torch.allclose(result, torch.tensor(2.0))


class _RecordingTeacherWorkerGroup(_MockTeacherWorkerGroup):
    """Capture the batch passed to a teacher for row-alignment assertions."""

    def __init__(self, fill_value=1.0, dp_size=4):
        super().__init__(fill_value=fill_value, dp_size=dp_size)
        self.received: BatchedDataDict | None = None

    def get_logprobs(self, data):
        self.received = data
        return super().get_logprobs(data)


def _row_marked_packed_tensor(markers):
    return PackedTensor(
        [
            None
            if marker is None
            else torch.full((1, 2), float(marker), dtype=torch.float32)
            for marker in markers
        ],
        dim_to_pack=0,
    ).enable_deduplication()


def _received_row_markers(packed):
    return [
        None if tensor is None else float(tensor[0, 0])
        for tensor in packed.iter_logical_segments()
    ]


def test_compute_teacher_logprobs_selects_multimodal_rows_per_teacher():
    """Each teacher receives media rows aligned with its selected token rows."""
    vision_twg = _RecordingTeacherWorkerGroup(fill_value=1.0, dp_size=1)
    text_twg = _RecordingTeacherWorkerGroup(fill_value=2.0, dp_size=1)
    collector = _make_collector(
        teacher_worker_groups={"vision": vision_twg, "text": text_twg},
        alias_to_group_alias={"vision_agent": "vision", "text_agent": "text"},
        on_policy_distillation_cfg={
            "teacher_model_by_agent_name": {
                "vision_agent": "/ckpt/vision",
                "text_agent": "/ckpt/text",
            },
        },
        _has_distillation_teachers=True,
    )

    collector._compute_teacher_logprobs(
        torch.randint(0, 100, (4, 8)),
        [
            {"name": "vision_agent"},
            {"name": "text_agent"},
            {"name": "vision_agent"},
            {"name": "text_agent"},
        ],
        multimodal_data={
            "pixel_values": _row_marked_packed_tensor([0, None, 2, None]),
            "imgs_sizes": _row_marked_packed_tensor([10, None, 12, None]),
        },
    )

    assert vision_twg.received is not None
    assert text_twg.received is not None
    assert _received_row_markers(vision_twg.received["pixel_values"]) == [0.0, 2.0]
    assert _received_row_markers(vision_twg.received["imgs_sizes"]) == [10.0, 12.0]
    assert "pixel_values" not in text_twg.received
    assert "imgs_sizes" not in text_twg.received


def test_compute_teacher_logprobs_dp_padding_repeats_multimodal_row():
    """DP padding repeats the media row paired with the repeated token row."""
    twg = _RecordingTeacherWorkerGroup(fill_value=3.0, dp_size=4)
    collector = _make_collector(
        teacher_worker_groups={"vision": twg},
        alias_to_group_alias={"vision_agent": "vision"},
        on_policy_distillation_cfg={
            "teacher_model_by_agent_name": {"vision_agent": "/ckpt/vision"},
        },
        _has_distillation_teachers=True,
    )

    result, _ = collector._compute_teacher_logprobs(
        torch.randint(0, 100, (1, 8)),
        [{"name": "vision_agent"}],
        multimodal_data={
            "pixel_values": _row_marked_packed_tensor([7]),
            "num_frames": _row_marked_packed_tensor([1]),
        },
    )

    assert twg.received is not None
    assert twg.received["input_ids"].shape[0] == 4
    assert _received_row_markers(twg.received["pixel_values"]) == [7.0] * 4
    assert _received_row_markers(twg.received["num_frames"]) == [1.0] * 4
    assert result.shape == (1, 8)


def test_compute_teacher_logprobs_mixed_media_and_text_rows_per_teacher():
    """Mixed image and text-only rows in one group keep the empty rows aligned."""
    twg = _RecordingTeacherWorkerGroup(fill_value=4.0, dp_size=1)
    collector = _make_collector(
        teacher_worker_groups={"mixed": twg},
        alias_to_group_alias={"mixed_agent": "mixed"},
        on_policy_distillation_cfg={
            "teacher_model_by_agent_name": {"mixed_agent": "/ckpt/mixed"},
        },
        _has_distillation_teachers=True,
    )

    result, _ = collector._compute_teacher_logprobs(
        torch.randint(0, 100, (3, 8)),
        [{"name": "mixed_agent"}] * 3,
        multimodal_data={
            "pixel_values": _row_marked_packed_tensor([5, None, 6]),
            "imgs_sizes": _row_marked_packed_tensor([15, None, 16]),
        },
    )

    assert twg.received is not None
    # The text-only row keeps its slot so media rows stay paired with token rows.
    assert _received_row_markers(twg.received["pixel_values"]) == [5.0, None, 6.0]
    assert _received_row_markers(twg.received["imgs_sizes"]) == [15.0, None, 16.0]
    assert result.shape == (3, 8)


def test_compute_teacher_logprobs_routes_to_correct_teacher():
    """Samples are routed to the right teacher and results stitched back."""
    math_twg = _MockTeacherWorkerGroup(fill_value=1.0, dp_size=1)
    code_twg = _MockTeacherWorkerGroup(fill_value=2.0, dp_size=1)

    collector = _make_collector(
        teacher_worker_groups={"math": math_twg, "code": code_twg},
        alias_to_group_alias={"math_agent": "math", "code_agent": "code"},
        on_policy_distillation_cfg={
            "teacher_model_by_agent_name": {
                "math_agent": "/ckpt/math",
                "code_agent": "/ckpt/code",
            },
        },
        _has_distillation_teachers=True,
    )

    B, S = 4, 8
    input_ids = torch.randint(0, 100, (B, S))
    agent_refs = [
        {"name": "math_agent"},
        {"name": "code_agent"},
        {"name": "math_agent"},
        {"name": "code_agent"},
    ]

    result, _ = collector._compute_teacher_logprobs(input_ids, agent_refs)

    assert result.shape == (B, S)
    assert torch.allclose(result[0], torch.tensor(1.0))
    assert torch.allclose(result[1], torch.tensor(2.0))
    assert torch.allclose(result[2], torch.tensor(1.0))
    assert torch.allclose(result[3], torch.tensor(2.0))


def test_compute_teacher_logprobs_deduplication():
    """alias_to_group_alias routes multiple aliases to one teacher group."""
    shared_twg = _MockTeacherWorkerGroup(fill_value=3.0, dp_size=1)

    collector = _make_collector(
        teacher_worker_groups={"primary": shared_twg},
        alias_to_group_alias={"mcqa": "primary", "terminal": "primary"},
        on_policy_distillation_cfg={
            "teacher_model_by_agent_name": {
                "mcqa": "/ckpt/shared",
                "terminal": "/ckpt/shared",
            },
        },
        _has_distillation_teachers=True,
    )

    B, S = 2, 4
    input_ids = torch.randint(0, 100, (B, S))
    agent_refs = [{"name": "mcqa"}, {"name": "terminal"}]

    result, _ = collector._compute_teacher_logprobs(input_ids, agent_refs)
    assert result.shape == (B, S)
    assert torch.allclose(result, torch.tensor(3.0))


def test_compute_teacher_logprobs_default_alias_fallback_routes():
    """Unmapped agent_ref falls back to default_teacher_alias and routes to a valid group."""
    math_twg = _MockTeacherWorkerGroup(fill_value=7.0, dp_size=1)
    collector = _make_collector(
        teacher_worker_groups={"math": math_twg},
        alias_to_group_alias={"math_agent": "math"},
        on_policy_distillation_cfg={
            "teacher_model_by_agent_name": {"math_agent": "/ckpt/math"},
            "default_teacher_alias": "math_agent",
        },
        _has_distillation_teachers=True,
    )
    B, S = 2, 5
    input_ids = torch.randint(0, 100, (B, S))
    # second agent ("surprise_agent") is unmapped -> must fall back to math_agent
    agent_refs = [{"name": "math_agent"}, {"name": "surprise_agent"}]
    result, _ = collector._compute_teacher_logprobs(input_ids, agent_refs)
    assert result.shape == (B, S)
    assert torch.allclose(result, torch.tensor(7.0))


# ---------------------------------------------------------------------------
# SingleController TQ teacher enrichment
# ---------------------------------------------------------------------------


def _teacher_record(agent_name: str):
    from nemo_rl.experience.interfaces import PromptGroupRecord

    return PromptGroupRecord(
        prompt_idx=0,
        prompt=[],
        extra_env_info={"agent_ref": {"name": agent_name}},
        metadata={},
        completions=[],
        rollout_metrics={},
    )


def _teacher_meta(prefix: str, batch_size: int, seq_len: int):
    from nemo_rl.data_plane import KVBatchMeta

    return KVBatchMeta(
        partition_id="rollout_data",
        task_name="train",
        sample_ids=[f"{prefix}_{index}" for index in range(batch_size)],
        fields=["input_ids", "input_lengths"],
        sequence_lengths=[seq_len] * batch_size,
    )


def test_tq_teacher_enrichment_pads_dp_and_writes_teacher_column(monkeypatch):
    """The SC coordinator preserves DP padding while TQ remains source and sink."""
    import asyncio

    from nemo_rl.algorithms import opd

    class MetaTeacher:
        def __init__(self):
            self.sharding_annotations = _MockShardingAnnotations(4)
            self.received_meta = None

        def get_logprobs_from_meta(self, meta):
            self.received_meta = meta

    class FakeDataPlane:
        def __init__(self):
            self.clear_calls = []

        def clear_samples(self, sample_ids, partition_id):
            self.clear_calls.append((list(sample_ids), partition_id))

    teacher = MetaTeacher()
    dp_client = FakeDataPlane()
    writes = []

    def fake_read_columns(dp_client, meta, select_fields, pad_value_dict):
        del dp_client, select_fields, pad_value_dict
        batch_size = len(meta.sample_ids)
        seq_len = max(meta.sequence_lengths)
        return BatchedDataDict(
            {
                "input_ids": torch.arange(batch_size * seq_len).reshape(
                    batch_size, seq_len
                ),
                "input_lengths": torch.tensor(meta.sequence_lengths),
            }
        )

    def fake_write_columns(dp_client, meta, fields):
        del dp_client
        writes.append((meta, fields))

    monkeypatch.setattr(opd, "read_columns", fake_read_columns)
    monkeypatch.setattr(opd, "write_columns", fake_write_columns)
    coordinator = opd.TQTeacherLogprobCoordinator(
        dp_client=dp_client,
        teacher_worker_groups={"primary": teacher},
        alias_to_group_alias={"math": "primary"},
        on_policy_distillation_cfg={
            "teacher_model_by_agent_name": {"math": "/ckpt/shared"}
        },
    )
    meta = _teacher_meta("group", batch_size=3, seq_len=5)

    enriched = asyncio.run(coordinator.enrich(meta, _teacher_record("math")))

    assert teacher.received_meta is not None
    assert teacher.received_meta.size == 4
    assert teacher.received_meta.sample_ids[:3] == meta.sample_ids
    assert "__teacher_pad_" in teacher.received_meta.sample_ids[-1]
    assert len(writes) == 1
    pad_meta, padding_fields = writes[0]
    assert pad_meta.size == 1
    assert padding_fields["input_ids"].shape == (1, 5)
    assert dp_client.clear_calls == [(pad_meta.sample_ids, "rollout_data")]
    assert "teacher_reference_logprobs" in enriched.fields
    metrics = coordinator.drain_metrics()
    assert metrics["on_policy_distillation/teacher_batches"] == 1.0
    assert metrics["on_policy_distillation/teacher_samples"] == 3.0
    assert metrics["on_policy_distillation/teacher_model_unique"] == 1.0


def test_tq_teacher_enrichment_skips_padding_for_dp_divisible_batch(monkeypatch):
    """DP-divisible teacher batches do not create or clean temporary TQ rows."""
    import asyncio

    from nemo_rl.algorithms import opd

    teacher = _MockTeacherWorkerGroup(dp_size=2)
    teacher.received_meta = None
    teacher.get_logprobs_from_meta = lambda meta: setattr(
        teacher, "received_meta", meta
    )

    class FakeDataPlane:
        def clear_samples(self, **kwargs):
            raise AssertionError(f"unexpected temporary-row cleanup: {kwargs}")

    monkeypatch.setattr(
        opd,
        "read_columns",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("DP-divisible batches must not read a padding source")
        ),
    )
    monkeypatch.setattr(
        opd,
        "write_columns",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("DP-divisible batches must not write padding rows")
        ),
    )
    coordinator = opd.TQTeacherLogprobCoordinator(
        dp_client=FakeDataPlane(),
        teacher_worker_groups={"teacher": teacher},
        alias_to_group_alias={"math": "teacher"},
        on_policy_distillation_cfg={
            "teacher_model_by_agent_name": {"math": "/ckpt/teacher"}
        },
    )
    meta = _teacher_meta("group", batch_size=2, seq_len=5)

    enriched = asyncio.run(coordinator.enrich(meta, _teacher_record("math")))

    assert teacher.received_meta is meta
    assert "teacher_reference_logprobs" in enriched.fields


def test_tq_teacher_routing_rejects_missing_agent_ref_without_retry_hint():
    """Missing Gym routing metadata gets a diagnostic that identifies the cause."""
    from nemo_rl.algorithms import opd
    from nemo_rl.experience.interfaces import PromptGroupRecord

    coordinator = opd.TQTeacherLogprobCoordinator(
        dp_client=object(),
        teacher_worker_groups={"teacher": _MockTeacherWorkerGroup(dp_size=1)},
        alias_to_group_alias={},
        on_policy_distillation_cfg={
            "teacher_model_by_agent_name": {"teacher": "/ckpt/teacher"}
        },
    )
    record = PromptGroupRecord(
        prompt_idx=0,
        prompt=[],
        extra_env_info={},
        metadata={},
        completions=[],
        rollout_metrics={},
    )

    with pytest.raises(
        ValueError,
        match="requires the NeMo-Gym rollout path.*cannot repair",
    ):
        coordinator._resolve_teacher(record)


def test_tq_teacher_routing_uses_default_teacher_for_unmapped_agent():
    """An unmapped Gym agent follows the configured default teacher alias."""
    from nemo_rl.algorithms import opd

    coordinator = opd.TQTeacherLogprobCoordinator(
        dp_client=object(),
        teacher_worker_groups={"default": _MockTeacherWorkerGroup(dp_size=1)},
        alias_to_group_alias={"default": "default"},
        on_policy_distillation_cfg={
            "teacher_model_by_agent_name": {"default": "/ckpt/default"},
            "default_teacher_alias": "default",
        },
    )

    assert coordinator._resolve_teacher(_teacher_record("unmapped")) == (
        "default",
        "default",
    )


def test_tq_teacher_padding_rows_are_cleaned_when_write_partially_fails(monkeypatch):
    """Temporary IDs enter cleanup scope before their first TQ write."""
    import asyncio

    from nemo_rl.algorithms import opd

    cleared = []

    class FakeDataPlane:
        def clear_samples(self, sample_ids, partition_id):
            cleared.append((list(sample_ids), partition_id))

    monkeypatch.setattr(
        opd,
        "read_columns",
        lambda *args, **kwargs: BatchedDataDict(
            {
                "input_ids": torch.ones(1, 3, dtype=torch.long),
                "input_lengths": torch.tensor([3]),
            }
        ),
    )

    def partially_failing_write(_dp_client, meta, fields):
        del fields
        assert meta.sample_ids
        raise RuntimeError("partial pad write")

    monkeypatch.setattr(opd, "write_columns", partially_failing_write)
    coordinator = opd.TQTeacherLogprobCoordinator(
        dp_client=FakeDataPlane(),
        teacher_worker_groups={"teacher": _MockTeacherWorkerGroup(dp_size=4)},
        alias_to_group_alias={"math": "teacher"},
        on_policy_distillation_cfg={
            "teacher_model_by_agent_name": {"math": "/ckpt/teacher"}
        },
    )

    with pytest.raises(RuntimeError, match="partial pad write"):
        asyncio.run(
            coordinator.enrich(
                _teacher_meta("group", batch_size=3, seq_len=3),
                _teacher_record("math"),
            )
        )

    assert len(cleared) == 1
    assert len(cleared[0][0]) == 1
    assert "__teacher_pad_" in cleared[0][0][0]


def test_tq_teacher_enrichment_drains_background_thread_before_cancellation():
    """Cancellation waits for teacher TQ activity to finish before propagating."""
    import asyncio
    import threading

    from nemo_rl.algorithms import opd

    started = threading.Event()
    release = threading.Event()
    finished = threading.Event()

    class BlockingTeacher(_MockTeacherWorkerGroup):
        def get_logprobs_from_meta(self, meta):
            del meta
            started.set()
            release.wait(timeout=2)
            finished.set()

    coordinator = opd.TQTeacherLogprobCoordinator(
        dp_client=object(),
        teacher_worker_groups={"teacher": BlockingTeacher(dp_size=1)},
        alias_to_group_alias={"math": "teacher"},
        on_policy_distillation_cfg={
            "teacher_model_by_agent_name": {"math": "/ckpt/teacher"}
        },
    )

    async def cancel_during_inference():
        task = asyncio.create_task(
            coordinator.enrich(
                _teacher_meta("group", batch_size=1, seq_len=3),
                _teacher_record("math"),
            )
        )
        assert await asyncio.to_thread(started.wait, 1)
        task.cancel()
        await asyncio.sleep(0)
        assert not task.done()
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await task

    asyncio.run(cancel_during_inference())
    assert finished.is_set()


def test_tq_teacher_waiters_do_not_occupy_executor_threads(monkeypatch):
    """Only the active inference for a physical teacher enters to_thread."""
    import asyncio
    import threading

    from nemo_rl.algorithms import opd

    started = threading.Event()
    release = threading.Event()
    submissions = 0
    real_to_thread = asyncio.to_thread

    class BlockingTeacher(_MockTeacherWorkerGroup):
        def get_logprobs_from_meta(self, meta):
            del meta
            started.set()
            release.wait(timeout=2)

    async def counted_to_thread(func, /, *args, **kwargs):
        nonlocal submissions
        submissions += 1
        return await real_to_thread(func, *args, **kwargs)

    monkeypatch.setattr(opd.asyncio, "to_thread", counted_to_thread)
    coordinator = opd.TQTeacherLogprobCoordinator(
        dp_client=object(),
        teacher_worker_groups={"teacher": BlockingTeacher(dp_size=1)},
        alias_to_group_alias={"math": "teacher"},
        on_policy_distillation_cfg={
            "teacher_model_by_agent_name": {"math": "/ckpt/teacher"}
        },
    )

    async def run_waiters():
        first = asyncio.create_task(
            coordinator.enrich(
                _teacher_meta("first", batch_size=1, seq_len=3),
                _teacher_record("math"),
            )
        )
        while not started.is_set():
            await asyncio.sleep(0.001)
        second = asyncio.create_task(
            coordinator.enrich(
                _teacher_meta("second", batch_size=1, seq_len=3),
                _teacher_record("math"),
            )
        )
        await asyncio.sleep(0.01)
        assert submissions == 1
        release.set()
        await asyncio.gather(first, second)

    asyncio.run(run_waiters())
    assert submissions == 2


def test_tq_teacher_enrichment_serializes_deduplicated_teacher(monkeypatch):
    """Two aliases sharing one physical teacher never overlap collectives."""
    import asyncio
    import threading
    import time

    from nemo_rl.algorithms import opd

    active = 0
    max_active = 0
    active_lock = threading.Lock()

    class SlowTeacher(_MockTeacherWorkerGroup):
        def get_logprobs_from_meta(self, meta):
            del meta
            nonlocal active, max_active
            with active_lock:
                active += 1
                max_active = max(max_active, active)
            time.sleep(0.05)
            try:
                return None
            finally:
                with active_lock:
                    active -= 1

    def fake_read_columns(dp_client, meta, select_fields, pad_value_dict):
        del dp_client, select_fields, pad_value_dict
        return BatchedDataDict(
            {
                "input_ids": torch.ones(len(meta.sample_ids), 4, dtype=torch.long),
                "input_lengths": torch.full(
                    (len(meta.sample_ids),), 4, dtype=torch.long
                ),
            }
        )

    monkeypatch.setattr(opd, "read_columns", fake_read_columns)
    monkeypatch.setattr(opd, "write_columns", lambda *args, **kwargs: None)
    teacher = SlowTeacher(dp_size=1)
    coordinator = opd.TQTeacherLogprobCoordinator(
        dp_client=object(),
        teacher_worker_groups={"primary": teacher},
        alias_to_group_alias={"math": "primary", "code": "primary"},
        on_policy_distillation_cfg={
            "teacher_model_by_agent_name": {
                "math": "/ckpt/shared",
                "code": "/ckpt/shared",
            }
        },
    )

    async def run_both():
        await asyncio.gather(
            coordinator.enrich(_teacher_meta("math", 1, 4), _teacher_record("math")),
            coordinator.enrich(_teacher_meta("code", 1, 4), _teacher_record("code")),
        )

    asyncio.run(run_both())

    assert max_active == 1


def test_tq_teacher_enrichment_runs_distinct_teachers_concurrently():
    """Distinct physical teachers hold distinct locks, so they overlap."""
    import asyncio
    import threading

    from nemo_rl.algorithms import opd

    barrier = threading.Barrier(2)

    class BarrierTeacher(_MockTeacherWorkerGroup):
        def get_logprobs_from_meta(self, meta):
            del meta
            # Both teachers must enter inference concurrently. One shared lock
            # would serialize them and trip BrokenBarrierError.
            barrier.wait(timeout=5)

    coordinator = opd.TQTeacherLogprobCoordinator(
        dp_client=object(),
        teacher_worker_groups={
            "primary": BarrierTeacher(dp_size=1),
            "secondary": BarrierTeacher(dp_size=1),
        },
        alias_to_group_alias={"math": "primary", "code": "secondary"},
        on_policy_distillation_cfg={
            "teacher_model_by_agent_name": {
                "math": "/ckpt/math",
                "code": "/ckpt/code",
            }
        },
    )

    async def run_both():
        await asyncio.gather(
            coordinator.enrich(_teacher_meta("math", 1, 4), _teacher_record("math")),
            coordinator.enrich(_teacher_meta("code", 1, 4), _teacher_record("code")),
        )

    asyncio.run(run_both())

    metrics = coordinator.drain_metrics()
    assert metrics["on_policy_distillation/teacher_model_unique"] == 2.0


def test_tq_teacher_metrics_omit_routing_cardinality_on_idle_drain():
    """An idle interval reports zero activity without claiming zero teachers."""
    from nemo_rl.algorithms import opd

    coordinator = opd.TQTeacherLogprobCoordinator(
        dp_client=object(),
        teacher_worker_groups={"teacher": _MockTeacherWorkerGroup(dp_size=1)},
        alias_to_group_alias={"math": "teacher"},
        on_policy_distillation_cfg={
            "teacher_model_by_agent_name": {"math": "/ckpt/teacher"}
        },
    )

    metrics = coordinator.drain_metrics()

    assert metrics["on_policy_distillation/teacher_batches"] == 0.0
    assert metrics["on_policy_distillation/teacher_samples"] == 0.0
    assert metrics["on_policy_distillation/teacher_logprob_time_s"] == 0.0
    assert metrics["on_policy_distillation/teacher_inference_time_s"] == 0.0
    assert metrics["on_policy_distillation/teacher_lock_wait_time_s"] == 0.0
    assert "on_policy_distillation/teacher_alias_unique" not in metrics
    assert "on_policy_distillation/teacher_model_unique" not in metrics
    assert "on_policy_distillation/teacher_alias_to_model_compression" not in metrics


# ---------------------------------------------------------------------------
# Unsort / reorder_data regression test
# ---------------------------------------------------------------------------


def test_reorder_data_vs_direct_gather():
    """Verify reorder_data inverts the permutation, while direct gather does not.

    This is the root cause of the num_gen>1 teacher logprob misalignment bug:
    shard_by_batch_size returns a forward permutation (sorted_pos → orig_idx).
    To restore original order we need the *inverse* (argsort), which
    reorder_data computes.  A direct gather ``result[indices]`` applies
    the forward permutation and silently produces wrong results.
    """
    # Simulate: 4 samples reordered by sequence packing as [3, 0, 2, 1]
    forward_perm = [3, 0, 2, 1]
    # After inference, results are in sorted order:
    #   position 0 = result for orig sample 3
    #   position 1 = result for orig sample 0  etc.
    sorted_results = BatchedDataDict(
        {"logprobs": torch.tensor([[30.0], [0.0], [20.0], [10.0]])}
    )
    # label: sorted_results[i] holds the value for original sample forward_perm[i]
    #   sorted_results[0]=30 → orig 3,  sorted_results[1]=0 → orig 0, etc.

    # --- WRONG: direct gather (the old bug) ---
    wrong = sorted_results["logprobs"][forward_perm]
    # wrong[0] = sorted_results[3] = 10  (should be 0 for orig 0)
    assert not torch.equal(wrong, torch.tensor([[0.0], [10.0], [20.0], [30.0]])), (
        "Direct gather should NOT produce the correct original order"
    )

    # --- CORRECT: reorder_data (inverse permutation) ---
    correct = BatchedDataDict({"logprobs": sorted_results["logprobs"].clone()})
    correct.reorder_data(forward_perm)
    assert torch.equal(
        correct["logprobs"], torch.tensor([[0.0], [10.0], [20.0], [30.0]])
    ), "reorder_data should restore the original sample order"


# ---------------------------------------------------------------------------
# Teacher logprob alignment with variable-length sequences (num_gen > 1)
# ---------------------------------------------------------------------------


def test_reorder_data_inverse_permutation_various():
    """reorder_data correctly inverts arbitrary permutations, including identity."""
    # Identity permutation
    bdd = BatchedDataDict({"x": torch.tensor([[0.0], [1.0], [2.0]])})
    bdd.reorder_data([0, 1, 2])
    assert torch.equal(bdd["x"], torch.tensor([[0.0], [1.0], [2.0]]))

    # Reversal
    bdd = BatchedDataDict({"x": torch.tensor([[0.0], [1.0], [2.0]])})
    bdd.reorder_data([2, 1, 0])
    # batch_sorted_indices=[2,1,0] means sorted[0] came from orig 2, etc.
    # Inverse: orig[2]=sorted[0]=0.0, orig[1]=sorted[1]=1.0, orig[0]=sorted[2]=2.0
    assert torch.equal(bdd["x"], torch.tensor([[2.0], [1.0], [0.0]]))

    # Non-trivial: simulate 4 samples reordered as [2, 3, 0, 1]
    bdd = BatchedDataDict({"x": torch.tensor([[20.0], [30.0], [0.0], [10.0]])})
    bdd.reorder_data([2, 3, 0, 1])
    assert torch.equal(bdd["x"], torch.tensor([[0.0], [10.0], [20.0], [30.0]])), (
        "After reorder_data, row i should hold the result for original sample i"
    )


def test_is_opd_enabled():
    from nemo_rl.algorithms.opd import is_opd_enabled

    assert is_opd_enabled({"on_policy_distillation": {"enabled": True}})
    assert not is_opd_enabled({"on_policy_distillation": {"enabled": False}})
    assert not is_opd_enabled({})


def test_is_opd_enabled_object_config():
    # _opd_cfg must also handle a config object (not just a dict): math recipes
    # have no on_policy_distillation attribute at all.
    from types import SimpleNamespace

    from nemo_rl.algorithms.opd import is_opd_enabled

    assert is_opd_enabled(SimpleNamespace(on_policy_distillation={"enabled": True}))
    assert not is_opd_enabled(
        SimpleNamespace(on_policy_distillation={"enabled": False})
    )
    assert not is_opd_enabled(SimpleNamespace())
    assert not is_opd_enabled(SimpleNamespace(on_policy_distillation=None))


def test_is_non_colocated_teachers_enabled():
    from nemo_rl.algorithms.opd import is_non_colocated_teachers_enabled

    assert is_non_colocated_teachers_enabled(
        {
            "on_policy_distillation": {
                "enabled": True,
                "non_colocated_teachers": {"enabled": True},
            }
        }
    )
    assert not is_non_colocated_teachers_enabled(
        {
            "on_policy_distillation": {
                "enabled": True,
                "non_colocated_teachers": {"enabled": False},
            }
        }
    )


def test_resolve_reference_aliases_bad_agent_ref():
    from nemo_rl.algorithms.opd import resolve_reference_aliases

    with pytest.raises(KeyError):
        resolve_reference_aliases([{"not_name": "oops"}], {"math": "/ckpt/math"})


def test_resolve_reference_aliases_fallback():
    from nemo_rl.algorithms.opd import resolve_reference_aliases

    aliases = resolve_reference_aliases(
        [{"name": "math_agent"}, {"name": "unknown"}, {"name": "code_agent"}],
        {"math_agent": "/ckpt/math", "code_agent": "/ckpt/code"},
        default_teacher_alias="math_agent",
    )
    assert aliases == ["math_agent", "math_agent", "code_agent"]


def test_resolve_reference_aliases_strict_raises():
    from nemo_rl.algorithms.opd import resolve_reference_aliases

    with pytest.raises(ValueError, match="No teacher model mapping"):
        resolve_reference_aliases(
            [{"name": "unknown"}], {"math": "/ckpt/math"}, strict_agent_name_match=True
        )


def test_get_teacher_routing_metrics():
    from nemo_rl.algorithms.opd import get_teacher_routing_metrics

    metrics = get_teacher_routing_metrics(
        ["math_a", "math_b", "if", "math_a"],
        {"math_a": "t_math", "math_b": "t_math", "if": "t_if"},
    )
    assert metrics["on_policy_distillation/teacher_alias_unique"] == 3.0
    assert metrics["on_policy_distillation/teacher_model_unique"] == 2.0


# ---------------------------------------------------------------------------
# teacher_seq_pad_multiple: teacher pre-pad multiple + packing-mode guard
# ---------------------------------------------------------------------------


def _twg(packed, pad_multiple=1):
    """TeacherWorkerGroup stand-in with the two attrs the helper reads."""
    from types import SimpleNamespace

    return SimpleNamespace(
        use_sequence_packing=packed, sequence_length_pad_multiple=pad_multiple
    )


def test_teacher_seq_pad_multiple_no_teachers_is_one():
    from nemo_rl.algorithms.opd import teacher_seq_pad_multiple

    assert teacher_seq_pad_multiple({}, 8) == 1


def test_teacher_seq_pad_multiple_all_packed_is_one():
    from nemo_rl.algorithms.opd import teacher_seq_pad_multiple

    assert teacher_seq_pad_multiple({"a": _twg(True), "b": _twg(True)}, 8) == 1


def test_teacher_seq_pad_multiple_mixed_packing_raises():
    from nemo_rl.algorithms.opd import teacher_seq_pad_multiple

    with pytest.raises(ValueError, match="same sequence-packing mode"):
        teacher_seq_pad_multiple({"a": _twg(True), "b": _twg(False)}, 8)


def test_teacher_seq_pad_multiple_non_packed_uses_policy_divisor():
    # Non-packed teachers pre-pad to the policy divisor when it is a multiple of
    # every teacher's requirement (here 2 and 4 both divide 8).
    from nemo_rl.algorithms.opd import teacher_seq_pad_multiple

    teachers = {"a": _twg(False, pad_multiple=2), "b": _twg(False, pad_multiple=4)}
    assert teacher_seq_pad_multiple(teachers, 8) == 8


def test_teacher_seq_pad_multiple_non_packed_incompatible_divisor_raises():
    # policy divisor 8 is not a multiple of the teacher requirement 16.
    from nemo_rl.algorithms.opd import teacher_seq_pad_multiple

    with pytest.raises(ValueError, match="make_sequence_length_divisible_by"):
        teacher_seq_pad_multiple({"a": _twg(False, pad_multiple=16)}, 8)


# ---------------------------------------------------------------------------
# Teacher placement-group reservation
# ---------------------------------------------------------------------------


def _teacher_setup_config():
    """Return a minimal config with two distinct non-colocated teachers."""
    return {
        "on_policy_distillation": {
            "enabled": True,
            "teacher_model_by_agent_name": {
                "math": "/checkpoints/math",
                "code": "/checkpoints/code",
            },
            "non_colocated_teachers": {
                "enabled": True,
                "default_teacher_cfg": {
                    "num_nodes": 2,
                    "gpus_per_node": 4,
                },
            },
        }
    }


def test_reserve_teacher_clusters_claims_each_topology_segment(monkeypatch):
    """Every teacher placement group is claimed before planning the next one."""
    from nemo_rl.algorithms import opd

    events = []

    class FakeRayVirtualCluster:
        def __init__(self, **kwargs):
            self.name = kwargs["name"]
            self.kwargs = kwargs
            events.append(("create", self.name))

        def get_placement_groups(self):
            events.append(("reserve", self.name))
            return [object()]

        def shutdown(self):
            events.append(("shutdown", self.name))

    def fake_prepare_segment_topology(segment_size, num_nodes, *, topology, role):
        assert segment_size == num_nodes == 2
        events.append(("prepare", role, tuple(topology)))
        selected_ids = list(topology)[:num_nodes]
        remaining_ids = [node_id for node_id in topology if node_id not in selected_ids]
        constraints = [{"nvlink_domain_0": 0.001}] * num_nodes
        return constraints, remaining_ids, topology

    monkeypatch.setattr(opd, "RayVirtualCluster", FakeRayVirtualCluster)
    monkeypatch.setattr(opd, "prepare_segment_topology", fake_prepare_segment_topology)

    topology = {f"node_{index}": ("nvlink_domain_0", index) for index in range(4)}
    clusters = opd.reserve_teacher_clusters(
        _teacher_setup_config(),
        segment_size=16,
        teacher_segment_topology=topology,
    )

    assert list(clusters) == ["math", "code"]
    assert events == [
        (
            "prepare",
            "teacher:math",
            ("node_0", "node_1", "node_2", "node_3"),
        ),
        ("create", "teacher_math"),
        ("reserve", "teacher_math"),
        ("prepare", "teacher:code", ("node_2", "node_3")),
        ("create", "teacher_code"),
        ("reserve", "teacher_code"),
    ]
    assert clusters["math"].kwargs["segment_size"] == 2
    assert clusters["code"].kwargs["node_resource_constraints"] == [
        {"nvlink_domain_0": 0.001},
        {"nvlink_domain_0": 0.001},
    ]


def test_reserve_teacher_clusters_claims_without_topology_constraints(monkeypatch):
    """Teachers are claimed before Gym even when segment topology is disabled."""
    from nemo_rl.algorithms import opd

    events = []

    class FakeRayVirtualCluster:
        def __init__(self, **kwargs):
            self.name = kwargs["name"]
            self.kwargs = kwargs
            events.append(("create", self.name))

        def get_placement_groups(self):
            events.append(("reserve", self.name))
            return [object()]

        def shutdown(self):
            events.append(("shutdown", self.name))

    def fail_if_topology_is_prepared(*args, **kwargs):
        raise AssertionError("topology must not be prepared when segment_size is unset")

    monkeypatch.setattr(opd, "RayVirtualCluster", FakeRayVirtualCluster)
    monkeypatch.setattr(opd, "prepare_segment_topology", fail_if_topology_is_prepared)

    clusters = opd.reserve_teacher_clusters(_teacher_setup_config())

    assert events == [
        ("create", "teacher_math"),
        ("reserve", "teacher_math"),
        ("create", "teacher_code"),
        ("reserve", "teacher_code"),
    ]
    assert all(cluster.kwargs["segment_size"] is None for cluster in clusters.values())
    assert all(
        cluster.kwargs["node_resource_constraints"] is None
        for cluster in clusters.values()
    )


def test_create_teacher_worker_groups_reuses_reserved_clusters(monkeypatch):
    """Deferred worker initialization uses the already claimed clusters."""
    from types import SimpleNamespace

    from nemo_rl.algorithms import opd
    from nemo_rl.models.policy import teacher_worker_group

    math_cluster = object()
    code_cluster = object()
    reserved_clusters = {"math": math_cluster, "code": code_cluster}
    initialized_clusters = []

    class FakeTeacherWorkerGroup:
        def __init__(self, *, teacher_cfg, cluster, policy_config, tokenizer):
            initialized_clusters.append((teacher_cfg.alias, cluster))
            self.worker_group = SimpleNamespace(workers=[])
            self.use_sequence_packing = True
            self.sequence_length_pad_multiple = 1

    def fail_if_reservation_repeats(*args, **kwargs):
        raise AssertionError("teacher placement groups must not be reserved twice")

    monkeypatch.setattr(
        teacher_worker_group, "TeacherWorkerGroup", FakeTeacherWorkerGroup
    )
    monkeypatch.setattr(opd, "reserve_teacher_clusters", fail_if_reservation_repeats)
    monkeypatch.setattr(opd.ray, "get", lambda refs, timeout: [])

    worker_groups, alias_to_group_alias = opd.create_teacher_worker_groups(
        _teacher_setup_config(),
        {"make_sequence_length_divisible_by": 8},
        tokenizer=object(),
        teacher_clusters=reserved_clusters,
    )

    assert initialized_clusters == [
        ("math", math_cluster),
        ("code", code_cluster),
    ]
    assert list(worker_groups) == ["math", "code"]
    assert alias_to_group_alias == {"math": "math", "code": "code"}


# ---------------------------------------------------------------------------
# Teacher-logprob seq-length padding + the "opd" advantage-estimator branch
# ---------------------------------------------------------------------------


def test_pad_teacher_logprobs():
    from nemo_rl.algorithms.grpo import _pad_teacher_logprobs

    # teacher_S < train_S -> right zero-pad
    padded = _pad_teacher_logprobs(torch.ones(2, 3), 5)
    assert padded.shape == (2, 5)
    assert (padded[:, :3] == 1).all() and (padded[:, 3:] == 0).all()
    # teacher_S == train_S -> unchanged
    assert _pad_teacher_logprobs(torch.ones(2, 4), 4).shape == (2, 4)
    # teacher_S > train_S -> raises
    with pytest.raises(ValueError, match="seq length"):
        _pad_teacher_logprobs(torch.ones(2, 6), 4)


def test_create_advantage_estimator_opd_branch():
    import warnings
    from types import SimpleNamespace

    from nemo_rl.algorithms.advantage_estimator import (
        AdvEstimatorConfig,
        OPDAdvantageEstimator,
    )
    from nemo_rl.algorithms.grpo import GRPOConfig, _create_advantage_estimator

    # loss_fn not MOPD-configured -> the 3 recommendation warnings fire.
    loss_fn = SimpleNamespace(
        disable_ppo_ratio=False,
        use_importance_sampling_correction=False,
        truncated_importance_sampling_type="none",
    )
    master_config = SimpleNamespace(
        grpo=GRPOConfig(adv_estimator=AdvEstimatorConfig(name="opd")),
        loss_fn=loss_fn,
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        estimator = _create_advantage_estimator(master_config)
    assert isinstance(estimator, OPDAdvantageEstimator)
    assert len(caught) == 3
