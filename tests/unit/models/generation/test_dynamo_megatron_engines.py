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

"""Megatron prefill/decode engines behind Dynamo: the refit channel."""

from unittest.mock import MagicMock

import pytest
from ray.exceptions import GetTimeoutError

import nemo_rl.models.generation.dynamo.megatron_refit_channel as channel_module
from nemo_rl.models.generation.dynamo.protocols import (
    DynamoWorkerRecord,
    RefitChannel,
)
from nemo_rl.models.generation.dynamo.megatron_refit_channel import (
    MegatronDynamoWorkerClient,
    MegatronRefitChannel,
    PinnedHandoffTimeout,
)
from nemo_rl.models.generation.megatron.config import (
    generation_runs_megatron_engines,
)
from nemo_rl.weight_sync.membership import desired_membership

_WIDTH = 2
_ROLES = ["prefill", "decode", "decode"]
_NAMES = ["p0", "d0", "d1"]
_CONTROL_TIMEOUT_S = 1.0


class _Ref:
    """A dispatched worker call, resolved by the patched ``ray.get``."""

    def __init__(self, value=None, *, error=None, hang=False) -> None:
        self.value = value
        self.error = error
        self.hang = hang


class _FakeWorker:
    """One Ray worker of an engine: ``worker.method.remote(**kwargs)``."""

    def __init__(self, engine: "_FakeEngine", rank: int) -> None:
        self._engine = engine
        self._rank = rank

    def __getattr__(self, method: str):
        engine, rank = self._engine, self._rank
        return MagicMock(
            remote=lambda **kwargs: engine.call(rank, method, kwargs),
        )


class _FakeEngine:
    """One MegatronGeneration; rank zero's calls land in a shared log."""

    def __init__(self, name: str, log: list[str], pinned=None, width=_WIDTH) -> None:
        self.name = name
        self.log = log
        self._pinned = list(pinned or [0])
        # method name -> exception to raise, or "hang" to never finish.
        self.faults: dict[str, object] = {}
        workers = [_FakeWorker(self, rank) for rank in range(width)]
        self.worker_group = MagicMock(
            workers=workers, get_dp_leader_worker_idx=lambda shard: 0
        )

    def call(self, rank: int, method: str, kwargs: dict) -> _Ref:
        if rank == 0:
            self.log.append(self._entry(method, kwargs))
        fault = self.faults.get(method)
        if fault == "hang":
            return _Ref(hang=True)
        if fault is not None:
            return _Ref(error=fault)
        if method == "pinned_handoff_block_count":
            # Scripted sequence; the last value repeats.
            return _Ref(
                self._pinned.pop(0) if len(self._pinned) > 1 else self._pinned[0]
            )
        return _Ref(True)

    def _entry(self, method: str, kwargs: dict) -> str:
        short = {
            "pause_for_refit": "pause",
            "suspend_for_refit": "suspend",
            "resume_after_refit": "resume",
            "pinned_handoff_block_count": "pinned",
            "prepare_refit_info": "prepare",
            "update_weights_from_collective": "update",
        }
        if method == "init_collective":
            return (
                f"{self.name}.init(ws={kwargs['world_size']},"
                f"train={kwargs['train_world_size']},off={kwargs['rank_offset']})"
            )
        if method == "prepare_for_generation":
            return f"{self.name}.prepare_for_generation({kwargs['tags']})"
        return f"{self.name}.{short[method]}"


@pytest.fixture
def get_timeouts(monkeypatch) -> list:
    """Patch Ray so dispatched fake calls resolve; record every get timeout."""
    timeouts: list = []

    def resolve(ref: _Ref):
        if ref.hang:
            raise GetTimeoutError("still running")
        if ref.error is not None:
            raise ref.error
        return ref.value

    def get(refs, timeout=None):
        timeouts.append(timeout)
        if isinstance(refs, list):
            return [resolve(ref) for ref in refs]
        return resolve(refs)

    monkeypatch.setattr(channel_module.ray, "get", get)
    monkeypatch.setattr(channel_module.ray, "put", lambda value: value)
    return timeouts


def _fleet(prefill_pinned=None, widths=(_WIDTH,) * len(_ROLES)):
    log: list[str] = []
    engines = {
        name: _FakeEngine(
            name,
            log,
            pinned=prefill_pinned if role == "prefill" else None,
            width=width,
        )
        for role, width, name in zip(_ROLES, widths, _NAMES, strict=True)
    }
    records = [
        DynamoWorkerRecord(role, width, name, "10.0.0.1", engines[name])
        for role, width, name in zip(_ROLES, widths, _NAMES, strict=True)
    ]
    channel = MegatronRefitChannel(
        records,
        control_timeout_s=_CONTROL_TIMEOUT_S,
        idle_poll_interval_s=0.0,
    )
    return channel, log, engines


def test_channel_satisfies_the_engine_refit_protocol() -> None:
    channel, _, engines = _fleet()
    assert isinstance(channel, RefitChannel)
    assert channel.num_engines == 3
    assert channel.inference_world_size == 6
    assert channel.client_copy() is channel
    assert channel.flush_cache() is False
    assert [record.instance_id for record in channel.records] == _NAMES


def test_records_hold_only_worker_handles_never_the_generation() -> None:
    # A copy holding the MegatronGeneration would shut the engine down when it
    # is garbage collected; the channel keeps only the workers' actor handles.
    channel, _, engines = _fleet()
    for record in channel.records:
        client = record.control
        assert isinstance(client, MegatronDynamoWorkerClient)
        workers = engines[record.instance_id].worker_group.workers
        assert client.workers == tuple(workers)
        assert client.leader is workers[0]
    assert {field for field in MegatronDynamoWorkerClient.__dataclass_fields__} == {
        "workers",
        "leader",
    }


def test_collective_fans_out_in_record_order_with_rank_prefixes(
    get_timeouts,
) -> None:
    channel, log, _ = _fleet()
    channel.prepare_refit_info({"w": ((1,), "float32")})
    futures = channel.init_collective("10.0.0.9", 1234, 14, train_world_size=8)
    futures += channel.update_weights_from_collective(5.0, refit_with_reload_api=True)
    # One future per rank of every engine.
    assert len(futures) == 2 * len(_NAMES) * _WIDTH
    assert log == [
        "p0.prepare",
        "d0.prepare",
        "d1.prepare",
        "p0.init(ws=14,train=8,off=8)",
        "d0.init(ws=14,train=8,off=10)",
        "d1.init(ws=14,train=8,off=12)",
        "p0.update",
        "d0.update",
        "d1.update",
    ]
    # prepare_refit_info is bounded by the control timeout.
    assert get_timeouts and all(
        timeout is not None and timeout <= _CONTROL_TIMEOUT_S
        for timeout in get_timeouts
    )
    with pytest.raises(ValueError, match="does not match expected 14"):
        channel.init_collective("10.0.0.9", 1234, 13, train_world_size=8)
    with pytest.raises(ValueError, match="must not be None"):
        channel.prepare_refit_info(None)


def test_pause_and_resume_order_prefill_pins_around_decode(get_timeouts) -> None:
    channel, log, _ = _fleet(prefill_pinned=[3, 1, 0])
    channel.pause(clear_cache=False, timeout_s=None)
    assert log == [
        "p0.pause",
        # Prefill is suspended only after its pins drained, which takes three
        # reads; decode keeps serving (no decode call) the whole time.
        "p0.pinned",
        "p0.pinned",
        "p0.pinned",
        "p0.suspend",
        "d0.suspend",
        "d1.suspend",
        "p0.prepare_for_generation(['weights'])",
        "d0.prepare_for_generation(['weights'])",
        "d1.prepare_for_generation(['weights'])",
    ]
    log.clear()
    channel.resume(timeout_s=None)
    assert log == [
        "d0.prepare_for_generation(['kv_cache'])",
        "d1.prepare_for_generation(['kv_cache'])",
        "d0.resume",
        "d1.resume",
        "p0.prepare_for_generation(['kv_cache'])",
        "p0.resume",
    ]
    # Every wait on an engine is bounded by the control timeout.
    assert all(
        timeout is not None and timeout <= _CONTROL_TIMEOUT_S
        for timeout in get_timeouts
    )


def test_pause_times_out_on_stuck_prefill_pins_and_unpauses_prefill(
    get_timeouts,
) -> None:
    channel, log, _ = _fleet(prefill_pinned=[2])
    with pytest.raises(PinnedHandoffTimeout, match="'p0': 2"):
        channel.pause(clear_cache=False, timeout_s=0.0)
    # Prefill was paused, then handed back to serving; nothing was suspended.
    assert log == ["p0.pause", "p0.pinned", "p0.resume"]


@pytest.mark.parametrize(
    ("engine", "method", "restored"),
    [
        ("p0", "pause_for_refit", ["p0.resume"]),
        ("p0", "pinned_handoff_block_count", ["p0.resume"]),
        ("p0", "suspend_for_refit", ["p0.resume"]),
        ("d1", "suspend_for_refit", ["d0.resume", "d1.resume", "p0.resume"]),
        ("d0", "prepare_for_generation", ["d0.resume", "d1.resume", "p0.resume"]),
    ],
    ids=["prefill_pause", "pin_read", "prefill_suspend", "decode_suspend", "weights"],
)
def test_any_pause_failure_resumes_every_engine_it_reached(
    get_timeouts, engine, method, restored
) -> None:
    channel, log, engines = _fleet()
    engines[engine].faults[method] = ConnectionError("engine gone")
    with pytest.raises(ConnectionError, match="engine gone") as caught:
        channel.pause(clear_cache=False, timeout_s=None)
    assert f"failed on engine {engine}" in "".join(caught.value.__notes__)
    # Decode engines are resumed before prefill, as after a refit.
    assert [entry for entry in log if entry.endswith(".resume")] == restored


def test_a_hung_engine_call_times_out_and_the_fleet_is_resumed(
    get_timeouts,
) -> None:
    channel, log, engines = _fleet()
    engines["d0"].faults["suspend_for_refit"] = "hang"
    with pytest.raises(GetTimeoutError) as caught:
        channel.pause(clear_cache=False, timeout_s=0.5)
    assert "suspend_for_refit failed on engine d0" in "".join(caught.value.__notes__)
    assert [entry for entry in log if entry.endswith(".resume")] == [
        "d0.resume",
        "d1.resume",
        "p0.resume",
    ]
    # The pause waits share one deadline no longer than timeout_s.
    pause_waits = get_timeouts[: -len(_NAMES)]
    assert all(timeout is not None and timeout <= 0.5 for timeout in pause_waits)


def test_failed_restore_is_reported_on_the_original_error(get_timeouts) -> None:
    channel, _, engines = _fleet()
    engines["d0"].faults["suspend_for_refit"] = ConnectionError("engine gone")
    engines["d0"].faults["resume_after_refit"] = RuntimeError("still gone")
    with pytest.raises(ConnectionError) as caught:
        channel.pause(clear_cache=False, timeout_s=None)
    notes = "".join(caught.value.__notes__)
    assert "resuming engines after the failed pause also failed" in notes
    assert "d0: resume_after_refit" in notes


def test_resume_attempts_every_engine_and_reports_all_failures(get_timeouts) -> None:
    channel, log, engines = _fleet()
    engines["d0"].faults["prepare_for_generation"] = RuntimeError("cache")
    with pytest.raises(RuntimeError, match="d0: prepare_for_generation") as caught:
        channel.resume(timeout_s=None)
    assert "d1" not in str(caught.value) and "p0" not in str(caught.value)
    # d0 is not woken on a cache it failed to rebuild; the others still resume.
    assert [entry for entry in log if entry.endswith(".resume")] == [
        "d1.resume",
        "p0.resume",
    ]


def test_pause_rejects_clear_cache(get_timeouts) -> None:
    channel, log, _ = _fleet()
    with pytest.raises(NotImplementedError, match="recompute_kv_cache"):
        channel.pause(clear_cache=True, timeout_s=None)
    assert log == []


@pytest.mark.parametrize(
    ("generation_cfg", "expected"),
    [
        ({"backend": "megatron"}, True),
        ({"backend": "dynamo", "dynamo_cfg": {"engine": "megatron"}}, True),
        ({"backend": "dynamo", "dynamo_cfg": {"engine": "vllm"}}, False),
        ({"backend": "vllm"}, False),
    ],
    ids=["megatron", "dynamo_megatron", "dynamo_vllm", "vllm"],
)
def test_megatron_engines_follow_the_engine_behind_the_backend(
    generation_cfg, expected
) -> None:
    assert generation_runs_megatron_engines(generation_cfg) is expected


@pytest.mark.parametrize(
    ("records", "match"),
    [
        (
            [
                DynamoWorkerRecord("prefill", 4, "p0", "ip", MagicMock()),
                DynamoWorkerRecord("decode", 0, "d0", "ip", MagicMock()),
            ],
            "must span at least one",
        ),
        ([], "at least one engine record"),
    ],
    ids=["zero_width", "empty"],
)
def test_channel_rejects_malformed_fleets(records, match) -> None:
    with pytest.raises(ValueError, match=match):
        MegatronRefitChannel(records, control_timeout_s=1.0)


def test_check_full_fleet_rejects_a_partial_membership() -> None:
    channel, _, _ = _fleet()
    full = channel.full_fleet_membership(train_world_size=8)
    channel.check_full_fleet(full)
    partial = desired_membership(
        absent_shards=[1], dp_size=3, total_gen_workers=6, train_world_size=8
    )
    with pytest.raises(NotImplementedError, match="fixed fleet"):
        channel.check_full_fleet(partial)
    with pytest.raises(NotImplementedError, match="fixed fleet"):
        channel.rebuild_collective(partial, "10.0.0.9", 1234)


def test_mixed_widths_place_each_engine_after_the_ones_before_it(
    get_timeouts,
) -> None:
    """A prefill engine at TP 4 next to two decode engines at TP 2."""
    channel, log, _ = _fleet(widths=(4, 2, 2))
    assert channel.shard_widths == (4, 2, 2)
    assert channel.inference_world_size == 8

    full = channel.full_fleet_membership(train_world_size=8)
    assert full.shard_prefixes == {0: 0, 1: 4, 2: 6}
    assert full.workers_per_shard == (4, 2, 2)
    channel.check_full_fleet(full)
    channel.init_collective("10.0.0.9", 1234, 16, train_world_size=8)
    # Each engine is offset to the trainers plus the widths of the engines before it.
    assert log == [
        f"{name}.init(ws=16,train=8,off={8 + prefix})"
        for name, prefix in zip(_NAMES, (0, 4, 6), strict=True)
    ]
    with pytest.raises(ValueError, match="does not match expected 16"):
        channel.init_collective("10.0.0.9", 1234, 14, train_world_size=8)


def test_check_full_fleet_rejects_uniform_prefixes_against_mixed_widths() -> None:
    """An evenly sharded membership would start the decode engines at the wrong ranks."""
    channel, _, _ = _fleet(widths=(4, 2, 2))
    uniform = desired_membership(
        absent_shards=[], dp_size=3, total_gen_workers=6, train_world_size=8
    )
    with pytest.raises(NotImplementedError, match=r"ranks \[4, 2, 2\]"):
        channel.check_full_fleet(uniform)
    # Same prefixes but a width that is not the engine's is refused too.
    mismatched = desired_membership(
        absent_shards=[],
        dp_size=3,
        total_gen_workers=8,
        train_world_size=8,
        shard_widths=(4, 3, 1),
    )
    with pytest.raises(NotImplementedError, match="fixed fleet"):
        channel.check_full_fleet(mismatched)


@pytest.mark.parametrize(
    "call",
    [
        lambda c: c.rebuild_nccl_reshard_comm_group(
            c.full_fleet_membership(8),
            pp_ips=[],
            pp_ports=[],
            pp_size=1,
            train_ranks_per_stage=8,
            sub_world_size=14,
        ),
        lambda c: c.prepare_nccl_reshard_refit_info({}),
        lambda c: c.nccl_reshard_refit(None),
    ],
    ids=["comm_group", "refit_info", "refit"],
)
def test_reshard_transport_is_not_available(call) -> None:
    channel, _, _ = _fleet()
    with pytest.raises(NotImplementedError, match="nccl_reshard"):
        call(channel)
