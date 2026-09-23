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
"""Direct ledger receipt equivalence, bounded concurrency and stream lifecycle."""

import asyncio
import gc
import threading
import time
import weakref
from pathlib import Path

import pytest

from nemo_rl.environments.nemo_gym import NemoGym
from nemo_rl.environments.nemo_gym_manifest import ManifestReceiptReader

pytestmark = pytest.mark.nemo_gym


def _commit(rollout_id: str, index: int, parent: str | None = None, prev_len: int = 0):
    from nemo_gym.token_id_capture.staging.digest import (
        EMPTY_EXTRAS_DIGEST,
        compute_chain_hash,
        hash_token_ids,
    )
    from nemo_gym.token_id_capture.staging.records import (
        CallRecord,
        CaptureLedgerCommit,
    )

    tokens = list(range(prev_len, prev_len + 16))
    record = CallRecord(
        model_call_id=f"c{index}",
        parent_call_id=parent,
        staging_key=f"{rollout_id}/c{index}",
        weight_version=1,
        prev_len=prev_len,
        delta_len=16,
        cum_len=prev_len + 16,
        digest="a" * 64,
        extras_digest=EMPTY_EXTRAS_DIGEST,
        mode="text" if parent is None else "token_in",
        chain_hash=compute_chain_hash(None, tokens),
        cumulative_hash=hash_token_ids(tokens),
        response_id=f"resp-{rollout_id}-{index}",
        admitted_at=1.0,
        output_fingerprint=None,
        continuation_fingerprint=None,
        fingerprint_version=1,
    )
    return CaptureLedgerCommit(
        rollout_id=rollout_id,
        record=record,
        staging_chain=(f"{rollout_id}/c{index}",),
        request_items=[{"role": "user", "content": f"q{index}"}],
        response_items=[{"role": "assistant", "content": f"a{index}"}],
    )


@pytest.fixture
def ledger(tmp_path: Path):
    """A writer-side FileLineageStore and a reader on the same root."""
    from nemo_gym.token_id_capture.lineage import FileLineageStore, FileManifestReader

    root = tmp_path / "lineage"
    store = FileLineageStore(root)
    reader = FileManifestReader(root)
    return store, reader, root


@pytest.fixture(params=["asyncio", "uvloop"])
def runner(request):
    factory = asyncio.new_event_loop
    if request.param == "uvloop":
        factory = pytest.importorskip("uvloop").new_event_loop
    with asyncio.Runner(loop_factory=factory) as value:
        yield value


def _pool(reader, *, workers=2, queue_size=2, timeout_s=2, assemble=None):
    return ManifestReceiptReader(
        reader,
        assemble or NemoGym._assemble_receipt,
        workers=workers,
        queue_size=queue_size,
        timeout_s=timeout_s,
    )


def _read(pool, rollout_id="r1"):
    return pool.read_receipt(
        rollout_id, terminal_response_id=None, scored_response=None, reward=1.0
    )


async def _until(predicate):
    async with asyncio.timeout(2):
        while not predicate():
            await asyncio.sleep(0.001)


class BlockingReader:
    def __init__(self):
        self.calls = []
        self.release = threading.Event()
        self.cancelled = threading.Event()

    def read_manifest(self, rollout_id, *, deadline, cancel_event):
        self.calls.append(rollout_id)
        if rollout_id == "slow":
            while not self.release.wait(0.001):
                if cancel_event.is_set():
                    self.cancelled.set()
                    break
        elif rollout_id == "stuck":
            self.release.wait(5)  # Deliberately ignores cooperative cancellation.
        return {"rollout_id": rollout_id, "records": [], "failures": []}


def _env(reader=None, *, transport="local_file", workers=2, queue_size=2):
    env = NemoGym.__ray_metadata__.modified_class({})
    env._token_capture_enabled = True
    env._manifest_transport = transport
    env._tokenizer = object()
    env._require_spinup = lambda: None
    if reader is not None:
        env._manifest_reader = _pool(reader, workers=workers, queue_size=queue_size)
    return env


def _rows(ids):
    return [
        {
            "_rowidx": i,
            "_ng_rollout_id": rid,
            "agent_ref": {"name": "agent"},
            "responses_create_params": {"input": []},
        }
        for i, rid in enumerate(ids)
    ]


class Rollouts:
    def __init__(self, pending=None, fail=False):
        self.pending = pending
        self.fail = fail
        self.cancelled = asyncio.Event()

    def run_examples(self, examples, head_server_config=None):
        async def complete(row):
            if row["_ng_rollout_id"] == "pending":
                try:
                    await self.pending.wait()
                finally:
                    self.cancelled.set()
                if self.fail:
                    raise RuntimeError("rollout failed")
            return row, {"reward": 1.0}

        return (complete(row) for row in examples)


def test_local_receipts_match_http_including_missing_ledgers(ledger, runner):
    store, reader, _ = ledger

    async def run():
        await store.record(_commit("r1", 1))
        await store.record(_commit("r1", 2, parent="c1", prev_len=16))
        await store.record_failure("r1", "c3", "capture_failed")
        env = _env(reader)
        rows = _rows(["r1", "missing"])
        try:
            for row in rows:
                rid = row["_ng_rollout_id"]
                result = await env._postprocess_receipt_mode(row, {"reward": 1.0})
                expected = NemoGym._assemble_receipt(
                    rid,
                    await store.manifest(rid),
                    terminal_response_id=None,
                    reward=1.0,
                )
                assert result["receipt"] == expected
                assert result["message_log"] == []
                assert result["rollout_id"] == rid
        finally:
            await env._manifest_reader.close()

    runner.run(run())


@pytest.mark.parametrize(
    "data",
    [
        b"{invalid",
        b"[]\n",
        b'{"staging_key":"k","response_id":"r","chain_hash":"h","cumulative_hash":"h"}\n',
    ],
)
def test_bad_ledger_becomes_placeholder_and_next_read_succeeds(ledger, runner, data):
    store, reader, root = ledger
    (root / "bad.lineage.jsonl").write_bytes(data)

    async def run():
        await store.record(_commit("good", 1))
        env = _env(reader)
        try:
            bad, good = _rows(["bad", "good"])
            assert (await env._postprocess_receipt_mode(bad, {}))["receipt"] is None
            assert (await env._postprocess_receipt_mode(good, {}))[
                "receipt"
            ] is not None
        finally:
            await env._manifest_reader.close()

    runner.run(run())


def test_timeout_bounds_admission_and_keeps_slots_until_executor_finishes(runner):
    reader = BlockingReader()

    async def run():
        pool = _pool(reader, workers=1, queue_size=1, timeout_s=0.1)
        try:
            running = asyncio.create_task(_read(pool, "stuck"))
            await _until(lambda: reader.calls == ["stuck"])
            queued = asyncio.create_task(_read(pool, "queued"))
            await _until(lambda: len(pool._jobs) == 2)
            blocked = asyncio.create_task(_read(pool, "blocked"))
            outcomes = await asyncio.wait_for(
                asyncio.gather(running, queued, blocked, return_exceptions=True), 1
            )
            assert all(isinstance(result, TimeoutError) for result in outcomes)
            assert len(pool._jobs) == 2  # Caller expiry did not release real capacity.
            assert reader.calls == ["stuck"]
            reader.release.set()
            await _until(lambda: not pool._jobs)
            assert reader.calls == [
                "stuck"
            ]  # Expired queued work never enters the reader.
            assert (await _read(pool, "later"))["rollout_id"] == "later"
        finally:
            reader.release.set()
            await pool.close()

    runner.run(run())


def test_cancellation_retains_capacity_and_skips_cancelled_queued_work(runner):
    reader = BlockingReader()

    async def run():
        pool = _pool(reader, workers=1, queue_size=1)
        tasks = [
            asyncio.create_task(_read(pool, rid))
            for rid in ["stuck", "queued", "blocked"]
        ]
        try:
            await _until(lambda: len(pool._jobs) == 2 and reader.calls == ["stuck"])
            for task in tasks:
                task.cancel()
            results = await asyncio.gather(*tasks, return_exceptions=True)
            assert all(isinstance(result, asyncio.CancelledError) for result in results)
            assert len(pool._jobs) == 2
            reader.release.set()
            await _until(lambda: not pool._jobs)
            assert reader.calls == ["stuck"]
            assert (await _read(pool, "later"))["rollout_id"] == "later"
        finally:
            reader.release.set()
            await pool.close()

    runner.run(run())


def test_close_wakes_waiters_without_waiting_for_os_reads(runner):
    reader = BlockingReader()

    async def run():
        pool = _pool(reader, workers=1, queue_size=1)
        tasks = [
            asyncio.create_task(_read(pool, rid))
            for rid in ["stuck", "queued", "blocked"]
        ]
        try:
            await _until(lambda: len(pool._jobs) == 2 and reader.calls == ["stuck"])
            await asyncio.wait_for(pool.close(), 0.5)
            assert not reader.release.is_set()
            assert all(task.cancelled() for task in tasks)
            await pool.close()
            with pytest.raises(RuntimeError, match="closed"):
                await _read(pool)
        finally:
            reader.release.set()
            await _until(lambda: not pool._jobs)

    runner.run(run())


def test_late_receipt_and_unexpected_errors_do_not_poison_the_reader(runner):
    reader = BlockingReader()

    def assemble(rollout_id, manifest, **kwargs):
        if rollout_id == "late":
            time.sleep(0.15)
        if rollout_id == "error":
            raise LookupError("assembly bug")
        return manifest

    async def run():
        pool = _pool(reader, workers=1, queue_size=1, timeout_s=0.05, assemble=assemble)
        try:
            with pytest.raises(TimeoutError):
                await _read(pool, "late")
            assert len(pool._jobs) == 1
            await _until(lambda: not pool._jobs)
            with pytest.raises(RuntimeError, match="assembly bug"):
                await _read(pool, "error")
            assert (await _read(pool, "good"))["rollout_id"] == "good"
        finally:
            await pool.close()

    runner.run(run())


def test_stream_yields_ready_receipt_before_slow_read_or_rollout(runner):
    reader = BlockingReader()

    async def run():
        env = _env(reader)
        env.rch = Rollouts(pending=asyncio.Event())
        before = asyncio.all_tasks()
        stream = env.run_rollouts(_rows(["slow", "fast", "pending"]), "test")
        try:
            first = await asyncio.wait_for(anext(stream), 1)
            assert first[0] == 1
            assert first[2]["receipt"]["rollout_id"] == "fast"
            assert not env.rch.pending.is_set()
            await stream.aclose()
            await _until(reader.cancelled.is_set)
            assert env.rch.cancelled.is_set()
            assert asyncio.all_tasks() == before
        finally:
            reader.release.set()
            await stream.aclose()
            await env._manifest_reader.close()

    runner.run(run())


@pytest.mark.parametrize("ending", ["cancel", "rollout_error"])
def test_stream_failure_cleans_up_running_queued_and_pending_work(runner, ending):
    reader = BlockingReader()

    async def run():
        env = _env(reader, workers=1, queue_size=1)
        env.rch = Rollouts(pending=asyncio.Event(), fail=True)
        before = asyncio.all_tasks()
        stream = env.run_rollouts(
            _rows(["slow", "queued", "blocked", "pending"]), "test"
        )
        advance = asyncio.create_task(anext(stream))
        try:
            await _until(
                lambda: reader.calls == ["slow"]
                and len(env._manifest_reader._jobs) == 2
            )
            if ending == "cancel":
                advance.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await advance
            else:
                env.rch.pending.set()
                with pytest.raises(RuntimeError, match="rollout failed"):
                    await asyncio.wait_for(advance, 1)
            await _until(reader.cancelled.is_set)
            await _until(lambda: not env._manifest_reader._jobs)
            assert reader.calls == ["slow"]
            assert env.rch.cancelled.is_set()
            assert asyncio.all_tasks() == before
        finally:
            reader.release.set()
            await stream.aclose()
            await env._manifest_reader.close()

    runner.run(run())


def test_concurrent_streams_share_worker_bound_and_report_timing(runner):
    class Reader:
        active = 0
        high_water = 0
        guard = threading.Lock()

        def read_manifest(self, rollout_id, **kwargs):
            with self.guard:
                self.active += 1
                self.high_water = max(self.high_water, self.active)
            time.sleep(0.002)
            with self.guard:
                self.active -= 1
            return {"rollout_id": rollout_id, "records": [], "failures": []}

    reader = Reader()

    async def run():
        env = _env(reader, workers=2, queue_size=2)
        env.rch = Rollouts()

        async def collect(prefix):
            return [
                item
                async for item in env.run_rollouts(
                    _rows([f"{prefix}{i}" for i in range(20)]), "test"
                )
            ]

        try:
            first, second = await asyncio.gather(collect("a"), collect("b"))
            for batch in (first, second):
                assert len(batch) == 20
                assert batch[-1][3]["test/manifest_read_total"] > 0
                assert {item[0] for item in batch} == set(range(20))
            assert reader.high_water == 2
        finally:
            await env._manifest_reader.close()

    runner.run(run())


def test_http_mode_keeps_sequential_receipt_processing(runner):
    async def run():
        env = _env(transport="http")
        env.rch = Rollouts()
        calls = []
        started, release = asyncio.Event(), asyncio.Event()

        async def receipt(row, result):
            calls.append(row["_rowidx"])
            if row["_rowidx"] == 0:
                started.set()
                await release.wait()
            return {"receipt": row["_rowidx"]}

        env._postprocess_receipt_mode = receipt
        stream = env.run_rollouts(_rows(["first", "second"]), "test")
        first = asyncio.create_task(anext(stream))
        try:
            await asyncio.wait_for(started.wait(), 1)
            assert calls == [0] and not first.done()
            release.set()
            assert (await first)[0] == 0
            assert calls == [0]
            assert (await anext(stream))[0] == 1
            assert calls == [0, 1]
            assert env._manifest_reader is None
        finally:
            first.cancel()
            await asyncio.gather(first, return_exceptions=True)
            await stream.aclose()

    runner.run(run())


def test_startup_requires_matching_writer_root(ledger, tmp_path, monkeypatch, runner):
    _, _, root = ledger
    monkeypatch.setattr("nemo_rl.environments.nemo_gym._MANIFEST_ROOT_WAIT_S", 0)

    async def run():
        env = _env()
        with pytest.raises(RuntimeError, match="manifest_transport: http"):
            env._init_local_manifest_reader(str(tmp_path / "missing"))
        env._init_local_manifest_reader(str(root))
        assert env._manifest_reader is not None
        await env.shutdown()
        await env.shutdown()

    runner.run(run())


def test_stream_releases_delivered_payloads_before_batch_finishes(runner):
    class Payload:
        pass

    async def run():
        references = {}
        pending = asyncio.Event()

        class Helper:
            def run_examples(self, examples, **kwargs):
                async def complete(row):
                    index = row["_rowidx"]
                    if index == 2:
                        await pending.wait()
                    payload = Payload()
                    references[index] = weakref.ref(payload)
                    return row, {"reward": 1.0, "payload": payload}

                return (complete(row) for row in examples)

        env = _env(BlockingReader())
        env.rch = Helper()
        stream = env.run_rollouts(_rows(["first", "second", "pending"]), "test")
        try:
            first = await anext(stream)
            first_index = first[0]
            second = await anext(stream)
            assert second[0] != first_index
            del first
            gc.collect()
            assert references[first_index]() is None
            assert not pending.is_set()
        finally:
            await stream.aclose()
            await env._manifest_reader.close()

    runner.run(run())
