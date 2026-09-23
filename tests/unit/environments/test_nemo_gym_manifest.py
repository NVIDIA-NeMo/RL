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
"""Direct ledger reads (``token_capture.manifest_transport: local_file``).

Covers the actor-owned :class:`ManifestReceiptService` (bounded admission,
deadline, cancellation, shutdown), its equivalence with the HTTP manifest
route, and the ``run_rollouts`` streaming integration.
"""

from __future__ import annotations

import asyncio
import fcntl
import threading
import time
from pathlib import Path
from typing import Any

import pytest

from nemo_rl.environments.nemo_gym import NemoGym
from nemo_rl.environments.nemo_gym_manifest import (
    ManifestReadJob,
    ManifestReceiptService,
    format_metrics_line,
)

pytestmark = pytest.mark.nemo_gym


# ── fixtures ─────────────────────────────────────────────────────────────────


def _capture_env() -> NemoGym:
    env_cls = NemoGym.__ray_metadata__.modified_class
    env = env_cls({})
    env._token_capture_enabled = True
    env._manifest_transport = "local_file"
    env._manifest_read_workers = 2
    env._manifest_queue_size = 4
    env._control_timeout_s = 5.0
    env._manifest_service = None
    env._manifest_reader = None
    env._manifest_last_report = 0.0
    env._tokenizer = object()
    env.rh = object()
    env.cfg = {}
    return env


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


def _service(reader, **overrides) -> ManifestReceiptService:
    kwargs: dict[str, Any] = dict(workers=2, queue_size=4, timeout_s=5.0)
    kwargs.update(overrides)
    return ManifestReceiptService(reader, NemoGym._assemble_receipt, **kwargs)


# ── equivalence with the HTTP manifest route ─────────────────────────────────


def test_local_receipt_matches_http_receipt(ledger) -> None:
    store, reader, _ = ledger

    async def run():
        await store.record(_commit("r1", 1))
        await store.record(_commit("r1", 2, parent="c1", prev_len=16))
        await store.record_failure(
            "r1", "c3", "request_finished_without_staged_coordinates"
        )
        http_manifest = await store.manifest("r1")
        http_receipt = NemoGym._assemble_receipt(
            "r1",
            http_manifest,
            terminal_response_id="resp-r1-2",
            scored_response=None,
            reward=1.0,
        )
        service = _service(reader)
        service.start()
        try:
            read = await service.receipt(ManifestReadJob("r1", "resp-r1-2", None, 1.0))
        finally:
            await service.close()
        return http_receipt, read

    http_receipt, read = asyncio.run(run())
    assert read.error is None
    assert read.receipt == http_receipt
    assert read.receipt["terminal_model_call_id"] == "c2"
    assert read.receipt["capture_poisoned"] is False


def test_missing_ledger_yields_the_same_empty_manifest_receipt(ledger) -> None:
    store, reader, _ = ledger

    async def run():
        http_manifest = await store.manifest("never-ran")
        service = _service(reader)
        service.start()
        try:
            read = await service.receipt(ManifestReadJob("never-ran", None, None, 0.0))
        finally:
            await service.close()
        return http_manifest, read

    http_manifest, read = asyncio.run(run())
    assert http_manifest["records"] == []
    assert read.error is None
    assert read.receipt == NemoGym._assemble_receipt(
        "never-ran", http_manifest, terminal_response_id=None, reward=0.0
    )
    assert read.receipt["capture_poisoned"] is True
    assert read.receipt["failure_reason"] == "no_records"


def test_malformed_ledger_row_is_a_corrupt_error_not_a_crash(ledger) -> None:
    _, reader, root = ledger
    (root / "bad.lineage.jsonl").write_bytes(b'{"model_call_id": "c1"}\n{not json\n')

    async def run():
        service = _service(reader)
        service.start()
        try:
            return await service.receipt(ManifestReadJob("bad", None, None, 0.0))
        finally:
            await service.close()

    read = asyncio.run(run())
    assert read.receipt is None
    assert read.error_kind == "corrupt"
    assert "malformed" in (read.error or "")


# ── admission, deadline, cancellation ────────────────────────────────────────


class _BlockingReader:
    """Reader whose reads block until released; counts concurrent readers."""

    def __init__(self):
        self.release = threading.Event()
        self.active = 0
        self.peak = 0
        self._lock = threading.Lock()
        self.calls = 0

    def read_manifest(
        self, rollout_id, *, deadline=None, cancel_event=None, stats=None
    ):
        from nemo_gym.token_id_capture.lineage import (
            ManifestReadCancelled,
            ManifestReadTimeout,
        )

        with self._lock:
            self.calls += 1
            self.active += 1
            self.peak = max(self.peak, self.active)
        try:
            while not self.release.wait(0.005):
                if cancel_event is not None and cancel_event.is_set():
                    raise ManifestReadCancelled(rollout_id)
                if deadline is not None and time.monotonic() >= deadline:
                    raise ManifestReadTimeout(rollout_id)
            return {"rollout_id": rollout_id, "records": [], "failures": []}
        finally:
            with self._lock:
                self.active -= 1


def test_active_jobs_never_exceed_workers_and_queue_backpressures() -> None:
    reader = _BlockingReader()

    async def run():
        service = _service(reader, workers=2, queue_size=3, timeout_s=10.0)
        service.start()
        futures = []
        # 2 run, 3 queue, the 6th submit must block until a slot frees.
        for index in range(5):
            futures.append(
                await service.submit(ManifestReadJob(f"r{index}", None, None, 0.0))
            )
        await asyncio.sleep(0.05)
        assert reader.peak <= 2
        assert service.active_jobs() == 2
        assert service.queue_depth() == 3
        sixth = asyncio.ensure_future(
            service.submit(ManifestReadJob("r5", None, None, 0.0))
        )
        await asyncio.sleep(0.05)
        assert not sixth.done(), "submit must back-pressure on a full queue"
        reader.release.set()
        futures.append(await asyncio.wait_for(sixth, 2.0))
        results = await asyncio.gather(*futures)
        metrics = service.metrics_snapshot()
        await service.close()
        return results, metrics, reader.peak

    results, metrics, peak = asyncio.run(run())
    assert all(result.error is None for result in results)
    assert peak == 2
    assert metrics["active_high_water"] <= 2
    assert metrics["queue_high_water"] <= 3
    assert metrics["count/receipt_ok"] == 6
    assert "manifest_local_file" in format_metrics_line("t", metrics)


def test_deadline_covers_admission_wait_and_keeps_slot_until_thread_exits() -> None:
    reader = _BlockingReader()

    async def run():
        service = _service(reader, workers=1, queue_size=8, timeout_s=0.3)
        service.start()
        blocked = asyncio.ensure_future(
            service.receipt(ManifestReadJob("slow", None, None, 0.0))
        )
        queued = asyncio.ensure_future(
            service.receipt(ManifestReadJob("queued", None, None, 0.0))
        )
        results = await asyncio.gather(blocked, queued)
        # Both timed out: the running one at its deadline, the queued one
        # because admission ate its whole budget.
        assert {result.error_kind for result in results} == {"timeout"}
        # The worker thread observed its deadline and exited on its own; the
        # service has capacity again and is still healthy.
        await asyncio.sleep(0.1)
        assert service.active_jobs() == 0
        reader.release.set()
        follow_up = await service.receipt(ManifestReadJob("after", None, None, 0.0))
        assert follow_up.error is None
        assert reader.calls == 2, (
            "the queued job must not start after its deadline passed"
        )
        await service.close()

    asyncio.run(run())


def test_cancelled_waiter_signals_the_job_and_frees_the_slot_only_when_it_exits() -> (
    None
):
    reader = _BlockingReader()

    async def run():
        service = _service(reader, workers=1, queue_size=4, timeout_s=10.0)
        service.start()
        future = await service.submit(ManifestReadJob("r0", None, None, 0.0))
        await asyncio.sleep(0.05)
        assert service.active_jobs() == 1
        future.cancel()
        # The job sees cancel_event at its next checkpoint and returns; until
        # then the slot stays occupied.
        for _ in range(100):
            if service.active_jobs() == 0:
                break
            await asyncio.sleep(0.01)
        assert service.active_jobs() == 0
        metrics = service.metrics_snapshot()
        assert metrics["count/cancelled"] == 1
        await service.close()

    asyncio.run(run())


def test_shutdown_drops_queued_jobs_and_is_idempotent() -> None:
    reader = _BlockingReader()

    async def run():
        service = _service(reader, workers=1, queue_size=4, timeout_s=10.0)
        service.close_nowait()  # before start
        service.close_nowait()  # again
        service = _service(reader, workers=1, queue_size=4, timeout_s=10.0)
        service.start()
        running = await service.submit(ManifestReadJob("running", None, None, 0.0))
        queued = await service.submit(ManifestReadJob("queued", None, None, 0.0))
        await asyncio.sleep(0.05)
        await service.close()
        await service.close()
        assert queued.done() and queued.result().error_kind == "cancelled"
        reader.release.set()
        with pytest.raises(RuntimeError):
            await service.submit(ManifestReadJob("late", None, None, 0.0))
        return running

    running = asyncio.run(run())
    assert running.done() or running.cancelled()


def test_held_writer_lock_times_out_within_the_deadline(ledger) -> None:
    store, reader, root = ledger

    async def run():
        await store.record(_commit("r1", 1))
        lock = open(root / "r1.tokens.lock", "a+b")
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        service = _service(reader, timeout_s=0.3)
        service.start()
        started = time.monotonic()
        try:
            read = await service.receipt(ManifestReadJob("r1", None, None, 0.0))
        finally:
            fcntl.flock(lock.fileno(), fcntl.LOCK_UN)
            lock.close()
            await service.close()
        return read, time.monotonic() - started

    read, elapsed = asyncio.run(run())
    assert read.error_kind == "timeout"
    assert "deadline" in (read.error or "")
    assert elapsed < 2.0


# ── streaming integration through run_rollouts ───────────────────────────────


class _FakeRCH:
    def __init__(self, completions: list[tuple[dict, dict]], delay_s: float = 0.0):
        self._completions = completions
        self._delay_s = delay_s

    def run_examples(self, examples, head_server_config=None):
        async def _one(row, result):
            if self._delay_s:
                await asyncio.sleep(self._delay_s)
            return row, result

        return (_one(row, result) for row, result in self._completions)


def _rows(rollout_ids: list[str]) -> list[dict]:
    return [
        {
            "_rowidx": index,
            "_ng_rollout_id": rollout_id,
            "agent_ref": {"name": "agent"},
            "responses_create_params": {"input": []},
        }
        for index, rollout_id in enumerate(rollout_ids)
    ]


def test_http_mode_preserves_sequential_receipt_processing(monkeypatch) -> None:
    """Only local-file mode overlaps manifests; HTTP keeps the existing path."""
    monkeypatch.setattr(
        "nemo_rl.environments.nemo_gym.normalize_media_in_examples", lambda rows: None
    )
    monkeypatch.setattr(
        "nemo_rl.utils.fastokens.maybe_patch_fastokens", lambda enabled: None
    )

    async def run():
        env = _capture_env()
        env._manifest_transport = "http"
        rows = _rows(["first", "second"])
        env.rch = _FakeRCH([(row, {}) for row in rows])
        env.head_server_config = None
        env._require_spinup = lambda: None
        calls = []
        started = asyncio.Event()
        release = asyncio.Event()

        async def receipt(row, result):
            calls.append(row["_rowidx"])
            if row["_rowidx"] == 0:
                started.set()
                await release.wait()
            return {"receipt": row["_rowidx"]}

        env._postprocess_receipt_mode = receipt
        stream = env.run_rollouts(rows, "http")
        first = asyncio.create_task(anext(stream))
        try:
            await asyncio.wait_for(started.wait(), 2)
            assert calls == [0]
            assert not first.done()
            release.set()
            assert (await asyncio.wait_for(first, 2))[0] == 0
            assert calls == [0]
            assert (await asyncio.wait_for(anext(stream), 2))[0] == 1
            assert calls == [0, 1]
            assert env._manifest_service is None
        finally:
            first.cancel()
            await asyncio.gather(first, return_exceptions=True)
            await stream.aclose()

    asyncio.run(run())


def test_run_rollouts_local_mode_streams_receipts_and_reports_timing(
    ledger, monkeypatch
) -> None:
    store, reader, _ = ledger
    monkeypatch.setattr(
        "nemo_rl.environments.nemo_gym.normalize_media_in_examples", lambda rows: None
    )
    monkeypatch.setattr(
        "nemo_rl.utils.fastokens.maybe_patch_fastokens", lambda enabled: None
    )

    async def run():
        for index in range(6):
            await store.record(_commit(f"r{index}", 1))
        env = _capture_env()
        env._manifest_reader = reader
        rows = _rows([f"r{index}" for index in range(6)])
        results = [
            {
                "reward": float(index),
                "response": {"id": f"resp-r{index}-1", "output": []},
            }
            for index in range(6)
        ]
        env.rch = _FakeRCH(list(zip(rows, results)), delay_s=0.01)
        env.head_server_config = None
        env._require_spinup = lambda: None
        yielded = []
        async for item in env.run_rollouts(rows, "nemo_gym"):
            yielded.append(item)
        env.shutdown = lambda: None
        service = env._manifest_service
        metrics = service.metrics_snapshot()
        await service.close()
        return yielded, metrics

    yielded, metrics = asyncio.run(run())
    assert len(yielded) == 6
    rowidxs = sorted(item[0] for item in yielded)
    assert rowidxs == list(range(6))
    for rowidx, _agent_ref, result, _timing in yielded:
        assert result["message_log"] == []
        assert result["receipt"]["rollout_id"] == f"r{rowidx}"
        assert result["receipt"]["terminal_model_call_id"] == "c1"
        assert result["receipt"]["reward"] == float(rowidx)
    timings = [item[3] for item in yielded if item[3] is not None]
    assert len(timings) == 1
    timing = timings[0]
    assert "nemo_gym/postprocess_results_pct" in timing
    assert timing["nemo_gym/manifest_read_total"] > 0.0
    assert "nemo_gym/manifest_admission_wait" in timing
    assert metrics["count/receipt_ok"] == 6


def test_run_rollouts_local_mode_shares_one_service_across_concurrent_calls(
    ledger, monkeypatch
) -> None:
    store, reader, _ = ledger
    monkeypatch.setattr(
        "nemo_rl.environments.nemo_gym.normalize_media_in_examples", lambda rows: None
    )
    monkeypatch.setattr(
        "nemo_rl.utils.fastokens.maybe_patch_fastokens", lambda enabled: None
    )

    async def run():
        env = _capture_env()
        env._manifest_reader = reader
        env._manifest_queue_size = 2
        env._manifest_read_workers = 1
        env._require_spinup = lambda: None
        env.head_server_config = None

        async def one_stream(prefix: str, count: int):
            ids = [f"{prefix}{index}" for index in range(count)]
            for rollout_id in ids:
                await store.record(_commit(rollout_id, 1))
            rows = _rows(ids)
            results = [
                {"reward": 1.0, "response": {"id": f"resp-{rid}-1"}} for rid in ids
            ]
            env.rch = _FakeRCH(list(zip(rows, results)))
            return [item async for item in env.run_rollouts(rows, "nemo_gym")]

        # Two concurrent streams larger than the queue: both must drain via one
        # shared, bounded service without deadlock.
        a, b = await asyncio.gather(one_stream("a", 8), one_stream("b", 8))
        service = env._manifest_service
        metrics = service.metrics_snapshot()
        await service.close()
        return a, b, metrics

    a, b, metrics = asyncio.run(run())
    assert len(a) == 8 and len(b) == 8
    assert all(item[2]["receipt"] is not None for item in a + b)
    assert metrics["count/receipt_ok"] == 16
    assert metrics["queue_high_water"] <= 2
    assert metrics["active_high_water"] <= 1


def test_postprocess_receipt_local_reports_failures_as_placeholder_rows(
    ledger, capsys
) -> None:
    _, reader, root = ledger
    (root / "bad.lineage.jsonl").write_bytes(b"{oops\n")
    env = _capture_env()
    env._manifest_reader = reader

    async def run():
        result = await env._postprocess_receipt_local(
            {"_ng_rollout_id": "bad"}, {"reward": 0.5, "response": {"id": "x"}}
        )
        await env._manifest_service.close()
        return result

    result = asyncio.run(run())
    assert result["receipt"] is None
    assert result["rollout_id"] == "bad"
    assert result["full_result"]["reward"] == 0.5
    assert "manifest(bad) fetch failed (local_file corrupt)" in capsys.readouterr().out


def test_spinup_local_mode_requires_a_visible_writer_root(tmp_path: Path) -> None:
    env = _capture_env()
    env._control_timeout_s = 1.0
    import nemo_rl.environments.nemo_gym as module

    original = module._MANIFEST_ROOT_WAIT_S
    module._MANIFEST_ROOT_WAIT_S = 0.2
    try:
        with pytest.raises(RuntimeError, match="does not see the ledger"):
            env._init_local_manifest_reader(str(tmp_path / "missing"))
        empty = tmp_path / "empty"
        empty.mkdir()
        with pytest.raises(RuntimeError, match="does not see the ledger"):
            env._init_local_manifest_reader(str(empty))
    finally:
        module._MANIFEST_ROOT_WAIT_S = original


def test_spinup_local_mode_accepts_a_root_with_a_matching_writer(
    tmp_path: Path, capsys
) -> None:
    from nemo_gym.token_id_capture.lineage import FileLineageStore

    root = tmp_path / "lineage"
    FileLineageStore(root)  # writes the identity marker
    env = _capture_env()
    env._init_local_manifest_reader(str(root))
    assert env._manifest_reader is not None
    out = capsys.readouterr().out
    assert "manifest transport: local_file" in out
    assert "matching_writers=1" in out


def test_spinup_rejects_unknown_manifest_transport() -> None:
    from nemo_rl.algorithms.single_controller_utils.config import TokenCaptureConfig

    with pytest.raises(ValueError):
        TokenCaptureConfig(enabled=True, manifest_transport="carrier_pigeon")
    with pytest.raises(ValueError):
        TokenCaptureConfig(enabled=True, manifest_read_workers=0)
    config = TokenCaptureConfig(enabled=True, manifest_transport="local_file")
    assert config.manifest_read_workers == 2
    assert config.manifest_queue_size == 256
    assert TokenCaptureConfig().manifest_transport == "http"


@pytest.fixture(params=["asyncio", "uvloop"])
def loop_factory(request):
    if request.param == "uvloop":
        import uvloop

        return uvloop.new_event_loop
    return asyncio.new_event_loop


async def _until(predicate, timeout=2.0):
    async with asyncio.timeout(timeout):
        while not predicate():
            await asyncio.sleep(0.005)


class _UninterruptibleReader:
    """Model a filesystem operation that cannot observe cancellation yet."""

    def __init__(self):
        self.started = threading.Event()
        self.release = threading.Event()
        self.calls = 0

    def read_manifest(self, rollout_id, **kwargs):
        self.calls += 1
        self.started.set()
        assert self.release.wait(5), "test must release the worker"
        return {"rollout_id": rollout_id, "records": [], "failures": []}


def test_submit_deadline_bounds_full_queue_and_retains_running_slot(loop_factory):
    reader = _UninterruptibleReader()

    async def run():
        service = _service(reader, workers=1, queue_size=1, timeout_s=0.15)
        service.start()
        try:
            running = await service.submit(ManifestReadJob("running", None, None, 0.0))
            await _until(reader.started.is_set)
            queued = await service.submit(ManifestReadJob("queued", None, None, 0.0))
            started = time.monotonic()
            blocked = asyncio.create_task(
                service.submit(ManifestReadJob("blocked", None, None, 0.0))
            )
            await asyncio.sleep(0)
            assert not blocked.done()
            admitted = await asyncio.wait_for(blocked, 1)
            results = await asyncio.wait_for(
                asyncio.gather(running, queued, admitted), 1
            )
            assert time.monotonic() - started < 0.8
            assert [result.error_kind for result in results] == ["timeout"] * 3
            assert service.active_jobs() == 1
            assert reader.calls == 1
            reader.release.set()
            await _until(lambda: service.active_jobs() == 0)
            followup = await service.receipt(ManifestReadJob("after", None, None, 0.0))
            assert followup.error is None
            assert reader.calls == 2
            metrics = service.metrics_snapshot()
            assert metrics["count/completed"] == 4
            assert metrics["count/timeout"] == 3
            assert metrics["count/receipt_ok"] == 1
        finally:
            reader.release.set()
            await service.close()

    with asyncio.Runner(loop_factory=loop_factory) as runner:
        runner.run(run())


def test_receipt_assembly_cannot_return_success_after_deadline(ledger, loop_factory):
    _, reader, _ = ledger
    release = threading.Event()
    started = threading.Event()

    def assemble(*args, **kwargs):
        started.set()
        assert release.wait(5)
        return {"late": True}

    async def run():
        service = ManifestReceiptService(
            reader, assemble, workers=1, queue_size=1, timeout_s=0.15
        )
        service.start()
        try:
            future = await service.submit(ManifestReadJob("empty", None, None, 0.0))
            await _until(started.is_set)
            result = await asyncio.wait_for(future, 1)
            assert result.receipt is None and result.error_kind == "timeout"
            assert service.active_jobs() == 1
            release.set()
            await _until(lambda: service.active_jobs() == 0)
            assert future.result() is result
            metrics = service.metrics_snapshot()
            assert metrics["count/completed"] == 1
            assert metrics["count/receipt_ok"] == 0
        finally:
            release.set()
            await service.close()

    with asyncio.Runner(loop_factory=loop_factory) as runner:
        runner.run(run())


@pytest.mark.parametrize("invalid_fields", [{}, {"prev_len": None}])
def test_schema_invalid_json_does_not_kill_a_consumer(
    ledger, loop_factory, invalid_fields
):
    import orjson

    store, reader, root = ledger
    row = {
        "model_call_id": "bad-call",
        "staging_key": "bad-key",
        "response_id": "bad-response",
        "chain_hash": "chain",
        "cumulative_hash": "cumulative",
        **invalid_fields,
    }
    (root / "bad.lineage.jsonl").write_bytes(orjson.dumps(row) + b"\n")

    async def run():
        await store.record(_commit("good", 1))
        service = _service(reader, workers=1)
        service.start()
        try:
            bad = await service.submit(ManifestReadJob("bad", None, None, 0.0))
            good = await service.submit(
                ManifestReadJob("good", "resp-good-1", None, 1.0)
            )
            invalid, valid = await asyncio.wait_for(asyncio.gather(bad, good), 2)
            assert invalid.error_kind == "corrupt"
            assert "invalid fields" in invalid.error
            assert valid.error is None
            assert valid.receipt["terminal_model_call_id"] == "c1"
            assert service.active_jobs() == 0
        finally:
            await service.close()

    with asyncio.Runner(loop_factory=loop_factory) as runner:
        runner.run(run())


def test_unexpected_worker_error_resolves_waiter_and_preserves_consumer(loop_factory):
    class Reader:
        def read_manifest(self, rollout_id, **kwargs):
            if rollout_id == "bad":
                raise LookupError("unexpected reader failure")
            return {"rollout_id": rollout_id, "records": [], "failures": []}

    async def run():
        service = _service(Reader(), workers=1)
        service.start()
        try:
            bad = await service.submit(ManifestReadJob("bad", None, None, 0.0))
            good = await service.submit(ManifestReadJob("good", None, None, 0.0))
            invalid, valid = await asyncio.wait_for(asyncio.gather(bad, good), 2)
            assert invalid.error_kind == "error"
            assert "unexpected reader failure" in invalid.error
            assert valid.error is None
            assert service.active_jobs() == 0
        finally:
            await service.close()

    with asyncio.Runner(loop_factory=loop_factory) as runner:
        runner.run(run())


def test_shutdown_wakes_blocked_admission_without_waiting_for_thread(loop_factory):
    reader = _UninterruptibleReader()

    async def run():
        service = _service(reader, workers=1, queue_size=1, timeout_s=30)
        service.start()
        try:
            running = await service.submit(ManifestReadJob("running", None, None, 0.0))
            await _until(reader.started.is_set)
            queued = await service.submit(ManifestReadJob("queued", None, None, 0.0))
            blocked = asyncio.create_task(
                service.submit(ManifestReadJob("blocked", None, None, 0.0))
            )
            await asyncio.sleep(0)
            assert not blocked.done()
            await asyncio.wait_for(service.close(), 1)
            last = await asyncio.wait_for(blocked, 1)
            assert all(
                future.result().error_kind == "cancelled"
                for future in (running, queued, last)
            )
            assert not reader.release.is_set()
            assert service.metrics_snapshot()["count/completed"] == 3
        finally:
            reader.release.set()
            await service.close()

    with asyncio.Runner(loop_factory=loop_factory) as runner:
        runner.run(run())


def test_cancelled_admission_never_runs_after_capacity_returns(loop_factory):
    reader = _UninterruptibleReader()

    async def run():
        service = _service(reader, workers=1, queue_size=1, timeout_s=5)
        service.start()
        try:
            running = await service.submit(ManifestReadJob("running", None, None, 0.0))
            await _until(reader.started.is_set)
            queued = await service.submit(ManifestReadJob("queued", None, None, 0.0))
            blocked = asyncio.create_task(
                service.submit(ManifestReadJob("cancelled", None, None, 0.0))
            )
            await asyncio.sleep(0)
            assert not blocked.done()
            blocked.cancel()
            with pytest.raises(asyncio.CancelledError):
                await blocked
            reader.release.set()
            results = await asyncio.wait_for(asyncio.gather(running, queued), 2)
            assert all(result.error is None for result in results)
            assert reader.calls == 2
            assert service.metrics_snapshot()["count/cancelled"] == 1
        finally:
            reader.release.set()
            await service.close()

    with asyncio.Runner(loop_factory=loop_factory) as runner:
        runner.run(run())


@pytest.mark.parametrize("ending", ["cancel", "rollout_error", "close"])
def test_local_stream_cleans_up_reads_and_rollout_tasks(loop_factory, ending):
    reader = _BlockingReader()
    cancelled = threading.Event()
    original_read = reader.read_manifest

    def read(rollout_id, **kwargs):
        if rollout_id == "fast":
            return {"rollout_id": rollout_id, "records": [], "failures": []}
        try:
            return original_read(rollout_id, **kwargs)
        finally:
            cancelled.set()

    reader.read_manifest = read

    async def run():
        env = _capture_env()
        env._manifest_reader = reader
        env._manifest_read_workers = 2 if ending == "close" else 1
        env._manifest_queue_size = 1
        rows = _rows(["slow", "fast" if ending == "close" else "queued", "pending"])
        reached_pending = asyncio.Event()
        release_rollout = asyncio.Event()

        class Helper:
            def run_examples(self, **kwargs):
                async def complete(row):
                    if row["_ng_rollout_id"] == "pending":
                        reached_pending.set()
                        await release_rollout.wait()
                        raise RuntimeError("rollout failed")
                    return row, {"reward": 1.0}

                return (complete(row) for row in rows)

        env.rch = Helper()
        before = asyncio.all_tasks()
        stream = env.run_rollouts(rows, "test")
        advance = asyncio.create_task(anext(stream))
        try:
            await _until(lambda: reader.active == 1)
            if ending == "close":
                first = await asyncio.wait_for(advance, 2)
                assert first[0] == 1
                await stream.aclose()
            else:
                await asyncio.wait_for(reached_pending.wait(), 2)
                await _until(lambda: env._manifest_service.queue_depth() == 1)
                if ending == "cancel":
                    advance.cancel()
                    with pytest.raises(asyncio.CancelledError):
                        await advance
                else:
                    release_rollout.set()
                    with pytest.raises(RuntimeError, match="rollout failed"):
                        await asyncio.wait_for(advance, 2)
            await _until(cancelled.is_set)
            await _until(lambda: env._manifest_service.active_jobs() == 0)
            assert reader.calls == 1, (
                "cancelled queued work must never enter the reader"
            )
        finally:
            reader.release.set()
            if not advance.done():
                advance.cancel()
            await asyncio.gather(advance, return_exceptions=True)
            await stream.aclose()
            await env._manifest_service.close()
        assert asyncio.all_tasks() == before

    with asyncio.Runner(loop_factory=loop_factory) as runner:
        runner.run(run())


def test_rollout_error_does_not_wait_for_a_blocked_manifest_admission(loop_factory):
    reader = _UninterruptibleReader()

    async def run():
        env = _capture_env()
        env._manifest_reader = reader
        env._manifest_read_workers = 1
        env._manifest_queue_size = 1
        rows = _rows(["running", "queued", "blocked", "failed"])
        reached_failure = asyncio.Event()
        release_failure = asyncio.Event()

        class Helper:
            def run_examples(self, **kwargs):
                async def complete(row):
                    if row["_ng_rollout_id"] == "failed":
                        reached_failure.set()
                        await release_failure.wait()
                        raise RuntimeError("rollout failed with full manifest queue")
                    return row, {"reward": 0.0}

                return (complete(row) for row in rows)

        env.rch = Helper()
        stream = env.run_rollouts(rows, "test")
        advance = asyncio.create_task(anext(stream))
        try:
            await asyncio.wait_for(reached_failure.wait(), 2)
            assert env._manifest_service.active_jobs() == 1
            assert env._manifest_service.queue_depth() == 1
            release_failure.set()
            with pytest.raises(RuntimeError, match="full manifest queue"):
                await asyncio.wait_for(advance, 1)
            assert not reader.release.is_set()
        finally:
            reader.release.set()
            if not advance.done():
                advance.cancel()
            await asyncio.gather(advance, return_exceptions=True)
            await stream.aclose()
            await env._manifest_service.close()

    with asyncio.Runner(loop_factory=loop_factory) as runner:
        runner.run(run())
