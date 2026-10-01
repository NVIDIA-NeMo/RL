# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Dependency-light tests; run directly with unittest to avoid GPU conftest."""

import asyncio
import importlib.util
import tempfile
import threading
import time
import unittest
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

from tools.benchmark_prefix_checkpoint_proxy import (
    BoundedStager,
    CachedRecordBuilder,
    PerWorkerStager,
    RealTQCutBackend,
    TimedPutClient,
    latency_summary,
    load_checkpoint_driver,
    worker_for_row,
)


class ProxyTests(unittest.TestCase):
    def test_per_worker_routes_id_only_batches_without_shared_gate(self) -> None:
        args = SimpleNamespace(rows=8, generation_workers=2)
        measured = TimedPutClient(None)
        stager = PerWorkerStager(measured, args)
        received: list[tuple[int, list[int]]] = []

        def actor(owner: int) -> SimpleNamespace:
            async def remote(indices: list[int], checkpoint_id: str) -> dict:
                self.assertEqual(checkpoint_id, "checkpoint")
                self.assertTrue(all(isinstance(index, int) for index in indices))
                received.append((owner, indices))
                await asyncio.sleep(0.01)
                return {
                    "results": [dict(ok=True, staging_key=f"key-{i}") for i in indices],
                    "stage_seconds": 0.005,
                    "put_latencies": [0.004],
                    "batch_rows": [len(indices)],
                }

            return SimpleNamespace(stage=SimpleNamespace(remote=remote))

        stager.actors = [actor(0), actor(1)]

        def record(index: int) -> SimpleNamespace:
            return SimpleNamespace(rollout_id=f"rollout-{index:09d}")

        async def run() -> None:
            results = await asyncio.gather(
                stager.stage([record(0), record(1)], "checkpoint"),
                stager.stage([record(4)], "checkpoint"),
            )
            self.assertEqual(results[0][0].staging_key, "key-0")
            with self.assertRaisesRegex(ValueError, "crosses"):
                await stager.stage([record(0), record(4)], "checkpoint")

        asyncio.run(run())
        self.assertEqual(received, [(0, [0, 1]), (1, [4])])
        self.assertEqual(stager.peak_requests, 2)
        self.assertEqual(stager.active_requests, 0)
        self.assertEqual(measured.batch_rows, [2, 1])
        self.assertEqual(len(measured.latencies), 2)
        self.assertEqual(stager.waits, [])

    def test_per_worker_remote_failure_propagates(self) -> None:
        async def remote(indices: list[int], checkpoint_id: str) -> None:
            raise RuntimeError("injected writer failure")

        stager = PerWorkerStager(
            TimedPutClient(None), SimpleNamespace(rows=1, generation_workers=1)
        )
        stager.actors = [SimpleNamespace(stage=SimpleNamespace(remote=remote))]

        async def run() -> None:
            with self.assertRaisesRegex(RuntimeError, "injected writer failure"):
                await stager.stage(
                    [SimpleNamespace(rollout_id="rollout-000000000")], "checkpoint"
                )

        asyncio.run(run())
        self.assertEqual(stager.active_requests, 0)
        self.assertEqual(stager.stages, [])

    @unittest.skipUnless(
        importlib.util.find_spec("nemo_gym"), "requires Gym dependencies"
    )
    def test_cached_records_match_production_encoder(self) -> None:
        # Optional Gym dependency, not needed for scheduler-only tests.
        from nemo_gym.token_id_capture.staging.digest import (
            compute_chain_hash,
            compute_extras_digest,
            compute_staging_digest,
            hash_token_ids,
        )
        from nemo_gym.token_id_capture.staging.records import StagedCallRecord

        for length in (1, 17, 128, 16384):
            for child in (False, True):
                tokens = [i % 32000 for i in range(length)]
                masks = [float(i % 2) for i in range(length)]
                logprobs = [-0.125 if mask else 0.0 for mask in masks]
                extras = {"test": [1, "π"]} if child else None
                values = dict(
                    schema_version=2,
                    digest_version=2,
                    extras_digest_version=1,
                    rollout_id="original",
                    model_call_id="call-0",
                    parent_call_id="parent" if child else None,
                    mode="token_in" if child else "text",
                    prev_len=3 if child else 0,
                    delta_len=length,
                    cum_len=length + (3 if child else 0),
                    weight_version=7,
                    token_ids_delta=tokens,
                    token_mask_delta=masks,
                    generation_log_probs_delta=logprobs,
                    extras_digest=compute_extras_digest(extras),
                    chain_hash=compute_chain_hash("a" * 64 if child else None, tokens),
                    cumulative_hash=hash_token_ids(
                        ([1, 2, 3] if child else []) + tokens
                    ),
                )
                template = StagedCallRecord(
                    **values, extras=extras, digest=compute_staging_digest(**values)
                )
                builder = CachedRecordBuilder(template)
                for rollout_id, call_id in (
                    ("a", "b"),
                    ("rollout-000131071", "call-000131071"),
                    ("π/日本語", "call/é"),
                ):
                    changed = values | {
                        "rollout_id": rollout_id,
                        "model_call_id": call_id,
                    }
                    expected = StagedCallRecord(
                        **changed,
                        extras=extras,
                        digest=compute_staging_digest(**changed),
                    )
                    actual = builder.build(rollout_id=rollout_id, model_call_id=call_id)
                    self.assertEqual(actual.model_dump(), expected.model_dump())
                    StagedCallRecord.model_validate(actual.model_dump())
                another = builder.build(rollout_id="next", model_call_id="next")
                self.assertIs(actual.token_ids_delta, another.token_ids_delta)
                self.assertIsNot(template.token_ids_delta, actual.token_ids_delta)
                with self.assertRaises(ValueError):
                    builder.build(rollout_id="", model_call_id="call")
                with self.assertRaises(ValueError):
                    CachedRecordBuilder(
                        template.model_copy(update={"digest": "0" * 64})
                    )

    @unittest.skipUnless(
        importlib.util.find_spec("nemo_gym"), "requires Gym dependencies"
    )
    def test_real_coordinator_commit_and_coordinate_restore(self) -> None:
        """Real Gym checkpoint lifecycle with a fake acknowledged storage sink."""
        driver = load_checkpoint_driver()
        args = SimpleNamespace(
            rows=16, generation_workers=4, batch_size=3, prefix_tokens=8
        )
        records = [
            SimpleNamespace(
                rollout_id=f"rollout-{i:09d}",
                model_call_id=f"call-{i:09d}",
                digest="a" * 64,
            )
            for i in range(args.rows)
        ]

        def key(record: SimpleNamespace, checkpoint_id: str) -> str:
            return f"__generation_cut__/{checkpoint_id}/{record.rollout_id}/{record.model_call_id}/0"

        class Sink:
            def stage_generation_prefix_batch(
                self,
                batch: list[SimpleNamespace],
                *,
                checkpoint_id: str,
                chunk_sequences: list[int],
            ) -> list[SimpleNamespace]:
                return [
                    SimpleNamespace(ok=True, staging_key=key(record, checkpoint_id))
                    for record in batch
                ]

        async def run(root: Path) -> None:
            stager = BoundedStager(Sink(), concurrency=2)
            try:
                result = await driver.run_checkpoint_benchmark(
                    root,
                    workers=2,
                    cuts=args.rows,
                    hot_worker_fraction=0.5,
                    prefix_tokens=8,
                    staging_key_bytes=0,
                    timeout_s=30,
                    backend_factory=lambda owner: RealTQCutBackend(
                        owner, records, stager, args
                    ),
                    staging_keys_for_ticket=lambda ticket, checkpoint_id: (
                        key(ticket, checkpoint_id),
                    ),
                )
                self.assertEqual(result["generation_cuts_restored"], args.rows)
                self.assertTrue(result["restored_prefix_coordinates_match"])
                timings = result["restore_timings"]
                for phase in (
                    "metadata_validation_seconds",
                    "archive_validation_seconds",
                    "tar_inventory_seconds",
                    "member_read_validation_seconds",
                    "lineage_parse_seconds",
                    "file_materialization_seconds",
                    "namespace_inventory_seconds",
                    "directory_publish_seconds",
                    "receipt_reconstruction_seconds",
                    "result_serialization_seconds",
                ):
                    self.assertGreater(timings[phase], 0, phase)
                self.assertAlmostEqual(
                    sum(
                        value
                        for name, value in timings.items()
                        if name != "total_seconds"
                    ),
                    timings["total_seconds"],
                )
                self.assertLessEqual(
                    timings["total_seconds"], result["restore_seconds"]
                )
                self.assertEqual(len(stager.stages), 8)
            finally:
                stager.close()

        with tempfile.TemporaryDirectory(prefix="proxy-unit-") as root:
            asyncio.run(run(Path(root)))

    def test_ownership_and_expected_batch_counts(self) -> None:
        for rows, expected in (
            (8192, 128),
            (16384, 128),
            (65536, 256),
            (131072, 512),
            (262144, 1024),
        ):
            counts = Counter(
                worker_for_row(index, rows=rows, workers=128) for index in range(rows)
            )
            self.assertEqual(len(counts), 128)
            self.assertEqual(
                sum((count + 255) // 256 for count in counts.values()), expected
            )
        self.assertEqual(
            [worker_for_row(i, rows=3, workers=8) for i in range(3)], [0, 1, 2]
        )
        self.assertEqual(
            [worker_for_row(i, rows=5, workers=2) for i in range(5)], [0, 0, 0, 1, 1]
        )

    def test_percentiles(self) -> None:
        summary = latency_summary(list(range(1, 101)))
        self.assertEqual(summary["p50_s"], 50)
        self.assertEqual(summary["p95_s"], 95)
        self.assertEqual(summary["p99_s"], 99)
        self.assertEqual(latency_summary([]), {"count": 0})

    def test_put_failure_is_measured_and_propagated(self) -> None:
        class BrokenClient:
            def put_samples(self, **kwargs: object) -> None:
                raise RuntimeError("injected")

        measured = TimedPutClient(BrokenClient())
        with self.assertRaisesRegex(RuntimeError, "injected"):
            measured.put_samples(sample_ids=["a", "b"])
        self.assertEqual(measured.failures, 1)
        self.assertEqual(measured.batch_rows, [2])
        self.assertEqual(measured.active, 0)

    def test_concurrency_is_bounded_and_all_batches_finish(self) -> None:
        class Sink:
            def __init__(self) -> None:
                self.lock = threading.Lock()
                self.active = 0
                self.peak = 0
                self.rows = 0

            def stage_generation_prefix_batch(
                self, records: list[int], **kwargs: object
            ) -> list[SimpleNamespace]:
                with self.lock:
                    self.active += 1
                    self.peak = max(self.peak, self.active)
                time.sleep(0.01)
                with self.lock:
                    self.active -= 1
                    self.rows += len(records)
                return [SimpleNamespace(ok=True) for _ in records]

        sink = Sink()

        async def run() -> None:
            stager = BoundedStager(sink, concurrency=2)
            try:
                await asyncio.gather(
                    *(stager.stage([index], "checkpoint") for index in range(20))
                )
                self.assertEqual(len(stager.stages), 20)
                self.assertEqual(len(stager.waits), 20)
                self.assertGreater(max(stager.waits), 0.01)
            finally:
                stager.close()

        asyncio.run(run())
        self.assertEqual(sink.peak, 2)
        self.assertEqual(sink.active, 0)
        self.assertEqual(sink.rows, 20)

    def test_failed_batch_cannot_become_evidence(self) -> None:
        class Sink:
            def stage_generation_prefix_batch(
                self, records: list[int], **kwargs: object
            ) -> list[SimpleNamespace]:
                return [SimpleNamespace(ok=False)]

        async def run() -> None:
            stager = BoundedStager(Sink(), concurrency=1)
            try:
                with self.assertRaisesRegex(RuntimeError, "Prefix staging failed"):
                    await stager.stage([0], "checkpoint")
            finally:
                stager.close()

        asyncio.run(run())


if __name__ == "__main__":
    unittest.main()
