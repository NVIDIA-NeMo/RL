# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
"""Distinguish scheduler byte pressure from preparation and acknowledgement."""

import json
import logging
import threading
import time
from types import SimpleNamespace

from nemo_rl.models.generation.generation_cut_pipeline import (
    GenerationCutPipeline,
    PrefixCutWrite,
)
from nemo_rl.utils.checkpoint_latency import CheckpointLatencyConfig


def test_pipeline_reports_buffer_wait_and_completion_work(caplog):
    caplog.set_level(logging.INFO)
    entered, release = threading.Event(), threading.Event()

    class Sink:
        def stage_generation_prefix_batch(self, records, **kwargs):
            entered.set()
            assert release.wait(3)
            return ["key"]

    def prepare(item, *, deadline):
        keys = yield PrefixCutWrite(
            item.checkpoint_id,
            (SimpleNamespace(token_ids_delta=[1]),),
            (0,),
            (None,),
            64,
            deadline,
            lambda: [],
            False,
        )
        return keys

    pipeline = GenerationCutPipeline(
        Sink(),
        prepare=prepare,
        max_buffered_bytes=64,
        max_batch_bytes=64,
        max_rows=1,
        max_tokens=1,
        write_concurrency=1,
        coalesce_wait_seconds=0,
        latency_diagnostics=CheckpointLatencyConfig(
            enabled=True, progress_interval_s=0.01
        ),
    )
    try:
        jobs = [
            pipeline.submit(
                SimpleNamespace(checkpoint_id="cp", inventory_digest=str(i)),
                deadline=None,
            )
            for i in range(2)
        ]
        assert entered.wait(3)
        # Let the scheduler enter its condition wait while the native write owns the bytes.
        deadline = time.monotonic() + 3
        while pipeline.stats()["buffer_wait_s"] == 0 and time.monotonic() < deadline:
            with pipeline._condition:
                pipeline._condition.notify_all()
            time.sleep(0.005)
        release.set()
        assert [job.result(3) for job in jobs] == [["key"], ["key"]]
    finally:
        release.set()
        pipeline.close(wait=True)
    summaries = [
        json.loads(r.message.split(" ", 1)[1])
        for r in caplog.records
        if r.message.startswith("generation_cut_pipeline_progress ")
    ]
    final = next(s for s in summaries if s["final"])
    assert final["checkpoint_id"] == "cp"
    assert final["pending_inventories"] == final["buffered_payload_bytes"] == 0
    assert final["scheduler_phases"]["buffer_capacity_wait"]["wall_s"] > 0
    assert final["scheduler_phases"]["prepare"]["thread_cpu_s"] >= 0
    assert final["scheduler_phases"]["completion_processing"]["wall_s"] >= 0
