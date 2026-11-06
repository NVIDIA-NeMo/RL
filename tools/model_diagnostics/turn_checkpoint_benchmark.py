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
"""Turn-only generation/TQ proxy; excludes Gym's multi-turn agent graph.

Each HTTP call is one model turn. Completed calls name immutable acknowledged
TQ rows. In-flight calls are recorded for regeneration with fresh call IDs.
There are no prefix rows to delete and no consumer clearing terminal rows, so
those completed rows remain stable while later calls continue staging.
"""

from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path
from typing import Any


def capture_turn_boundary(loads: list[Any]) -> dict[str, Any]:
    """Copy one driver-loop boundary without awaiting or draining active calls."""
    workers = []
    for load in loads:
        completed = list(load.completed_turns.values())
        if not completed:
            raise RuntimeError(f"{load.name} has no completed durable turns to save")
        pending = [
            {
                "rollout_id": call.rollout_id,
                "previous_model_call_id": call.model_call_id,
                "request_index": call.request_index,
                "admitted_at": call.admitted_at,
            }
            for call in load.active.values()
        ]
        workers.append(
            {
                "name": load.name,
                "completed_turns": completed,
                "pending_calls": pending,
                "next_request_index": load.next_id + 1,
            }
        )
    return {
        "request_factory": "prefix_cut_benchmark.Load._run/v1",
        "pending_recovery": "regenerate_with_fresh_model_call_id",
        "workers": workers,
    }


async def run_turn_snapshot(
    generations: list[Any],
    loads: list[Any],
    storage: Any,
    *,
    round_index: int,
    output: Path,
    staging_partition: str,
    worker_call: Any,
) -> dict[str, Any]:
    """Save completed turn rows while unfinished model calls keep running."""
    # Ray is optional in CPU protocol tests and plan validation.
    import ray

    checkpoint_id = f"benchmark-turn-round-{round_index}"
    started = time.monotonic()
    timings: dict[str, float] = {}
    for load in loads:
        load.allow_new.clear()
    try:
        before = time.monotonic()
        cut = capture_turn_boundary(loads)
        timings["turn_boundary_capture_s"] = time.monotonic() - before
        keys = [
            record["commit_coords"]["staging_key"]
            for worker in cut["workers"]
            for record in worker["completed_turns"]
        ]
        before = time.monotonic()
        verified = await asyncio.to_thread(
            ray.get, storage.verify_keys.remote(keys, staging_partition)
        )
        timings["verify_live_turn_rows_s"] = time.monotonic() - before
        snapshot = output / f"snapshot-{round_index:03d}"
        snapshot.mkdir()
        before = time.monotonic()
        storage_seconds = await asyncio.to_thread(
            ray.get, storage.save.remote(str(snapshot / "data_plane"))
        )
        timings["tq_save_s"] = time.monotonic() - before
        timings["tq_storage_save_s"] = storage_seconds
        before = time.monotonic()
        saved = await asyncio.to_thread(
            ray.get,
            storage.verify_saved_turn_keys.remote(
                str(snapshot / "data_plane"), keys, staging_partition
            ),
        )
        if saved != verified:
            raise RuntimeError("Saved controller snapshot lost completed turn rows")
        timings["verify_saved_turn_rows_s"] = time.monotonic() - before
        before = time.monotonic()
        state_text = json.dumps(cut)
        (snapshot / "turn_state.json").write_text(state_text)
        manifest = {
            "kind": "generation_only_turn_checkpoint_proxy",
            "scope": "completed model turns and TQ; no Gym agent graph or trainer",
            "checkpoint_id": checkpoint_id,
            "committed": True,
            "checkpoint_level": "turn",
            "generation_prefix_cuts_enabled": False,
            "generation_fence_used": False,
            "verified_completed_turn_rows": saved,
            "prefix_rows": 0,
            "recovery_validation": "saved row presence; end-to-end restore not exercised",
        }
        temporary = snapshot / "manifest.json.tmp"
        temporary.write_text(json.dumps(manifest, indent=2))
        temporary.replace(snapshot / "manifest.json")
        timings["turn_state_publication_s"] = time.monotonic() - before
    finally:
        for load in loads:
            load.allow_new.set()
    timings["total_s"] = time.monotonic() - started
    statuses = await asyncio.gather(
        *(
            asyncio.to_thread(worker_call, generation, "benchmark_status")
            for generation in generations
        )
    )
    workers = []
    for load, state, status in zip(loads, cut["workers"], statuses, strict=True):
        workers.append(
            {
                "name": load.name,
                "slots": status["slots"],
                "calls": len(state["pending_calls"]),
                "completed_turns": len(state["completed_turns"]),
                "pending_calls_to_regenerate": len(state["pending_calls"]),
                "durable_prefixes": 0,
                "http_successes": 0,
                "inventory_http_latency_s": [],
                "diagnostics": status,
                "load": load.summary(),
            }
        )
    result = {
        "round": round_index,
        "checkpoint_id": checkpoint_id,
        "checkpoint_level": "turn",
        "generation_fence_used": False,
        "timings": timings,
        "workers": workers,
    }
    (output / f"round-{round_index:03d}.json").write_text(json.dumps(result, indent=2))
    return result
