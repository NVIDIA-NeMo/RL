# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

import asyncio
import json
import sys
from types import SimpleNamespace

import pytest

from tools.model_diagnostics.turn_checkpoint_benchmark import (
    capture_turn_boundary,
    run_turn_snapshot,
)


def make_load():
    admitted = asyncio.Event()
    admitted.set()
    return SimpleNamespace(
        name="engine",
        next_id=2,
        allow_new=admitted,
        completed_turns={
            "done": {"request_index": 1, "commit_coords": {"staging_key": "key"}}
        },
        active={
            "pending": SimpleNamespace(
                rollout_id="pending",
                model_call_id="physical",
                request_index=2,
                admitted_at=10.0,
            )
        },
        summary=lambda: {"errors": 0},
    )


def test_boundary_does_not_wait_for_inflight_call():
    async def exercise():
        load = make_load()
        cut = capture_turn_boundary([load])
        assert cut["workers"][0]["pending_calls"][0]["request_index"] == 2
        assert cut["pending_recovery"] == "regenerate_with_fresh_model_call_id"
        load.completed_turns["late"] = {"commit_coords": {"staging_key": "late"}}
        assert len(cut["workers"][0]["completed_turns"]) == 1
        load.completed_turns.clear()
        with pytest.raises(RuntimeError, match="no completed durable turns"):
            capture_turn_boundary([load])

    asyncio.run(exercise())


@pytest.mark.parametrize("fail_at", [None, "save", "verify"])
def test_snapshot_uses_no_prefix_or_fence_and_reopens_admission(
    tmp_path, monkeypatch, fail_at
):
    class Remote:
        def __init__(self, method):
            self.remote = method

    def save(path):
        if fail_at == "save":
            raise OSError("storage unavailable")
        return 0.1

    storage = SimpleNamespace(
        verify_keys=Remote(lambda keys, partition: 1),
        save=Remote(save),
        verify_saved_turn_keys=Remote(
            lambda path, keys, partition: 0 if fail_at == "verify" else 1
        ),
    )
    monkeypatch.setitem(
        sys.modules, "ray", SimpleNamespace(get=lambda reference: reference)
    )
    methods = []

    def worker_call(generation, method):
        methods.append(method)
        assert method == "benchmark_status"
        return {"slots": 4}

    async def exercise():
        load = make_load()
        args = dict(
            generations=[object()],
            loads=[load],
            storage=storage,
            round_index=0,
            output=tmp_path,
            staging_partition="staging",
            worker_call=worker_call,
        )
        if fail_at is not None:
            with pytest.raises((OSError, RuntimeError)):
                await run_turn_snapshot(**args)
            assert not (tmp_path / "snapshot-000/manifest.json").exists()
        else:
            result = await run_turn_snapshot(**args)
            assert result["workers"][0]["durable_prefixes"] == 0
            assert result["workers"][0]["completed_turns"] == 1
            assert result["workers"][0]["pending_calls_to_regenerate"] == 1
            manifest = json.loads((tmp_path / "snapshot-000/manifest.json").read_text())
            assert manifest["generation_fence_used"] is False
            assert manifest["verified_completed_turn_rows"] == 1
            assert methods == ["benchmark_status"]
        assert load.allow_new.is_set()

    asyncio.run(exercise())
