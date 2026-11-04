# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
"""Ensure pending-group time is visible before the inner admission timer."""

import asyncio
import json
import logging
from types import SimpleNamespace

import pytest

from nemo_rl.models.generation.generation_cut_grouping import GenerationCutGrouper
from nemo_rl.utils.prefix_cut_diagnostics import prefix_cut_probe


@pytest.mark.asyncio
async def test_group_logs_arrivals_occupancy_and_each_members_pending_wait(caplog):
    entered = asyncio.Event()
    release = asyncio.Event()

    async def execute(items, deadline):
        entered.set()
        await release.wait()
        return [SimpleNamespace(validate_for=lambda inventory: None) for _ in items]

    group = GenerationCutGrouper(
        max_groups=1,
        max_inventories=8,
        max_calls=256,
        wait_seconds=0,
        max_payload_bytes=1024,
        execute=execute,
    )

    def inventory(index):
        return SimpleNamespace(
            checkpoint_id="checkpoint",
            inventory_digest=str(index),
            active_prefixes=[
                SimpleNamespace(model_call_id=str(index), ticket_id=str(index))
            ],
        )

    caplog.set_level(
        logging.INFO, logger="nemo_rl.models.generation.generation_cut_grouping"
    )
    with prefix_cut_probe(
        enabled=True,
        checkpoint_id="checkpoint",
        inventory_digest="outer",
        calls=1,
        deadline=None,
    ):
        first = asyncio.create_task(group.run(inventory(1), deadline=None))
        await entered.wait()
        others = [
            asyncio.create_task(group.run(inventory(i), deadline=None)) for i in (2, 3)
        ]
        await asyncio.sleep(0)
        assert group.stats()["pending_members"] == 2
        release.set()
        await asyncio.gather(first, *others)
    group.close()
    events = [
        json.loads(record.getMessage().split("generation_cut_group ", 1)[1])
        for record in caplog.records
        if record.getMessage().startswith("generation_cut_group ")
    ]
    arrivals = [event for event in events if event["event"] == "enqueue"]
    dispatches = [event for event in events if event["event"] == "dispatch"]
    assert len(arrivals) == 3
    assert arrivals[1]["active_groups"] == arrivals[2]["active_groups"] == 1
    assert [len(event["members"]) for event in dispatches] == [1, 2]
    assert dispatches[1]["active_groups"] == 1
    assert dispatches[1]["pending_inventories"] == 0
    for member in dispatches[1]["members"]:
        assert member["pending_wait_s"] > 0
        assert member["enqueued_unix_s"] <= dispatches[1]["unix_s"]
