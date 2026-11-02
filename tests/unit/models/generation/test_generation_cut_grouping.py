# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may obtain a copy at http://www.apache.org/licenses/LICENSE-2.0
"""CPU tests for grouped admission, original receipts, and native ownership."""

import asyncio
import time
from collections.abc import Awaitable, Callable
from typing import Any

import pytest
from nemo_gym._checkpoint.generation_cut import (
    GenerationCutInventory,
    GenerationCutPrefix,
    GenerationCutPrefixAck,
    GenerationCutReceipt,
)

from nemo_rl.models.generation.generation_cut_control import GenerationCutControl
from nemo_rl.models.generation.generation_cut_grouping import GenerationCutGrouper


def inventory(
    index: int, *, checkpoint: str = "checkpoint", calls: int = 1
) -> GenerationCutInventory:
    return GenerationCutInventory.build(
        checkpoint_id=checkpoint,
        server_name=f"participant-{index}",
        active_prefixes=[
            GenerationCutPrefix(
                ticket_id=f"ticket-{index}-{i}",
                model_call_id=f"call-{index}-{i}",
                rollout_id=f"rollout-{index}-{i}",
                attempt=0,
                admitted_at=1.0,
            )
            for i in range(calls)
        ],
    )


def receipt(item: GenerationCutInventory) -> GenerationCutReceipt:
    return GenerationCutReceipt(
        checkpoint_id=item.checkpoint_id,
        cut_id="cut",
        inventory_digest=item.inventory_digest,
        inventory=item,
        backend_snapshot_id="persisted-backend",
        prefixes=tuple(
            GenerationCutPrefixAck.failure(prefix) for prefix in item.active_prefixes
        ),
    )


def grouper(
    execute: Callable[..., Awaitable[Any]],
    *,
    max_groups: int = 4,
    max_calls: int = 256,
    max_inventories: int = 8,
    wait: float = 0.001,
) -> GenerationCutGrouper:
    return GenerationCutGrouper(
        max_groups=max_groups,
        max_inventories=max_inventories,
        max_calls=max_calls,
        wait_seconds=wait,
        max_payload_bytes=1024 * 1024,
        execute=execute,
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("max_groups", [1, 4])
async def test_staggered_arrivals_combine_only_after_busy_slot_finishes(
    max_groups: int,
) -> None:
    entered: asyncio.Queue[tuple[list[Any], asyncio.Event]] = asyncio.Queue()
    executions = []
    active = 0
    peak_active = 0

    async def execute(items: list[Any], deadline: float | None) -> list[Any]:
        nonlocal active, peak_active
        active += 1
        peak_active = max(peak_active, active)
        release = asyncio.Event()
        executions.append(items)
        entered.put_nowait((items, release))
        try:
            await release.wait()
            return [receipt(item) for item in items]
        finally:
            active -= 1

    group = grouper(execute, max_groups=max_groups, wait=0)
    initial = [inventory(i, calls=32) for i in range(max_groups)]
    tasks = []
    initial_releases = []
    for item in initial:
        tasks.append(asyncio.create_task(group.run(item, deadline=None)))
        items, release = await asyncio.wait_for(entered.get(), 1)
        assert items == [item]
        initial_releases.append(release)

    queued = [inventory(max_groups + i, calls=32) for i in range(16)]
    for item in queued:
        tasks.append(asyncio.create_task(group.run(item, deadline=None)))
        # Each arrival gets its own event-loop turn, rather than arriving in
        # the initial collection window. Busy slots must leave it unclaimed.
        await asyncio.sleep(0.001)
    assert len(executions) == max_groups
    assert group.stats()["active_groups"] == max_groups
    assert group.stats()["pending_members"] == 16

    # A completion must drain pending work without adding a collection timer.
    group._wait = 60
    initial_releases[0].set()
    items, release = await asyncio.wait_for(entered.get(), 1)
    assert items == queued[:8]
    assert group.stats()["pending_members"] == 8
    release.set()
    items, release = await asyncio.wait_for(entered.get(), 1)
    assert items == queued[8:]
    release.set()
    for initial_release in initial_releases[1:]:
        initial_release.set()
    answers = await asyncio.wait_for(asyncio.gather(*tasks), 1)
    for original, answer in zip(initial + queued, answers, strict=True):
        answer.validate_for(original)
    await asyncio.sleep(0)
    assert peak_active == max_groups
    assert group.stats()["groups"] == max_groups + 2
    assert group.stats()["pending_members"] == 0
    assert group.stats()["active_groups"] == 0


@pytest.mark.asyncio
async def test_cancelled_and_expired_members_are_removed_while_capacity_is_full() -> (
    None
):
    entered, release = asyncio.Event(), asyncio.Event()
    executions = []

    async def execute(items: list[Any], deadline: float | None) -> list[Any]:
        executions.append(items)
        entered.set()
        await release.wait()
        return [receipt(item) for item in items]

    group = grouper(execute, max_groups=1, wait=0)
    active = asyncio.create_task(group.run(inventory(0), deadline=None))
    await asyncio.wait_for(entered.wait(), 1)
    cancelled = asyncio.create_task(group.run(inventory(1), deadline=None))
    expired = asyncio.create_task(
        group.run(inventory(2), deadline=time.monotonic() + 0.02)
    )
    survivor = asyncio.create_task(group.run(inventory(3), deadline=None))
    await asyncio.sleep(0)
    cancelled.cancel()
    with pytest.raises(asyncio.CancelledError):
        await cancelled
    with pytest.raises(TimeoutError):
        await expired
    assert executions == [[inventory(0)]]
    assert group.stats()["pending_members"] == 1
    release.set()
    await asyncio.wait_for(asyncio.gather(active, survivor), 1)
    assert executions == [[inventory(0)], [inventory(3)]]


@pytest.mark.asyncio
async def test_timed_out_caller_keeps_group_capacity_until_native_operation_finishes() -> (
    None
):
    entered, release = asyncio.Event(), asyncio.Event()
    control = GenerationCutControl(max_workers=1)
    executions = []

    async def execute(items: list[Any], deadline: float | None) -> list[Any]:
        async def operation() -> list[Any]:
            executions.append(items)
            entered.set()
            await release.wait()
            return [receipt(item) for item in items]

        return await control.run(
            checkpoint_id=items[0].checkpoint_id,
            inventory_digest=items[0].inventory_digest,
            call_ids=frozenset(
                prefix.model_call_id
                for item in items
                for prefix in item.active_prefixes
            ),
            deadline=deadline,
            operation=operation,
        )

    group = grouper(execute, max_groups=1, wait=0)
    expiring = asyncio.create_task(
        group.run(inventory(0), deadline=time.monotonic() + 0.04)
    )
    await asyncio.wait_for(entered.wait(), 1)
    pending_inputs = [inventory(i) for i in range(1, 9)]
    pending = [
        asyncio.create_task(group.run(item, deadline=None)) for item in pending_inputs
    ]
    with pytest.raises(TimeoutError):
        await expiring
    assert group.stats()["active_groups"] == len(control._active) == 1
    assert group.stats()["pending_members"] == 8
    assert not control._pending and executions == [[inventory(0)]]
    release.set()
    answers = await asyncio.wait_for(asyncio.gather(*pending), 1)
    assert executions == [[inventory(0)], pending_inputs]
    for original, answer in zip(pending_inputs, answers, strict=True):
        answer.validate_for(original)


@pytest.mark.asyncio
async def test_64_original_inventories_become_eight_disjoint_groups() -> None:
    groups = []

    async def execute(items: list[Any], deadline: float | None) -> list[Any]:
        groups.append(items)
        return [receipt(item) for item in items]

    group = grouper(execute)
    inputs = [inventory(i, calls=32) for i in range(64)]
    outputs = await asyncio.gather(*(group.run(item, deadline=None) for item in inputs))
    assert len(groups) == 8 and all(len(items) == 8 for items in groups)
    assert all(
        sum(len(item.active_prefixes) for item in items) == 256 for items in groups
    )
    for original, answer in zip(inputs, outputs, strict=True):
        answer.validate_for(original)
    await asyncio.sleep(0)
    assert group.stats() == {
        "groups": 8,
        "members": 64,
        "max_members": 8,
        "max_calls": 256,
        "pending_members": 0,
        "active_groups": 0,
    }


@pytest.mark.asyncio
async def test_call_cap_and_oversized_original_stays_alone() -> None:
    sizes = []

    async def execute(items: list[Any], deadline: float | None) -> list[Any]:
        sizes.append([len(item.active_prefixes) for item in items])
        return [receipt(item) for item in items]

    group = grouper(execute, max_calls=3)
    await asyncio.gather(
        *(
            group.run(inventory(i, calls=n), deadline=None)
            for i, n in enumerate([2, 2, 5, 1, 1])
        )
    )
    assert sizes == [[2], [2], [5], [1, 1]]


@pytest.mark.asyncio
@pytest.mark.parametrize("barrier", ["checkpoint", "call", "ticket"])
async def test_barriers_never_get_skipped(barrier: str) -> None:
    inputs = [inventory(0), inventory(1), inventory(2)]
    if barrier == "checkpoint":
        inputs[1] = inventory(1, checkpoint="other")
    else:
        prefix = (
            inputs[1]
            .active_prefixes[0]
            .model_copy(
                update={
                    "model_call_id" if barrier == "call" else "ticket_id": inputs[0]
                    .active_prefixes[0]
                    .model_call_id
                    if barrier == "call"
                    else inputs[0].active_prefixes[0].ticket_id,
                }
            )
        )
        inputs[1] = GenerationCutInventory.build(
            checkpoint_id="checkpoint", server_name="second", active_prefixes=[prefix]
        )
    groups = []

    async def execute(items: list[Any], deadline: float | None) -> list[Any]:
        groups.append(items)
        return [receipt(item) for item in items]

    group = grouper(execute)
    await asyncio.gather(*(group.run(item, deadline=None) for item in inputs))
    assert groups[0] == [inputs[0]]
    assert [item for items in groups for item in items] == inputs
    if barrier == "checkpoint":
        assert groups == [[item] for item in inputs]


@pytest.mark.asyncio
async def test_duplicate_join_keeps_native_work_alive_after_one_waiter_cancels() -> (
    None
):
    entered, release = asyncio.Event(), asyncio.Event()
    executions = 0
    control = GenerationCutControl(max_workers=1)

    async def execute(items: list[Any], deadline: float | None) -> list[Any]:
        async def operation() -> list[Any]:
            nonlocal executions
            executions += 1
            entered.set()
            await release.wait()
            return [receipt(item) for item in items]

        return await control.run(
            checkpoint_id=items[0].checkpoint_id,
            inventory_digest=items[0].inventory_digest,
            call_ids=frozenset(
                prefix.model_call_id
                for item in items
                for prefix in item.active_prefixes
            ),
            deadline=deadline,
            operation=operation,
        )

    group = grouper(execute)
    original = inventory(0)
    first = asyncio.create_task(group.run(original, deadline=None))
    second = asyncio.create_task(group.run(original, deadline=None))
    await asyncio.wait_for(entered.wait(), 1)
    first.cancel()
    with pytest.raises(asyncio.CancelledError):
        await first
    assert len(control._active) == 1 and not second.done()
    release.set()
    (await second).validate_for(original)
    assert executions == 1 and not control._active


@pytest.mark.asyncio
async def test_individual_deadline_does_not_cancel_shared_operation() -> None:
    entered, release = asyncio.Event(), asyncio.Event()
    seen = []

    async def execute(items: list[Any], deadline: float | None) -> list[Any]:
        seen.append(deadline)
        entered.set()
        await release.wait()
        return [receipt(item) for item in items]

    group = grouper(execute)
    early = time.monotonic() + 0.08
    first = asyncio.create_task(group.run(inventory(0), deadline=early))
    second = asyncio.create_task(group.run(inventory(1), deadline=time.monotonic() + 2))
    await asyncio.wait_for(entered.wait(), 1)
    with pytest.raises(TimeoutError):
        await first
    assert seen == [early] and not second.done() and group.stats()["active_groups"] == 1
    release.set()
    (await second).validate_for(inventory(1))


@pytest.mark.asyncio
async def test_expiry_or_cancellation_before_dispatch_does_not_start_work() -> None:
    groups = []

    async def execute(items: list[Any], deadline: float | None) -> list[Any]:
        groups.append(items)
        return [receipt(item) for item in items]

    group = grouper(execute, wait=0.05)
    expiring = asyncio.create_task(
        group.run(inventory(0), deadline=time.monotonic() + 0.005)
    )
    cancelled = asyncio.create_task(group.run(inventory(1), deadline=None))
    survivor = asyncio.create_task(group.run(inventory(2), deadline=None))
    await asyncio.sleep(0)
    cancelled.cancel()
    with pytest.raises(asyncio.CancelledError):
        await cancelled
    with pytest.raises(TimeoutError):
        await expiring
    await survivor
    assert groups == [[inventory(2)]]


@pytest.mark.asyncio
async def test_close_refuses_new_and_pending_but_keeps_claimed_work() -> None:
    entered, release = asyncio.Event(), asyncio.Event()

    async def execute(items: list[Any], deadline: float | None) -> list[Any]:
        entered.set()
        await release.wait()
        return [receipt(item) for item in items]

    group = grouper(execute, max_groups=1, wait=0)
    active = asyncio.create_task(group.run(inventory(0), deadline=None))
    await asyncio.wait_for(entered.wait(), 1)
    pending = asyncio.create_task(group.run(inventory(1), deadline=None))
    await asyncio.sleep(0)
    assert group.stats()["pending_members"] == 1
    group.close()
    with pytest.raises(RuntimeError, match="closed"):
        await pending
    with pytest.raises(RuntimeError, match="closed"):
        await group.run(inventory(2), deadline=None)
    assert not active.done()
    release.set()
    await active


@pytest.mark.asyncio
async def test_failed_group_releases_capacity_for_pending_members() -> None:
    entered, release = asyncio.Event(), asyncio.Event()
    executions = []

    async def execute(items: list[Any], deadline: float | None) -> list[Any]:
        executions.append(items)
        if len(executions) == 1:
            entered.set()
            await release.wait()
            raise RuntimeError("write failed")
        return [receipt(item) for item in items]

    group = grouper(execute, max_groups=1, wait=0)
    failing = asyncio.create_task(group.run(inventory(0), deadline=None))
    await asyncio.wait_for(entered.wait(), 1)
    inputs = [inventory(1), inventory(2)]
    pending = [asyncio.create_task(group.run(item, deadline=None)) for item in inputs]
    await asyncio.sleep(0)
    release.set()
    with pytest.raises(RuntimeError, match="write failed"):
        await failing
    answers = await asyncio.wait_for(asyncio.gather(*pending), 1)
    assert executions == [[inventory(0)], inputs]
    for original, answer in zip(inputs, answers, strict=True):
        answer.validate_for(original)


@pytest.mark.asyncio
async def test_invalid_receipt_fails_all_members_before_publishing_success() -> None:
    async def execute(items: list[Any], deadline: float | None) -> list[Any]:
        return [receipt(items[0]), receipt(items[0])]

    group = grouper(execute)
    answers = await asyncio.gather(
        *(group.run(inventory(i), deadline=None) for i in range(2)),
        return_exceptions=True,
    )
    assert all(isinstance(answer, ValueError) for answer in answers)


@pytest.mark.asyncio
async def test_duplicate_digest_with_changed_content_is_rejected() -> None:
    entered, release = asyncio.Event(), asyncio.Event()

    async def execute(items: list[Any], deadline: float | None) -> list[Any]:
        entered.set()
        await release.wait()
        return [receipt(item) for item in items]

    group = grouper(execute)
    original = inventory(0)
    running = asyncio.create_task(group.run(original, deadline=None))
    await entered.wait()
    with pytest.raises(ValueError, match="different content"):
        await group.run(
            original.model_copy(update={"server_name": "forged"}), deadline=None
        )
    release.set()
    await running


@pytest.mark.asyncio
async def test_real_group_receipt_split_and_original_cache() -> None:
    import threading

    from nemo_rl.models.generation.generation_cut_capture import (
        GenerationCutCaptureMixin,
    )

    class Worker(GenerationCutCaptureMixin):
        def __init__(self) -> None:
            self._capture_registry_lock = threading.Lock()
            self._generation_cut_receipts = {}
            self.admitted = []

        async def _admit_generation_cut_inventory(
            self, item: Any, *, deadline: float | None
        ) -> Any:
            self.admitted.append((item, deadline))
            return receipt(item)

    worker = Worker()
    inputs = [inventory(i, calls=2) for i in range(8)]
    worker._generation_cut_grouper = grouper(worker._checkpoint_generation_cut_group)
    answers = await asyncio.gather(
        *(
            worker._checkpoint_generation_cut_control(item, deadline=None)
            for item in inputs
        )
    )
    assert (
        len(worker.admitted) == 1 and len(worker.admitted[0][0].active_prefixes) == 16
    )
    for item, answer in zip(inputs, answers, strict=True):
        answer.validate_for(item)
        assert (
            await worker._checkpoint_generation_cut_control(item, deadline=None)
            is answer
        )
    assert len(worker.admitted) == 1


@pytest.mark.asyncio
async def test_checkpoint_barrier_uses_existing_controller_and_rejects_retired_epoch() -> (
    None
):
    control = GenerationCutControl(max_workers=4)
    order = []

    async def execute(items: list[Any], deadline: float | None) -> list[Any]:
        async def operation() -> list[Any]:
            order.append(items[0].checkpoint_id)
            await asyncio.sleep(0)
            return [receipt(item) for item in items]

        return await control.run(
            checkpoint_id=items[0].checkpoint_id,
            inventory_digest=items[0].inventory_digest,
            call_ids=frozenset(
                prefix.model_call_id
                for item in items
                for prefix in item.active_prefixes
            ),
            deadline=deadline,
            operation=operation,
        )

    group = grouper(execute)
    inputs = [
        inventory(0, checkpoint="a"),
        inventory(1, checkpoint="b"),
        inventory(2, checkpoint="a"),
    ]
    answers = await asyncio.gather(
        *(group.run(item, deadline=None) for item in inputs), return_exceptions=True
    )
    assert isinstance(answers[-1], RuntimeError)
    assert "superseded" in str(answers[-1])
    assert order == ["a", "b"]
    assert not control._active and not control._pending
