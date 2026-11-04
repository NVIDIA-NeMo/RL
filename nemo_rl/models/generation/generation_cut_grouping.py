# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may obtain a copy at http://www.apache.org/licenses/LICENSE-2.0
"""Combine pending generation-cut inventories before reserving executor slots."""

from __future__ import annotations

import asyncio
import functools
import json
import logging
import math
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any

from nemo_rl.utils.prefix_cut_diagnostics import current_prefix_cut_probe

LOGGER = logging.getLogger(__name__)


@dataclass(eq=False)
class _Member:
    inventory: Any
    deadline: float | None
    result: asyncio.Future[Any]
    calls: frozenset[str]
    tickets: frozenset[str]
    waiters: int
    claimed: bool
    enqueued_at: float
    enqueued_unix_s: float
    diagnostics: bool


class GenerationCutGrouper:
    """HTTP-loop grouping; the existing admission controller owns reservations.

    A FIFO prefix is combined only across the same checkpoint and disjoint call
    and ticket identities. Barriers are never skipped. The earliest member
    deadline bounds the shared operation; every waiter retains its own timeout.
    Once claimed, work survives caller cancellation through native completion.
    ``execute`` must validate/split/cache original receipts before returning.
    At most ``max_groups`` operations are dispatched. Further arrivals remain
    unclaimed so they can combine when an operation actually finishes.
    """

    def __init__(
        self,
        *,
        max_groups: int,
        max_inventories: int,
        max_calls: int,
        wait_seconds: float,
        max_payload_bytes: int,
        execute: Callable[[list[Any], float | None], Awaitable[list[Any]]],
    ) -> None:
        for value in (max_groups, max_inventories, max_calls, max_payload_bytes):
            if type(value) is not int or value < 1:
                raise ValueError(
                    "Generation-cut group limits must be positive integers"
                )
        if not math.isfinite(wait_seconds) or wait_seconds < 0:
            raise ValueError("Generation-cut group wait must be finite and nonnegative")
        self._max_groups = max_groups
        self._max_inventories = max_inventories
        self._max_calls = max_calls
        self.max_payload_bytes = max_payload_bytes
        self._wait = wait_seconds
        self._execute = execute
        self._loop: asyncio.AbstractEventLoop | None = None
        self._jobs: dict[tuple[str, str], _Member] = {}
        self._pending: list[_Member] = []
        self._tasks: set[asyncio.Task[None]] = set()
        self._timer: asyncio.TimerHandle | None = None
        self._closed = False
        self._counts = {"groups": 0, "members": 0, "max_members": 0, "max_calls": 0}

    @staticmethod
    def _consume_result(future: asyncio.Future[Any]) -> None:
        if not future.cancelled():
            future.exception()

    async def run(self, inventory: Any, *, deadline: float | None) -> Any:
        loop = asyncio.get_running_loop()
        if self._loop is None:
            self._loop = loop
        elif self._loop is not loop:
            raise RuntimeError("Generation-cut grouping belongs to another HTTP loop")
        if self._closed:
            raise RuntimeError("Generation-cut grouping is closed")
        if deadline is not None and deadline <= time.monotonic():
            raise TimeoutError("Generation-cut group member deadline expired")
        prefixes = inventory.active_prefixes
        calls = frozenset(prefix.model_call_id for prefix in prefixes)
        tickets = frozenset(prefix.ticket_id for prefix in prefixes)
        if len(calls) != len(prefixes) or len(tickets) != len(prefixes):
            raise ValueError(
                "Generation-cut inventory has duplicate call/ticket identities"
            )
        key = (inventory.checkpoint_id, inventory.inventory_digest)
        member = self._jobs.get(key)
        if member is None:
            result = loop.create_future()
            result.add_done_callback(self._consume_result)
            member = _Member(
                inventory,
                deadline,
                result,
                calls,
                tickets,
                0,
                False,
                time.monotonic(),
                time.time(),
                current_prefix_cut_probe() is not None,
            )
            self._jobs[key] = member
            self._pending.append(member)
            if member.diagnostics:
                LOGGER.info(
                    "generation_cut_group %s",
                    json.dumps(
                        {
                            "event": "enqueue",
                            "unix_s": member.enqueued_unix_s,
                            "checkpoint_id": inventory.checkpoint_id,
                            "inventory_digest": inventory.inventory_digest,
                            "calls": len(calls),
                            "active_groups": len(self._tasks),
                            "pending_inventories": len(self._pending),
                        },
                        sort_keys=True,
                    ),
                )
        elif member.inventory != inventory:
            raise ValueError("Duplicate inventory digest has different content")
        member.waiters += 1
        if (
            self._timer is None
            and self._pending
            and len(self._tasks) < self._max_groups
        ):
            # Never run work inline: adjacent HTTP tasks get a bounded window
            # to register before groups enter the admission queue.
            self._timer = loop.call_later(self._wait, functools.partial(self._dispatch))
        try:
            async with asyncio.timeout_at(deadline):
                return await asyncio.shield(member.result)
        finally:
            member.waiters -= 1
            if member.waiters == 0 and not member.claimed and not member.result.done():
                self._pending.remove(member)
                self._jobs.pop(key, None)
                member.result.cancel()

    def _dispatch(self) -> None:
        if self._timer is not None:
            self._timer.cancel()
        self._timer = None
        while (
            self._pending and not self._closed and len(self._tasks) < self._max_groups
        ):
            group: list[_Member] = []
            calls: set[str] = set()
            tickets: set[str] = set()
            while self._pending and len(group) < self._max_inventories:
                candidate = self._pending[0]
                if (
                    candidate.deadline is not None
                    and candidate.deadline <= time.monotonic()
                ):
                    self._pending.pop(0)
                    self._jobs.pop(
                        (
                            candidate.inventory.checkpoint_id,
                            candidate.inventory.inventory_digest,
                        ),
                        None,
                    )
                    candidate.result.set_exception(
                        TimeoutError(
                            "Generation-cut group member expired before dispatch"
                        )
                    )
                    continue
                if group and (
                    candidate.inventory.checkpoint_id
                    != group[0].inventory.checkpoint_id
                    or calls & candidate.calls
                    or tickets & candidate.tickets
                    or len(calls) + len(candidate.calls) > self._max_calls
                ):
                    break
                # A pre-existing oversized single inventory runs alone; grouping
                # never enlarges it. Backend row/token/byte limits still apply.
                self._pending.pop(0)
                candidate.claimed = True
                group.append(candidate)
                calls.update(candidate.calls)
                tickets.update(candidate.tickets)
                if len(calls) >= self._max_calls:
                    break
            if not group:
                continue
            self._counts["groups"] += 1
            self._counts["members"] += len(group)
            self._counts["max_members"] = max(self._counts["max_members"], len(group))
            self._counts["max_calls"] = max(self._counts["max_calls"], len(calls))
            if any(member.diagnostics for member in group):
                now = time.monotonic()
                LOGGER.info(
                    "generation_cut_group %s",
                    json.dumps(
                        {
                            "event": "dispatch",
                            "unix_s": time.time(),
                            "checkpoint_id": group[0].inventory.checkpoint_id,
                            "active_groups": len(self._tasks) + 1,
                            "pending_inventories": len(self._pending),
                            "group_calls": len(calls),
                            "members": [
                                {
                                    "inventory_digest": member.inventory.inventory_digest,
                                    "calls": len(member.calls),
                                    "enqueued_unix_s": member.enqueued_unix_s,
                                    "pending_wait_s": now - member.enqueued_at,
                                }
                                for member in group
                            ],
                        },
                        sort_keys=True,
                    ),
                )
            task = asyncio.create_task(self._run_group(group))
            self._tasks.add(task)
            task.add_done_callback(functools.partial(self._group_finished))

    def _group_finished(self, task: asyncio.Task[None]) -> None:
        self._tasks.discard(task)
        # Only actual operation completion frees capacity. Caller timeouts and
        # cancellation leave claimed work alive. Pending arrivals have already
        # collected while the slots were busy; do not add another timer window.
        self._dispatch()

    async def _run_group(self, group: list[_Member]) -> None:
        deadlines = [member.deadline for member in group if member.deadline is not None]
        deadline = min(deadlines) if deadlines else None
        try:
            receipts = await self._execute(
                [member.inventory for member in group], deadline
            )
            if len(receipts) != len(group):
                raise RuntimeError("Grouped cut returned the wrong number of receipts")
            # Validate all answers before publishing any successful result.
            for member, receipt in zip(group, receipts, strict=True):
                receipt.validate_for(member.inventory)
            for member, receipt in zip(group, receipts, strict=True):
                if not member.result.done():
                    member.result.set_result(receipt)
        except asyncio.CancelledError:
            for member in group:
                if not member.result.done():
                    member.result.cancel()
            raise
        except Exception as error:
            for member in group:
                if not member.result.done():
                    member.result.set_exception(error)
        finally:
            for member in group:
                self._jobs.pop(
                    (member.inventory.checkpoint_id, member.inventory.inventory_digest),
                    None,
                )

    def stats(self) -> dict[str, int]:
        """Read on the HTTP loop, or while it is quiescent at a status boundary."""
        return dict(
            self._counts,
            pending_members=len(self._pending),
            active_groups=len(self._tasks),
        )

    def close(self) -> None:
        """Refuse new work; active groups keep their underlying reservations."""
        self._closed = True
        if self._loop is not None and not self._loop.is_closed():
            self._loop.call_soon_threadsafe(functools.partial(self._close_pending))

    def _close_pending(self) -> None:
        if self._timer is not None:
            self._timer.cancel()
            self._timer = None
        for member in self._pending:
            self._jobs.pop(
                (member.inventory.checkpoint_id, member.inventory.inventory_digest),
                None,
            )
            if not member.result.done():
                member.result.set_exception(
                    RuntimeError("Generation-cut grouping is closed")
                )
        self._pending.clear()
