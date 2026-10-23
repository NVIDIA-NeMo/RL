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

"""CPU-only admission regressions; no Ray actor or vLLM process is started."""

import ast
import asyncio
import importlib.util
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Annotated

import pytest
from pydantic import BaseModel, Field, ValidationError


@pytest.fixture(scope="module")
def implementation():
    # The generation package initializer imports optional GPU dependencies.
    # Load this stdlib-only implementation directly instead of stubbing it.
    path = (
        Path(__file__).resolve().parents[4]
        / "nemo_rl/models/generation/generation_cut_control.py"
    )
    spec = importlib.util.spec_from_file_location("_cut_control_test", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    yield module.GenerationCutControl
    sys.modules.pop(spec.name, None)


def submit(control, key, calls, operation, *, checkpoint="cp", deadline=None):
    return asyncio.create_task(
        control.run(
            checkpoint_id=checkpoint,
            inventory_digest=key,
            call_ids=frozenset(calls),
            deadline=deadline,
            operation=operation,
        )
    )


async def turns():
    for _ in range(4):
        await asyncio.sleep(0)


@pytest.mark.parametrize("workers", [1, 2])
def test_independent_inventories_respect_worker_limit(implementation, workers):
    async def scenario():
        control = implementation(max_workers=workers)
        entered_a, entered_b, release = (asyncio.Event() for _ in range(3))

        async def a():
            entered_a.set()
            await release.wait()
            return "a"

        async def b():
            entered_b.set()
            return "b"

        first = submit(control, "a", {"x"}, a)
        await asyncio.wait_for(entered_a.wait(), 2)
        second = submit(control, "b", {"y"}, b)
        await turns()
        assert entered_b.is_set() == (workers == 2)
        release.set()
        assert await asyncio.wait_for(asyncio.gather(first, second), 2) == ["a", "b"]
        control.close()

    asyncio.run(scenario())


def test_overlapping_lists_serialize_but_independent_work_can_pass(implementation):
    async def scenario():
        control = implementation(max_workers=2)
        a_started, b_started, c_started, release = (asyncio.Event() for _ in range(4))

        async def a():
            a_started.set()
            await release.wait()

        async def b():
            b_started.set()

        async def c():
            c_started.set()

        first = submit(control, "a", ["x", "y"], a)
        await asyncio.wait_for(a_started.wait(), 2)
        # Opposite call order would allow a two-lock deadlock without admission.
        overlap = submit(control, "b", ["y", "x"], b)
        independent = submit(control, "c", ["z"], c)
        await asyncio.wait_for(c_started.wait(), 2)
        assert not b_started.is_set()
        release.set()
        await asyncio.wait_for(asyncio.gather(first, overlap, independent), 2)
        assert b_started.is_set()
        control.close()

    asyncio.run(scenario())


def test_checkpoint_barrier_prevents_overtaking(implementation):
    async def scenario():
        control = implementation(max_workers=2)
        a_started, b_started, late_started, release_a, release_b = (
            asyncio.Event() for _ in range(5)
        )

        async def a():
            a_started.set()
            await release_a.wait()

        async def b():
            b_started.set()
            await release_b.wait()

        async def late():
            late_started.set()

        first = submit(control, "a", {"x"}, a, checkpoint="old")
        await asyncio.wait_for(a_started.wait(), 2)
        next_epoch = submit(control, "b", {"y"}, b, checkpoint="new")
        late_old = submit(control, "late", {"z"}, late, checkpoint="old")
        await turns()
        assert not b_started.is_set() and not late_started.is_set()
        release_a.set()
        await asyncio.wait_for(b_started.wait(), 2)
        assert not late_started.is_set()
        release_b.set()
        await asyncio.wait_for(asyncio.gather(first, next_epoch), 2)
        with pytest.raises(RuntimeError, match="superseded"):
            await late_old
        assert not late_started.is_set()
        control.close()

    asyncio.run(scenario())


def test_duplicate_shares_work_after_original_caller_cancels(implementation):
    async def scenario():
        control = implementation(max_workers=2)
        entered, release = asyncio.Event(), asyncio.Event()
        executions = 0
        receipt = object()

        async def operation():
            nonlocal executions
            executions += 1
            entered.set()
            await release.wait()
            return receipt

        original = submit(control, "same", {"x"}, operation)
        await asyncio.wait_for(entered.wait(), 2)
        retry = submit(control, "same", {"x"}, operation)
        await turns()
        original.cancel()
        with pytest.raises(asyncio.CancelledError):
            await original
        assert executions == 1 and not retry.done()
        release.set()
        assert await asyncio.wait_for(retry, 2) is receipt
        assert executions == 1
        control.close()

    asyncio.run(scenario())


def test_caller_cancellation_keeps_actual_thread_reserved(implementation):
    async def scenario():
        control = implementation(max_workers=2)
        entered, release = threading.Event(), threading.Event()
        next_started = asyncio.Event()
        with ThreadPoolExecutor(max_workers=2) as executor:

            def blocked_write():
                entered.set()
                assert release.wait(5)
                return "durable"

            async def write():
                return await asyncio.get_running_loop().run_in_executor(
                    executor, blocked_write
                )

            async def next_cut():
                next_started.set()

            original = submit(control, "a", {"x"}, write, checkpoint="old")
            try:
                assert await asyncio.to_thread(entered.wait, 2)
                original.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await original
                following = submit(control, "b", {"y"}, next_cut, checkpoint="new")
                await turns()
                assert not next_started.is_set()
                retry = submit(control, "a", {"x"}, write, checkpoint="old")
                await turns()
                assert not retry.done()  # No acknowledgement before the write.
            finally:
                release.set()
            assert await asyncio.wait_for(retry, 2) == "durable"
            await asyncio.wait_for(following, 2)
        control.close()

    asyncio.run(scenario())


def test_cancelled_pending_inventory_is_never_executed(implementation):
    async def scenario():
        control = implementation(max_workers=1)
        entered, release = asyncio.Event(), asyncio.Event()
        executions = []

        async def first():
            entered.set()
            await release.wait()

        async def cancelled():
            executions.append("cancelled")

        active = submit(control, "active", {"x"}, first)
        await asyncio.wait_for(entered.wait(), 2)
        pending = submit(control, "cancelled", {"y"}, cancelled)
        await turns()
        pending.cancel()
        with pytest.raises(asyncio.CancelledError):
            await pending
        release.set()
        await asyncio.wait_for(active, 2)
        await turns()
        assert executions == []
        assert not control._pending and not control._jobs
        control.close()

    asyncio.run(scenario())


def test_retry_of_cancelled_pending_inventory_can_start(implementation):
    async def scenario():
        control = implementation(max_workers=1)
        entered, release = asyncio.Event(), asyncio.Event()
        executions = []

        async def blocker():
            entered.set()
            await release.wait()

        async def operation():
            executions.append("retry")
            return 42

        active = submit(control, "active", {"x"}, blocker)
        await asyncio.wait_for(entered.wait(), 2)
        original = submit(control, "same", {"y"}, operation)
        await turns()
        original.cancel()
        with pytest.raises(asyncio.CancelledError):
            await original
        retry = submit(control, "same", {"y"}, operation)
        release.set()
        assert await asyncio.wait_for(retry, 2) == 42
        await active
        assert executions == ["retry"]
        control.close()

    asyncio.run(scenario())


@pytest.mark.parametrize("occupied", [False, True])
def test_expired_inventory_does_not_start(implementation, occupied):
    async def scenario():
        control = implementation(max_workers=1)
        entered, release = asyncio.Event(), asyncio.Event()
        executions = []

        async def blocker():
            entered.set()
            await release.wait()

        async def expired():
            executions.append("expired")

        active = None
        if occupied:
            active = submit(control, "blocker", {"x"}, blocker)
            await asyncio.wait_for(entered.wait(), 2)
        pending = submit(
            control,
            "expired",
            {"y"},
            expired,
            deadline=time.monotonic() + (0.02 if occupied else -1),
        )
        with pytest.raises(TimeoutError, match="before admission"):
            await asyncio.wait_for(pending, 2)
        release.set()
        if active is not None:
            await asyncio.wait_for(active, 2)
        assert executions == []
        control.close()

    asyncio.run(scenario())


def test_failure_releases_reservations_for_retry_and_next_checkpoint(implementation):
    async def scenario():
        control = implementation(max_workers=2)
        entered, release = asyncio.Event(), asyncio.Event()

        async def failed():
            entered.set()
            await release.wait()
            raise ValueError("write failed")

        async def healthy():
            return 42

        first = submit(control, "same", {"x"}, failed)
        await asyncio.wait_for(entered.wait(), 2)
        duplicate = submit(control, "same", {"x"}, failed)
        await turns()
        release.set()
        results = await asyncio.gather(first, duplicate, return_exceptions=True)
        assert all(isinstance(result, ValueError) for result in results)
        assert await submit(control, "same", {"x"}, healthy) == 42
        assert await submit(control, "new", {"x"}, healthy, checkpoint="new") == 42
        control.close()

    asyncio.run(scenario())


def test_shutdown_from_other_thread_cancels_pending_not_running(implementation):
    async def scenario():
        control = implementation(max_workers=1)
        entered, release = asyncio.Event(), asyncio.Event()

        async def operation():
            entered.set()
            await release.wait()
            return 42

        active = submit(control, "active", {"x"}, operation)
        await asyncio.wait_for(entered.wait(), 2)
        pending = submit(control, "pending", {"y"}, operation)
        await turns()
        await asyncio.to_thread(control.close)
        with pytest.raises(asyncio.CancelledError):
            await pending
        with pytest.raises(RuntimeError, match="closed"):
            await submit(control, "refused", {"z"}, operation)
        assert not active.done()
        release.set()
        assert await asyncio.wait_for(active, 2) == 42
        control.close()

    asyncio.run(scenario())


def test_duplicate_identity_mismatch_is_rejected(implementation):
    async def scenario():
        control = implementation(max_workers=2)
        entered, release = asyncio.Event(), asyncio.Event()

        async def operation():
            entered.set()
            await release.wait()

        active = submit(control, "same", {"x"}, operation)
        await asyncio.wait_for(entered.wait(), 2)
        with pytest.raises(ValueError, match="different model-call"):
            await submit(control, "same", {"y"}, operation)
        release.set()
        await active
        control.close()

    asyncio.run(scenario())


@pytest.mark.parametrize("value", [0, 3, -1, True, 1.5, "2"])
def test_control_config_rejects_invalid_worker_counts(implementation, value):
    path = (
        Path(__file__).resolve().parents[4] / "nemo_rl/models/generation/vllm/config.py"
    )
    tree = ast.parse(path.read_text())
    model = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "GenerationCutControlConfig"
    )
    namespace = {"BaseModel": BaseModel, "Field": Field, "Annotated": Annotated}
    exec(
        compile(ast.Module(body=[model], type_ignores=[]), str(path), "exec"), namespace
    )
    config = namespace["GenerationCutControlConfig"]
    assert config().max_workers == 1
    assert config(max_workers=2).max_workers == 2
    with pytest.raises(ValidationError):
        config(max_workers=value)
