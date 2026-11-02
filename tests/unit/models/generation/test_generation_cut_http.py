# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""CPU coverage of the real cut HTTP handler, admission and Gym requester."""

import ast
import asyncio
import importlib.util
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from fastapi import FastAPI, Header, HTTPException, Request

from nemo_rl.utils.prefix_cut_diagnostics import prefix_cut_count, prefix_cut_probe

pytest.importorskip("nemo_gym._checkpoint.generation_cut")
from nemo_gym._checkpoint.generation_cut import (  # noqa: E402
    GenerationCutInventory,
    GenerationCutPrefix,
    GenerationCutReceipt,
)

pytestmark = pytest.mark.nemo_gym
ROOT = Path(__file__).resolve().parents[4]


@pytest.fixture(scope="module")
def control_module():
    path = ROOT / "nemo_rl/models/generation/generation_cut_control.py"
    spec = importlib.util.spec_from_file_location("_cut_http_control", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    yield module
    sys.modules.pop(spec.name, None)


def inventory():
    return GenerationCutInventory.build(
        checkpoint_id="cp",
        server_name="policy",
        active_prefixes=[
            GenerationCutPrefix(
                ticket_id="t0",
                rollout_id="r0",
                attempt=0,
                model_call_id="c0",
                admitted_at=time.time(),
            )
        ],
    )


def request(headers=None):
    messages = asyncio.Queue()
    raw = Request(
        {
            "type": "http",
            "headers": [(k.encode(), v.encode()) for k, v in (headers or {}).items()],
        },
        messages.get,
    )
    return raw, messages


def route(module, operation, *, timeout_s=120.0):
    """Compile the production handler with its own cut-control imports."""
    path = ROOT / "nemo_rl/models/generation/vllm/vllm_worker_async.py"
    tree = ast.parse(path.read_text())
    handler = next(
        n
        for n in ast.walk(tree)
        if isinstance(n, ast.AsyncFunctionDef) and n.name == "checkpoint_generation_cut"
    )
    handler.decorator_list = []  # Only the FastAPI route-registration decorator.
    core_import = next(
        n
        for n in tree.body
        if isinstance(n, ast.ImportFrom)
        and n.module == "nemo_rl.models.generation.generation_cut_control"
    )
    namespace = {
        alias.asname or alias.name: getattr(module, alias.name)
        for alias in core_import.names
    }
    control = module.GenerationCutControl(max_workers=4)
    deadlines = []

    async def run(inv, *, deadline):
        deadlines.append(deadline)
        return await control.run(
            checkpoint_id=inv.checkpoint_id,
            inventory_digest=inv.inventory_digest,
            call_ids=frozenset(p.model_call_id for p in inv.active_prefixes),
            deadline=deadline,
            operation=operation,
        )

    namespace.update(
        Any=Any,
        worker_self=SimpleNamespace(
            _generation_prefix_cuts_enabled=True,
            _generation_cut_control_token="test-token",
            _generation_cut_control_timeout_s=timeout_s,
            _checkpoint_generation_cut_control=run,
        ),
        Header=Header,
        HTTPException=HTTPException,
        Request=Request,
        prefix_cut_count=prefix_cut_count,
        prefix_cut_probe=prefix_cut_probe,
        telemetry_metrics_exporting=lambda: True,
    )
    future = ast.ImportFrom(
        module="__future__", names=[ast.alias(name="annotations")], level=0
    )
    exec(
        compile(
            ast.fix_missing_locations(
                ast.Module(body=[future, handler], type_ignores=[])
            ),
            str(path),
            "exec",
        ),
        namespace,
    )
    return namespace["checkpoint_generation_cut"], control, deadlines


def receipt():
    return SimpleNamespace(prefixes=(), model_dump=lambda mode: {"ok": True})


@pytest.mark.parametrize(
    "header,status", [("nan", 400), ("inf", 400), ("invalid", 400), ("0", 408)]
)
def test_bad_or_expired_caller_deadline_never_enters_admission(
    control_module, header, status
):
    async def run():
        executed = []

        async def work():
            executed.append(True)
            return receipt()

        handler, control, deadlines = route(control_module, work)
        raw, _ = request({"x-nemo-gym-cut-deadline": header})
        with pytest.raises(HTTPException) as error:
            await handler(
                inventory().model_dump(), raw, authorization="Bearer test-token"
            )
        assert error.value.status_code == status
        assert not executed and not deadlines and not control._jobs

    asyncio.run(run())


@pytest.mark.parametrize(
    "timeout_s,caller_s", [(120.0, 0.5), (0.5, 120.0), (None, 0.5), (0.5, None)]
)
def test_handler_uses_earliest_deadline(control_module, timeout_s, caller_s):
    async def run():
        async def work():
            return receipt()

        handler, control, deadlines = route(control_module, work, timeout_s=timeout_s)
        headers = (
            {}
            if caller_s is None
            else {"x-nemo-gym-cut-deadline": str(time.time() + caller_s)}
        )
        raw, _ = request(headers)
        now = time.monotonic()
        assert await handler(
            inventory().model_dump(), raw, authorization="Bearer test-token"
        ) == {"ok": True}
        expected = min(x for x in [timeout_s, caller_s] if x is not None)
        assert now + expected - 0.05 <= deadlines[0] <= now + expected + 0.01
        assert not control._active

    asyncio.run(run())


@pytest.mark.parametrize("event", ["disconnect", "cancel", "expire"])
def test_queued_http_cut_is_removed_without_starting(control_module, event):
    async def run():
        executed = []

        async def work():
            executed.append(True)
            return receipt()

        handler, control, _ = route(control_module, work)
        release = asyncio.Event()
        entered = [asyncio.Event() for _ in range(4)]

        async def blocker(index):
            entered[index].set()
            await release.wait()

        blockers = [
            asyncio.create_task(
                control.run(
                    checkpoint_id="cp",
                    inventory_digest=f"block-{i}",
                    call_ids=frozenset({f"other-{i}"}),
                    deadline=None,
                    operation=lambda i=i: blocker(i),
                )
            )
            for i in range(4)
        ]
        await asyncio.wait_for(asyncio.gather(*(e.wait() for e in entered)), 2)
        assert len(control._active) == 4
        raw, messages = request(
            {
                "x-nemo-gym-cut-deadline": str(
                    time.time() + (0.05 if event == "expire" else 5)
                )
            }
        )
        waiter = asyncio.create_task(
            handler(inventory().model_dump(), raw, authorization="Bearer test-token")
        )
        for _ in range(10):
            await asyncio.sleep(0)
        assert len(control._pending) == 1 and not executed
        if event == "disconnect":
            await messages.put({"type": "http.disconnect"})
        elif event == "cancel":
            waiter.cancel()
        if event == "expire":
            with pytest.raises(HTTPException) as error:
                await asyncio.wait_for(waiter, 2)
            assert error.value.status_code == 408
        elif event == "cancel":
            with pytest.raises(asyncio.CancelledError):
                await asyncio.wait_for(waiter, 2)
        else:
            assert await asyncio.wait_for(waiter, 2) is None
        for _ in range(5):
            await asyncio.sleep(0)
        assert not control._pending and not executed
        assert len(control._active) == 4
        release.set()
        await asyncio.wait_for(asyncio.gather(*blockers), 2)
        assert not executed

    asyncio.run(run())


def test_disconnect_does_not_release_a_running_thread_reservation(control_module):
    async def run():
        started, release = threading.Event(), threading.Event()
        next_started = asyncio.Event()
        with ThreadPoolExecutor(max_workers=1) as executor:

            def write():
                started.set()
                assert release.wait(5)
                return receipt()

            async def work():
                return await asyncio.get_running_loop().run_in_executor(executor, write)

            handler, control, _ = route(control_module, work)
            raw, messages = request()
            task = asyncio.create_task(
                handler(
                    inventory().model_dump(), raw, authorization="Bearer test-token"
                )
            )
            try:
                assert await asyncio.to_thread(started.wait, 2)
                await messages.put({"type": "http.disconnect"})
                assert await asyncio.wait_for(task, 2) is None
                assert len(control._active) == 1

                async def next_cut():
                    next_started.set()

                next_task = asyncio.create_task(
                    control.run(
                        checkpoint_id="cp",
                        inventory_digest="next",
                        call_ids=frozenset({"c0"}),
                        deadline=None,
                        operation=next_cut,
                    )
                )
                for _ in range(8):
                    await asyncio.sleep(0)
                assert not next_started.is_set()
                release.set()
                await asyncio.wait_for(next_task, 2)
                assert next_started.is_set() and not control._active
            finally:
                release.set()

    asyncio.run(run())


@pytest.mark.parametrize("middleware", [False, True])
def test_asgi_disconnect_cancels_queued_cut_through_http_middleware(
    control_module, middleware
):
    """Exercise FastAPI body parsing and the ASGI receive channel end to end."""

    async def run():
        executed = []

        async def work():
            executed.append(True)
            return receipt()

        handler, control, _ = route(control_module, work)
        app = FastAPI()
        app.post("/cut")(handler)
        if middleware:

            @app.middleware("http")
            async def passthrough(raw, call_next):
                return await call_next(raw)

        release = asyncio.Event()
        entered = [asyncio.Event() for _ in range(4)]

        async def blocker(index):
            entered[index].set()
            await release.wait()

        blockers = [
            asyncio.create_task(
                control.run(
                    checkpoint_id="cp",
                    inventory_digest=f"block-{i}",
                    call_ids=frozenset({f"other-{i}"}),
                    deadline=None,
                    operation=lambda i=i: blocker(i),
                )
            )
            for i in range(4)
        ]
        await asyncio.wait_for(asyncio.gather(*(e.wait() for e in entered)), 2)
        messages = asyncio.Queue()
        await messages.put(
            {
                "type": "http.request",
                "body": inventory().model_dump_json().encode(),
                "more_body": False,
            }
        )
        sent = []

        async def send(message):
            sent.append(message)

        scope = dict(
            type="http",
            asgi={"version": "3.0"},
            http_version="1.1",
            method="POST",
            scheme="http",
            server=("test", 80),
            client=("client", 1234),
            path="/cut",
            raw_path=b"/cut",
            root_path="",
            query_string=b"",
            headers=[
                (b"content-type", b"application/json"),
                (b"authorization", b"Bearer test-token"),
            ],
        )
        task = asyncio.create_task(app(scope, messages.get, send))
        try:
            for _ in range(100):
                if control._pending:
                    break
                await asyncio.sleep(0)
            assert len(control._pending) == 1 and not executed
            await messages.put({"type": "http.disconnect"})
            await asyncio.wait_for(task, 2)
            for _ in range(5):
                await asyncio.sleep(0)
            assert not control._pending and not executed
            assert len(control._active) == 4
        finally:
            release.set()
            await asyncio.wait_for(asyncio.gather(*blockers), 2)
            if not task.done():
                task.cancel()
                await asyncio.gather(task, return_exceptions=True)

    asyncio.run(run())


def test_gym_sends_the_actual_cut_deadline_and_keeps_contexts_separate(monkeypatch):
    gym_model = pytest.importorskip("nemo_gym._checkpoint.model")
    import nemo_gym.server_utils as server_utils
    from nemo_gym._checkpoint.control import CheckpointRequest

    seen = []

    async def send(**kwargs):
        await asyncio.sleep(0)
        inv = GenerationCutInventory.model_validate(kwargs["json"])
        seen.append((inv.checkpoint_id, kwargs["headers"]))
        return GenerationCutReceipt(
            checkpoint_id=inv.checkpoint_id,
            cut_id="test",
            inventory_digest=inv.inventory_digest,
            inventory=inv,
            backend_snapshot_id="test",
            prefixes=(),
        ).model_dump(mode="json")

    async def ok(response):
        return None

    async def data(response):
        return response

    monkeypatch.setattr(server_utils, "request", send)
    monkeypatch.setattr(server_utils, "raise_for_status", ok)
    monkeypatch.setattr(server_utils, "get_response_json", data)

    async def run():
        start = time.time()
        gates = [
            gym_model.PolicyModelParticipant(
                server_name=f"policy-{i}",
                cut_requester=gym_model.generation_cut_requester("test-token"),
            ).gate
            for i in range(2)
        ]
        await asyncio.gather(
            *(
                gate._cut(
                    CheckpointRequest(
                        checkpoint_id=f"cp-{i}", deadline_ts=start + remaining
                    ),
                    "http://worker/v1",
                    [],
                )
                for i, (gate, remaining) in enumerate(zip(gates, [2.0, 4.0]))
            )
        )
        assert gym_model._CURRENT_CUT_DEADLINE.get() is None
        bounds = {
            cid: float(headers["x-nemo-gym-cut-deadline"]) for cid, headers in seen
        }
        assert start + 0.9 <= bounds["cp-0"] <= start + 1.1
        assert start + 1.9 <= bounds["cp-1"] <= start + 2.1
        await gym_model.generation_cut_requester("test-token")(
            "http://worker/v1",
            GenerationCutInventory.build(
                checkpoint_id="outside", server_name="policy", active_prefixes=[]
            ),
        )
        assert "x-nemo-gym-cut-deadline" not in seen[-1][1]

    asyncio.run(run())
