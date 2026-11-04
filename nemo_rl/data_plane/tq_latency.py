# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
"""Opt-in phase probes for pinned TQ SimpleStorage and Mooncake clients.

Instance-local wrappers preserve native calls, retries and exceptions. Verified
source seams preserve context through executors and expose native RPC phases;
we never replace an executor, storage implementation, socket or data buffer.
Controller RPC timings include client scheduling/network/server time. Native
upsert timings include the backend's retries: neither is pure server time.
"""

from __future__ import annotations

import ast
import asyncio
import contextvars
import functools
import inspect
import threading
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from textwrap import dedent
from types import MethodType
from typing import Any

from nemo_rl.utils.checkpoint_latency import (
    G_GATE_WORK,
    CheckpointLatencyDiagnostics,
    emit_latency,
)
from nemo_rl.utils.prefix_cut_diagnostics import current_prefix_cut_probe


@dataclass
class _PutTrace:
    max_events: int
    started_unix_s: float
    events: list[dict[str, Any]] = field(default_factory=list)
    dropped: int = 0
    lock: Any = field(default_factory=threading.Lock)

    @contextmanager
    def phase(
        self,
        name: str,
        *,
        cpu: bool = True,
        storage_unit: str | None = None,
        controller_request: str | None = None,
    ) -> Iterator[None]:
        started, wall, thread_cpu = time.time(), time.perf_counter(), time.thread_time()
        failed = True
        try:
            yield
            failed = False
        finally:
            event = {
                "phase": name,
                "storage_unit": storage_unit,
                "controller_request": controller_request,
                "started_unix_s": started,
                "finished_unix_s": time.time(),
                "wall_s": time.perf_counter() - wall,
                # An async interval can include other tasks' CPU; don't attribute it.
                "thread_cpu_s": time.thread_time() - thread_cpu if cpu else None,
                "thread_id": threading.get_ident(),
                "native_thread_id": threading.get_native_id(),
                "failed": failed,
            }
            with self.lock:
                if len(self.events) < self.max_events:
                    self.events.append(event)
                else:
                    self.dropped += 1


G_TQ_TRACE: contextvars.ContextVar[_PutTrace | None] = contextvars.ContextVar(
    "tq_latency_trace", default=None
)
G_TQ_HOOKS: TransferQueueLatencyHooks | None = None


@contextmanager
def _phase(
    name: str,
    *,
    cpu: bool = True,
    storage_unit: str | None = None,
    controller_request: str | None = None,
) -> Iterator[None]:
    trace = G_TQ_TRACE.get()
    if trace is None:
        yield
    else:
        with trace.phase(
            name,
            cpu=cpu,
            storage_unit=storage_unit,
            controller_request=controller_request,
        ):
            yield


def _context_call(function: Callable[..., Any]) -> Callable[..., Any]:
    """Copy at executor submission, not when its thread eventually starts."""
    context = contextvars.copy_context()
    submitted = time.time()

    def run(*args: Any, **kwargs: Any) -> Any:
        def invoke() -> Any:
            trace = G_TQ_TRACE.get()
            if trace is not None:
                with trace.lock:
                    if len(trace.events) < trace.max_events:
                        trace.events.append(
                            {
                                "phase": "executor_dispatch_wait",
                                "started_unix_s": submitted,
                                "finished_unix_s": time.time(),
                                "wall_s": time.time() - submitted,
                                "thread_cpu_s": None,
                                "thread_id": threading.get_ident(),
                                "native_thread_id": threading.get_native_id(),
                                "failed": False,
                            }
                        )
                    else:
                        trace.dropped += 1
            return function(*args, **kwargs)

        return context.run(invoke)

    return run


def instrument_native_function(
    function: Callable[..., Any], seam: str
) -> Callable[..., Any]:
    """Validate exact executor/serialization call shapes before compiling a probe."""
    native = inspect.unwrap(function)
    tree = ast.parse(dedent(inspect.getsource(native)))

    class Instrument(ast.NodeTransformer):
        def __init__(self) -> None:
            self.changed = 0

        def timed(self, node: ast.stmt, name: str, *, cpu: bool = True) -> ast.With:
            self.changed += 1
            keywords = [ast.keyword("cpu", ast.Constant(cpu))]
            if seam == "simple_put":
                keywords.append(
                    ast.keyword(
                        "storage_unit", ast.Name("target_storage_unit", ast.Load())
                    )
                )
            if seam == "controller":
                keywords.append(
                    ast.keyword(
                        "controller_request",
                        ast.Call(
                            ast.Name("str", ast.Load()),
                            [ast.Name("request_type", ast.Load())],
                            [],
                        ),
                    )
                )
            return ast.With(
                items=[
                    ast.withitem(
                        ast.Call(
                            ast.Name("_nrl_tq_phase", ast.Load()),
                            [ast.Constant(name)],
                            keywords,
                        )
                    )
                ],
                body=[node],
            )

        def visit_Call(self, node: ast.Call) -> ast.AST:
            self.generic_visit(node)
            if (
                seam == "manager"
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "run_in_executor"
            ):
                if (
                    len(node.args) != 4
                    or ast.unparse(node.args[1]) != "self.storage_client.put"
                ):
                    raise ValueError("Unsupported TQ storage executor call shape")
                node.args[1] = ast.Call(
                    ast.Name("_nrl_context_call", ast.Load()), [node.args[1]], []
                )
                self.changed += 1
            elif (
                seam == "put"
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "submit"
            ):
                if not node.args or ast.unparse(node.args[0]) not in {
                    "self._put_tensors_thread_worker",
                    "self._put_bytes_thread_worker",
                }:
                    raise ValueError("Unsupported Mooncake put executor call shape")
                node.args[0] = ast.Call(
                    ast.Name("_nrl_context_call", ast.Load()), [node.args[0]], []
                )
                self.changed += 1
            return node

        def visit_Assign(self, node: ast.Assign) -> ast.AST:
            self.generic_visit(node)
            if seam == "controller":
                if isinstance(node.value, ast.Call):
                    call = ast.unparse(node.value.func)
                    if call in {"ZMQMessage.create", "ZMQMessage.deserialize"}:
                        return self.timed(
                            node,
                            "controller_build"
                            if call == "ZMQMessage.create"
                            else "controller_deserialize",
                        )
                if (
                    isinstance(node.value, ast.Await)
                    and isinstance(node.value.value, ast.Call)
                    and ast.unparse(node.value.value.func) == "socket.recv_multipart"
                ):
                    return self.timed(node, "controller_response_wait", cpu=False)
            if seam == "simple_manager":
                if isinstance(node.value, ast.Call):
                    call = ast.unparse(node.value.func)
                    if call in {"extract_field_schema", "self._group_by_hash"}:
                        return self.timed(
                            node,
                            "storage_schema"
                            if call == "extract_field_schema"
                            else "storage_routing",
                        )
                if isinstance(node.value, ast.DictComp) and any(
                    ast.unparse(t) == "storage_data" for t in node.targets
                ):
                    return self.timed(node, "storage_slice")
            if seam == "simple_put":
                if isinstance(node.value, ast.Call):
                    call = ast.unparse(node.value.func)
                    if call in {"request_msg.serialize", "ZMQMessage.deserialize"}:
                        return self.timed(
                            node,
                            "storage_serialize"
                            if call == "request_msg.serialize"
                            else "storage_deserialize",
                        )
                if (
                    isinstance(node.value, ast.Await)
                    and isinstance(node.value.value, ast.Call)
                    and ast.unparse(node.value.value.func) == "socket.recv_multipart"
                ):
                    return self.timed(node, "storage_response_wait", cpu=False)
            if (
                seam == "bytes"
                and isinstance(node.value, ast.Call)
                and ast.unparse(node.value.func) == "serial_utils.batch_encode_into"
            ):
                self.changed += 1
                return ast.With(
                    items=[
                        ast.withitem(
                            ast.Call(
                                ast.Name("_nrl_tq_phase", ast.Load()),
                                [ast.Constant("serialization")],
                                [],
                            )
                        )
                    ],
                    body=[node],
                )
            return node

        def visit_Expr(self, node: ast.Expr) -> ast.AST | list[ast.stmt]:
            self.generic_visit(node)
            if (
                seam == "controller"
                and isinstance(node.value, ast.Await)
                and isinstance(node.value.value, ast.Call)
                and ast.unparse(node.value.value.func) == "socket.send_multipart"
            ):
                call = node.value.value
                if (
                    len(call.args) != 1
                    or ast.unparse(call.args[0]) != "request_msg.serialize()"
                ):
                    raise ValueError("Unsupported controller send/serialize seam")
                serialize = ast.parse("_nrl_controller_frames = None").body[0]
                assert isinstance(serialize, ast.Assign)
                serialize.value = call.args[0]
                call.args[0] = ast.Name("_nrl_controller_frames", ast.Load())
                return [
                    self.timed(serialize, "controller_serialize"),
                    self.timed(node, "controller_send", cpu=False),
                ]
            if (
                seam == "simple_put"
                and isinstance(node.value, ast.Await)
                and isinstance(node.value.value, ast.Call)
                and ast.unparse(node.value.value.func) == "socket.send_multipart"
            ):
                return self.timed(node, "storage_send", cpu=False)
            return node

    transformer = Instrument()
    tree = transformer.visit(tree)
    changed = transformer.changed
    if (
        changed
        != {
            "manager": 1,
            "put": 2,
            "bytes": 1,
            "simple_manager": 3,
            "simple_put": 4,
            "controller": 5,
        }[seam]
    ):
        raise ValueError(f"Unsupported pinned TQ {seam} seam: found {changed} sites")
    definition = tree.body[0]
    if seam == "simple_put":
        if [ast.unparse(d) for d in definition.decorator_list] != [
            "with_storage_unit_socket"
        ]:
            raise ValueError("Unsupported SimpleStorage socket decorator")
    else:
        definition.decorator_list = []
    ast.increment_lineno(tree, inspect.getsourcelines(native)[1] - 1)
    namespace: dict[str, Any] = {
        **native.__globals__,
        "_nrl_context_call": _context_call,
        "_nrl_tq_phase": _phase,
    }
    exec(
        compile(
            ast.fix_missing_locations(tree),
            inspect.getsourcefile(function) or "<tq-probe>",
            "exec",
        ),
        namespace,
    )
    return functools.update_wrapper(namespace[function.__name__], function)


class TransferQueueLatencyHooks:
    """Own reversible, instance-local probes; install before serving requests."""

    def __init__(
        self,
        client: Any,
        *,
        max_events: int,
        slow_call_s: float,
        diagnostics: CheckpointLatencyDiagnostics | None = None,
    ) -> None:
        self.max_events = max_events
        self.slow_call_s = slow_call_s
        self.diagnostics = diagnostics
        self._patches: list[tuple[Any, str, bool, Any, Any]] = []
        self._closed = False
        self._monitor_task: asyncio.Task[None] | None = None
        self._client_loop: asyncio.AbstractEventLoop | None = None
        manager = client.storage_manager
        # Compile and validate every private seam before changing a live object.
        methods = [
            (client, "async_kv_retrieve_meta", "metadata_lookup"),
            (client, "async_put", "put_async"),
            (client, "async_set_custom_meta", "custom_metadata_ack"),
            (client, "_request_controller", "controller_rpc"),
            (manager, "put_data", "storage_put"),
            (manager, "notify_data_update", "data_ready_notify"),
            (manager, "_notify_and_wait", "data_ready_ack"),
        ]
        if callable(getattr(manager, "_put_to_single_storage_unit", None)):
            replacements = [
                (
                    manager,
                    "put_data",
                    instrument_native_function(
                        type(manager).put_data, "simple_manager"
                    ),
                ),
                (
                    manager,
                    "_put_to_single_storage_unit",
                    instrument_native_function(
                        type(manager)._put_to_single_storage_unit, "simple_put"
                    ),
                ),
            ]
            methods.extend(
                [
                    (manager, "_request_with_retry", "storage_rpc_with_retries"),
                    (
                        manager,
                        "_put_to_single_storage_unit",
                        "storage_unit_rpc_including_socket_lease",
                    ),
                ]
            )
        elif getattr(manager, "storage_client", None) is not None:
            storage = manager.storage_client
            replacements = [
                (
                    manager,
                    "put_data",
                    instrument_native_function(type(manager).put_data, "manager"),
                ),
                (storage, "put", instrument_native_function(type(storage).put, "put")),
                (
                    storage,
                    "_put_bytes_thread_worker",
                    instrument_native_function(
                        type(storage)._put_bytes_thread_worker, "bytes"
                    ),
                ),
            ]
            methods.extend(
                [
                    (storage, "put", "backend_put"),
                    (storage, "_preprocess_tensors_for_put", "tensor_preprocess"),
                    (storage, "_register_all_buffers", "buffer_register"),
                    (
                        storage,
                        "_batch_upsert_with_retry",
                        "backend_upsert_with_retries",
                    ),
                    (storage, "_unregister_all_buffers", "buffer_unregister"),
                ]
            )
        else:
            raise ValueError(
                f"Unsupported TQ latency backend: {type(manager).__qualname__}"
            )
        replacements.append(
            (
                client,
                "_request_controller",
                instrument_native_function(
                    type(client)._request_controller, "controller"
                ),
            )
        )
        for obj, name, _ in methods:
            if not callable(getattr(obj, name, None)):
                raise ValueError(f"Unsupported pinned TQ diagnostic method: {name}")
        try:
            for obj, name, function in replacements:
                self._replace(obj, name, MethodType(function, obj))
            for obj, name, phase in methods:
                original = getattr(obj, name)
                self._replace(obj, name, self._wrap(original, phase))
            original_run = client._run_coroutine

            def run_coroutine(coro: Any) -> Any:
                submitted = time.time()

                async def observed() -> Any:
                    trace = G_TQ_TRACE.get()
                    if trace is not None:
                        with trace.lock:
                            if len(trace.events) < trace.max_events:
                                trace.events.append(
                                    {
                                        "phase": "client_loop_dispatch_wait",
                                        "started_unix_s": submitted,
                                        "finished_unix_s": time.time(),
                                        "wall_s": time.time() - submitted,
                                        "thread_cpu_s": None,
                                        "thread_id": threading.get_ident(),
                                        "native_thread_id": threading.get_native_id(),
                                        "failed": False,
                                    }
                                )
                            else:
                                trace.dropped += 1
                    with _phase("client_async_execution", cpu=False):
                        return await coro

                with _phase("client_sync_roundtrip"):
                    return original_run(observed())

            self._replace(client, "_run_coroutine", run_coroutine)
            if diagnostics is not None:
                self._client_loop = client._loop
                self._client_loop.call_soon_threadsafe(self._start_monitor)
        except Exception:
            self.close()
            raise

    def _start_monitor(self) -> None:
        if not self._closed:
            self._monitor_task = asyncio.create_task(self._monitor_loop())

    async def _monitor_loop(self) -> None:
        # One timer on the native client loop. Interval CPU includes other
        # tasks and native work; it is not exclusive CPU of a pending RPC.
        while not self._closed:
            started, unix = time.perf_counter(), time.time()
            thread_cpu, process_cpu = time.thread_time(), time.process_time()
            await asyncio.sleep(1.0)
            elapsed = time.perf_counter() - started
            if elapsed - 1.0 >= self.slow_call_s:
                emit_latency(
                    "tq_client_loop_delay",
                    checkpoint_id=(
                        self.diagnostics.checkpoint_id
                        if self.diagnostics is not None
                        else None
                    ),
                    started_unix_s=unix,
                    elapsed_s=elapsed,
                    lag_s=elapsed - 1.0,
                    thread_cpu_interval_s=time.thread_time() - thread_cpu,
                    process_cpu_interval_s=time.process_time() - process_cpu,
                    native_thread_id=threading.get_native_id(),
                )

    def _cancel_monitor(self) -> None:
        if self._monitor_task is not None:
            self._monitor_task.cancel()

    def _replace(self, obj: Any, name: str, value: Any) -> None:
        self._patches.append((obj, name, name in vars(obj), getattr(obj, name), value))
        setattr(obj, name, value)

    @staticmethod
    def _wrap(original: Callable[..., Any], phase: str) -> Callable[..., Any]:
        if inspect.iscoroutinefunction(original):

            @functools.wraps(original)
            async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
                with _phase(phase, cpu=False):
                    return await original(*args, **kwargs)

            return async_wrapper

        @functools.wraps(original)
        def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
            with _phase(phase):
                return original(*args, **kwargs)

        return sync_wrapper

    def close(self) -> None:
        """Restore only hooks still owned by this handle, including partial setup."""
        if self._closed:
            return
        self._closed = True
        if self._client_loop is not None and not self._client_loop.is_closed():
            self._client_loop.call_soon_threadsafe(
                functools.partial(self._cancel_monitor)
            )
        for obj, name, owned, original, replacement in reversed(self._patches):
            if getattr(obj, name) is replacement:
                if owned:
                    setattr(obj, name, original)
                else:
                    delattr(obj, name)
        global G_TQ_HOOKS
        if G_TQ_HOOKS is self:
            G_TQ_HOOKS = None


def install_transfer_queue_latency(
    client: Any,
    *,
    max_events: int,
    slow_call_s: float,
    diagnostics: CheckpointLatencyDiagnostics | None = None,
) -> TransferQueueLatencyHooks:
    """Install once on the capture client's process-local native TQ handle."""
    global G_TQ_HOOKS
    if G_TQ_HOOKS is not None:
        raise RuntimeError("TQ latency probes are already installed")
    hooks = TransferQueueLatencyHooks(
        client,
        max_events=max_events,
        slow_call_s=slow_call_s,
        diagnostics=diagnostics,
    )
    G_TQ_HOOKS = hooks
    emit_latency(
        "tq_latency_hooks", event="installed", client_type=type(client).__qualname__
    )
    return hooks


@contextmanager
def trace_tq_put() -> Iterator[None]:
    """One bounded correlated record per prefix put or slow terminal put."""
    hooks = G_TQ_HOOKS
    if hooks is None or hooks._closed:
        yield
        return
    trace = _PutTrace(hooks.max_events, time.time())
    token = G_TQ_TRACE.set(trace)
    probe, gate_work = current_prefix_cut_probe(), G_GATE_WORK.get()
    failed = True
    try:
        yield
        failed = False
    finally:
        G_TQ_TRACE.reset(token)
        elapsed = time.time() - trace.started_unix_s
        if probe is not None or elapsed >= hooks.slow_call_s:
            emit_latency(
                "tq_put_internal_phases",
                checkpoint_id=(
                    probe.checkpoint_id
                    if probe is not None
                    else hooks.diagnostics.checkpoint_id
                    if hooks.diagnostics is not None
                    else None
                ),
                batch_or_inventory_id=None if probe is None else probe.inventory_digest,
                operation="prefix"
                if probe is not None
                else (gate_work.operation if gate_work else "put"),
                model_call_id=None if gate_work is None else gate_work.model_call_id,
                started_unix_s=trace.started_unix_s,
                elapsed_s=elapsed,
                failed=failed,
                phases=trace.events,
                dropped_events=trace.dropped,
            )
