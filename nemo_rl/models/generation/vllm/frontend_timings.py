# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
"""Sample CPU stages inside the pinned vLLM frontend without per-token hooks."""

from __future__ import annotations

import ast
import contextlib
import functools
import importlib.metadata
import inspect
import textwrap
import time
import types
from collections.abc import Iterator
from typing import Any


class _StageTransformer(ast.NodeTransformer):
    """Wrap six disjoint statements in the upstream per-request output loop."""

    def __init__(self) -> None:
        self.matches: dict[str, int] = {}

    def visit(self, node: ast.AST) -> Any:
        stage = None
        if isinstance(node, ast.Expr) and isinstance(node.value, ast.Call):
            call = ast.unparse(node.value.func)
            if call == "self._update_stats_from_output":
                stage = "stats"
            elif call == "req_state.logprobs_processor.update_from_output":
                stage = "logprobs"
        elif isinstance(node, ast.Assign) and isinstance(node.value, ast.Call):
            if ast.unparse(node.value.func) == "req_state.detokenizer.update":
                stage = "detokenize_stop"
        elif isinstance(node, ast.If):
            condition = ast.unparse(node.test)
            if condition == "engine_core_output.routed_experts is not None":
                stage = "router_collect"
            elif isinstance(node.test, ast.NamedExpr) and isinstance(
                node.test.value, ast.Call
            ):
                if ast.unparse(node.test.value.func) == "req_state.make_request_output":
                    stage = "completion_delivery"
            elif condition == "finish_reason is not None":
                stage = "finish_cleanup"
        if stage is None:
            return super().visit(node)
        assert isinstance(node, ast.stmt)
        self.matches[stage] = self.matches.get(stage, 0) + 1
        return ast.copy_location(
            ast.With(
                items=[
                    ast.withitem(
                        context_expr=ast.Call(
                            func=ast.Name(id="_nrl_frontend_stage", ctx=ast.Load()),
                            args=[ast.Constant(stage)],
                            keywords=[],
                        )
                    )
                ],
                body=[node],
            ),
            node,
        )


class FrontendStageTimings:
    """Use the original function except for one sampled chunk every N chunks.

    The sampled function is the upstream source with timing contexts inserted
    around six non-overlapping stages. No per-request objects or method wrappers
    are retained. Install before the token publication wrapper so its existing
    locking and completion-retention behavior still encloses both paths.
    """

    def __init__(self, processor: Any, *, sample_every: int) -> None:
        if type(sample_every) is not int or sample_every < 1:
            raise ValueError("Frontend timing sample_every must be positive")
        self._processor = processor
        self._original = processor.process_outputs
        self._sample_every = sample_every
        self._stats: dict[str, int | float] = {
            "process_calls": 0,
            "sampled_calls": 0,
            "sampled_request_updates": 0,
        }
        function = self._original.__func__
        source = textwrap.dedent(inspect.getsource(function))
        tree = ast.parse(source)
        transform = _StageTransformer()
        tree = transform.visit(tree)
        expected = {
            "stats",
            "logprobs",
            "detokenize_stop",
            "router_collect",
            "completion_delivery",
            "finish_cleanup",
        }
        if transform.matches != dict.fromkeys(expected, 1):
            raise RuntimeError(
                f"Unverified vLLM output processor timing boundaries: {transform.matches}"
            )
        # Defer annotations just as in the upstream module. Its globals include
        # the exact native types/helpers, avoiding replacement implementations.
        tree.body.insert(
            0,
            ast.ImportFrom(
                module="__future__", names=[ast.alias(name="annotations")], level=0
            ),
        )
        namespace = dict(function.__globals__)
        namespace["_nrl_frontend_stage"] = self.phase
        exec(
            compile(
                ast.fix_missing_locations(tree), function.__code__.co_filename, "exec"
            ),
            namespace,
        )
        sampled = types.MethodType(namespace[function.__name__], processor)

        @functools.wraps(self._original)
        def process_outputs(engine_core_outputs: Any, *args: Any, **kwargs: Any) -> Any:
            self._stats["process_calls"] += 1
            measured = (self._stats["process_calls"] - 1) % self._sample_every == 0
            with self.phase("process_total"):
                if not measured:
                    return self._original(engine_core_outputs, *args, **kwargs)
                self._stats["sampled_calls"] += 1
                self._stats["sampled_request_updates"] += len(engine_core_outputs)
                with self.phase("sampled_process_total"):
                    return sampled(engine_core_outputs, *args, **kwargs)

        self._wrapper = process_outputs
        processor.process_outputs = process_outputs

    @contextlib.contextmanager
    def phase(self, name: str) -> Iterator[None]:
        started, cpu_started = time.perf_counter(), time.thread_time()
        try:
            yield
        finally:
            wall = time.perf_counter() - started
            cpu = time.thread_time() - cpu_started
            for suffix, amount in (
                ("calls", 1),
                ("wall_s", wall),
                ("thread_cpu_s", cpu),
            ):
                key = f"{name}_{suffix}"
                self._stats[key] = self._stats.get(key, 0) + amount

    def stats(self) -> dict[str, int | float]:
        """Cumulative measurements; child stages cover sampled chunks only."""
        return dict(self._stats)

    def observe_decoder(self, decoder: Any) -> None:
        """Time native wire decoding in either the receiver or compact consumer."""
        original = decoder.decode

        @functools.wraps(original)
        def decode(*args: Any, **kwargs: Any) -> Any:
            with self.phase("wire_decode"):
                return original(*args, **kwargs)

        decoder.decode = decode


def install_frontend_stage_timings(
    processor: Any, *, sample_every: int
) -> FrontendStageTimings:
    if importlib.metadata.version("vllm").split("+")[0] != "0.29.0":
        raise RuntimeError("Frontend stage timings are verified on vLLM 0.29.0")
    return FrontendStageTimings(processor, sample_every=sample_every)
