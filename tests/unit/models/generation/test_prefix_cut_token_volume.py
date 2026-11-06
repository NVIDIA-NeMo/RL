# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may obtain a copy at http://www.apache.org/licenses/LICENSE-2.0
"""Read-only volume accounting and token-target scheduling contracts."""

import ast
import asyncio
import threading
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import pytest

from nemo_rl.utils.prefix_cut_diagnostics import (
    current_prefix_cut_probe,
    prefix_cut_count,
    prefix_cut_probe,
)
from tools.model_diagnostics.prefix_cut_token_volume import (
    CaptureTokenHeader,
    count_engine_token_volume,
    token_match_report,
    wait_for_token_volume,
)


class LengthOnly:
    def __init__(self, count: int) -> None:
        self.count = count

    def __len__(self) -> int:
        return self.count

    def __iter__(self):
        raise AssertionError("Volume sampling must not read or copy token arrays")


def state(tokens: int, logprobs: int | None = None) -> SimpleNamespace:
    return SimpleNamespace(
        detokenizer=SimpleNamespace(output_token_ids=LengthOnly(tokens)),
        logprobs_processor=SimpleNamespace(
            logprobs=LengthOnly(tokens if logprobs is None else logprobs)
        ),
    )


def test_lengths_count_first_chunks_existing_chunks_and_retained_requests() -> None:
    processor = SimpleNamespace(
        external_req_ids={"first": ["a"], "next": ["b"], "zero": ["z"]},
        request_states={"a": state(30), "b": state(70), "z": state(0)},
    )
    retained = {"done": {"c": SimpleNamespace(request_state=state(100))}}
    headers = [
        CaptureTokenHeader("first", 10, 0, False),
        CaptureTokenHeader("next", 10, 50, True),
        CaptureTokenHeader("done", 10, 0, False),
        CaptureTokenHeader("zero", 99, 0, False),
        CaptureTokenHeader(None, 20, 0, False),
    ]
    before = dict(processor.request_states)
    result = count_engine_token_volume(headers, processor, retained)
    assert result == {
        "calls": 5,
        "positive_calls": 3,
        "not_in_engine": 1,
        "cumulative_generation_tokens": 200,
        "new_token_positions": 170,
        "unstaged_prompt_tokens": 20,
        "max_generated_tokens": 100,
    }
    assert processor.request_states == before
    assert set(retained["done"]) == {"c"}


@pytest.mark.parametrize("kind", ["logprobs", "shrunk", "multiple"])
def test_inconsistent_or_unsupported_states_fail(kind: str) -> None:
    processor = SimpleNamespace(
        external_req_ids={"r": ["a", "b"] if kind == "multiple" else ["a"]},
        request_states={"a": state(10, 9 if kind == "logprobs" else 10)},
    )
    with pytest.raises(ValueError):
        count_engine_token_volume(
            [CaptureTokenHeader("r", 2, 11 if kind == "shrunk" else 0, False)],
            processor,
            {},
        )


def test_missing_engine_state_is_counted_without_reading_other_requests() -> None:
    processor = SimpleNamespace(
        external_req_ids={"missing": ["gone"]}, request_states={}
    )
    result = count_engine_token_volume(
        [CaptureTokenHeader("missing", 99, 0, False)], processor, {}
    )
    assert result["not_in_engine"] == 1
    assert result["new_token_positions"] == 0


@pytest.mark.asyncio
async def test_wait_requires_both_metrics_on_every_engine() -> None:
    samples = iter(
        [
            [
                {"cumulative_generation_tokens": 101, "new_token_positions": 60},
                {"cumulative_generation_tokens": 99, "new_token_positions": 60},
            ],
            [
                {"cumulative_generation_tokens": 101, "new_token_positions": 60},
                {"cumulative_generation_tokens": 101, "new_token_positions": 59},
            ],
            [{"cumulative_generation_tokens": 102, "new_token_positions": 61}] * 2,
        ]
    )
    reports = []

    async def collect():
        return next(samples)

    result = await wait_for_token_volume(
        collect,
        cumulative_tokens=100,
        new_token_positions=60,
        timeout_s=1,
        poll_s=0,
        check_healthy=lambda: None,
        report=reports.append,
    )
    assert [entry["ready"] for entry in reports] == [False, False, True]
    assert result["workers"][1]["new_token_positions"] == 61


@pytest.mark.asyncio
async def test_timeout_does_not_return_a_false_match() -> None:
    async def collect():
        return [{"cumulative_generation_tokens": 1, "new_token_positions": 1}]

    with pytest.raises(TimeoutError):
        await wait_for_token_volume(
            collect,
            cumulative_tokens=100,
            new_token_positions=60,
            timeout_s=0.005,
            poll_s=0.01,
            check_healthy=lambda: None,
            report=lambda _: None,
        )


@pytest.mark.asyncio
async def test_cancelled_sampling_propagates_without_running_a_cut() -> None:
    async def collect():
        raise asyncio.CancelledError

    with pytest.raises(asyncio.CancelledError):
        await wait_for_token_volume(
            collect,
            cumulative_tokens=100,
            new_token_positions=60,
            timeout_s=1,
            poll_s=0,
            check_healthy=lambda: None,
            report=lambda _: None,
        )


@pytest.mark.parametrize(
    "actual,matched",
    [
        ({"cumulative_generation_tokens": 103, "batch_token_positions": 61}, True),
        ({"cumulative_generation_tokens": 100, "batch_token_positions": 80}, False),
        ({"cumulative_generation_tokens": 130, "batch_token_positions": 60}, False),
        ({}, False),
    ],
)
def test_match_uses_actual_captured_values_not_only_readiness(
    actual: dict[str, int], matched: bool
) -> None:
    result = token_match_report(
        actual, cumulative_tokens=100, new_token_positions=60, tolerance=0.1
    )
    assert result["matched"] is matched


def test_worker_counts_survive_pipeline_yields_and_deduplicate_retries() -> None:
    # Compile the actual extension methods without importing the GPU-only vLLM
    # actor module. The parent simulates a production preparation generator.
    worker_path = (
        Path(__file__).parents[4]
        / "tools/model_diagnostics/prefix_cut_benchmark_worker.py"
    )
    tree = ast.parse(worker_path.read_text())
    actor = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "PrefixCutBenchmarkWorker"
    )
    methods = [
        node
        for node in actor.body
        if isinstance(node, ast.FunctionDef)
        and node.name
        in {
            "_checkpoint_generation_cut_pipeline_steps",
            "_benchmark_saved_token_volumes",
        }
    ]

    class Parent:
        def _checkpoint_generation_cut_pipeline_steps(self, inventory, *, deadline):
            prefix_cut_count("cumulative_generation_tokens", 100)
            yield "write"
            prefix_cut_count("batch_token_positions", 60)
            return "receipt"

    harness = ast.ClassDef(
        name="Harness",
        bases=[ast.Name(id="Parent", ctx=ast.Load())],
        keywords=[],
        body=methods,
        decorator_list=[],
        type_params=[],
    )
    module = ast.Module(
        body=[
            ast.ImportFrom(
                module="__future__", names=[ast.alias(name="annotations")], level=0
            ),
            harness,
        ],
        type_ignores=[],
    )
    environment = {
        "Parent": Parent,
        "current_prefix_cut_probe": current_prefix_cut_probe,
        "Counter": Counter,
    }
    exec(
        compile(ast.fix_missing_locations(module), str(worker_path), "exec"),
        environment,
    )
    worker = environment["Harness"]()
    worker.cfg = {"_benchmark_token_volume_targets": True}
    worker._bench_token_volume_lock = threading.Lock()
    worker._bench_token_volume_records = {}
    inventory = SimpleNamespace(checkpoint_id="cp", inventory_digest="inventory")
    for _ in range(2):
        with prefix_cut_probe(
            enabled=True,
            checkpoint_id="cp",
            inventory_digest="inventory",
            calls=1,
            deadline=None,
        ):
            steps = worker._checkpoint_generation_cut_pipeline_steps(
                inventory, deadline=None
            )
            assert next(steps) == "write"
            with pytest.raises(StopIteration) as result:
                next(steps)
            assert result.value.value == "receipt"
    actual = worker._benchmark_saved_token_volumes()["cp"]
    assert actual["cumulative_generation_tokens"] == 100
    assert actual["batch_token_positions"] == 60
    assert len(worker._bench_token_volume_records) == 1
