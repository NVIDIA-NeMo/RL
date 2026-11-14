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
"""Protocol and diagnostics tests; no vLLM, GPU allocation or Ray actors."""

import asyncio
import importlib.util
import json
import os
import pickle
import shlex
import subprocess
import sys
from collections.abc import Generator
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from tools.model_diagnostics.prefix_checkpoint_benchmark import (
    Call,
    inventories,
    parse_args,
    prefix_options,
    verify_saved_prefixes,
)


def test_inventory_coverage() -> None:
    calls = [Call(str(index), float(index)) for index in range(17)]
    groups = inventories(calls, checkpoint_id="cut", count=4)
    ids = [prefix.model_call_id for group in groups for prefix in group.active_prefixes]
    assert sorted(ids) == sorted(call.identity for call in calls)
    assert sorted(len(group.active_prefixes) for group in groups) == [4, 4, 4, 5]
    for group in groups:
        assert type(group).model_validate_json(group.model_dump_json()) == group


@pytest.mark.parametrize(
    "calls,count",
    [
        ([Call("one", 0), Call("one", 1)], 1),
        ([Call("one", 0)], 2),
        ([Call("one", 0)], 0),
        ([], 1),
    ],
)
def test_invalid_inventory(calls: list[Call], count: int) -> None:
    with pytest.raises(ValueError):
        inventories(calls, checkpoint_id="cut", count=count)


@pytest.mark.parametrize(
    "mode,expected",
    [
        ("baseline", (False, False, False)),
        ("encoder", (False, False, True)),
        ("pipeline", (False, True, True)),
        ("writer", (True, True, True)),
    ],
)
def test_production_switches(mode: str, expected: tuple[bool, bool, bool]) -> None:
    options = prefix_options(mode)
    assert (
        options["prefix_writer_actor"],
        options["prefix_write_pipeline"]["enabled"],
        options["prefix_write_pipeline"]["direct_column_encoding"],
    ) == expected


def test_unknown_mode() -> None:
    with pytest.raises(ValueError):
        prefix_options("unsupported")


@pytest.mark.parametrize(
    "extra",
    [
        ["--outstanding", "0"],
        ["--outstanding", "3", "--inventories", "4"],
        ["--token-targets", "10", "--new-token-targets", "0", "1"],
        ["--new-token-targets", "-1", "1"],
        ["--cut-timeout-s", "nan"],
        ["--target-wait-s", "inf"],
        ["--batch-sizes", "0", "256"],
        ["--max-output-tokens", "131072"],
    ],
)
def test_bad_workload_is_rejected(extra: list[str]) -> None:
    with pytest.raises(SystemExit):
        parse_args(["--model", "/model", "--output", "/output", *extra])


def test_cli_explicit_options() -> None:
    args = parse_args(
        [
            "--model",
            "/model",
            "--output",
            "/output",
            "--mode",
            "encoder",
            "--no-router-replay",
            "--token-targets",
            "10",
            "--new-token-targets",
            "0",
        ]
    )
    assert args.mode == "encoder"
    assert not args.router_replay
    assert len(args.token_targets) == 1


def saved_dump(root: Path) -> None:
    (root / "metadata.json").write_text(json.dumps({"storage_saved": True}))
    with (root / "controller_state.pkl").open("wb") as stream:
        pickle.dump(
            {
                "partitions": {
                    "rollout_staging": SimpleNamespace(keys_mapping={"a": 7, "b": 8})
                }
            },
            stream,
        )
    (root / "simple_storage").mkdir()
    with (root / "simple_storage" / "su_0.pkl").open("wb") as stream:
        pickle.dump({"field_data": {"tokens": {7: [1]}, "routes": {7: [2]}}}, stream)


def test_saved_dump_checks_all_required_columns(tmp_path: Path) -> None:
    saved_dump(tmp_path)
    verify_saved_prefixes(tmp_path, ["a"], ["tokens", "routes"])
    with pytest.raises(RuntimeError, match="routes"):
        verify_saved_prefixes(tmp_path, ["a", "b"], ["routes"])
    with pytest.raises(RuntimeError, match="unknown"):
        verify_saved_prefixes(tmp_path, ["a"], ["unknown"])
    with pytest.raises(RuntimeError, match="controller"):
        verify_saved_prefixes(tmp_path, ["missing"], ["tokens"])


@pytest.mark.parametrize("marker", [False, "false", None])
def test_incomplete_saved_dump(tmp_path: Path, marker: object) -> None:
    (tmp_path / "metadata.json").write_text(json.dumps({"storage_saved": marker}))
    with pytest.raises(RuntimeError, match="completion"):
        verify_saved_prefixes(tmp_path, ["a"], ["tokens"])


def test_launcher_dry_run_preserves_arguments(tmp_path: Path) -> None:
    model = tmp_path / "model with spaces"
    model.mkdir()
    (model / "config.json").write_text("{}")
    image = tmp_path / "container.sqsh"
    image.touch()
    results = tmp_path / "results"
    root = Path(__file__).parents[3]
    environment = os.environ | {
        "BENCH_IMAGE": str(image),
        "BENCH_MODEL": str(model),
        "BENCH_RESULTS": str(results),
        "BENCH_ACCOUNT": "example",
        "BENCH_MOUNTS": f"{tmp_path}:{tmp_path}",
    }
    result = subprocess.run(
        [
            "bash",
            str(root / "tools/model_diagnostics/launch_prefix_checkpoint_benchmark.sh"),
            "--dry-run",
            "--mode",
            "encoder",
            "--no-router-replay",
            "--project",
            "project with spaces",
        ],
        env=environment,
        capture_output=True,
        text=True,
        check=True,
    )
    command = shlex.split(result.stdout)
    assert command[0] == "sbatch"
    assert command[-5:] == [
        "--mode",
        "encoder",
        "--no-router-replay",
        "--project",
        "project with spaces",
    ]
    assert not results.exists()


@pytest.mark.parametrize("pipelined", [False, True])
def test_cancelled_http_still_drains_native_cut(
    monkeypatch: pytest.MonkeyPatch, pipelined: bool
) -> None:
    """Exercise the actual benchmark wrappers with lightweight engine stand-ins."""

    class BaseWorker:
        async def setup_token_capture(self) -> None:
            pass

    monkeypatch.setitem(sys.modules, "ray", SimpleNamespace(remote=lambda cls: cls))
    monkeypatch.setitem(
        sys.modules,
        "nemo_rl.models.generation.vllm.vllm_worker_async",
        SimpleNamespace(VllmAsyncGenerationWorkerImpl=BaseWorker),
    )
    source = (
        Path(__file__).parents[3]
        / "tools/model_diagnostics/prefix_checkpoint_benchmark_worker.py"
    )
    spec = importlib.util.spec_from_file_location("benchmark_worker_test", source)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    async def scenario() -> None:
        entered, release = asyncio.Event(), asyncio.Event()

        class Recovery:
            def __init__(self) -> None:
                self._parts = SimpleNamespace(
                    generation_prefix_batch_size=8,
                    sink=SimpleNamespace(
                        stage_generation_prefix_batch=lambda rows, **kw: [],
                        _prefix_store=None,
                    ),
                )
                self._control_executor = ThreadPoolExecutor(max_workers=1)
                self._pipeline = (
                    SimpleNamespace(
                        _prepare=lambda inventory, deadline: self._checkpoint_steps(
                            inventory, deadline=deadline
                        ),
                        stats=lambda: {"pending_inventories": 0},
                    )
                    if pipelined
                    else None
                )

            def _checkpoint_steps(
                self, inventory: Any, *, deadline: float | None
            ) -> Generator[None, Any, str]:
                yield None
                return "durable"

            async def run_cut(self, inventory: Any) -> str | None:
                prepare = (
                    self._checkpoint_steps
                    if self._pipeline is None
                    else self._pipeline._prepare
                )
                iterator = prepare(inventory, deadline=None)
                next(iterator)
                entered.set()
                await release.wait()
                try:
                    iterator.send(None)
                except StopIteration as finished:
                    return finished.value

            def counts(self) -> dict[str, int]:
                return {}

        worker = module.PrefixCheckpointBenchmarkWorker()
        worker.is_model_owner = True
        recovery = worker._prefix_recovery = Recovery()
        # __init__ created the benchmark state before the real engine constructor.
        await worker.setup_token_capture()
        try:
            inventory = SimpleNamespace(
                checkpoint_id="cut", inventory_digest="id", active_prefixes=[1]
            )
            request = asyncio.create_task(recovery.run_cut(inventory))
            await entered.wait()
            request.cancel()
            with pytest.raises(asyncio.CancelledError):
                await request
            drain = asyncio.create_task(worker.benchmark_drain())
            await asyncio.sleep(0)
            assert not drain.done()
            release.set()
            diagnostics = await drain
            (event,) = diagnostics["events"]
            assert event["success"]
            assert event["calls"] == 1
            assert event["queue_s"] >= 0
            assert event["prepare_wall_s"] >= event["thread_cpu_s"] - 0.01
            assert not worker._bench_pending
        finally:
            recovery._control_executor.shutdown(wait=True)

    asyncio.run(scenario())
