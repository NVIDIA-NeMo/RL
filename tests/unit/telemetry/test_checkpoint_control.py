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

"""Exercise real checkpoint control methods without loading training engines.

The coordinator imports normally. Controller and engine classes import CUDA/
training packages, so their actual method ASTs execute against lightweight
resources. This tests the instrumented production flow, not a duplicate of it.
"""

from __future__ import annotations

import ast
import asyncio
import contextlib
import functools
import json
import shutil
import threading
import time
import uuid
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest
from nemo.lens import span_cm
from opentelemetry import trace

import nemo_rl.telemetry.rollout_checkpoint as checkpoint
from nemo_rl.environments.gym_checkpoint_adapter import GymCheckpointPrepareSummary
from nemo_rl.environments.gym_checkpoint_coordinator import (
    GymCheckpointCoordinator,
    GymCheckpointOperationError,
)
from nemo_rl.telemetry.instrumentation import accepts_trace_context
from tests.unit.telemetry.test_rollout_checkpoint import spans

_REPO = Path(__file__).resolve().parents[3]


def production_method(path, name, **extra):
    source = _REPO / path
    tree = ast.parse(source.read_text())
    method = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name == name
    )
    future = ast.parse("from __future__ import annotations").body[0]
    namespace = dict(
        cast=cast,
        Callable=Callable,
        Any=Any,
        asyncio=asyncio,
        contextlib=contextlib,
        functools=functools,
        partial=functools.partial,
        time=time,
        uuid=uuid,
        Path=Path,
        json=json,
        shutil=shutil,
        accepts_trace_context=accepts_trace_context,
        checkpoint_attempt=checkpoint.checkpoint_attempt,
        checkpoint_phase=checkpoint.checkpoint_phase,
        checkpoint_phase_async=checkpoint.checkpoint_phase_async,
        checkpoint_phase_sync=checkpoint.checkpoint_phase_sync,
        checkpoint_barrier=checkpoint.checkpoint_barrier,
        checkpoint_lock=checkpoint.checkpoint_lock,
        checkpoint_identity=checkpoint.checkpoint_identity,
        trace_checkpoint_save=checkpoint.trace_checkpoint_save,
        **extra,
    )
    exec(
        compile(
            ast.Module(body=[future, method], type_ignores=[]), str(source), "exec"
        ),
        namespace,
    )
    return namespace[name]


class Remote:
    def __init__(self, function):
        self.function = function

    def remote(self, *args, **kwargs):
        return self.function(*args, **kwargs)


def test_actual_gym_rpc_accepts_checkpoint_parent(checkpoint_exporting):
    exporter, _ = checkpoint_exporting
    contexts = []

    class Adapter:
        async def prepare(self, checkpoint_id, *, deadline_ts):
            # This is the boundary where Gym #3903 opens its own spans; RL must
            # deliver the ambient parent before invoking the library.
            with span_cm("gym.checkpoint.coordinate.prepare"):
                contexts.append(trace.get_current_span().get_span_context())
            return SimpleNamespace(prepared=True, blockers=lambda: {})

    receive = production_method(
        "nemo_rl/environments/nemo_gym.py",
        "checkpoint_prepare",
        GymCheckpointPrepareSummary=GymCheckpointPrepareSummary,
    )
    actor = SimpleNamespace(_require_checkpoint_adapter=lambda: Adapter())
    handle = SimpleNamespace(
        checkpoint_prepare=Remote(functools.partial(receive, actor))
    )
    coordinator = GymCheckpointCoordinator(
        SimpleNamespace(checkpoint_handles={"gym/0": handle}),
        control_timeout_s=1,
        prepare_timeout_s=1,
    )

    async def run():
        async with checkpoint.checkpoint_attempt("save"):
            return await coordinator.prepare(checkpoint.checkpoint_identity())

    assert asyncio.run(run())["gym/0"].prepared
    emitted = spans(exporter)
    assert (
        contexts[0].trace_id == emitted["rl.rollout_checkpoint.save"].context.trace_id
    )
    assert (
        emitted["gym.checkpoint.coordinate.prepare"].parent.span_id
        == emitted["rl.rollout_checkpoint.save"].context.span_id
    )


def test_gym_timeout_reports_only_unanswered_deployment(checkpoint_exporting, caplog):
    exporter, _ = checkpoint_exporting
    coordinator = GymCheckpointCoordinator(
        SimpleNamespace(checkpoint_handles={"unused": None}),
        control_timeout_s=0.005,
        prepare_timeout_s=0.005,
    )

    async def run():
        async def fast():
            return "done"

        async def hung():
            await asyncio.Event().wait()

        async with checkpoint.checkpoint_attempt("save"):
            return await coordinator._collect(
                "prepare",
                {"fast": fast(), "hung": hung()},
                deadline_ts=time.time() + 0.005,
            )

    with pytest.raises(GymCheckpointOperationError) as caught:
        asyncio.run(run())
    assert set(caught.value.failures) == {"hung"}
    phase = spans(exporter)["rl.rollout_checkpoint.gym_prepare"]
    assert phase.attributes["rl.checkpoint.pending_instances"] == ("hung",)
    assert phase.attributes["rl.checkpoint.pending_deployments"] == 1
    assert "hung" in caplog.text


def test_prefix_executor_preserves_parent_context(checkpoint_exporting):
    import contextvars

    exporter, _ = checkpoint_exporting
    execute = production_method(
        "nemo_rl/models/generation/generation_cut_capture.py",
        "_run_generation_cut_control",
        contextvars=contextvars,
    )
    with ThreadPoolExecutor(max_workers=1) as pool:
        worker = SimpleNamespace(_generation_cut_control_executor=pool)

        def cut():
            with checkpoint.checkpoint_phase("prefix_cut"):
                return checkpoint.checkpoint_identity(), threading.get_ident()

        async def run():
            async with checkpoint.checkpoint_attempt("save") as attempt:
                identity, thread = await execute(worker, cut)
                assert identity == attempt.checkpoint_id
                assert thread != threading.get_ident()

        asyncio.run(run())
    emitted = spans(exporter)
    assert (
        emitted["rl.rollout_checkpoint.prefix_cut"].parent.span_id
        == emitted["rl.rollout_checkpoint.prefix_control"].context.span_id
    )


@pytest.mark.parametrize("fail_tq", [False, True])
def test_actual_save_releases_gym_before_publication_and_on_failure(
    checkpoint_exporting, tmp_path, fail_tq
):
    exporter, _ = checkpoint_exporting
    events = []
    failure = OSError("TQ save failed")
    anchor = tmp_path / "bootstrap"
    anchor.mkdir()
    temporary = anchor / "snapshot.tmp"
    temporary.mkdir()
    final = anchor / "snapshot_1"

    @contextlib.asynccontextmanager
    async def barrier():
        yield object()

    @contextlib.asynccontextmanager
    async def gate():
        events.append("gate_closed")
        try:
            yield
        finally:
            events.append("gate_open")

    @contextlib.asynccontextmanager
    async def prepared(checkpoint_id):
        assert checkpoint_id == checkpoint.checkpoint_identity()
        events.append("gym_prepared")
        try:
            yield
        finally:
            events.append("gym_resumed")

    async def commit(*args):
        return "gym_commit"

    async def noop(*args, **kwargs):
        return None

    async def capture(*args, **kwargs):
        with checkpoint.checkpoint_phase("tq_save"):
            events.append("tq_save")
            if fail_tq:
                raise failure
        return SimpleNamespace(
            tq_save_seconds=0.01,
            sampler_dispatch_index=0,
            mutation_version=1,
            rolled_back_train_group_count=0,
            replay_metadata={"groups": []},
            rollout_recovery_group_count=0,
            replay_row_count=0,
            staging_row_count=0,
        )

    async def sidecars(*args):
        return 0

    def publish(*args, **kwargs):
        assert (
            "gym_resumed" in events and "gate_open" in events and "fence_open" in events
        )
        events.append("publication")
        temporary.rename(final)

    cls = SimpleNamespace
    actor = SimpleNamespace(
        _checkpoint_save_lock=asyncio.Lock(),
        _optimizer_commit_in_progress=False,
        _checkpointer=SimpleNamespace(
            finalize_pending=lambda: None, checkpoint_dir=tmp_path
        ),
        _train_steps=0,
        _trainer_version=0,
        _current_epoch=0,
        _bootstrap_identity=SimpleNamespace(fingerprint=lambda: "identity"),
        _master_config=SimpleNamespace(
            rollout_checkpointing=SimpleNamespace(keep_latest_k=2),
            token_capture=SimpleNamespace(control_timeout_s=1),
        ),
        _generation_prefix_cuts_enabled=True,
        _gen=SimpleNamespace(
            begin_token_capture_snapshot_fence=lambda **kw: events.append(
                "fence_closed"
            ),
            end_token_capture_snapshot_fence=lambda **kw: events.append("fence_open"),
        ),
        _gym_checkpoint_coordinator=SimpleNamespace(prepared=prepared, commit=commit),
        _rollout_dispatch_admission_gate=SimpleNamespace(closed=gate),
        _rollout_manager=SimpleNamespace(
            suspend_request_deadlines=lambda *args: None,
            resume_request_deadlines=lambda *args: None,
        ),
        _retire_dropped_gym_episodes=noop,
        _gym_checkpoint_inventory=lambda co: {"gym": ()},
        _queued_gym_retirements=lambda co: {"gym": set()},
        _drain_non_exported_gym_candidates=noop,
        _data_plane_checkpoint_barrier=SimpleNamespace(checkpoint=barrier),
        _capture_coordinated_rollout_checkpoint_cut=capture,
        _write_rollout_checkpoint_sidecars=sidecars,
        _log_telemetry_metrics=lambda *args, **kwargs: None,
    )
    save = production_method(
        "nemo_rl/algorithms/single_controller.py",
        "_save_rollout_checkpoint",
        _RolloutCheckpointSaveResult=cls,
        ensure_bootstrap_anchor=lambda *args, **kwargs: anchor,
        prepare_snapshot_paths=lambda *args: (temporary, final, 1),
        RolloutSnapshotManifest=lambda **kwargs: SimpleNamespace(
            to_dict=lambda: kwargs
        ),
        ROLLOUT_SNAPSHOT_SCHEMA_VERSION=1,
        ROLLOUT_SNAPSHOT_MANIFEST_FILENAME="manifest.json",
        commit_snapshot=publish,
    )
    prepare = production_method(
        "nemo_rl/algorithms/single_controller.py",
        "_prepared_gym_checkpoint",
        _GYM_CHECKPOINT_DEADLINE_HOLDER="checkpoint",
    )
    actor._prepared_gym_checkpoint = functools.partial(prepare, actor)
    if fail_tq:
        with pytest.raises(OSError) as caught:
            asyncio.run(save(actor, force=True))
        assert caught.value is failure and not temporary.exists() and not final.exists()
        assert "publication" not in events
    else:
        assert asyncio.run(save(actor, force=True)).saved
        emitted = spans(exporter)
        assert (
            emitted["rl.rollout_checkpoint.dispatch_pause"].end_time
            <= emitted["rl.rollout_checkpoint.publication"].start_time
        )
        assert final.is_dir()
    assert (
        events.count("fence_open") == 1
        and events.count("gym_resumed") == 1
        and events.count("gate_open") == 1
    )


def test_new_checkpoint_methods_are_context_receivers():
    # Source guard plus runtime prepare test: a receiver decorator without the
    # caller's carrier would appear instrumented while silently starting roots.
    source = ast.parse((_REPO / "nemo_rl/environments/nemo_gym.py").read_text())
    expected = {
        "checkpoint_prepare",
        "checkpoint_commit",
        "checkpoint_restore",
        "checkpoint_resume",
        "checkpoint_retire",
        "checkpoint_renew",
    }
    found = {
        node.name
        for node in ast.walk(source)
        if isinstance(node, ast.AsyncFunctionDef)
        and any(
            isinstance(d, ast.Name) and d.id == "accepts_trace_context"
            for d in node.decorator_list
        )
    }
    assert expected <= found
