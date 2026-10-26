# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""SFT v2 spans and metric declarations use the shared RL telemetry path."""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from nemo.lens import NemoLensConfig, setup_telemetry
from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
    InMemorySpanExporter,
)

from nemo_rl.algorithms.sft_v2 import (
    SFT_V2_TEED_METRICS,
    SFTSingleControllerActor,
    SFTV2SaveState,
)
from opentelemetry import trace as otel_trace

from nemo_rl.telemetry.instrumentation import (
    RL_ITERATION_ATTR,
    iteration_scope,
    umbrella_span,
)
from nemo_rl.telemetry.metrics import map_teed_scalars
from nemo_rl.telemetry.span_groups import RLSpanGroup


def test_sft_v2_step_metrics_have_otel_series() -> None:
    values = {
        "loader_latency_max": 0.2,
        "policy_time": 1.5,
        "total_step_time": 1.8,
        "valid_tokens_per_second": 100.0,
    }
    mapped = map_teed_scalars(values)
    assert mapped == {row.key: values[row.logger_key] for row in SFT_V2_TEED_METRICS}


@pytest.mark.parametrize(("groups", "expected"), [("per_step", True), ("setup", False)])
def test_sft_v2_step_spans_follow_the_enabled_groups(
    groups: str, expected: bool
) -> None:
    exporter = InMemorySpanExporter()
    handle = setup_telemetry(
        NemoLensConfig(enabled=True, span_groups=groups),
        span_exporter=exporter,
    )
    controller_cls = SFTSingleControllerActor.__ray_metadata__.modified_class
    controller = object.__new__(controller_cls)
    controller._tracer = handle.tracer
    controller._save_state = SFTV2SaveState(0, 0, 0, "placement")
    controller._loss_fn = object()
    controller._load_envelopes = MagicMock(
        return_value=[
            SimpleNamespace(
                meta=SimpleNamespace(sample_ids=["sample"]),
                valid_tokens=4,
                source_ids=("sample",),
                load_seconds=0.2,
            )
        ]
    )
    controller._owner_call = MagicMock()
    controller._trainer = MagicMock()
    controller._trainer.finish_train_step.return_value = {
        "loss": 1.0,
        "grad_norm": 0.5,
        "all_mb_metrics": {},
    }

    with umbrella_span(RLSpanGroup.U_JOB, "rl.sft_v2.job", tracer=handle.tracer):
        controller._run_train_step()
    handle.shutdown()

    spans = {span.name: span for span in exporter.get_finished_spans()}
    if not expected:
        assert spans == {}
        return
    assert set(spans) == {
        "rl.sft_v2.job",
        "rl.sft_v2.step",
        "rl.sft_v2.policy_training",
    }
    assert (
        spans["rl.sft_v2.step"].parent.span_id == spans["rl.sft_v2.job"].context.span_id
    )
    assert (
        spans["rl.sft_v2.policy_training"].parent.span_id
        == spans["rl.sft_v2.step"].context.span_id
    )
    assert "rl.bucket" not in spans["rl.sft_v2.job"].attributes
    assert "rl.bucket" not in spans["rl.sft_v2.step"].attributes
    assert spans["rl.sft_v2.policy_training"].attributes["rl.bucket"] == "productive"


def test_sft_v2_checkpointing_span_covers_the_whole_save(monkeypatch, tmp_path) -> None:
    """The save blocks the loop with ``async_save`` off, so the span has to hold
    all of it: a span around the policy write alone leaves the loader dump and
    the finalization handoff outside, and ``rl.checkpoint.finalize`` parents to
    whatever is open when ``begin_finalization`` runs.
    """
    exporter = InMemorySpanExporter()
    handle = setup_telemetry(
        NemoLensConfig(enabled=True, span_groups="per_step"),
        span_exporter=exporter,
    )
    monkeypatch.setattr("nemo_rl.algorithms.sft_v2.save_checkpoint_object", MagicMock())

    controller_cls = SFTSingleControllerActor.__ray_metadata__.modified_class
    controller = object.__new__(controller_cls)
    controller._tracer = handle.tracer
    controller._save_state = SFTV2SaveState(7, 0, 0, "placement")
    controller._max_steps = 10
    controller._master_config = MagicMock()
    controller._placement_plan = SimpleNamespace(logical_world_size=1)
    controller._loader_state_dicts = MagicMock(return_value=[{"loader": 0}])
    controller._trainer = MagicMock()
    controller._checkpointer = MagicMock(save_optimizer=True)
    # A real path: the save builds the policy subpaths with os.path.join.
    controller._checkpointer.init_tmp_checkpoint.return_value = str(tmp_path)

    finalize_parent: dict[str, int] = {}

    def _record_open_span(*args, **kwargs):
        context = otel_trace.get_current_span().get_span_context()
        finalize_parent["span_id"] = context.span_id

    controller._checkpointer.begin_finalization = MagicMock(
        side_effect=_record_open_span
    )

    with iteration_scope(7):
        controller._save_checkpoint({"loss": 1.0})
    handle.shutdown()

    spans = {span.name: span for span in exporter.get_finished_spans()}
    checkpointing = spans["rl.sft_v2.checkpointing"]
    assert checkpointing.attributes["rl.bucket"] == "overhead"
    # Stamped by the caller's iteration_scope rather than passed at the site.
    assert checkpointing.attributes[RL_ITERATION_ATTR] == 7
    assert finalize_parent["span_id"] == checkpointing.context.span_id
