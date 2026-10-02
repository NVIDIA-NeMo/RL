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
from nemo_rl.telemetry.instrumentation import umbrella_span
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
