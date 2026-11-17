# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""SFT v2 spans and metric declarations use the shared RL telemetry path."""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
from nemo.lens import NemoLensConfig, setup_telemetry
from opentelemetry import trace as otel_trace
from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
    InMemorySpanExporter,
)

from nemo_rl.algorithms.metric_utils import SETUP_TIMING_PREFIX, SetupTimingMetrics
from nemo_rl.algorithms.sft_v2 import (
    ENERGON_IN_FLIGHT_ATTR,
    ENERGON_WORKERS_PER_RANK_ATTR,
    SFT_V2_TEED_METRICS,
    SFTSingleControllerActor,
    SFTV2SaveState,
    _energon_fanout_attributes,
)
from nemo_rl.data.energon.config import EnergonLoaderConfig
from nemo_rl.telemetry.instrumentation import (
    RL_ITERATION_ATTR,
    TRACE_CARRIER_KWARG,
    current_trace_carrier,
    iteration_scope,
    umbrella_span,
)
from nemo_rl.telemetry.metrics import map_teed_scalars
from nemo_rl.telemetry.span_groups import RLSpanGroup


def _loader_setup_controller(tracer, **energon: object) -> object:
    """A controller with just enough state for _setup_loaders, over 2 DP ranks."""
    controller_cls = SFTSingleControllerActor.__ray_metadata__.modified_class
    controller = object.__new__(controller_cls)
    controller._tracer = tracer
    controller._master_config = SimpleNamespace(
        data={
            "max_input_seq_length": 128,
            "energon": EnergonLoaderConfig(model_family="qwen", **energon),
        },
        policy={
            "train_global_batch_size": 4,
            "make_sequence_length_divisible_by": 8,
        },
        sft=SimpleNamespace(only_unmask_final=False),
    )
    controller._placement_plan = SimpleNamespace(
        logical_world_size=2, placement_hash="placement"
    )
    controller._loader_states = None
    controller._trainer = MagicMock()
    controller._trainer.worker_group.run_all_workers_single_data.return_value = []
    return controller


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
    """The checkpointing span covers the whole blocking save, including a deferred-save wait."""
    exporter = InMemorySpanExporter()
    handle = setup_telemetry(
        NemoLensConfig(enabled=True, span_groups="per_step"),
        span_exporter=exporter,
    )
    # Which phases were open under which span. A span wrapping only the policy
    # write, or only the handoff, still satisfies a single-phase assertion.
    open_span: dict[str, int] = {}

    def _record(phase: str, result=None):
        def _side_effect(*args, **kwargs):
            context = otel_trace.get_current_span().get_span_context()
            open_span[phase] = context.span_id
            return result

        return _side_effect

    monkeypatch.setattr(
        "nemo_rl.algorithms.sft_v2.save_checkpoint_object",
        MagicMock(side_effect=_record("loader_dump")),
    )

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
    controller._checkpointer.init_tmp_checkpoint = MagicMock(
        side_effect=_record("init_tmp", result=str(tmp_path))
    )
    controller._trainer.save_checkpoint = MagicMock(side_effect=_record("policy_write"))
    controller._checkpointer.begin_finalization = MagicMock(
        side_effect=_record("finalize_handoff")
    )
    # A save from the previous step still waiting to be published.
    controller._pending_checkpoint_path = str(tmp_path / "tmp_step_6")
    controller._trainer.finalize_async_save = MagicMock(
        side_effect=_record("pending_wait")
    )

    with iteration_scope(7):
        controller._save_checkpoint({"loss": 1.0})
    handle.shutdown()

    spans = {span.name: span for span in exporter.get_finished_spans()}
    checkpointing = spans["rl.sft_v2.checkpointing"]
    assert checkpointing.attributes["rl.bucket"] == "overhead"
    # Stamped by the caller's iteration_scope rather than passed at the site.
    assert checkpointing.attributes[RL_ITERATION_ATTR] == 7
    assert open_span == {
        phase: checkpointing.context.span_id
        for phase in (
            "pending_wait",
            "init_tmp",
            "policy_write",
            "loader_dump",
            "finalize_handoff",
        )
    }


def test_sft_v2_controller_constructor_parents_setup_loaders_to_the_driver() -> None:
    """setup_loaders joins the driver's trace through the constructor's trace_carrier."""
    exporter = InMemorySpanExporter()
    handle = setup_telemetry(
        NemoLensConfig(enabled=True, span_groups="per_step"),
        span_exporter=exporter,
    )
    controller_cls = SFTSingleControllerActor.__ray_metadata__.modified_class
    controller = object.__new__(controller_cls)
    template = _loader_setup_controller(handle.tracer, num_workers=2)
    actor_args = SimpleNamespace(
        trainer=template._trainer,
        loss_fn=MagicMock(),
        train_cluster=MagicMock(),
        placement_plan=template._placement_plan,
        save_state=SFTV2SaveState(0, 0, 0, "placement"),
        loader_states=None,
        setup_timing=MagicMock(),
    )
    master_config = MagicMock()
    master_config.data = template._master_config.data
    master_config.policy = template._master_config.policy
    master_config.sft = SimpleNamespace(
        only_unmask_final=False, defer_async_checkpoint_finalize=False
    )

    with umbrella_span(RLSpanGroup.U_JOB, "rl.sft_v2.driver", tracer=handle.tracer):
        # Captured on the driver, inside its job span, as run_sft_v2 does.
        carrier = current_trace_carrier()

    # Outside the span: the actor is another process, so running __init__ under
    # it would pass whether or not the carrier is read.
    with (
        patch(
            "nemo_rl.algorithms.sft_v2.init_telemetry_worker",
            return_value=SimpleNamespace(tracer=handle.tracer),
        ),
        patch("nemo_rl.algorithms.sft_v2.Logger"),
        patch("nemo_rl.algorithms.sft_v2.CheckpointManager"),
        patch("nemo_rl.algorithms.sft_v2.TimeoutChecker"),
        patch("nemo_rl.algorithms.sft_v2._max_train_steps", return_value=1),
        patch("nemo_rl.algorithms.sft_v2.ray.get", return_value=[True, True]),
    ):
        controller_cls.__init__(
            controller, master_config, actor_args, trace_carrier=carrier
        )
    handle.shutdown()

    spans = {span.name: span for span in exporter.get_finished_spans()}
    assert set(spans) >= {"rl.sft_v2.driver", "rl.sft_v2.setup_loaders"}
    assert (
        spans["rl.sft_v2.setup_loaders"].parent.span_id
        == spans["rl.sft_v2.driver"].context.span_id
    )


def test_sft_v2_setup_loaders_span_records_the_energon_worker_pool() -> None:
    """The setup_loaders span carries the per-rank Energon worker count and queue depth."""
    exporter = InMemorySpanExporter()
    handle = setup_telemetry(
        NemoLensConfig(enabled=True, span_groups="per_step"),
        span_exporter=exporter,
    )
    controller = _loader_setup_controller(
        handle.tracer, num_workers=4, prefetch_factor=3
    )

    with patch("nemo_rl.algorithms.sft_v2.ray.get", return_value=[True, True]):
        controller._setup_loaders()
    handle.shutdown()

    (span,) = exporter.get_finished_spans()
    assert span.name == "rl.sft_v2.setup_loaders"
    assert span.attributes[ENERGON_WORKERS_PER_RANK_ATTR] == 4
    assert span.attributes[ENERGON_IN_FLIGHT_ATTR] == 12
    # Startup work fans out across the loader ranks, so it stays unbucketed.
    assert "rl.bucket" not in span.attributes


def test_sft_v2_in_process_loading_reports_no_worker_pool() -> None:
    """num_workers=0 loads in-process, so the worker count and queue depth are both 0."""
    controller = _loader_setup_controller(None, num_workers=0)

    attributes = _energon_fanout_attributes(controller._master_config.data)

    assert attributes[ENERGON_WORKERS_PER_RANK_ATTR] == 0
    assert attributes[ENERGON_IN_FLIGHT_ATTR] == 0


def test_sft_v2_owner_calls_carry_the_caller_trace_context() -> None:
    """_owner_call sends a trace carrier so commit_sft_batch joins the step's trace."""
    handle = setup_telemetry(
        NemoLensConfig(enabled=True, span_groups="per_step"),
        span_exporter=InMemorySpanExporter(),
    )
    controller_cls = SFTSingleControllerActor.__ray_metadata__.modified_class
    controller = object.__new__(controller_cls)
    controller._trainer = MagicMock()
    controller._trainer.worker_group.run_all_workers_single_data.return_value = []

    with (
        umbrella_span(RLSpanGroup.U_JOB, "rl.sft_v2.job", tracer=handle.tracer),
        patch("nemo_rl.algorithms.sft_v2.ray.get", return_value=[]),
    ):
        controller._owner_call("commit_sft_batch")
    handle.shutdown()

    kwargs = (
        controller._trainer.worker_group.run_all_workers_single_data.call_args.kwargs
    )
    assert kwargs[TRACE_CARRIER_KWARG]


def test_sft_v2_setup_timing_reaches_the_logger_under_the_teed_prefix() -> None:
    """Setup timing is logged under SETUP_TIMING_PREFIX, which is what emits rl.setup.duration."""
    controller_cls = SFTSingleControllerActor.__ray_metadata__.modified_class
    controller = object.__new__(controller_cls)
    logger = MagicMock()

    setup_timing = SetupTimingMetrics(
        policy_init_time_s=12.0,
        total_setup_time_s=20.0,
        extras={"cluster_init_time_s": 5.0},
    )
    controller._logger = logger
    controller._log_setup_timing(setup_timing)

    logger.log_metrics.assert_called_once_with(
        {
            "policy_init_time_s": 12.0,
            "total_setup_time_s": 20.0,
            "cluster_init_time_s": 5.0,
        },
        step=0,
        prefix=SETUP_TIMING_PREFIX,
    )
    # The phases the metric will carry, which is what a dashboard filters on.
    assert set(SetupTimingMetrics.phase_seconds(setup_timing.to_metrics_dict())) == {
        "policy_init",
        "total_setup",
        "cluster_init",
    }
