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
    iteration_scope,
    umbrella_span,
)
from nemo_rl.telemetry.metrics import map_teed_scalars
from nemo_rl.telemetry.span_groups import RLSpanGroup


def _loader_setup_controller(tracer, **energon: object) -> object:
    """A controller with just enough state for ``_setup_loaders``, over 2 DP ranks.

    ``data["energon"]`` is a model, not a mapping: ``MasterConfig`` validates it
    on the driver, so the controller never sees the raw YAML.
    """
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
    """The save blocks the loop with ``async_save`` off, so the span has to hold
    all of it: a span around the policy write alone leaves the loader dump and
    the finalization handoff outside, and ``rl.checkpoint.finalize`` parents to
    whatever is open when ``begin_finalization`` runs.

    The wait on a deferred save counts too. It is the previous checkpoint's,
    but it blocks this step, and leaving it outside made the span read shorter
    than the gap it is supposed to explain.
    """
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


def test_sft_v2_setup_loaders_span_records_the_energon_worker_pool() -> None:
    """The subprocesses emit nothing, so the span is the only place they appear.

    Per rank, because the job-wide total is this times the loader rank count,
    which every ``rl.sft_v2.build_loader`` child of this span already carries.
    """
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
    """``num_workers=0`` loads in the policy process, so the honest count is 0.

    Reported rather than omitted: an absent attribute reads as telemetry that
    failed, where 0 reads as the pool that is genuinely not there.
    """
    controller = _loader_setup_controller(None, num_workers=0)

    attributes = _energon_fanout_attributes(controller._master_config.data)

    assert attributes[ENERGON_WORKERS_PER_RANK_ATTR] == 0
    assert attributes[ENERGON_IN_FLIGHT_ATTR] == 0


def test_sft_v2_worker_pool_attributes_are_absent_without_an_energon_block() -> None:
    """A span must not be the thing that fails a run.

    ``build_energon_sft_loader`` rejects a missing block with a message worth
    reaching, so the span cannot raise a ``KeyError`` ahead of it.
    """
    assert _energon_fanout_attributes({"max_input_seq_length": 128}) == {}


def test_sft_v2_owner_calls_carry_the_caller_trace_context() -> None:
    """Without this the loader's clear lands in its own root trace, once a step.

    ``_owner_call`` names its method with a runtime string, so the carrier is
    the only thing tying ``commit_sft_batch`` -- and the ``rl.data_plane.clear``
    it opens on the worker -- back to the step that asked for it.
    """
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
    """``setup_sft_v2`` measures on the driver, but the Logger lives in the actor.

    The prefix is the whole mechanism: ``rl.setup.duration`` is teed off it, so
    logging the same dict without it emits no metric at all.
    """
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
