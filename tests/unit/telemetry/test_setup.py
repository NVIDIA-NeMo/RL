# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the telemetry setup module (driver init, resource attrs, digging)."""

import logging
import os
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from nemo_rl.telemetry.config import TelemetryConfig
from nemo_rl.telemetry.setup import (
    _build_resource_attributes,
    _dig,
    _worker_resource_attributes,
    get_telemetry_handle,
    init_telemetry_driver,
    init_telemetry_worker,
    shutdown_telemetry,
    telemetry_enabled_in_env,
    traced_worker_init,
    vllm_native_tracing_requested,
)


class _FakeMasterConfig:
    def __init__(self, telemetry=None, policy=None):
        self.telemetry = telemetry
        self.policy = policy or {
            "model_name": "org/Model-1B",
            "precision": "bfloat16",
            "megatron_cfg": {
                "tensor_model_parallel_size": 2,
                "pipeline_model_parallel_size": 1,
            },
        }


def test_dig_handles_dicts_objects_and_missing():
    assert _dig({"a": {"b": 7}}, "a", "b") == 7
    assert _dig({"a": {}}, "a", "missing") is None
    assert _dig(None, "a") is None

    class Node:
        x = {"y": 9}

    assert _dig(Node(), "x", "y") == 9


def test_build_resource_attributes():
    attrs = _build_resource_attributes(_FakeMasterConfig(), "grpo")
    assert attrs["rl.algorithm"] == "grpo"
    assert attrs["rl.model"] == "org/Model-1B"
    assert attrs["nemo.precision"] == "bfloat16"
    assert attrs["dl.tensor_parallel.size"] == 2
    assert attrs["dl.pipeline_parallel.size"] == 1


def test_build_resource_attributes_dtensor_tp():
    cfg = _FakeMasterConfig(
        policy={
            "model_name": "org/Model-1B",
            "precision": "bfloat16",
            "dtensor_cfg": {"tensor_parallel_size": 4},
        }
    )
    attrs = _build_resource_attributes(cfg, "grpo")
    assert attrs["dl.tensor_parallel.size"] == 4
    assert "dl.pipeline_parallel.size" not in attrs


def test_init_driver_returns_none_when_disabled():
    handle = init_telemetry_driver(
        _FakeMasterConfig(TelemetryConfig(enabled=False)), "grpo"
    )
    assert handle is None
    assert get_telemetry_handle() is None


def test_init_driver_returns_none_when_no_telemetry_block():
    handle = init_telemetry_driver(_FakeMasterConfig(telemetry=None), "grpo")
    assert handle is None


def test_init_worker_returns_none_when_disabled():
    # No NEMO_RL_OTEL_ENABLED / NEMO_LENS_ENABLED — every actor takes this path.
    handle = init_telemetry_worker()
    assert handle is None
    assert get_telemetry_handle() is None


def test_config_rejects_an_unknown_exporter_at_parse_time():
    # The Literal is what a user actually hits: it fires wherever the YAML is
    # read, in every process, and regardless of `enabled` -- so a typo cannot
    # sit dormant in a disabled block until someone switches telemetry on.
    with pytest.raises(ValidationError):
        TelemetryConfig(exporter="consoel")


def test_config_has_no_rank_filtering_fields():
    """Rank filtering left with lens; the config must not imply it still works.

    Lens deleted its export-strategy registry and rank-aware sampler, so a
    ``telemetry.export_strategy`` in a YAML now reaches nothing. ``extra="allow"``
    means such a key is silently accepted rather than rejected, which is exactly
    how a field that quietly does nothing survives -- so the guard is here
    instead.
    """
    fields = set(TelemetryConfig.model_fields)
    assert not fields & {
        "export_strategy",
        "export_rank",
        "export_sample_rate",
        "sampler_enabled",
    }


def test_init_driver_warns_about_an_unknown_span_group_without_raising(caplog):
    """A typo has to be reported, but must not end the run.

    Lens returns an unrecognised spec entry as ``pending`` rather than raising,
    because a registry is per-process while a spec is job-wide. NeMo-RL registers
    everything it emits at import, so on the driver a pending entry really is a
    typo -- and the user would otherwise get silence and an empty trace.
    """
    pytest.importorskip("nemo.lens")
    config = SimpleNamespace(
        telemetry=TelemetryConfig(
            enabled=True, exporter="console", span_groups="per_stp"
        )
    )
    with caplog.at_level(logging.WARNING, logger="nemo_rl.telemetry.setup"):
        handle = init_telemetry_driver(config, algorithm="grpo")

    assert handle is not None
    assert "per_stp" in caplog.text


def test_init_driver_does_not_warn_for_a_valid_spec(caplog):
    pytest.importorskip("nemo.lens")
    cfg = TelemetryConfig(enabled=True, exporter="console", span_groups="per_step")
    with caplog.at_level(logging.WARNING, logger="nemo_rl.telemetry.setup"):
        assert init_telemetry_driver(_FakeMasterConfig(cfg), "grpo") is not None

    assert "match no registered" not in caplog.text


def test_init_driver_tags_itself_as_rank_zero_of_one(monkeypatch):
    """Lens has no rank of its own, so the driver has to state one.

    It is a singleton rather than a member of a group, and rank 0 of 1 is both
    the honest description and what silences lens's warning about a process that
    cannot be identified by rank downstream.
    """
    pytest.importorskip("nemo.lens")
    from nemo.lens.semconv import NV_DL_RANK, NV_DL_WORLD_SIZE

    captured = {}

    def _fake_setup_telemetry(config, **kwargs):
        captured.update(kwargs.get("resource_attributes") or {})
        return SimpleNamespace(is_exporting=True, tracer=None)

    monkeypatch.setattr("nemo.lens.setup_telemetry", _fake_setup_telemetry)
    cfg = TelemetryConfig(enabled=True, exporter="console")

    assert init_telemetry_driver(_FakeMasterConfig(cfg), "grpo") is not None
    assert captured[NV_DL_RANK] == 0
    assert captured[NV_DL_WORLD_SIZE] == 1
    # The run-identifying attributes still have to survive the merge.
    assert captured["rl.algorithm"] == "grpo"


def test_init_driver_publishes_default_service_name_to_env():
    # Workers rebuild their config from the environment, so a service name only
    # assigned on the config object would leave them defaulting to lens's
    # "nemo" and split one run across two services.
    pytest.importorskip("nemo.lens")
    # Empty rather than the field default, so _config_to_env skips it and only
    # the branch under test can populate the env var.
    cfg = TelemetryConfig(enabled=True, exporter="console", service_name="")
    handle = init_telemetry_driver(_FakeMasterConfig(cfg), "grpo")
    assert handle is not None
    assert os.environ["OTEL_SERVICE_NAME"] == "nemo-rl"


def test_init_driver_keeps_user_service_name(monkeypatch):
    pytest.importorskip("nemo.lens")
    monkeypatch.setenv("OTEL_SERVICE_NAME", "my-service")
    handle = init_telemetry_driver(
        _FakeMasterConfig(TelemetryConfig(enabled=True, exporter="console")), "grpo"
    )
    assert handle is not None
    assert os.environ["OTEL_SERVICE_NAME"] == "my-service"


def test_init_driver_publishes_env_even_when_telemetry_is_disabled():
    """``_config_to_env`` runs before the ``enabled`` early return.

    Deliberate -- the env is the only channel to a Ray worker, so it must not
    depend on which branch the driver took. It is also why anything reading
    these variables has to check the master switch itself; see
    :func:`telemetry_enabled_in_env`.
    """
    cfg = TelemetryConfig(enabled=False, span_groups="per_step")

    assert init_telemetry_driver(_FakeMasterConfig(cfg), "grpo") is None

    assert os.environ["NEMO_RL_OTEL_SPAN_GROUPS"] == "per_step"
    assert not telemetry_enabled_in_env()


def test_vllm_native_tracing_needs_the_master_switch(monkeypatch):
    # The field is exported regardless of `enabled`, so reading it alone would
    # leave per-request vLLM tracing on for a run that turned telemetry off.
    monkeypatch.setenv("NEMO_RL_OTEL_VLLM_NATIVE_TRACING", "1")
    monkeypatch.setenv("NEMO_RL_OTEL_ENABLED", "0")

    assert vllm_native_tracing_requested()
    assert not telemetry_enabled_in_env()

    monkeypatch.setenv("NEMO_RL_OTEL_ENABLED", "1")
    assert telemetry_enabled_in_env()


def test_worker_resource_attributes_carries_worker_group(monkeypatch):
    monkeypatch.setenv("NRL_WORKER_GROUP", "vllm_policy")
    assert _worker_resource_attributes(None) == {"rl.worker_group": "vllm_policy"}


def test_worker_resource_attributes_without_group_env():
    # Workers not created by RayWorkerGroup simply omit the attribute.
    assert _worker_resource_attributes(None) == {}


def test_worker_resource_attributes_explicit_extra_wins(monkeypatch):
    monkeypatch.setenv("NRL_WORKER_GROUP", "lm_policy")
    attrs = _worker_resource_attributes({"rl.worker_group": "override", "k": 1})
    assert attrs == {"rl.worker_group": "override", "k": 1}


def test_init_worker_sets_worker_group_attribute(monkeypatch):
    pytest.importorskip("nemo.lens")
    captured = {}

    def _fake_setup_telemetry(config, **kwargs):
        captured.update(kwargs)
        return SimpleNamespace(is_exporting=True)

    monkeypatch.setattr("nemo.lens.setup_telemetry", _fake_setup_telemetry)
    monkeypatch.setenv("NEMO_RL_OTEL_ENABLED", "1")
    monkeypatch.setenv("NRL_WORKER_GROUP", "vllm_policy")
    monkeypatch.setenv("RANK", "3")
    monkeypatch.setenv("WORLD_SIZE", "8")

    handle = init_telemetry_worker()
    assert handle is not None
    # Rank reaches lens as a resource attribute now, alongside the group name:
    # RANK is group-local, so the two are only useful together.
    assert captured["resource_attributes"] == {
        "rl.worker_group": "vllm_policy",
        "nv.dl.rank": 3,
        "nv.dl.world_size": 8,
    }


def test_init_worker_explicit_rank_overrides_env(monkeypatch):
    """Singleton actors pass their own rank instead of reading the env.

    The trajectory collector's runtime_env is a copy of the driver's
    environment, so a ``RANK`` inherited from the launcher would otherwise
    decide whether the collector exports.
    """
    pytest.importorskip("nemo.lens")
    captured = {}

    def _fake_setup_telemetry(config, **kwargs):
        captured.update(kwargs)
        return SimpleNamespace(is_exporting=True)

    monkeypatch.setattr("nemo.lens.setup_telemetry", _fake_setup_telemetry)
    monkeypatch.setenv("NEMO_RL_OTEL_ENABLED", "1")
    monkeypatch.setenv("RANK", "5")
    monkeypatch.setenv("WORLD_SIZE", "8")

    assert init_telemetry_worker(rank=0, world_size=1) is not None
    assert captured["resource_attributes"]["nv.dl.rank"] == 0
    assert captured["resource_attributes"]["nv.dl.world_size"] == 1


def test_every_worker_that_gets_here_exports(monkeypatch):
    """No rank is filtered out any more, so every initialised worker exports.

    Lens dropped both of its rank filters. A worker that would previously have
    been muted by ``export_rank`` or a low sample rate now exports and labels
    itself with ``nv.dl.rank`` instead, leaving the narrowing to the collector.
    """
    pytest.importorskip("nemo.lens")
    monkeypatch.setenv("NEMO_RL_OTEL_ENABLED", "1")
    monkeypatch.setenv("NEMO_RL_OTEL_EXPORTER", "console")

    handle = init_telemetry_worker(rank=7, world_size=8)
    assert handle is not None
    assert handle.is_exporting


def test_init_worker_records_a_ranked_worker_by_rank(monkeypatch):
    """The rank label is the only thing that makes one worker's spans findable.

    With no rank filtering left, a fleet exports as one undifferentiated stream
    unless every process says which rank it is -- so this attribute is what
    replaced the export strategy, not a nice-to-have.
    """
    pytest.importorskip("nemo.lens")
    captured = {}

    def _fake_setup_telemetry(config, **kwargs):
        captured.update(kwargs)
        return SimpleNamespace(is_exporting=True)

    monkeypatch.setattr("nemo.lens.setup_telemetry", _fake_setup_telemetry)
    monkeypatch.setenv("NEMO_RL_OTEL_ENABLED", "1")

    assert init_telemetry_worker(rank=2, world_size=8) is not None
    assert captured["resource_attributes"]["nv.dl.rank"] == 2
    assert captured["resource_attributes"]["nv.dl.world_size"] == 8


def test_init_worker_survives_a_failing_setup(monkeypatch, caplog):
    # A worker must not take a training run down over optional observability,
    # but the reason has to reach the log or the run is silently half-traced.
    pytest.importorskip("nemo.lens")

    def _boom(config, **kwargs):
        raise ImportError("OpenTelemetry SDK is required for telemetry export")

    monkeypatch.setattr("nemo.lens.setup_telemetry", _boom)
    monkeypatch.setenv("NEMO_RL_OTEL_ENABLED", "1")

    with caplog.at_level(logging.WARNING):
        assert init_telemetry_worker() is None
    assert "OpenTelemetry SDK is required" in caplog.text


def test_init_worker_stays_quiet_when_telemetry_is_disabled(monkeypatch, caplog):
    # Nothing to warn about: the user did not ask for telemetry.
    pytest.importorskip("nemo.lens")
    monkeypatch.delenv("NEMO_RL_OTEL_ENABLED", raising=False)
    monkeypatch.delenv("NEMO_LENS_ENABLED", raising=False)

    with caplog.at_level(logging.WARNING):
        assert init_telemetry_worker() is None
    assert caplog.text == ""


def test_init_worker_never_raises_on_setup_failure(monkeypatch):
    # A worker must not fail a training run over optional observability.
    pytest.importorskip("nemo.lens")

    def _boom(config, **kwargs):
        raise ValueError("unknown exporter 'typo'")

    monkeypatch.setattr("nemo.lens.setup_telemetry", _boom)
    monkeypatch.setenv("NEMO_RL_OTEL_ENABLED", "1")

    assert init_telemetry_worker() is None
    assert get_telemetry_handle() is None


def test_init_driver_enabled_is_idempotent():
    pytest.importorskip("nemo.lens")
    cfg = TelemetryConfig(enabled=True, span_groups="default", exporter="console")
    handle1 = init_telemetry_driver(_FakeMasterConfig(cfg), "grpo")
    assert handle1 is not None
    assert handle1.is_exporting
    assert get_telemetry_handle() is handle1
    # Second call must not re-init or raise; returns the same handle.
    handle2 = init_telemetry_driver(
        _FakeMasterConfig(TelemetryConfig(enabled=True)), "grpo"
    )
    assert handle2 is handle1


def test_init_driver_ignores_a_stale_rank_filtering_key(monkeypatch):
    """An old YAML must keep working, not fail on a key that no longer exists.

    ``export_strategy`` and friends were real fields until lens removed the
    machinery behind them, so configs in the wild still carry them.
    ``extra="allow"`` is what lets those parse, and ``_config_to_env`` only
    projects fields the map names -- so a leftover key reaches nothing rather
    than being pushed at lens, which would now reject it.
    """
    pytest.importorskip("nemo.lens")
    monkeypatch.delenv("NEMO_RL_OTEL_EXPORT_STRATEGY", raising=False)
    cfg = TelemetryConfig(
        enabled=True,
        exporter="console",
        export_strategy="single_rank",
        export_rank=3,
    )
    handle = init_telemetry_driver(_FakeMasterConfig(cfg), "grpo")
    assert handle is not None
    assert handle.is_exporting
    assert "NEMO_RL_OTEL_EXPORT_STRATEGY" not in os.environ


def test_shutdown_is_a_noop_without_telemetry():
    # The common case by far: every run with no `telemetry:` block reaches the
    # driver's `finally` and each actor's shutdown hook with no handle.
    import nemo_rl.telemetry.setup as setup_mod

    setup_mod._TELEMETRY_HANDLE = None
    shutdown_telemetry()


def test_shutdown_clears_the_handle_so_a_second_call_cannot_reach_a_dead_provider():
    # Both the driver's `finally` and the worker shutdown hooks call this, and
    # flushing an already-shut-down provider is what logs "Already shutdown".
    import nemo_rl.telemetry.setup as setup_mod

    calls = []
    setup_mod._TELEMETRY_HANDLE = SimpleNamespace(
        shutdown=lambda timeout_ms: calls.append(timeout_ms)
    )

    shutdown_telemetry(timeout_ms=1234)
    shutdown_telemetry(timeout_ms=1234)

    assert calls == [1234]
    assert get_telemetry_handle() is None


def test_shutdown_swallows_a_failing_flush():
    # It runs in a `finally`, so raising here would replace whatever exception
    # actually ended the run -- and still has to clear the handle.
    import nemo_rl.telemetry.setup as setup_mod

    def _boom(timeout_ms):
        raise RuntimeError("exporter is gone")

    setup_mod._TELEMETRY_HANDLE = SimpleNamespace(shutdown=_boom)

    shutdown_telemetry()

    assert get_telemetry_handle() is None


def test_a_traced_worker_init_runs_the_constructor_with_telemetry_off(monkeypatch):
    """Telemetry off is the common case, and must be transparent.

    A worker constructor builds the model; wrapping it must not change what it
    does, what it returns, or -- since ``nemo.lens`` may be absent -- reach the
    span imports at all.
    """
    monkeypatch.setenv("NEMO_RL_OTEL_ENABLED", "0")
    calls = []

    class Worker:
        @traced_worker_init("rl.policy.load_model", **{"rl.backend": "megatron"})
        def __init__(self, name):
            calls.append(name)
            self.name = name

    worker = Worker("policy")

    assert worker.name == "policy"
    assert calls == ["policy"]


def test_a_traced_worker_init_initialises_telemetry_before_the_body(monkeypatch):
    """The span cannot open before the provider exists.

    A worker calls init_telemetry_worker() a few lines into __init__, which is
    too late for a decorator to have opened a span, so the decorator hoists it.
    """
    order = []

    def _fake_init():
        order.append("telemetry")
        return None  # disabled: the body still has to run

    monkeypatch.setattr("nemo_rl.telemetry.setup.init_telemetry_worker", _fake_init)

    class Worker:
        @traced_worker_init("rl.value.load_model")
        def __init__(self):
            order.append("body")

    Worker()

    assert order == ["telemetry", "body"]


def test_a_traced_worker_init_preserves_the_wrapped_signature():
    class Worker:
        @traced_worker_init("rl.policy.load_model")
        def __init__(self, a, b=2):
            """Build the model."""
            self.total = a + b

    assert Worker.__init__.__doc__ == "Build the model."
    assert Worker(1).total == 3
    assert Worker(1, b=5).total == 6
