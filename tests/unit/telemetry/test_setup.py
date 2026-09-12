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


def test_config_rejects_an_unknown_export_strategy_at_parse_time():
    # The Literal is what a user actually hits: it fires wherever the YAML is
    # read, in every process, and regardless of `enabled` -- so a typo cannot
    # sit dormant in a disabled block until someone switches telemetry on.
    with pytest.raises(ValidationError):
        TelemetryConfig(enabled=True, export_strategy="single_ranks")


def test_init_driver_rejects_a_strategy_lens_does_not_register():
    # Second line of defence, for drift between the Literal above and lens's
    # registry rather than for user typos. It matters because the driver
    # overrides the strategy with _always_export, which bypasses the registry
    # lookup that would otherwise reject an unknown name. model_construct skips
    # validation, standing in for a Literal that has gained a value lens
    # dropped.
    pytest.importorskip("nemo.lens")
    tel = TelemetryConfig.model_construct(enabled=True, export_strategy="single_ranks")
    config = SimpleNamespace(telemetry=tel)
    with pytest.raises(ValueError, match="Unknown telemetry.export_strategy"):
        init_telemetry_driver(config, algorithm="grpo")
    # Again, because the init guard must not be set by a path that raised:
    # otherwise the second call reports success-with-nothing instead of the bug.
    with pytest.raises(ValueError, match="Unknown telemetry.export_strategy"):
        init_telemetry_driver(config, algorithm="grpo")


def test_config_rejects_out_of_range_export_bounds():
    # export_rank had no check anywhere before, and setup.py documents the
    # failure it causes: a rank that never matches silently mutes the actor.
    with pytest.raises(ValidationError):
        TelemetryConfig(export_rank=-2)
    with pytest.raises(ValidationError):
        TelemetryConfig(export_sample_rate=1.5)
    with pytest.raises(ValidationError):
        TelemetryConfig(exporter="consoel")


def test_init_driver_rejects_unknown_span_group():
    # Lens resolves span_groups only after installing the global tracer
    # provider, so without this check a typo kills the run mid-setup.
    pytest.importorskip("nemo.lens")
    config = SimpleNamespace(
        telemetry=TelemetryConfig(enabled=True, span_groups="per_stp")
    )
    with pytest.raises(ValueError):
        init_telemetry_driver(config, algorithm="grpo")


def test_documented_export_strategies_are_registered():
    pytest.importorskip("nemo.lens")
    from nemo.lens import registered_strategies

    documented = {"single_rank", "all_ranks", "sampled", "first_rank_per_node"}
    assert documented <= set(registered_strategies()), (
        "TelemetryConfig.export_strategy docstring lists a strategy lens does "
        "not register"
    )


@pytest.mark.parametrize(
    "strategy",
    ["single_rank", "all_ranks", "sampled", "first_rank_per_node"],
)
def test_init_driver_accepts_every_documented_export_strategy(strategy):
    # Exercises init end to end rather than just the registry: a documented
    # value has to survive the validation branch and lens's own setup. (Whether
    # the strategy *selects* this process is not in play here -- the driver
    # overrides it -- see the worker tests for that.)
    pytest.importorskip("nemo.lens")
    cfg = TelemetryConfig(enabled=True, exporter="console", export_strategy=strategy)
    handle = init_telemetry_driver(_FakeMasterConfig(cfg), "grpo")
    assert handle is not None
    assert handle.is_exporting


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


def test_init_driver_disables_the_rank_sampler(monkeypatch):
    # The driver is not one of the ranks the user asked to sample; leaving the
    # sampler on drops the training loop's own spans.
    pytest.importorskip("nemo.lens")
    captured = {}

    def _fake_setup_telemetry(config, **kwargs):
        captured["sampler_enabled"] = config.sampler_enabled
        return SimpleNamespace(is_exporting=True, tracer=None)

    monkeypatch.setattr("nemo.lens.setup_telemetry", _fake_setup_telemetry)
    cfg = TelemetryConfig(
        enabled=True,
        exporter="console",
        sampler_enabled=True,
        export_sample_rate=0.1,
    )

    assert init_telemetry_driver(_FakeMasterConfig(cfg), "grpo") is not None
    assert captured["sampler_enabled"] is False
    # Process-local: what ranked workers inherit must still say the user asked
    # for sampling.
    assert os.environ["NEMO_RL_OTEL_SAMPLER_ENABLED"] == "1"


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
    assert captured["rank"] == 3
    assert captured["world_size"] == 8
    assert captured["resource_attributes"] == {"rl.worker_group": "vllm_policy"}


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
    assert captured["rank"] == 0
    assert captured["world_size"] == 1


def test_init_worker_honours_export_strategy_by_default(monkeypatch):
    # The baseline for the always_export test below: a ranked worker must obey
    # whatever the user configured.
    pytest.importorskip("nemo.lens")
    monkeypatch.setenv("NEMO_RL_OTEL_ENABLED", "1")
    monkeypatch.setenv("NEMO_RL_OTEL_EXPORTER", "console")
    monkeypatch.setenv("NEMO_RL_OTEL_EXPORT_STRATEGY", "single_rank")
    monkeypatch.setenv("NEMO_RL_OTEL_EXPORT_RANK", "3")

    handle = init_telemetry_worker(rank=0, world_size=1)
    assert handle is not None
    assert not handle.is_exporting


def test_init_worker_always_export_overrides_export_rank(monkeypatch):
    """A singleton actor's synthetic rank must not be subject to the strategy.

    The trajectory collector reports ``rank=0, world_size=1`` because it is not
    a member of a ranked group. Applying ``export_rank: 3`` to that made-up rank
    would silently mute the actor -- taking every async rollout span with it.
    """
    pytest.importorskip("nemo.lens")
    monkeypatch.setenv("NEMO_RL_OTEL_ENABLED", "1")
    monkeypatch.setenv("NEMO_RL_OTEL_EXPORTER", "console")
    monkeypatch.setenv("NEMO_RL_OTEL_EXPORT_STRATEGY", "single_rank")
    monkeypatch.setenv("NEMO_RL_OTEL_EXPORT_RANK", "3")

    handle = init_telemetry_worker(rank=0, world_size=1, always_export=True)
    assert handle is not None
    assert handle.is_exporting


def test_init_worker_always_export_overrides_sample_rate(monkeypatch):
    # rank 0 hashes into the 0.785 bucket, so a low sample rate excludes it.
    pytest.importorskip("nemo.lens")
    monkeypatch.setenv("NEMO_RL_OTEL_ENABLED", "1")
    monkeypatch.setenv("NEMO_RL_OTEL_EXPORTER", "console")
    monkeypatch.setenv("NEMO_RL_OTEL_EXPORT_STRATEGY", "sampled")
    monkeypatch.setenv("NEMO_RL_OTEL_EXPORT_SAMPLE_RATE", "0.1")

    handle = init_telemetry_worker(rank=0, world_size=1, always_export=True)
    assert handle is not None
    assert handle.is_exporting


def test_init_worker_always_export_disables_the_rank_sampler(monkeypatch):
    """Exporting is not enough: the sampler drops spans before that decision.

    ``sampler_enabled`` installs lens's ``RankAwareSampler`` on the tracer
    provider, which filters on the same rank hash independently of
    ``export_strategy``. Left on, it discards every span of a synthetic rank 0
    while ``is_exporting`` still reports True -- telemetry that looks wired and
    produces nothing.
    """
    pytest.importorskip("nemo.lens")
    captured = {}

    def _fake_setup_telemetry(config, **kwargs):
        captured["sampler_enabled"] = config.sampler_enabled
        return SimpleNamespace(is_exporting=True)

    monkeypatch.setattr("nemo.lens.setup_telemetry", _fake_setup_telemetry)
    monkeypatch.setenv("NEMO_RL_OTEL_ENABLED", "1")
    monkeypatch.setenv("NEMO_RL_OTEL_SAMPLER_ENABLED", "1")
    monkeypatch.setenv("NEMO_RL_OTEL_EXPORT_SAMPLE_RATE", "0.1")

    assert init_telemetry_worker(rank=0, world_size=1, always_export=True) is not None
    assert captured["sampler_enabled"] is False


def test_init_worker_leaves_the_rank_sampler_alone_for_ranked_workers(monkeypatch):
    # The counterpart: a real group member is part of the population the user
    # asked to sample, so the sampler must stay on.
    pytest.importorskip("nemo.lens")
    captured = {}

    def _fake_setup_telemetry(config, **kwargs):
        captured["sampler_enabled"] = config.sampler_enabled
        return SimpleNamespace(is_exporting=True)

    monkeypatch.setattr("nemo.lens.setup_telemetry", _fake_setup_telemetry)
    monkeypatch.setenv("NEMO_RL_OTEL_ENABLED", "1")
    monkeypatch.setenv("NEMO_RL_OTEL_SAMPLER_ENABLED", "1")
    monkeypatch.setenv("NEMO_RL_OTEL_EXPORT_SAMPLE_RATE", "0.1")

    assert init_telemetry_worker(rank=2, world_size=8) is not None
    assert captured["sampler_enabled"] is True


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
        raise ValueError("unknown export_strategy 'typo'")

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


def test_init_driver_exports_despite_nonzero_export_rank():
    # export_rank selects among the Ray worker ranks; it must not switch off the
    # driver, whose rank=0/world_size=1 are synthetic.
    pytest.importorskip("nemo.lens")
    cfg = TelemetryConfig(
        enabled=True,
        exporter="console",
        export_strategy="single_rank",
        export_rank=3,
    )
    handle = init_telemetry_driver(_FakeMasterConfig(cfg), "grpo")
    assert handle is not None
    assert handle.is_exporting


def test_init_driver_exports_under_sampled_strategy(monkeypatch):
    # rank 0 hashes into the 0.785 bucket, so a lower sample rate would exclude
    # the driver without the export-strategy override.
    pytest.importorskip("nemo.lens")
    monkeypatch.setenv("NEMO_RL_OTEL_EXPORT_SAMPLE_RATE", "0.1")
    cfg = TelemetryConfig(enabled=True, exporter="console", export_strategy="sampled")
    handle = init_telemetry_driver(_FakeMasterConfig(cfg), "grpo")
    assert handle is not None
    assert handle.is_exporting


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
