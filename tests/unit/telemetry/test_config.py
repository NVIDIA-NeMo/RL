# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for TelemetryConfig and the YAML->env translation."""

import os

from nemo_rl.telemetry.config import TelemetryConfig
from nemo_rl.telemetry.setup import _ENV_FIELD_MAP, _config_to_env


def test_defaults():
    cfg = TelemetryConfig()
    assert cfg.enabled is False
    assert cfg.service_name == "nemo-rl"
    assert cfg.span_groups == "default"
    assert cfg.export_strategy == "single_rank"
    assert cfg.export_rank == -1
    # Must match the nemo-lens defaults, since propagating them is unconditional.
    assert cfg.export_sample_rate == 1.0
    assert cfg.sampler_enabled is False
    assert cfg.traces_enabled is True
    assert cfg.metrics_enabled is True
    assert cfg.logs_enabled is False
    assert cfg.exporter == "otlp"
    assert cfg.vllm_native_tracing is False


def test_extra_keys_allowed():
    cfg = TelemetryConfig(enabled=True, future_unknown_key="x")
    assert cfg.enabled is True


def test_config_to_env_translation():
    tel = TelemetryConfig(
        enabled=True,
        span_groups="per_step",
        export_rank=3,
        export_strategy="sampled",
        export_sample_rate=0.25,
        sampler_enabled=True,
        vllm_native_tracing=True,
        exporter="console",
        service_name="my-rl",
    )
    _config_to_env(tel)
    assert os.environ["NEMO_RL_OTEL_ENABLED"] == "1"
    assert os.environ["NEMO_RL_OTEL_SPAN_GROUPS"] == "per_step"
    assert os.environ["NEMO_RL_OTEL_EXPORT_RANK"] == "3"
    assert os.environ["NEMO_RL_OTEL_EXPORT_STRATEGY"] == "sampled"
    assert os.environ["NEMO_RL_OTEL_EXPORT_SAMPLE_RATE"] == "0.25"
    assert os.environ["NEMO_RL_OTEL_SAMPLER_ENABLED"] == "1"
    assert os.environ["NEMO_RL_OTEL_VLLM_NATIVE_TRACING"] == "1"
    assert os.environ["NEMO_RL_OTEL_EXPORTER"] == "console"
    assert os.environ["OTEL_SERVICE_NAME"] == "my-rl"


def test_disabled_translates_to_zero():
    _config_to_env(TelemetryConfig(enabled=False))
    assert os.environ["NEMO_RL_OTEL_ENABLED"] == "0"


def test_env_wins_over_yaml():
    os.environ["NEMO_RL_OTEL_SPAN_GROUPS"] = "all"
    _config_to_env(TelemetryConfig(enabled=True, span_groups="per_step"))
    # setdefault must not overwrite a pre-existing env var.
    assert os.environ["NEMO_RL_OTEL_SPAN_GROUPS"] == "all"


def test_env_field_map_fields_exist_on_config():
    cfg = TelemetryConfig()
    for field in _ENV_FIELD_MAP:
        assert hasattr(cfg, field), field


def test_every_config_field_propagates_to_workers():
    # A field that exists on the config but is missing from _ENV_FIELD_MAP is
    # settable in YAML yet never reaches a worker. service_name is the one
    # exception: it maps onto the unprefixed OTEL_SERVICE_NAME.
    declared = set(TelemetryConfig.model_fields) - {"service_name"}
    assert declared == set(_ENV_FIELD_MAP)
