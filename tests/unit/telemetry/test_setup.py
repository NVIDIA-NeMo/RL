"""Tests for the telemetry setup module (driver init, resource attrs, digging)."""

import pytest

from nemo_rl.telemetry.config import TelemetryConfig
from nemo_rl.telemetry.setup import (
    _build_resource_attributes,
    _dig,
    get_telemetry,
    init_telemetry_driver,
)


class _FakeMasterConfig:
    def __init__(self, telemetry=None):
        self.telemetry = telemetry
        self.policy = {
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
    attrs = _build_resource_attributes(_FakeMasterConfig(), "grpo", rank=0, world_size=1)
    assert attrs["rl.algorithm"] == "grpo"
    assert attrs["rl.model"] == "org/Model-1B"
    assert attrs["nemo.precision"] == "bfloat16"
    assert attrs["dl.tensor_parallel.size"] == 2
    assert attrs["dl.pipeline_parallel.size"] == 1


def test_init_driver_returns_none_when_disabled():
    handle = init_telemetry_driver(
        _FakeMasterConfig(TelemetryConfig(enabled=False)), "grpo"
    )
    assert handle is None
    assert get_telemetry() is None


def test_init_driver_returns_none_when_no_telemetry_block():
    handle = init_telemetry_driver(_FakeMasterConfig(telemetry=None), "grpo")
    assert handle is None


def test_init_driver_enabled_is_idempotent():
    pytest.importorskip("nemo.lens")
    cfg = TelemetryConfig(enabled=True, span_groups="default", exporter="console")
    handle1 = init_telemetry_driver(_FakeMasterConfig(cfg), "grpo")
    assert handle1 is not None
    assert handle1.is_exporting
    assert get_telemetry() is handle1
    # Second call must not re-init or raise; returns the same handle.
    handle2 = init_telemetry_driver(
        _FakeMasterConfig(TelemetryConfig(enabled=True)), "grpo"
    )
    assert handle2 is handle1
