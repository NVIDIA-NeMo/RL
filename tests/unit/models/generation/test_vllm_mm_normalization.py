# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Image normalization defaults across old and new engine constructor APIs."""

import sys
import types

import pytest

from nemo_rl.models.generation.vllm.vllm_worker import _set_mm_normalization_default


class OldEngineArgs:
    def __init__(self, model="unused"):
        self.model = model


class NewEngineArgs:
    def __init__(self, model="unused", mm_device_do_normalize=True):
        self.model = model
        self.mm_device_do_normalize = mm_device_do_normalize


def install_engine_args(monkeypatch, engine_args):
    module = types.ModuleType("vllm.engine.arg_utils")
    module.EngineArgs = engine_args
    monkeypatch.setitem(sys.modules, "vllm.engine.arg_utils", module)


@pytest.mark.parametrize("engine_args", [OldEngineArgs, NewEngineArgs])
def test_default_constructs_with_either_engine_api(monkeypatch, engine_args):
    install_engine_args(monkeypatch, engine_args)
    kwargs = {"model": "test-model"}
    _set_mm_normalization_default(kwargs)
    result = engine_args(**kwargs)
    assert result.model == "test-model"
    if engine_args is NewEngineArgs:
        assert result.mm_device_do_normalize is False
    else:
        assert "mm_device_do_normalize" not in kwargs


@pytest.mark.parametrize("value", [True, False])
def test_explicit_normalization_setting_is_preserved(monkeypatch, value):
    install_engine_args(monkeypatch, NewEngineArgs)
    kwargs = {"mm_device_do_normalize": value}
    _set_mm_normalization_default(kwargs)
    assert NewEngineArgs(**kwargs).mm_device_do_normalize is value


def test_unsupported_explicit_setting_is_not_silently_dropped(monkeypatch):
    install_engine_args(monkeypatch, OldEngineArgs)
    kwargs = {"mm_device_do_normalize": True}
    _set_mm_normalization_default(kwargs)
    with pytest.raises(TypeError, match="mm_device_do_normalize"):
        OldEngineArgs(**kwargs)
