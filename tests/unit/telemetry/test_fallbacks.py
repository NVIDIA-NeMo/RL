"""Tests that the fallback shims are pure no-ops when nemo-lens is absent.

Each test forces the ``except ImportError`` branch of ``_fallbacks`` by blocking
``nemo.lens*`` imports and re-importing the module, so the behaviour is verified
regardless of whether nemo-lens happens to be installed in the test env.
"""

import builtins
import importlib
import sys

import pytest


def _import_fallbacks_without_lens(monkeypatch):
    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name.startswith("nemo.lens"):
            raise ImportError("nemo.lens blocked for test")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    sys.modules.pop("nemo_rl.telemetry._fallbacks", None)
    return importlib.import_module("nemo_rl.telemetry._fallbacks")


@pytest.fixture(autouse=True)
def _restore_real_fallbacks():
    yield
    sys.modules.pop("nemo_rl.telemetry._fallbacks", None)
    importlib.import_module("nemo_rl.telemetry._fallbacks")


def test_managed_span_yields_none(monkeypatch):
    fb = _import_fallbacks_without_lens(monkeypatch)
    with fb.managed_span("reward", "rl.x", some_attr=1) as span:
        assert span is None


def test_span_cm_yields_none(monkeypatch):
    fb = _import_fallbacks_without_lens(monkeypatch)
    with fb.span_cm("rl.x", record_exception=True, attr=2) as span:
        assert span is None


def test_trace_fn_returns_function_unchanged(monkeypatch):
    fb = _import_fallbacks_without_lens(monkeypatch)

    @fb.trace_fn("job", "rl.job")
    def add_one(x):
        return x + 1

    assert add_one(41) == 42


def test_is_span_group_enabled_false(monkeypatch):
    fb = _import_fallbacks_without_lens(monkeypatch)
    assert fb.is_span_group_enabled("reward") is False


def test_safe_set_span_attributes_does_not_raise(monkeypatch):
    fb = _import_fallbacks_without_lens(monkeypatch)
    fb.safe_set_span_attributes(None, {"a": 1})  # must not raise
