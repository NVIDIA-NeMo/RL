"""Fixtures for NeMo-RL telemetry unit tests.

Resets global OpenTelemetry providers, the nemo-lens init guard, span-group
state, the process-global telemetry handle, and the ``NEMO_RL_OTEL_*`` /
``OTEL_SERVICE_NAME`` env vars before and after each test so nothing leaks. The
nemo-lens resets are guarded so this suite still imports when lens is absent.
"""

import os

import pytest


def _clear_telemetry_env() -> None:
    for key in list(os.environ):
        if key.startswith(("NEMO_RL_OTEL", "NEMO_LENS")) or key == "OTEL_SERVICE_NAME":
            del os.environ[key]


def _reset_rl_telemetry() -> None:
    import nemo_rl.telemetry.setup as setup_mod

    setup_mod._TELEMETRY_HANDLE = None
    setup_mod._TELEMETRY_INITIALISED = False


def _reset_otel_and_lens() -> None:
    try:
        import opentelemetry.metrics._internal as _metrics_mod
        import opentelemetry.trace as _trace_mod
        from opentelemetry.util._once import Once

        _trace_mod._TRACER_PROVIDER = None
        _trace_mod._TRACER_PROVIDER_SET_ONCE = Once()
        _metrics_mod._METER_PROVIDER = None
        _metrics_mod._METER_PROVIDER_SET_ONCE = Once()
    except Exception:
        pass
    try:
        import nemo.lens.handle as _handle_mod
        from nemo.lens.state import set_enabled_span_groups

        _handle_mod._INITIALIZED = False
        set_enabled_span_groups(frozenset())
    except ImportError:
        pass


@pytest.fixture(autouse=True)
def _reset_telemetry_state():
    _clear_telemetry_env()
    _reset_otel_and_lens()
    _reset_rl_telemetry()
    yield
    _reset_otel_and_lens()
    _reset_rl_telemetry()
    _clear_telemetry_env()
