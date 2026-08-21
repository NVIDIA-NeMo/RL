# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Fixtures for NeMo-RL telemetry unit tests.

Resets global OpenTelemetry providers, the nemo-lens init guard, span-group
state, the process-global telemetry handle, and the ``NEMO_RL_OTEL_*`` /
``OTEL_SERVICE_NAME`` / ``NRL_WORKER_GROUP`` env vars before and after each test
so nothing leaks.
"""

import ast
import os
from pathlib import Path

import pytest


def string_constants(node: ast.AST | None) -> set[str]:
    """The string literals in a list/set/tuple display, ignoring anything else."""
    return {
        elt.value
        for elt in getattr(node, "elts", [])
        if isinstance(elt, ast.Constant) and isinstance(elt.value, str)
    }


def algorithms_utils_categories(*names: str) -> dict[str, set[str]]:
    """Read named category collections out of ``nemo_rl/algorithms/utils.py``.

    That module owns the canonical efficiency-category lists but imports torch,
    which is far too heavy for this suite — and several telemetry constants
    deliberately restate those lists so they stay importable without the
    training stack. Parsing the source is what keeps the copies honest.

    Handles list, set, and ``frozenset({...})`` literals; string elements only.
    """
    source = Path(__file__).resolve().parents[3] / "nemo_rl" / "algorithms" / "utils.py"
    wanted = set(names)
    found: dict[str, set[str]] = {}
    for node in ast.parse(source.read_text()).body:
        if not isinstance(node, ast.AnnAssign | ast.Assign):
            continue
        targets = node.targets if isinstance(node, ast.Assign) else [node.target]
        for target in targets:
            if not isinstance(target, ast.Name) or target.id not in wanted:
                continue
            value = node.value
            # frozenset({...}) / set([...]) wrap the literal in a call.
            if isinstance(value, ast.Call) and value.args:
                value = value.args[0]
            found[target.id] = string_constants(value)
    assert wanted == set(found), f"could not locate {wanted - set(found)} in {source}"
    return found


def _clear_telemetry_env() -> None:
    for key in list(os.environ):
        if key.startswith(("NEMO_RL_OTEL", "NEMO_LENS")) or key in (
            "OTEL_SERVICE_NAME",
            "NRL_WORKER_GROUP",
        ):
            del os.environ[key]


def _reset_rl_telemetry() -> None:
    import nemo_rl.telemetry.setup as setup_mod

    setup_mod._TELEMETRY_HANDLE = None
    setup_mod._TELEMETRY_INITIALISED = False


def _reset_otel_and_lens() -> None:
    # Before the providers are dropped: several tests build real console
    # providers, and simply nulling the globals would leave their exporter
    # threads and periodic metric readers running for the rest of the session.
    try:
        from nemo_rl.telemetry.setup import shutdown_telemetry

        shutdown_telemetry()
    except Exception:
        pass
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
    import nemo.lens.handle as _handle_mod
    from nemo.lens.state import set_enabled_span_groups

    _handle_mod._INITIALIZED = False
    set_enabled_span_groups(frozenset())


@pytest.fixture(autouse=True)
def _reset_telemetry_state():
    _clear_telemetry_env()
    _reset_otel_and_lens()
    _reset_rl_telemetry()
    yield
    _reset_otel_and_lens()
    _reset_rl_telemetry()
    _clear_telemetry_env()
