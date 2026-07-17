# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Telemetry configuration schema for NeMo-RL.

The ``telemetry:`` block of a run config. :mod:`nemo_rl.telemetry.setup`
translates it into ``NEMO_RL_OTEL_*`` environment variables on the driver
*before* ``init_ray()``, so every Ray worker inherits the same settings via the
Ray ``runtime_env``. Raw ``NEMO_RL_OTEL_*`` / ``OTEL_EXPORTER_OTLP_*`` env vars
always win over these YAML values (they are applied with ``setdefault``).

This module imports only ``pydantic`` — it never requires nemo-lens, so it is
safe to import unconditionally from the algorithm ``MasterConfig`` classes.
"""

from __future__ import annotations

from pydantic import BaseModel


class TelemetryConfig(BaseModel, extra="allow"):
    """OpenTelemetry / nemo-lens configuration.

    Telemetry is optional: it activates only when ``enabled`` is true *and*
    nemo-lens is installed (``uv sync --extra telemetry``). When either is
    absent, every instrumentation site degrades to a ~0-cost no-op.
    """

    enabled: bool = False
    """Master switch. When false, all instrumentation is a ~0-cost no-op."""

    service_name: str = "nemo-rl"
    """``service.name`` reported to the OTLP backend."""

    span_groups: str = "default"
    """Span-group spec: a preset (``default`` | ``per_step`` | ``all``) or a
    comma-separated list of individual group names (e.g.
    ``"default,generation,reward"``). See ``RLSpanGroup``."""

    export_strategy: str = "single_rank"
    """Which ranks export: ``single_rank`` | ``all_ranks`` | ``sampled`` |
    ``first_rank_per_node``. The driver always exports (it runs the training
    loop and the metrics logger); this governs the Ray worker ranks."""

    export_rank: int = -1
    """For ``single_rank``: which rank exports (``-1`` = last rank)."""

    traces_enabled: bool = True
    """Emit trace spans."""

    metrics_enabled: bool = True
    """Emit metric instruments (the ``rl.*`` gauges/histograms)."""

    logs_enabled: bool = False
    """Bridge Python logging to OTel logs (exported with trace correlation)."""

    exporter: str = "otlp"
    """Exporter backend: ``otlp`` | ``console``. The OTLP endpoint / headers /
    protocol come from the standard ``OTEL_EXPORTER_OTLP_*`` env vars, so any
    OTLP-compatible backend or an OpenTelemetry Collector works."""

    vllm_native_tracing: bool = False
    """Enable vLLM's own OTLP tracing inside generation workers (opt-in). vLLM's
    exporter is gRPC-only, so this needs a gRPC OTLP endpoint / collector — it
    does not ride an ``http/protobuf`` OTLP endpoint used by lens."""
