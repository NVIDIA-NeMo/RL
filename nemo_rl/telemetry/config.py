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

from typing import Annotated, Literal

from pydantic import BaseModel, Field


class TelemetryConfig(BaseModel, extra="allow"):
    """OpenTelemetry / nemo-lens configuration.

    Telemetry activates only when ``enabled`` is true; otherwise every
    instrumentation site degrades to a ~0-cost no-op.

    Fields with a fixed set of valid values are typed so that a typo is
    rejected when the YAML is parsed, in every process, rather than surfacing
    later on a GPU node -- or not at all, when ``enabled`` is false and the
    driver returns before it validates anything.
    """

    enabled: bool = False
    """Master switch. When false, all instrumentation is a ~0-cost no-op."""

    service_name: str = "nemo-rl"
    """``service.name`` reported to the OTLP backend."""

    span_groups: str = "default"
    """Span-group spec: a preset (``default`` | ``per_step`` | ``all``) or a
    comma-separated list of individual group names (e.g.
    ``"default,generation,reward"``). See ``RLSpanGroup``."""

    export_strategy: Literal[
        "single_rank", "all_ranks", "sampled", "first_rank_per_node"
    ] = "single_rank"
    """Which ranks export. The driver always exports (it runs the training loop
    and the metrics logger); this governs the Ray worker ranks. nemo-lens owns
    the strategy registry, so the driver re-checks this name against it."""

    export_rank: Annotated[int, Field(ge=-1)] = -1
    """For ``single_rank``: which rank exports (``-1`` = last rank)."""

    export_sample_rate: Annotated[float, Field(ge=0.0, le=1.0)] = 1.0
    """For ``sampled``: fraction of worker ranks that export, in ``[0.0, 1.0]``.
    Also the sampling rate used by the span sampler when ``sampler_enabled`` is
    true. ``1.0`` means every rank considered by the strategy exports."""

    sampler_enabled: bool = False
    """Enable lens's rank-aware span sampler on the TracerProvider. It drops
    spans at the SDK level — cheaper than exporting and filtering downstream —
    and decides all-or-nothing per rank, from a hash of the rank against
    ``export_sample_rate``. This is a *second*, independent filter: a rank has
    to pass both it and ``export_strategy`` to emit anything. The driver and
    singleton actors are exempt from both, having no real rank."""

    traces_enabled: bool = True
    """Emit trace spans."""

    metrics_enabled: bool = True
    """Emit metric instruments (the ``rl.*`` gauges/histograms)."""

    logs_enabled: bool = False
    """Bridge Python logging to OTel logs (exported with trace correlation)."""

    exporter: Literal["otlp", "console"] = "otlp"
    """Exporter backend. The OTLP endpoint / headers / protocol come from the
    standard ``OTEL_EXPORTER_OTLP_*`` env vars, so any OTLP-compatible backend
    or an OpenTelemetry Collector works."""

    vllm_native_tracing: bool = False
    """Enable vLLM's own OTLP tracing inside generation workers (opt-in). vLLM's
    exporter is gRPC-only, so this needs a gRPC OTLP endpoint / collector — it
    does not ride an ``http/protobuf`` OTLP endpoint used by lens."""
