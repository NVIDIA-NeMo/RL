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

from typing import Literal

from pydantic import BaseModel


class TelemetryConfig(BaseModel, extra="allow"):
    """OpenTelemetry / nemo-lens configuration.

    Telemetry activates only when ``enabled`` is true; otherwise every
    instrumentation site degrades to a ~0-cost no-op.

    Fields with a fixed set of valid values are typed so that a typo is
    rejected when the YAML is parsed, in every process, rather than surfacing
    later on a GPU node -- or not at all, when ``enabled`` is false and the
    driver returns before it validates anything.
    """

    enabled: bool = True
    """Master switch. When false, all instrumentation is a ~0-cost no-op.

    On by default, but the driver turns it back off when no OTLP endpoint is
    configured -- see ``_otlp_endpoint_configured`` in ``setup.py``. So a run on
    a cluster whose ``OTEL_EXPORTER_OTLP_*`` environment is already set gets
    telemetry with no YAML at all, while a laptop or CI run that would otherwise
    export into a refused connection stays a no-op. Setting this false here (or
    via ``NEMO_RL_OTEL_ENABLED``) still wins over both."""

    service_name: str = "nemo-rl"
    """``service.name`` reported to the OTLP backend."""

    span_groups: str = "per_step"
    """Span-group spec: a preset (``default`` | ``per_step`` | ``all``) or a
    comma-separated list of individual group names (e.g.
    ``"default,generation,reward"``). See ``RLSpanGroup``.

    ``per_step`` is the default because the ``step`` group is what makes a run
    legible: without it the phases inside a step are the only thing exported and
    nothing covers the step itself, so a per-step goodput breakdown has no
    denominator. Every group in the preset emits a bounded number of spans per
    step, so the volume scales with step count rather than dataset size. Drop to
    ``default`` for run-level spans only; ``all`` adds ``per_prompt``, which
    scales with the dataset."""

    # NOTE: there is deliberately no rank-filtering field here. nemo-lens has no
    # notion of rank -- it dropped its export-strategy registry and rank-aware
    # sampler -- so every process that enables telemetry exports, and each one
    # tags itself with ``nv.dl.rank`` / ``nv.dl.world_size``. Narrowing a large
    # fleet down to a few ranks is now a downstream decision: filter on
    # ``nv.dl.rank`` in the collector, which is also the only place that can see
    # every rank at once and is where such a policy belongs.

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
    """Enable vLLM's own OTLP tracing inside generation workers (opt-in). vLLM
    builds its exporter itself and defaults it to gRPC, reading only
    ``OTEL_EXPORTER_OTLP_TRACES_PROTOCOL`` -- not the generic
    ``OTEL_EXPORTER_OTLP_PROTOCOL`` lens honours. To ride the same
    ``http/protobuf`` endpoint as lens, set the traces-specific var too.

    vLLM traces per **request**, so this emits one span per rollout -- thousands
    per step, against ~20 for the rest of the run -- and vLLM offers no sampling
    knob. Switch it on to debug a few steps, not for a training run. For engine
    behaviour in aggregate the ``vllm/*`` metrics are on by default and cost two
    RPCs per step (a snapshot before generation and a read after)."""
