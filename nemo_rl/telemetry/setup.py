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

"""Process-global nemo-lens telemetry lifecycle for NeMo-RL.

Two entry points, mirroring Megatron's ``global_vars._set_telemetry`` /
``get_telemetry_handle`` pattern but adapted to NeMo-RL's Ray driver + worker process
model:

* :func:`init_telemetry_driver` — called once on the driver, **before**
  ``init_ray()``. It reads the ``telemetry:`` config block, exports the settings
  as ``NEMO_RL_OTEL_*`` env vars so every Ray worker inherits them, and sets up
  the driver's own telemetry (the training loop and the metrics logger run
  here, so the driver always exports).
* :func:`init_telemetry_worker` — called once inside each Ray actor process,
  from the worker's ``__init__`` (policy, value and vLLM generation workers).
  It reads the propagated env and sets up that worker's telemetry, then
  :func:`shutdown_telemetry` flushes it from the worker's ``shutdown``.
  ``__init__`` rather than ``post_init``, because some ``post_init`` fan-outs
  run on only one rank per parallel group, while OTel providers have to be set
  up in *every* actor process.

nemo-lens is a base dependency, so ``telemetry.enabled`` is the only switch:
when it is false the init functions return ``None`` and every instrumentation
site is a ~0-cost no-op. Lens imports stay function-local to keep the cost off
the import path of modules that never emit anything.
"""

from __future__ import annotations

import logging
import os
import uuid
from typing import TYPE_CHECKING, Any, Optional

if TYPE_CHECKING:
    from nemo.lens import NemoLensConfig, TelemetryHandle

logger = logging.getLogger(__name__)

# Process-global handle. One per process (driver or Ray actor); ``None`` when
# lens is absent or telemetry is disabled.
_TELEMETRY_HANDLE: Optional["TelemetryHandle"] = None
_TELEMETRY_INITIALISED = False

# Env-var prefix for NeMo-RL. ``NemoLensConfig.from_env`` reads
# ``NEMO_RL_OTEL_<KEY>`` first, then falls back to ``NEMO_LENS_<KEY>``.
_OTEL_PREFIX = "NEMO_RL_OTEL"
_OTEL_FALLBACK_PREFIX = "NEMO_LENS"
_RUN_ID_ENV = f"{_OTEL_PREFIX}_RUN_ID"

# Set per worker by ``RayWorkerGroup`` from the group's ``name_prefix``.
_WORKER_GROUP_ENV = "NRL_WORKER_GROUP"

# TelemetryConfig field -> NEMO_RL_OTEL_* env var. ``service_name`` maps to the
# standard ``OTEL_SERVICE_NAME`` (lens reads it directly, unprefixed).
_ENV_FIELD_MAP = {
    "enabled": f"{_OTEL_PREFIX}_ENABLED",
    "span_groups": f"{_OTEL_PREFIX}_SPAN_GROUPS",
    "export_strategy": f"{_OTEL_PREFIX}_EXPORT_STRATEGY",
    "export_rank": f"{_OTEL_PREFIX}_EXPORT_RANK",
    "export_sample_rate": f"{_OTEL_PREFIX}_EXPORT_SAMPLE_RATE",
    "sampler_enabled": f"{_OTEL_PREFIX}_SAMPLER_ENABLED",
    "traces_enabled": f"{_OTEL_PREFIX}_TRACES_ENABLED",
    "metrics_enabled": f"{_OTEL_PREFIX}_METRICS_ENABLED",
    "logs_enabled": f"{_OTEL_PREFIX}_LOGS_ENABLED",
    "exporter": f"{_OTEL_PREFIX}_EXPORTER",
    # RL-owned flag consumed by the vLLM generation worker (not a lens field).
    "vllm_native_tracing": f"{_OTEL_PREFIX}_VLLM_NATIVE_TRACING",
}

# Standard-OTel env var that also propagates to workers via the Ray runtime_env.
_SERVICE_NAME_ENV = "OTEL_SERVICE_NAME"


def _is_env_truthy(name: str) -> bool:
    """Return True if env var ``name`` is set to a truthy value."""
    return os.environ.get(name, "").strip().lower() in ("1", "true", "yes", "on")


def telemetry_enabled_in_env() -> bool:
    """Whether ``telemetry.enabled`` reached this process as truthy.

    Exists for instrumentation that never asks for a handle: vLLM's native
    tracing runs on vLLM's own exporter, so :func:`get_telemetry_handle` would not
    gate it and the master switch would not reach it. Reads the environment
    rather than a config object because that is the only channel a worker
    process has.
    """
    return _is_env_truthy(f"{_OTEL_PREFIX}_ENABLED") or _is_env_truthy(
        f"{_OTEL_FALLBACK_PREFIX}_ENABLED"
    )


def vllm_native_tracing_requested() -> bool:
    """Whether ``telemetry.vllm_native_tracing`` reached this process as truthy.

    Callers must also honour :func:`telemetry_enabled_in_env`; this reports only
    the one field.
    """
    return _is_env_truthy(f"{_OTEL_PREFIX}_VLLM_NATIVE_TRACING")


def _config_to_env(tel: Any) -> None:
    """Translate a ``TelemetryConfig`` into ``NEMO_RL_OTEL_*`` env vars.

    Uses ``os.environ.setdefault`` so raw env vars always win over YAML. Runs on
    the driver before ``init_ray()``, so the resulting environment is snapshotted
    into the Ray ``runtime_env`` and inherited by every worker process.
    """
    for field, env_name in _ENV_FIELD_MAP.items():
        value = getattr(tel, field, None)
        if value is None:
            continue
        if isinstance(value, bool):
            os.environ.setdefault(env_name, "1" if value else "0")
        else:
            os.environ.setdefault(env_name, str(value))

    service_name = getattr(tel, "service_name", None)
    if service_name:
        os.environ.setdefault(_SERVICE_NAME_ENV, str(service_name))


def _dig(obj: Any, *path: str) -> Any:
    """Best-effort nested lookup that works for both dicts and objects.

    Returns ``None`` as soon as any level is missing. Used to pull resource
    attributes out of a ``MasterConfig`` whose nested nodes may be pydantic
    models (attribute access) or TypedDict-derived dicts (key access).
    """
    cur = obj
    for key in path:
        if cur is None:
            return None
        cur = cur.get(key) if isinstance(cur, dict) else getattr(cur, key, None)
    return cur


def _build_resource_attributes(
    master_config: Any,
    algorithm: str,
) -> dict:
    """Build process-lifetime resource attributes (Jaeger "Process" tags).

    Only stable-for-the-run values belong here (algorithm, model, precision,
    parallelism). Per-step values are span tags; time-series values are metrics.
    Best-effort: a missing key simply omits that attribute — never raises.
    ``dl.rank`` / ``dl.world_size`` are set by lens from the setup call, not here.
    """
    attrs: dict[str, Any] = {"rl.algorithm": algorithm}

    model = _dig(master_config, "policy", "model_name")
    if model:
        attrs["rl.model"] = model

    precision = _dig(master_config, "policy", "precision")
    if precision:
        attrs["nemo.precision"] = precision

    # Parallelism lives under the active policy backend (megatron vs dtensor).
    tp = _dig(
        master_config, "policy", "megatron_cfg", "tensor_model_parallel_size"
    ) or _dig(master_config, "policy", "dtensor_cfg", "tensor_parallel_size")
    if tp:
        attrs["dl.tensor_parallel.size"] = tp
    pp = _dig(master_config, "policy", "megatron_cfg", "pipeline_model_parallel_size")
    if pp:
        attrs["dl.pipeline_parallel.size"] = pp

    return attrs


def _always_export(
    config: "NemoLensConfig",
    rank: int,
    world_size: int,
) -> bool:
    """Export-strategy override for singleton processes: always export."""
    return True


def _unrank(config: "NemoLensConfig") -> Any:
    """Disable rank-based span filtering for a process that has no real rank.

    The driver and singleton actors such as ``AsyncTrajectoryCollector`` are not
    members of a distributed group, so they pass a synthetic ``rank=0`` /
    ``world_size=1``. Both of lens's rank filters then misfire on that made-up
    rank, and the two are independent, so both have to be neutralised:

    * ``export_strategy`` decides whether this process exports at all. A
      strategy that selects among the ranks of a *group* has no meaning for a
      singleton, and mutes it outright for any ``export_rank >= 1``.
    * ``sampler_enabled`` installs a ``RankAwareSampler`` on the tracer
      provider, which drops every span on ranks whose ``md5(rank)`` bucket
      lands above ``export_sample_rate``. Rank 0's bucket is 0.785, so any
      sample rate at or below that discards the process's spans *before* the
      export decision is ever consulted.

    The driver hosts the training loop and the metrics logger, and the
    collector generates every async rollout, so either filter silently drops
    the telemetry that matters most.

    Mutating ``config`` here is process-local: it is rebuilt from the
    environment in each process, and the propagated ``NEMO_RL_OTEL_*`` vars are
    untouched, so ranked workers still honour what the user configured.

    Returns the export-strategy override to hand to ``setup_telemetry``.
    """
    if config.sampler_enabled and config.export_sample_rate < 1.0:
        logger.info(
            "nemo-lens: disabling the rank sampler for this process "
            "(sample_rate=%s applies to ranked workers, not to the driver or a "
            "singleton actor)",
            config.export_sample_rate,
        )
        config.sampler_enabled = False
    return _always_export


def init_telemetry_driver(
    master_config: Any,
    algorithm: str,
) -> Optional["TelemetryHandle"]:
    """Initialise driver-side telemetry (call once, before ``init_ray()``).

    Reads ``master_config.telemetry``, exports the resolved settings as
    ``NEMO_RL_OTEL_*`` env vars (so workers inherit them), and sets up the
    driver's OTel providers. The driver always exports (it hosts the training
    loop and the metrics logger).

    Returns the :class:`TelemetryHandle`, or ``None`` if telemetry is disabled.
    Idempotent.

    Raises:
        ValueError: if ``telemetry.export_strategy`` names a strategy lens does
            not have. Deliberately fatal on the driver, where the user sees it.
    """
    global _TELEMETRY_HANDLE, _TELEMETRY_INITIALISED
    if _TELEMETRY_INITIALISED:
        return _TELEMETRY_HANDLE

    # Before building the config below, so the settings reach workers through
    # the environment even on the paths that return early here.
    tel = getattr(master_config, "telemetry", None)
    if tel is not None:
        _config_to_env(tel)

    from nemo.lens import NemoLensConfig, registered_strategies, setup_telemetry

    from nemo_rl.telemetry.span_groups import RLSpanGroup

    config = NemoLensConfig.from_env(
        prefix=_OTEL_PREFIX,
        fallback_prefix=_OTEL_FALLBACK_PREFIX,
        span_group_cls=RLSpanGroup,
    )
    if not config.enabled:
        _TELEMETRY_INITIALISED = True
        return None

    # A friendly default service name if the user set nothing. Exported like
    # run_id below, not just assigned: workers rebuild their config from the
    # environment, and lens falls back to "nemo" when this is unset — which
    # would file one run's spans under two different service names.
    if not os.environ.get(_SERVICE_NAME_ENV, "").strip():
        config.service_name = "nemo-rl"
        os.environ[_SERVICE_NAME_ENV] = config.service_name

    # One run_id shared by the driver and every worker. Written to the env
    # before init_ray() so workers inherit it and correlate to the same trace.
    if not config.run_id:
        run_id = os.environ.get("SLURM_JOB_ID", "").strip() or uuid.uuid4().hex[:12]
        os.environ[_RUN_ID_ENV] = run_id
        config.run_id = run_id

    # Passing _always_export below bypasses lens's registry lookup, which is
    # what would otherwise reject a misspelled strategy. Validate it here so a
    # typo fails on the driver, where the user sees it, instead of degrading to
    # a warning inside every worker.
    if config.export_strategy not in registered_strategies():
        raise ValueError(
            f"Unknown telemetry.export_strategy {config.export_strategy!r}. "
            f"Registered strategies: {registered_strategies()}."
        )

    # Same reasoning, different field: lens resolves span_groups lazily, *after*
    # it has installed the global tracer provider, so a typo there would take
    # the process down with telemetry half-built and no handle to flush.
    RLSpanGroup.resolve(config.span_groups)

    # Unguarded on purpose: _build_resource_attributes is total by construction
    # (missing keys omit an attribute), so a raise here is a real bug, and
    # swallowing it would drop rl.model / nemo.precision / dl.*_parallel.size
    # from every span and metric for the whole run.
    resource_attrs = _build_resource_attributes(master_config, algorithm)

    handle = setup_telemetry(
        config,
        rank=0,
        world_size=1,
        resource_attributes=resource_attrs,
        export_strategy=_unrank(config),
    )
    # Only now, past everything that can raise: setting the guard earlier would
    # turn a retry after a failed setup into a silent None instead of the same
    # error. Lens leaves its own guard clear on that path, so a retry is safe.
    _TELEMETRY_INITIALISED = True
    _TELEMETRY_HANDLE = handle

    if config.logs_enabled and handle.is_exporting:
        # Unguarded: the user asked for logs, so failing to install the bridge
        # should be loud rather than a warning followed by silently no logs.
        from nemo.lens.logging_bridge import setup_logging_bridge

        setup_logging_bridge()

    # Every resolved field, not just the headline ones: the env projection uses
    # setdefault, so a stray NEMO_RL_OTEL_* in the shell silently overrides the
    # YAML. Logging what was actually resolved keeps "how was this run
    # configured" answerable from the run's own log either way.
    resolved = ", ".join(
        f"{field}={getattr(config, field)!r}"
        for field in _ENV_FIELD_MAP
        if hasattr(config, field)
    )
    logger.info(
        "nemo-lens telemetry initialised (algorithm=%s, exporting=%s, run_id=%s, "
        "service_name=%s, %s)",
        algorithm,
        handle.is_exporting,
        config.run_id,
        config.service_name,
        resolved,
    )
    return handle


def _worker_resource_attributes(
    extra: Optional[dict[str, Any]],
) -> dict[str, Any]:
    """Build resource attributes identifying this worker process.

    ``RANK`` is group-local — the policy group and the generation group each
    number their workers from zero — so ``dl.rank`` alone cannot tell their
    spans apart. ``rl.worker_group`` carries the group's ``name_prefix``
    (``lm_policy``, ``vllm_policy``, ...), which ``RayWorkerGroup`` exports as
    ``NRL_WORKER_GROUP``. Explicit ``extra`` attributes win.
    """
    attrs: dict[str, Any] = {}
    worker_group = os.environ.get(_WORKER_GROUP_ENV, "").strip()
    if worker_group:
        attrs["rl.worker_group"] = worker_group
    if extra:
        attrs.update(extra)
    return attrs


def init_telemetry_worker(
    rank: Optional[int] = None,
    world_size: Optional[int] = None,
    resource_attributes: Optional[dict[str, Any]] = None,
    always_export: bool = False,
) -> Optional["TelemetryHandle"]:
    """Initialise telemetry inside a Ray actor (call once per worker process).

    Reads the ``NEMO_RL_OTEL_*`` env propagated from the driver via the Ray
    ``runtime_env``. ``rank`` / ``world_size`` default to the ``RANK`` /
    ``WORLD_SIZE`` env vars the worker was launched with, which — together with
    the export strategy — decide whether this worker exports.

    Args:
        rank: This process's rank. Defaults to the ``RANK`` env var.
        world_size: Size of this process's group. Defaults to ``WORLD_SIZE``.
        resource_attributes: Extra resource attributes for this process.
        always_export: Bypass the configured rank filters for this process, the
            way the driver does (see :func:`_unrank`). Set it for a *singleton*
            actor passing a synthetic ``rank`` / ``world_size``: the filters
            select among the ranks of a distributed group, so applying them to
            a made-up rank has no meaning and silently mutes the actor — e.g.
            ``export_rank: 3`` never matches a synthetic rank 0. Ranked members
            of a real worker group must leave this false.

    Never raises: unlike the driver, a worker must not fail a training run over
    optional observability, and the driver has already validated the same
    config before any worker starts — so a genuine misconfiguration surfaces
    there, loudly, rather than here.

    Returns the :class:`TelemetryHandle`, or ``None`` if telemetry is disabled
    or setup failed. Idempotent per process.
    """
    global _TELEMETRY_HANDLE, _TELEMETRY_INITIALISED
    if _TELEMETRY_INITIALISED:
        return _TELEMETRY_HANDLE
    _TELEMETRY_INITIALISED = True

    if not telemetry_enabled_in_env():
        return None

    # Deliberately broad: everything from here on is best-effort, so that a bad
    # exporter endpoint or a malformed RANK cannot take a training worker down.
    try:
        from nemo.lens import NemoLensConfig, setup_telemetry

        from nemo_rl.telemetry.span_groups import RLSpanGroup

        if rank is None:
            rank = int(os.environ.get("RANK", "0"))
        if world_size is None:
            world_size = int(os.environ.get("WORLD_SIZE", "1"))

        config = NemoLensConfig.from_env(
            prefix=_OTEL_PREFIX,
            fallback_prefix=_OTEL_FALLBACK_PREFIX,
            span_group_cls=RLSpanGroup,
        )
        if not config.enabled:
            return None

        handle = setup_telemetry(
            config,
            rank=rank,
            world_size=world_size,
            resource_attributes=_worker_resource_attributes(resource_attributes),
            # None leaves lens to resolve config.export_strategy as usual.
            export_strategy=_unrank(config) if always_export else None,
        )
        logger.info(
            "nemo-lens worker telemetry initialised (group=%s, rank=%s/%s, exporting=%s)",
            os.environ.get(_WORKER_GROUP_ENV, "?"),
            rank,
            world_size,
            handle.is_exporting,
        )
    except Exception:
        logger.warning(
            "nemo-lens: worker telemetry setup failed; continuing without it",
            exc_info=True,
        )
        return None
    else:
        _TELEMETRY_HANDLE = handle
        return handle


def get_telemetry_handle() -> Optional["TelemetryHandle"]:
    """Return the process-global telemetry handle (``None`` if uninitialised).

    Named for the handle rather than the telemetry because callers reach through
    it -- ``.tracer``, ``.meter``, ``.is_exporting`` -- rather than using the
    return value as a value.
    """
    return _TELEMETRY_HANDLE


def shutdown_telemetry(timeout_ms: int = 5000) -> None:
    """Flush and shut down telemetry providers.

    Call on the driver at job end, and in each Ray actor's ``shutdown``: span
    and metric processors buffer in the background, so an actor that exits
    without flushing silently drops whatever it had not exported yet. A no-op
    when this process never initialised telemetry.
    """
    global _TELEMETRY_HANDLE
    handle = _TELEMETRY_HANDLE
    if handle is None:
        return
    try:
        handle.shutdown(timeout_ms=timeout_ms)
    except Exception:
        logger.warning("nemo-lens: error during telemetry shutdown", exc_info=True)
    finally:
        _TELEMETRY_HANDLE = None
