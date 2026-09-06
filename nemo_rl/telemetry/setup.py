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

import functools
import logging
import os
import threading
import uuid
from typing import TYPE_CHECKING, Any, Callable, Optional, TypeVar, cast

if TYPE_CHECKING:
    from nemo.lens import TelemetryHandle

logger = logging.getLogger(__name__)

_F = TypeVar("_F", bound=Callable[..., Any])

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
    Rank identity comes from :func:`_rank_attributes`, which the callers merge in.
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


def _rank_attributes(rank: int, world_size: int) -> dict[str, Any]:
    """Resource attributes identifying this process within its group.

    Lens has no notion of rank: it neither filters nor samples on one, so every
    process that sets up telemetry exports. Rank is recorded as a resource
    attribute instead, which is what makes a single rank's spans selectable
    afterwards -- filter on ``nv.dl.rank`` in the collector, or leave
    ``telemetry.enabled`` false on the ranks that should stay quiet.

    Passing rank at all is what keeps that filter available; lens warns when
    ``nv.dl.rank`` is missing, because without it a process cannot be told
    apart from its peers downstream.
    """
    from nemo.lens.semconv import NV_DL_RANK, NV_DL_WORLD_SIZE

    return {NV_DL_RANK: rank, NV_DL_WORLD_SIZE: world_size}


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
    """
    global _TELEMETRY_HANDLE, _TELEMETRY_INITIALISED
    if _TELEMETRY_INITIALISED:
        return _TELEMETRY_HANDLE

    # Before building the config below, so the settings reach workers through
    # the environment even on the paths that return early here.
    tel = getattr(master_config, "telemetry", None)
    if tel is not None:
        _config_to_env(tel)

    from nemo.lens import NemoLensConfig, setup_telemetry

    # Imported for its import side effect as much as for the name: importing
    # this module is what registers NeMo-RL's groups with lens's SpanRegistry,
    # and the spec below can only resolve against groups already registered.
    from nemo_rl.telemetry.span_groups import RLSpanGroup

    config = NemoLensConfig.from_env(
        prefix=_OTEL_PREFIX,
        fallback_prefix=_OTEL_FALLBACK_PREFIX,
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

    # Resolved eagerly so a typo is reported here rather than silently selecting
    # less than the user asked for. Lens treats an unknown entry as pending, not
    # an error, because the library that owns it may not have been imported yet.
    # On the driver that is usually a typo, but not always: workers receive the
    # raw spec and resolve it themselves, so a group owned by a library imported
    # only in the workers (Megatron's forward_backward, say) is pending here and
    # live there. Hence a warning that says so, rather than a raise.
    _, pending = RLSpanGroup.resolve_with_pending(config.span_groups)
    if pending:
        logger.warning(
            "nemo-lens: telemetry.span_groups names %s, which match nothing "
            "NeMo-RL registers. They select no driver spans -- check for a typo. "
            "They still apply in workers if another library registers them "
            "there. Registered groups: %s.",
            sorted(pending),
            sorted(RLSpanGroup.ALL_GROUPS),
        )

    # Unguarded on purpose: _build_resource_attributes is total by construction
    # (missing keys omit an attribute), so a raise here is a real bug, and
    # swallowing it would drop rl.model / nemo.precision / dl.*_parallel.size
    # from every span and metric for the whole run.
    resource_attrs = _build_resource_attributes(master_config, algorithm)
    # The driver is a singleton, not a member of a distributed group. Rank 0 of
    # 1 is the honest description, and stating it silences lens's warning about
    # a process that cannot be identified by rank downstream.
    resource_attrs.update(_rank_attributes(rank=0, world_size=1))

    handle = setup_telemetry(config, resource_attributes=resource_attrs)
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
    number their workers from zero — so ``nv.dl.rank`` alone cannot tell their
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
) -> Optional["TelemetryHandle"]:
    """Initialise telemetry inside a Ray actor (call once per worker process).

    Reads the ``NEMO_RL_OTEL_*`` env propagated from the driver via the Ray
    ``runtime_env``. ``rank`` / ``world_size`` default to the ``RANK`` /
    ``WORLD_SIZE`` env vars the worker was launched with, and are recorded as
    ``nv.dl.rank`` / ``nv.dl.world_size`` resource attributes.

    Every worker that gets here exports. Narrowing that down is a downstream
    decision now: filter on ``nv.dl.rank`` in the collector, or leave
    ``telemetry.enabled`` false for the ranks that should stay quiet.

    Args:
        rank: This process's rank. Defaults to the ``RANK`` env var.
        world_size: Size of this process's group. Defaults to ``WORLD_SIZE``.
        resource_attributes: Extra resource attributes for this process.

    Never raises: a worker must not fail a training run over optional
    observability. The driver resolves the same config before any worker
    starts, so a misconfiguration is already reported once, from the one
    process whose log the user is reading — rather than once per worker.

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

        # Imported for the side effect: registers NeMo-RL's span groups, which
        # the spec in the config below resolves against.
        import nemo_rl.telemetry.span_groups  # noqa: F401

        if rank is None:
            rank = int(os.environ.get("RANK", "0"))
        if world_size is None:
            world_size = int(os.environ.get("WORLD_SIZE", "1"))

        config = NemoLensConfig.from_env(
            prefix=_OTEL_PREFIX,
            fallback_prefix=_OTEL_FALLBACK_PREFIX,
        )
        if not config.enabled:
            return None

        attrs = _worker_resource_attributes(resource_attributes)
        attrs.update(_rank_attributes(rank=rank, world_size=world_size))

        handle = setup_telemetry(config, resource_attributes=attrs)
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


#: Set once ``instrument_aiohttp_client`` has patched the client, so a second
#: call is a no-op. The upstream instrumentor warns and skips on re-entry, and
#: an actor that spins up more than once would otherwise log that warning on a
#: path where nothing is wrong.
_AIOHTTP_INSTRUMENTED = False


def instrument_aiohttp_client() -> bool:
    """Make outgoing ``aiohttp`` requests carry the caller's trace context.

    The NeMo-Gym rollout path leaves this process over HTTP from inside Gym's
    own client, so there is no request NeMo-RL can add a header to. Patching the
    client library instead means every request Gym makes is stamped with a W3C
    ``traceparent`` taken from the ambient context, which is what lets a Gym
    server running :func:`nemo.lens.contrib.fastapi.instrument_fastapi` continue
    the trace rather than start one.

    Call after worker telemetry is up and before any ``ClientSession`` exists --
    the instrumentor patches the class, so sessions built earlier keep the
    unpatched behaviour.

    Returns:
        Whether the client is instrumented. ``False`` when telemetry is off, or
        when the optional dependency is missing -- neither is fatal, because
        losing the Gym half of a trace is not a reason to fail a rollout.
    """
    global _AIOHTTP_INSTRUMENTED
    if _AIOHTTP_INSTRUMENTED:
        return True
    if get_telemetry_handle() is None:
        return False
    try:
        from nemo.lens.contrib.aiohttp import instrument_aiohttp_client as _instrument

        _instrument()
    except ImportError:
        # nemo-lens[aiohttp] pulls this in, so it is present in a normal
        # install. Reachable when lens is pinned without the extra.
        logger.warning(
            "nemo-lens: aiohttp instrumentation unavailable, so NeMo-Gym HTTP "
            "calls will start their own traces; install nemo-lens[aiohttp]",
        )
        return False
    except Exception:
        logger.warning(
            "nemo-lens: aiohttp instrumentation failed; continuing without it",
            exc_info=True,
        )
        return False
    else:
        _AIOHTTP_INSTRUMENTED = True
        return True


def traced_worker_init(span_name: str, **attributes: Any) -> Callable[[_F], _F]:
    """Decorate a worker ``__init__`` so its model load lands in a span.

    Training workers build and shard the model inside ``__init__``, which on a
    large checkpoint is the longest single phase of a run's startup. Without
    this the time is real but unattributed: the driver's ``rl.setup.workers``
    span covers it as one block, and nothing says what the worker was doing.

    Initialises worker telemetry before the body runs. A worker normally calls
    :func:`init_telemetry_worker` itself a few lines into ``__init__``, but a
    span cannot open before its provider exists, so the decorator hoists that
    call ahead of the constructor; the body's own call then returns the handle
    already made, being idempotent per process.

    A ``None`` handle -- telemetry off, or its setup failed -- runs the
    constructor untouched. That is also what keeps this safe when ``nemo.lens``
    is absent: the span imports below are reached only once a handle exists,
    which cannot happen without lens, so an install without it neither imports
    nor pays for any of this.

    The span belongs to the ``model_init`` umbrella and so carries no
    ``rl.bucket``: it is startup rather than a phase of a training step, and
    the goodput rollup measures steps.
    """

    def decorate(fn: _F) -> _F:
        @functools.wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            if init_telemetry_worker() is None:
                return fn(*args, **kwargs)

            from nemo_rl.telemetry.instrumentation import umbrella_span
            from nemo_rl.telemetry.span_groups import RLSpanGroup

            with umbrella_span(RLSpanGroup.U_MODEL_INIT, span_name, **attributes):
                return fn(*args, **kwargs)

        return cast(_F, wrapper)

    return decorate


def get_telemetry_handle() -> Optional["TelemetryHandle"]:
    """Return the process-global telemetry handle (``None`` if uninitialised).

    Named for the handle rather than the telemetry because callers reach through
    it -- ``.tracer``, ``.meter``, ``.is_exporting`` -- rather than using the
    return value as a value.
    """
    return _TELEMETRY_HANDLE


def shutdown_telemetry(timeout_ms: int = 5000) -> None:
    """Flush and shut down telemetry providers, returning within *timeout_ms*.

    Call on the driver at job end, and in each Ray actor's ``shutdown``: span
    and metric processors buffer in the background, so an actor that exits
    without flushing silently drops whatever it had not exported yet. A no-op
    when this process never initialised telemetry.

    The budget is enforced here rather than passed down, because nothing below
    honours it. ``TelemetryHandle.shutdown`` forwards ``timeout_ms`` to
    ``force_flush(timeout_millis=...)``, and the SDK's ``BatchProcessor``
    accepts that argument and then drains the whole queue regardless; the
    ``provider.shutdown()`` calls that follow take no timeout at all and fall
    back to the SDK's own 30s. Four unbounded calls in total. Against an
    unreachable collector each buffered batch is retried with exponential
    backoff, so the cost scales with the queue: 3k spans took 36s under a 5s
    budget when measured.

    That mattered because callers size their own timeouts on this one --
    ``_flush_collector_telemetry`` in async GRPO allows 15s for a 3s quiesce
    plus "its 5s export", then ``ray.kill``s the collector when the budget
    blows, dropping the last rollout spans the flush exists to save.

    So: run the flush on a daemon thread and stop waiting at the deadline. A
    flush that overruns is abandoned rather than joined, which loses the same
    spans an unreachable collector was going to lose anyway, and being a daemon
    it never holds up process exit.
    """
    global _TELEMETRY_HANDLE
    handle = _TELEMETRY_HANDLE
    if handle is None:
        return
    # Cleared before the flush, not after: a second caller should return at once
    # rather than queue behind an export that is already overrunning, and the
    # handle's own _shutdown_done guards the underlying providers.
    _TELEMETRY_HANDLE = None

    def _flush() -> None:
        try:
            handle.shutdown(timeout_ms=timeout_ms)
        except Exception:
            logger.warning("nemo-lens: error during telemetry shutdown", exc_info=True)

    flusher = threading.Thread(target=_flush, name="nemo-lens-shutdown", daemon=True)
    flusher.start()
    flusher.join(timeout_ms / 1000)
    if flusher.is_alive():
        logger.warning(
            "nemo-lens: telemetry flush exceeded %dms and was abandoned; "
            "buffered spans and metrics may be lost. Usually means the OTLP "
            "endpoint is unreachable.",
            timeout_ms,
        )

    # Spans opened after this point would be built at full cost and then dropped
    # by the shut-down processor, which also logs per span. Nothing should be
    # emitting this late, but the paths that could are ordered by convention
    # rather than enforcement, so close the gate instead of relying on it.
    try:
        from nemo.lens.state import set_enabled_span_groups

        set_enabled_span_groups(frozenset())
    except Exception:
        logger.debug("nemo-lens: could not clear enabled span groups", exc_info=True)
