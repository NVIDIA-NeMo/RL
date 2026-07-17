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
``get_telemetry`` pattern but adapted to NeMo-RL's Ray driver + worker process
model:

* :func:`init_telemetry_driver` — called once on the driver, **before**
  ``init_ray()``. It reads the ``telemetry:`` config block, exports the settings
  as ``NEMO_RL_OTEL_*`` env vars so every Ray worker inherits them, and sets up
  the driver's own telemetry (the training loop and the metrics logger run
  here, so the driver always exports).
* :func:`init_telemetry_worker` — called once inside each Ray actor process
  (from ``__init__`` / ``post_init``). It reads the propagated env and sets up
  that worker's telemetry.

Importing this module never requires nemo-lens: every lens import is
function-local and guarded by ``try/except ImportError``. When lens is not
installed (or telemetry is disabled), the init functions return ``None`` and all
instrumentation sites stay no-ops via ``nemo_rl.telemetry._fallbacks``.
"""

from __future__ import annotations

import logging
import os
import uuid
from typing import TYPE_CHECKING, Any, Optional

if TYPE_CHECKING:
    from nemo.lens import TelemetryHandle

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

# TelemetryConfig field -> NEMO_RL_OTEL_* env var. ``service_name`` maps to the
# standard ``OTEL_SERVICE_NAME`` (lens reads it directly, unprefixed).
_ENV_FIELD_MAP = {
    "enabled": f"{_OTEL_PREFIX}_ENABLED",
    "span_groups": f"{_OTEL_PREFIX}_SPAN_GROUPS",
    "export_strategy": f"{_OTEL_PREFIX}_EXPORT_STRATEGY",
    "export_rank": f"{_OTEL_PREFIX}_EXPORT_RANK",
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
    rank: int,
    world_size: int,
) -> dict:
    """Build process-lifetime resource attributes (Jaeger "Process" tags).

    Only stable-for-the-run values belong here (algorithm, model, precision,
    parallelism). Per-step values are span tags; time-series values are metrics.
    Best-effort: a missing key simply omits that attribute — never raises.
    """
    attrs: dict[str, Any] = {"rl.algorithm": algorithm}

    model = _dig(master_config, "policy", "model_name")
    if model:
        attrs["rl.model"] = model

    precision = _dig(master_config, "policy", "precision")
    if precision:
        attrs["nemo.precision"] = precision

    # Parallelism lives under the active policy backend (megatron vs dtensor).
    tp = _dig(master_config, "policy", "megatron_cfg", "tensor_model_parallel_size") or _dig(
        master_config, "policy", "dtensor_cfg", "tensor_parallel_size"
    )
    if tp:
        attrs["dl.tensor_parallel.size"] = tp
    pp = _dig(master_config, "policy", "megatron_cfg", "pipeline_model_parallel_size")
    if pp:
        attrs["dl.pipeline_parallel.size"] = pp

    return attrs


def init_telemetry_driver(
    master_config: Any,
    algorithm: str,
) -> Optional["TelemetryHandle"]:
    """Initialise driver-side telemetry (call once, before ``init_ray()``).

    Reads ``master_config.telemetry``, exports the resolved settings as
    ``NEMO_RL_OTEL_*`` env vars (so workers inherit them), and sets up the
    driver's OTel providers. The driver always exports (it hosts the training
    loop and the metrics logger).

    Returns the :class:`TelemetryHandle`, or ``None`` if nemo-lens is not
    installed or telemetry is disabled. Idempotent.
    """
    global _TELEMETRY_HANDLE, _TELEMETRY_INITIALISED
    if _TELEMETRY_INITIALISED:
        return _TELEMETRY_HANDLE
    _TELEMETRY_INITIALISED = True

    try:
        from nemo.lens import NemoLensConfig, setup_telemetry
    except ImportError:
        return None

    from nemo_rl.telemetry.span_groups import RLSpanGroup

    tel = getattr(master_config, "telemetry", None)
    if tel is not None:
        _config_to_env(tel)

    config = NemoLensConfig.from_env(
        prefix=_OTEL_PREFIX,
        fallback_prefix=_OTEL_FALLBACK_PREFIX,
        span_group_cls=RLSpanGroup,
    )
    if not config.enabled:
        return None

    # A friendly default service name if the user set nothing.
    if not os.environ.get(_SERVICE_NAME_ENV, "").strip():
        config.service_name = "nemo-rl"

    # One run_id shared by the driver and every worker. Written to the env
    # before init_ray() so workers inherit it and correlate to the same trace.
    if not config.run_id:
        run_id = os.environ.get("SLURM_JOB_ID", "").strip() or uuid.uuid4().hex[:12]
        os.environ[_RUN_ID_ENV] = run_id
        config.run_id = run_id

    try:
        resource_attrs = _build_resource_attributes(
            master_config, algorithm, rank=0, world_size=1
        )
    except Exception:
        logger.warning("nemo-lens: failed to build resource attributes", exc_info=True)
        resource_attrs = {"rl.algorithm": algorithm}

    handle = setup_telemetry(
        config, rank=0, world_size=1, resource_attributes=resource_attrs
    )
    _TELEMETRY_HANDLE = handle

    if config.logs_enabled and handle.is_exporting:
        try:
            from nemo.lens.logging_bridge import setup_logging_bridge

            setup_logging_bridge()
        except Exception:
            logger.warning("nemo-lens: failed to set up logging bridge", exc_info=True)

    logger.info(
        "nemo-lens telemetry initialised (algorithm=%s, exporting=%s, run_id=%s, groups=%s)",
        algorithm,
        handle.is_exporting,
        config.run_id,
        config.span_groups,
    )
    return handle


def init_telemetry_worker(
    rank: Optional[int] = None,
    world_size: Optional[int] = None,
    resource_attributes: Optional[dict] = None,
) -> Optional["TelemetryHandle"]:
    """Initialise telemetry inside a Ray actor (call once per worker process).

    Reads the ``NEMO_RL_OTEL_*`` env propagated from the driver via the Ray
    ``runtime_env``. ``rank`` / ``world_size`` default to the ``RANK`` /
    ``WORLD_SIZE`` env vars the worker was launched with, which — together with
    the export strategy — decide whether this worker exports.

    Returns the :class:`TelemetryHandle`, or ``None`` if lens is absent or
    telemetry is disabled. Idempotent per process.
    """
    global _TELEMETRY_HANDLE, _TELEMETRY_INITIALISED
    if _TELEMETRY_INITIALISED:
        return _TELEMETRY_HANDLE
    _TELEMETRY_INITIALISED = True

    if not (_is_env_truthy(f"{_OTEL_PREFIX}_ENABLED") or _is_env_truthy(f"{_OTEL_FALLBACK_PREFIX}_ENABLED")):
        return None

    try:
        from nemo.lens import NemoLensConfig, setup_telemetry
    except ImportError:
        return None

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
        resource_attributes=resource_attributes,
    )
    _TELEMETRY_HANDLE = handle
    return handle


def get_telemetry() -> Optional["TelemetryHandle"]:
    """Return the process-global telemetry handle (``None`` if uninitialised)."""
    return _TELEMETRY_HANDLE


def shutdown_telemetry(timeout_ms: int = 5000) -> None:
    """Flush and shut down telemetry providers. Call on the driver at job end."""
    global _TELEMETRY_HANDLE
    handle = _TELEMETRY_HANDLE
    if handle is None:
        return
    try:
        handle.shutdown(timeout_ms=timeout_ms)
    except Exception:
        logger.warning("nemo-lens: error during telemetry shutdown", exc_info=True)
