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

"""Names shared between the code that produces metrics and the code that exports them.

Every name here has at least two readers on opposite sides of that split, and
the module imports nothing so either side can reach it: an owner is free to be
a torch-heavy training module, and telemetry is free to be importable without
the training stack.

The teed rows in particular live next to the code that produces their keys,
not here and not in :mod:`nemo_rl.telemetry.metrics`: a module that logs
``"reward"`` declares ``rl.reward.mean`` in the same file, so renaming the key
renames the declaration with it and there is no second copy to drift. Owners
call :func:`register_teed_metrics` at import; the rows are collected
process-wide and handed to lens once, on the first tee, so an owner that is
never imported (``ppo.py`` in a GRPO run) simply declares nothing.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Iterable, Optional

if TYPE_CHECKING:
    # Behind the guard to keep the no-runtime-imports promise above: an owner
    # declaring a row must not have to have OTel installed to do it.
    from opentelemetry.metrics import CallbackT

#: Prefix every NeMo-RL series carries, stripped to form the registry key.
METRIC_NAME_PREFIX = "rl."

#: Efficiency category covering startup, measured once before the training loop.
INIT_TOTAL_CATEGORY = "init/total"

#: Env var ``RayWorkerGroup`` exports so a worker's spans carry
#: ``rl.worker_group``; telemetry reads it back when setting a worker up.
WORKER_GROUP_ENV = "NRL_WORKER_GROUP"

#: Wall-clock axis the Single-Controller logs beside every event dict and plots
#: it on in W&B; the rollout save span reads it as its end time.
TELEMETRY_WALL_TIME_METRIC = "telemetry/wall_time_seconds"

#: Trainer step the Single-Controller logs beside every event dict; the rollout
#: save span reads it as ``rl.iteration``.
TELEMETRY_TRAIN_STEP_KEY = "train_step"

#: Keys ``algorithms/utils.py`` writes the efficiency ratio under and telemetry
#: reads back. Named here, not at either end, because a rename on one side
#: alone would leave the metric reading nothing with no error.
EFFICIENCY_PCT_KEY = "efficiency/efficiency_pct"
EFFICIENCY_PCT_PER_STEP_KEY = "efficiency/efficiency_pct_is_per_step"

# Wall-clock categories whose value covers the whole run rather than one step.
# The driver's Timer is reset every step, so its idle categories are per-step
# deltas -- but init/total is measured once before the loop and republished
# unchanged afterwards, so it cannot be compared against a single step's wall
# time. Read by algorithms/utils.py (which excludes it from the per-step
# efficiency ratio) and by telemetry (which labels its window and leaves its
# span unbucketed).
RUN_WINDOW_WALL_CLOCK_CATEGORIES: frozenset[str] = frozenset({INIT_TOTAL_CATEGORY})


def as_scalar(value: Any) -> Optional[float]:
    """Coerce a logged value to float, or None when it is not a usable scalar.

    Shared by every family that reads a raw ``Logger`` dict, so they agree on
    what counts as a number.
    """
    # bool is a subclass of int, so it has to be excluded explicitly.
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)


def registry_key(name: str) -> str:
    """Registry key for a series name (``rl.reward.mean`` -> ``reward_mean``).

    lens keys must be Python identifiers, so the shared prefix goes and the
    dots become underscores. Derived rather than typed out so a row cannot
    name its series one thing and record against another.
    """
    return name.removeprefix(METRIC_NAME_PREFIX).replace(".", "_")


@dataclass(frozen=True)
class RecordedMetric:
    """A series recorded directly, with no ``Logger`` key behind it.

    Declared by its owner for the same reason the teed rows are: the module
    that calls ``record_metrics`` for a series is the one that names it.

    Attributes:
        name: OTel series name emitted; :func:`registry_key` derives the key.
        kind: One of lens's ``gauge`` / ``counter`` / ``histogram`` /
            ``up_down_counter``, or one of its observable kinds
            (``observable_gauge`` / ``observable_counter`` /
            ``observable_up_down_counter``), which require *callback*.
        unit: UCUM-ish unit string, or empty when dimensionless.
        description: Help text carried to the backend.
        callback: For the observable kinds only, what the exporter calls to
            read the series. Each cycle's points are exactly what it returns,
            so an attribute set it stops yielding stops being exported, and it
            is read on every export, where a pushed gauge reports only in the
            cycles that follow a set. Must be a function that outlives the
            process's first export, not a closure built per call.
    """

    name: str
    kind: str = "gauge"
    unit: str = ""
    description: str = ""
    callback: Optional[CallbackT] = None

    @property
    def key(self) -> str:
        """Registry key recorded against."""
        return registry_key(self.name)


@dataclass(frozen=True)
class TeedMetric:
    """One ``Logger`` key mirrored into OTel, and how it is declared to lens.

    Attributes:
        logger_key: Key as it appears in the dict handed to ``Logger.log_metrics``.
        name: OTel series name emitted; :func:`registry_key` derives the key.
        kind: One of lens's ``gauge`` / ``counter`` / ``histogram`` / ``up_down_counter``.
        unit: UCUM-ish unit string, or empty when dimensionless.
        description: Help text carried to the backend.
    """

    logger_key: str
    name: str
    kind: str = "gauge"
    unit: str = ""
    description: str = ""

    @property
    def key(self) -> str:
        """Registry key recorded against."""
        return registry_key(self.name)


_REGISTERED: dict[str, TeedMetric] = {}

_FROZEN = False


def freeze_metrics() -> None:
    """Close registration, once the rows have been handed to lens.

    Called by ``ensure_metric_group_registered``. lens takes the specs once
    per process, so a row declared after that is read into the recorded values
    but was never declared -- lens drops it with a single warning and RL says
    nothing at all.
    """
    global _FROZEN
    _FROZEN = True


def _check_not_frozen(name: str) -> None:
    if _FROZEN:
        raise RuntimeError(
            f"{name!r} was declared after the metric group was registered with "
            "lens, so it has no instrument and would be silently dropped. "
            "Import the module that declares it before init_telemetry_driver "
            "runs, which is where registration now happens -- the observable "
            "host-memory series have to have their instruments built at "
            "startup, because nothing records them into existence later. "
            "Declare at import time from a module the entry point already "
            "imports, not lazily from inside the training loop."
        )


def register_teed_metrics(rows: Iterable[TeedMetric]) -> None:
    """Declare *rows* for teeing, from the module that logs their keys.

    Raises:
        ValueError: Two rows claim the same logger key or the same series name.
            Both would be silent otherwise: the first makes one row unreachable,
            and the second makes two rows record against one instrument.
        RuntimeError: Registration is already closed -- see :func:`freeze_metrics`.
    """
    for row in rows:
        _check_not_frozen(row.name)
        clash = _REGISTERED.get(row.logger_key)
        if clash is not None and clash != row:
            raise ValueError(
                f"logger key {row.logger_key!r} is already teed as {clash.name!r}"
            )
        by_name = {r.name: r for r in _REGISTERED.values()}
        named = by_name.get(row.name)
        if named is not None and named.logger_key != row.logger_key:
            raise ValueError(
                f"series {row.name!r} is already declared for logger key "
                f"{named.logger_key!r}"
            )
        _REGISTERED[row.logger_key] = row


def teed_metrics() -> tuple[TeedMetric, ...]:
    """Every row declared so far, in declaration order."""
    return tuple(_REGISTERED.values())


_RECORDED: dict[str, RecordedMetric] = {}


def register_recorded_metrics(rows: Iterable[RecordedMetric]) -> None:
    """Declare *rows* from the module that records them.

    Raises:
        ValueError: Two rows claim the same series name with different
            definitions, which would otherwise have them share one instrument.
        RuntimeError: Registration is already closed -- see :func:`freeze_metrics`.
    """
    for row in rows:
        _check_not_frozen(row.name)
        clash = _RECORDED.get(row.name)
        if clash is not None and clash != row:
            raise ValueError(f"series {row.name!r} is already declared as {clash!r}")
        _RECORDED[row.name] = row


def recorded_metrics() -> tuple[RecordedMetric, ...]:
    """Every directly-recorded row declared so far, in declaration order."""
    return tuple(_RECORDED.values())
