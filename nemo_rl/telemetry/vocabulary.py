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
from typing import Any, Iterable, Optional

#: Prefix every NeMo-RL series carries, stripped to form the registry key.
METRIC_NAME_PREFIX = "rl."

#: Efficiency category covering startup, measured once before the training loop.
INIT_TOTAL_CATEGORY = "init/total"

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
        kind: One of lens's ``gauge`` / ``counter`` / ``histogram`` / ``up_down_counter``.
        unit: UCUM-ish unit string, or empty when dimensionless.
        description: Help text carried to the backend.
    """

    name: str
    kind: str = "gauge"
    unit: str = ""
    description: str = ""

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


def register_teed_metrics(rows: Iterable[TeedMetric]) -> None:
    """Declare *rows* for teeing, from the module that logs their keys.

    Raises:
        ValueError: Two rows claim the same logger key or the same series name.
            Both would be silent otherwise: the first makes one row unreachable,
            and the second makes two rows record against one instrument.
    """
    for row in rows:
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
    """
    for row in rows:
        clash = _RECORDED.get(row.name)
        if clash is not None and clash != row:
            raise ValueError(f"series {row.name!r} is already declared as {clash!r}")
        _RECORDED[row.name] = row


def recorded_metrics() -> tuple[RecordedMetric, ...]:
    """Every directly-recorded row declared so far, in declaration order."""
    return tuple(_RECORDED.values())
