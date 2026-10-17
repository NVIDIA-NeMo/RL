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
"""Writer placement policies: order candidate units for a put.

Two policies, matching the two existing data planes in NeMo-RL:

- ``spread``       — every writer rotates over all ACTIVE units, so bytes
                     split evenly like TransferQueue SimpleStorage's
                     ``global_index % num_units`` (but by blob, via the
                     directory, so the unit count may change).
- ``local_first``  — ACTIVE units on the writer's node first, then the
                     rest; what NeMo-RL PR #4465 does for vLLM capture puts
                     (``prefer_storage_segment``), generalised to every writer.

Both spill to the next candidate when a unit raises ``UnitFull``.
"""

from __future__ import annotations

from typing import Any, Protocol, Sequence


class Placement(Protocol):
    def order(
        self, units: Sequence[dict[str, Any]], *, node_id: str, nbytes: int
    ) -> list[dict[str, Any]]: ...


def _active(units: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(
        (u for u in units if u.get("state", "ACTIVE") == "ACTIVE"),
        key=lambda u: int(u["unit_id"]),
    )


def _rotate(xs: list[dict[str, Any]], k: int) -> list[dict[str, Any]]:
    if not xs:
        return xs
    k %= len(xs)
    return xs[k:] + xs[:k]


class Spread:
    """Round-robin over all ACTIVE units; ``start`` staggers writers."""

    def __init__(self, start: int = 0) -> None:
        self._n = start

    def order(
        self, units: Sequence[dict[str, Any]], *, node_id: str, nbytes: int
    ) -> list[dict[str, Any]]:
        out = _rotate(_active(units), self._n)
        self._n += 1
        return out


class LocalFirst:
    """Same-socket units first, then the rest of the node, then remote ones.

    ``numa`` is the writer's socket (``None`` if unpinned or single-socket);
    units report theirs in ``info()["numa"]``.
    """

    def __init__(self, start: int = 0, numa: int | None = None) -> None:
        self._n = start
        self.numa = numa

    def order(
        self, units: Sequence[dict[str, Any]], *, node_id: str, nbytes: int
    ) -> list[dict[str, Any]]:
        active = _active(units)
        local = [u for u in active if u.get("node_id") == node_id]
        same = [
            u for u in local if self.numa is not None and u.get("numa") == self.numa
        ]
        other = [u for u in local if u not in same]
        remote = [u for u in active if u.get("node_id") != node_id]
        out = (
            _rotate(same, self._n) + _rotate(other, self._n) + _rotate(remote, self._n)
        )
        self._n += 1
        return out


POLICIES: dict[str, type] = {"spread": Spread, "local_first": LocalFirst}


def make_placement(name: str, *, start: int = 0, numa: int | None = None) -> Placement:
    try:
        if name == "local_first":
            return LocalFirst(start=start, numa=numa)
        return POLICIES[name](start=start)
    except KeyError:
        raise ValueError(
            f"unknown placement policy {name!r}; choose from {sorted(POLICIES)}"
        ) from None
