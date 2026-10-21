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
"""Writer placement: order candidate units for a put.

``Spread`` rotates every writer over all ACTIVE units, so bytes split evenly
like TransferQueue SimpleStorage's ``global_index % num_units`` (but by blob,
via the directory, so the unit count may change). A put goes to the next
candidate when a unit raises ``UnitFull``.
"""

from __future__ import annotations

from typing import Any, Protocol, Sequence


class Placement(Protocol):
    def order(
        self, units: Sequence[dict[str, Any]], *, node_id: str, nbytes: int
    ) -> list[dict[str, Any]]: ...


class Spread:
    """Round-robin over all ACTIVE units; ``start`` staggers writers."""

    def __init__(self, start: int = 0) -> None:
        self._n = start

    def order(
        self, units: Sequence[dict[str, Any]], *, node_id: str, nbytes: int
    ) -> list[dict[str, Any]]:
        active = sorted(
            (u for u in units if u.get("state", "ACTIVE") == "ACTIVE"),
            key=lambda u: int(u["unit_id"]),
        )
        if not active:
            return active
        k = self._n % len(active)
        self._n += 1
        return active[k:] + active[:k]
