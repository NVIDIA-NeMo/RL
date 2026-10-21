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
"""BlobDirectory: unit membership and blob → unit placement.

Deliberately tiny and not a catalog: no keys, fields, produced bits or
cursors (the TransferQueue controller owns those in Phase A). It exists
because TQ restores storage before the controller, so the locations the
controller stores must be placement-free; this actor holds the placement.
"""

from __future__ import annotations

from typing import Any

import ray

LOST = "__LOST__"

ACTIVE = "ACTIVE"
DRAINING = "DRAINING"
DEAD = "DEAD"


# No restarts: all state is in memory and units register only at bootstrap, so
# a restarted directory would be empty and fail every op without saying why.
@ray.remote(num_cpus=0, max_restarts=0, max_task_retries=0)
class BlobDirectory:
    def __init__(self) -> None:
        self._epoch = 0
        self._units: dict[int, dict[str, Any]] = {}
        self._blob_unit: dict[str, int] = {}

    # ------------------------------------------------------------------ membership
    def register_unit(self, info: dict[str, Any]) -> int:
        info = dict(info)
        info.setdefault("state", ACTIVE)
        self._units[int(info["unit_id"])] = info
        self._epoch += 1
        return self._epoch

    def set_unit_state(self, unit_id: int, state: str) -> int:
        if unit_id in self._units:
            self._units[unit_id]["state"] = state
            self._epoch += 1
            if state == DEAD:
                for b, u in list(self._blob_unit.items()):
                    if u == unit_id:
                        self._blob_unit[b] = -2
        return self._epoch

    def units(self, min_epoch: int = 0) -> tuple[int, list[dict[str, Any]]]:
        if self._epoch <= min_epoch:
            return self._epoch, []
        return self._epoch, [dict(u) for u in self._units.values()]

    def epoch(self) -> int:
        return self._epoch

    # ------------------------------------------------------------------ placement
    def put_blobs(self, items: list[tuple[str, int]]) -> None:
        """``(blob_id, unit_id)`` for every blob placed."""
        for blob_id, unit_id in items:
            self._blob_unit[blob_id] = int(unit_id)

    def where(self, blob_ids: list[str]) -> dict[str, int | str]:
        out: dict[str, int | str] = {}
        for b in blob_ids:
            u = self._blob_unit.get(b)
            if u is None or u == -2:
                out[b] = LOST
            else:
                st = self._units.get(u, {}).get("state", DEAD)
                out[b] = LOST if st == DEAD else u
        return out

    def stats(self) -> dict[str, Any]:
        return {
            "epoch": self._epoch,
            "units": {u: i.get("state") for u, i in self._units.items()},
            "blobs": len(self._blob_unit),
        }
