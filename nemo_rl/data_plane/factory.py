# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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
"""Single entrypoint that maps a :class:`DataPlaneConfig` to a client."""

from __future__ import annotations

from nemo_rl.data_plane.interfaces import DataPlaneClient, DataPlaneConfig


def build_data_plane_client(
    cfg: DataPlaneConfig | None, *, bootstrap: bool = True
) -> DataPlaneClient:
    """Construct a TransferQueue-backed client.

    Callers should reach this function only when the TQ-mediated trainer
    (``grpo_sync``) is in use — the legacy trainer never touches the
    data plane and therefore should not call the factory at all. There
    is intentionally no NoOp fallback here: a NoOp client running inside
    ``grpo_sync`` would silently divorce the per-step lifecycle from the
    storage backend the trainer is meant to exercise.

    ``bootstrap`` is honored by the TransferQueue adapter:
      * True (driver, default): bootstraps the TQ controller from ``cfg``.
      * False (worker process): connects this process to the existing
        controller — workers must use this so they don't try to create a
        second named actor in the Ray cluster.
    """
    if cfg is None or not cfg.get("enabled", False):
        raise ValueError(
            "build_data_plane_client called with data_plane disabled. "
            "Use the legacy nemo_rl.algorithms.grpo.grpo_train trainer "
            "(which never engages the data plane) for that case."
        )

    impl = cfg["impl"]
    if impl == "transfer_queue":
        from nemo_rl.data_plane.adapters.transfer_queue import TQDataPlaneClient

        client: DataPlaneClient = TQDataPlaneClient(cfg, bootstrap=bootstrap)
    else:
        raise ValueError(f"unknown data_plane impl: {impl!r}")

    obs = cfg.get("observability") or {}
    if obs.get("enabled", False):
        from nemo_rl.data_plane.observability import (
            MetricsDataPlaneClient,
            print_event,
        )

        on_event = obs.get("callback") or print_event
        client = MetricsDataPlaneClient(client, on_event=on_event)
    return client
