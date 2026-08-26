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


def maybe_configure_data_plane_env(cfg: DataPlaneConfig | None) -> None:
    """Set backend env vars that must be identical in every process.

    Call this on the driver **before** ``init_ray()``: ``init_ray`` snapshots the
    driver's environment into ``runtime_env["env_vars"]`` and hands it to every
    Ray worker, which are fresh processes, so the value is in place before they
    run any engine code.

    The binding constraint is not ``init_ray`` but that this must run before
    anything imports the backend's engine, which snapshots its configuration as
    it loads. :func:`~nemo_rl.data_plane.adapters.transfer_queue_env.configure_engine_env`
    raises rather than silently no-op'ing if that is violated.

    Subprocesses the driver did not spawn through Ray inherit whatever the
    driver's environment held at fork, so they are covered as long as this ran
    first.

    No-op when the data plane is disabled or the backend has no such knobs.

    Args:
        cfg: Data-plane config, or ``None`` when the data plane is off.
    """
    if cfg is None or not cfg["enabled"]:
        return

    impl = cfg["impl"]
    if impl == "transfer_queue":
        # transfer_queue_env, not the adapter — importing the adapter loads
        # mooncake, which is what this call has to precede.
        from nemo_rl.data_plane.adapters.transfer_queue_env import (
            configure_engine_env,
        )

        configure_engine_env(cfg)
    else:
        raise ValueError(f"unknown data_plane impl: {impl!r}")


def build_data_plane_client(
    cfg: DataPlaneConfig | None, *, bootstrap: bool = True
) -> DataPlaneClient:
    """Construct the configured data-plane client.

    Dispatches on ``cfg["impl"]``. Only ``"transfer_queue"`` ships today;
    other adapters can be added behind this factory without touching
    call sites. Raises if data_plane is disabled — the legacy trainer
    (``nemo_rl.algorithms.grpo.grpo_train``) should be used in that case
    rather than a NoOp fallback here.

    Args:
        cfg: Data-plane config; must have ``enabled=True``.
        bootstrap: ``True`` on the driver — bootstraps the TQ
            controller. ``False`` on worker processes — connects to the
            existing controller (avoids creating a second named actor).

    Returns:
        A configured ``DataPlaneClient``; wrapped in
        :class:`MetricsDataPlaneClient` when observability is enabled.
    """
    if cfg is None or not cfg["enabled"]:
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
            log_event,
        )

        on_event = obs.get("callback") or log_event
        # pyrefly: obs.get returns Any, can't narrow to the expected callback type.
        client = MetricsDataPlaneClient(client, on_event=on_event)  # type: ignore[bad-argument-type]
    return client
