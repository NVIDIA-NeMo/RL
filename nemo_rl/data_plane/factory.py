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

from nemo_rl.data_plane.interfaces import (
    DataPlaneClient,
    DataPlaneConfig,
    DataPlaneRuntimeConfig,
    LocalDataPlaneConfig,
)
from nemo_rl.telemetry.setup import telemetry_enabled_in_env


def data_plane_enabled(cfg: DataPlaneConfig | None) -> bool:
    """Whether the data plane is on. ``None`` (key absent) means off."""
    return cfg is not None and bool(cfg.get("enabled", False))


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
    cfg: DataPlaneRuntimeConfig | None,
    *,
    bootstrap: bool = True,
    checkpointing: bool = False,
    segment_size: int | None = None,
    storage_unit_node_ids: list[str] | None = None,
) -> DataPlaneClient:
    """Construct the configured data-plane client.

    Dispatches on the configured implementation. TransferQueue supports
    cross-process transfer; the local adapter keeps colocated SFT batches in
    one process. Raises if data_plane is disabled — the legacy trainer
    (``nemo_rl.algorithms.grpo.grpo_train``) should be used in that case
    rather than a NoOp fallback here.

    Args:
        cfg: Data-plane config; must have ``enabled=True``.
        bootstrap: ``True`` on the driver — bootstraps the TQ
            controller. ``False`` on worker processes — connects to the
            existing controller (avoids creating a second named actor).
        checkpointing: Prepare storage for saving or restoring data-plane state.
            Derived by the caller from its existing checkpoint settings and
            resume path. Only used at bootstrap; workers inherit the mode from TQ.
        segment_size: mooncake_cpu only, worker processes only: Mooncake memory
            this process owns, in place of the controller's
            ``global_segment_size`` (0: a client that owns nothing). ``None``
            keeps the controller's value.
        storage_unit_node_ids: Bootstrap only: one Ray node ID per storage unit,
            planned by the SingleController (see
            :func:`~nemo_rl.data_plane.mooncake_storage_unit.plan_storage_unit_nodes`).
            Storage-unit settings without a plan fail at bootstrap.

    Returns:
        A configured ``DataPlaneClient``; wrapped in
        :class:`MetricsDataPlaneClient` when observability is enabled.
    """
    if cfg is None:
        raise ValueError(
            "build_data_plane_client called with data_plane disabled. "
            "Use the legacy nemo_rl.algorithms.grpo.grpo_train trainer "
            "(which never engages the data plane) for that case."
        )
    if isinstance(cfg, LocalDataPlaneConfig):
        enabled = cfg.enabled
    else:
        enabled = cfg["enabled"]
    if not enabled:
        raise ValueError(
            "build_data_plane_client called with data_plane disabled. "
            "Use the legacy nemo_rl.algorithms.grpo.grpo_train trainer "
            "(which never engages the data plane) for that case."
        )

    impl = cfg.impl if isinstance(cfg, LocalDataPlaneConfig) else cfg["impl"]
    if impl == "transfer_queue":
        from nemo_rl.data_plane.adapters.transfer_queue import TQDataPlaneClient

        assert not isinstance(cfg, LocalDataPlaneConfig)
        client: DataPlaneClient = TQDataPlaneClient(
            cfg,
            bootstrap=bootstrap,
            checkpointing=checkpointing,
            segment_size=segment_size,
            storage_unit_node_ids=storage_unit_node_ids,
        )
    elif impl == "local":
        from nemo_rl.data_plane.adapters.local import LocalDataPlaneClient

        local_cfg = (
            cfg
            if isinstance(cfg, LocalDataPlaneConfig)
            else LocalDataPlaneConfig.model_validate(cfg)
        )
        client = LocalDataPlaneClient(local_cfg)
    else:
        raise ValueError(f"unknown data_plane impl: {impl!r}")

    obs = (
        cfg.observability
        if isinstance(cfg, LocalDataPlaneConfig)
        else cfg.get("observability")
    ) or {}
    obs_enabled = obs.get("enabled", False)
    # Telemetry alone installs the wrapper, for its spans.
    if obs_enabled or telemetry_enabled_in_env():
        from nemo_rl.data_plane.observability import MetricsDataPlaneClient

        # Callback and hash check are observability features; off for
        # telemetry-only runs.
        # pyrefly: obs.get returns Any, can't narrow to the expected callback type.
        client = MetricsDataPlaneClient(
            client,  # type: ignore[bad-argument-type]
            on_event=obs.get("callback") if obs_enabled else None,  # type: ignore[bad-argument-type]
            verify_tensor_hash=obs_enabled and bool(obs.get("verify_tensor_hash")),
            observability_enabled=obs_enabled,
        )
    return client
