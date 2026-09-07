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

from typing import TYPE_CHECKING, Any, Callable, Optional

from nemo_rl.data_plane.interfaces import DataPlaneClient, DataPlaneConfig

if TYPE_CHECKING:
    from nemo_rl.algorithms.grpo import MasterConfig


def data_plane_enabled(cfg: DataPlaneConfig | None) -> bool:
    """Whether the data plane is on. ``None`` (key absent) means off."""
    return cfg is not None and bool(cfg.get("enabled", False))


def select_sync_trainer(
    master_config: "MasterConfig", *, label: str = "GRPO"
) -> Callable[..., Any]:
    """Pick the synchronous trainer based on ``data_plane.enabled``.

    Shared by every launcher so trainer choice cannot drift between them.
    Pairs with :func:`make_policy_factory`: turning the data plane on means
    picking *both* the TQ-mediated trainer and a ``TQPolicy``, and a launcher
    that picked only one would fail after a full model load.

    Args:
        master_config: The resolved config; only ``data_plane`` is read.
        label: Algorithm name for the progress line (e.g. ``"VLM GRPO"``).
    """
    if data_plane_enabled(master_config.data_plane):
        from nemo_rl.algorithms.grpo_sync import grpo_train_sync

        print(f"🚀 Running synchronous {label} training (TransferQueue)")
        return grpo_train_sync

    from nemo_rl.algorithms.grpo import grpo_train

    print(f"🚀 Running synchronous {label} training (legacy)")
    return grpo_train


def make_policy_factory(
    cfg: DataPlaneConfig | None,
) -> Optional[Callable[..., Any]]:
    """The ``policy_factory`` for ``setup()``, or ``None`` for a plain ``Policy``.

    Lives at the launcher level so the legacy trainer stays data-plane-agnostic
    (architectural invariant — see
    ``tests/unit/data_plane/test_architecture_invariants.py``).
    """
    if not data_plane_enabled(cfg):
        return None

    from nemo_rl.models.policy.tq_policy import TQPolicy

    def _make_policy(**kwargs: Any) -> TQPolicy:
        return TQPolicy(**kwargs, dp_cfg=cfg)

    return _make_policy


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
