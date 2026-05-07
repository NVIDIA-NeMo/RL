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
"""Adapter wiring :class:`DataPlaneClient` onto the ``transfer_queue`` package.

Pure plumbing — it owns the TQ controller / client handle and translates
:class:`KVBatchMeta` ↔ TQ's own ``BatchMeta`` / ``KVBatchMeta``. No
business logic. Backend init is lifted from
``rl-arena/arena/backends.py``; the call shapes are lifted from
``rl-arena/arena/dataplane_client.py``.
"""

from __future__ import annotations

import os
import socket
import subprocess
import time
from dataclasses import dataclass, field
from importlib import resources
from typing import Any

import torch
from tensordict import TensorDict

from nemo_rl.data_plane.interfaces import (
    DataPlaneClient,
    DataPlaneConfig,
    KVBatchMeta,
)


# ──────────────────────────────────────────────────────────────────────────
# Lazy import of transfer_queue. Mirrors verl's pattern at
# ``verl/utils/transferqueue_utils.py:35-57`` — NeMo-RL still imports
# cleanly without TQ installed; failure is deferred to construction time.
# ──────────────────────────────────────────────────────────────────────────


def _tq():  # pragma: no cover - trivially exercised by smoke tests
    try:
        import transfer_queue as tq
    except ImportError as e:  # noqa: F841
        raise ImportError(
            "transfer_queue is not installed. It is a base dependency of "
            "nemo-rl — try `uv sync` to refresh, or `pip install "
            "TransferQueue==0.1.6` if you're not using uv."
        ) from e
    return tq


# ──────────────────────────────────────────────────────────────────────────
# Backend init — lifted from rl-arena/arena/backends.py.
# ──────────────────────────────────────────────────────────────────────────


def _get_head_node_ip() -> str:
    try:
        return socket.gethostbyname(socket.gethostname())
    except Exception:
        return ""


def _usb0_down() -> None:
    cmds = [
        "ifconfig usb0 0.0.0.0 2>/dev/null",
        "ifconfig usb0 down 2>/dev/null",
        "ip link set usb0 down 2>/dev/null",
        "ip addr flush dev usb0 2>/dev/null",
    ]
    try:
        subprocess.run(
            ["sh", "-c", "; ".join(cmds)], check=False, capture_output=True
        )
    except Exception:
        pass


def _mooncake_transport_config() -> dict:
    protocol = os.environ.get("MC_MOONCAKE_PROTOCOL", "tcp")
    if protocol != "rdma":
        return {"protocol": "tcp"}
    device = os.environ.get("MC_MOONCAKE_DEVICE", "")
    if not device:
        try:
            out = subprocess.run(
                [
                    "sh",
                    "-c",
                    "for d in /sys/class/infiniband/mlx5_*/ports/1/link_layer; do "
                    "  test -f $d && grep -q Ethernet $d && basename $(dirname $(dirname $d)); "
                    "done | head -1",
                ],
                check=False,
                capture_output=True,
                text=True,
            ).stdout.strip()
            device = out or ""
        except Exception:
            device = ""
    if device:
        os.environ.setdefault("MC_GID_INDEX", os.environ.get("MC_GID_INDEX", "3"))
    return {"protocol": "rdma", "device_name": device}


def _connect_existing() -> None:
    """Worker-process path: connect this process's client to the
    already-running named controller actor in the Ray cluster. Mirrors
    rl-arena/arena/dataplane_client.py's `tq.init()` (no args) call.
    """
    _tq().init()


_TQ_RUNTIME_ENV_PATCHED = False


def _patch_tq_actor_runtime_env() -> None:
    """Inject Ray ``runtime_env={"pip": ["TransferQueue==0.1.6"]}`` into the
    ``.options()`` calls on TQ's internal actor classes (``SimpleStorageUnit``,
    ``TransferQueueController``).

    **Why**: TQ spawns these actors via ``Cls.options(...).remote(...)`` with
    no runtime_env. They inherit the *job-level* runtime_env that the driver
    set when calling ``ray.init``. In a multi-node container deployment where
    each node has its own ``/opt/nemo_rl_venv`` (per-container filesystem),
    ``uv sync`` on the driver only updates ray-head's venv — ray-worker-N's
    venv is stale and lacks ``transfer_queue``. The TQ storage actor on a
    worker node then dies with ``ModuleNotFoundError: No module named
    'transfer_queue'``.

    This monkey-patch makes Ray pip-install ``TransferQueue==0.1.6`` into a
    per-actor runtime_env on first spawn (cached per-node by Ray after that),
    sidestepping the per-node venv divergence entirely. Idempotent — only
    patches once per process.

    Trade-offs:
      * Requires PyPI access from each Ray worker node. The user's cluster
        has it (we resolved TQ via PyPI when building the driver venv).
      * Couples our adapter to TQ's *internal* class layout. If TQ renames or
        restructures these classes in a future release, the patch becomes a
        no-op (with a logged warning) and we fall back to the
        per-node-uv-sync workaround.
    """
    global _TQ_RUNTIME_ENV_PATCHED
    if _TQ_RUNTIME_ENV_PATCHED:
        return

    runtime_env = {"pip": ["TransferQueue==0.1.6"]}

    def _install(cls) -> bool:
        if not hasattr(cls, "options"):
            return False
        original = cls.options

        def patched(*args, **kwargs):
            kwargs.setdefault("runtime_env", runtime_env)
            return original(*args, **kwargs)

        cls.options = patched  # type: ignore[method-assign]
        return True

    patched_any = False
    try:
        from transfer_queue.storage.simple_backend import SimpleStorageUnit

        patched_any |= _install(SimpleStorageUnit)
    except ImportError:
        pass
    try:
        from transfer_queue.controller import TransferQueueController

        patched_any |= _install(TransferQueueController)
    except ImportError:
        pass

    if not patched_any:
        # Soft-fail: TQ may have moved its actor classes. The driver will
        # still work; multi-node TQ may need the per-node `uv sync` workaround.
        import warnings

        warnings.warn(
            "Could not patch TQ actor classes for runtime_env injection. "
            "Multi-node TQ may fail with ModuleNotFoundError: 'transfer_queue' "
            "on worker nodes. Workaround: run `uv sync` inside each node's "
            "container before the driver runs.",
            RuntimeWarning,
            stacklevel=2,
        )
    _TQ_RUNTIME_ENV_PATCHED = True


def _init_tq(cfg: DataPlaneConfig) -> None:
    """Driver-process path: bootstrap the TQ controller for the chosen backend."""
    from omegaconf import OmegaConf

    tq = _tq()
    base = OmegaConf.load(str(resources.files("transfer_queue") / "config.yaml"))

    backend = cfg.get("backend", "simple")
    storage_capacity = cfg.get("storage_capacity", 1_000_000)
    num_storage_units = cfg.get("num_storage_units", 2)

    # polling_mode=True: controller returns empty BatchMeta instead of raising
    # TimeoutError when no samples are ready yet. The client-side blocking
    # loop in `get_meta` drives the retry cadence.
    controller_overlay = {"controller": {"polling_mode": True}}

    if backend == "simple":
        overlay = {
            **controller_overlay,
            "backend": {
                "storage_backend": "SimpleStorage",
                "SimpleStorage": {
                    "total_storage_size": storage_capacity,
                    "num_data_storage_units": num_storage_units,
                },
            },
        }
    elif backend == "mooncake_cpu":
        _usb0_down()
        head_ip = _get_head_node_ip()
        if head_ip and not head_ip.startswith("169.254."):
            os.environ.setdefault("MC_TCP_BIND_ADDRESS", head_ip)
        overlay = {
            **controller_overlay,
            "backend": {
                "storage_backend": "MooncakeStore",
                "MooncakeStore": {
                    "global_segment_size": 4 * 1024**3,
                    "local_buffer_size": 1 * 1024**3,
                    "metadata_server": f"{head_ip}:50050",
                    "master_server_address": f"{head_ip}:50051",
                    **_mooncake_transport_config(),
                },
            },
        }
    else:
        raise ValueError(f"unknown TQ backend: {backend!r}")

    conf = OmegaConf.merge(base, overlay)

    # Inject runtime_env into TQ's actor spawn so SimpleStorageUnit /
    # TransferQueueController land on workers with transfer_queue available
    # — see _patch_tq_actor_runtime_env() docstring for the why.
    _patch_tq_actor_runtime_env()

    tq.init(conf=conf)


# ──────────────────────────────────────────────────────────────────────────
# P3 — adapter-level enforcement that nothing but tensors crosses the bus.
# ──────────────────────────────────────────────────────────────────────────


def _to_wire(td: TensorDict) -> TensorDict:
    # Walk via keys() + get() rather than items() — see noop adapter for
    # the rationale (NonTensorData entries can slip past items()).
    bad = []
    for k in td.keys(include_nested=True, leaves_only=True):
        v = td.get(k)
        if not isinstance(v, torch.Tensor):
            bad.append(k)
    if bad:
        raise TypeError(
            f"kv_batch_put received non-tensor leaves: {bad}. "
            "Tensorize via codec helpers, use `tags=` for primitives, "
            "or use the Ray object store for arbitrary Python objects."
        )
    return td.detach().contiguous()


# ──────────────────────────────────────────────────────────────────────────
# Per-partition record kept client-side for register_partition semantics
# (TQ creates partitions implicitly on first put — this is bookkeeping
# that lets `kv_clear(keys=None)` and the consumer-task list survive
# without a controller round-trip).
# ──────────────────────────────────────────────────────────────────────────


@dataclass
class _PartitionRecord:
    fields: list[str]
    num_samples: int
    consumer_tasks: list[str]
    grpo_group_size: int | None
    enums: dict[str, list[str]]
    seen_keys: set[str] = field(default_factory=set)


class TQDataPlaneClient(DataPlaneClient):
    """Adapter façade — maps NeMo-RL calls onto TransferQueue's public API."""

    def __init__(self, cfg: DataPlaneConfig, *, bootstrap: bool = True) -> None:
        """Construct a TQ-backed client.

        Args:
            cfg: data-plane config (backend selection, poll cadence, …).
            bootstrap: True (driver) bootstraps the TQ controller using
                ``cfg``. False (worker) connects this process to an
                already-running named controller actor in the Ray
                cluster — ``cfg`` is then only consulted for client-side
                knobs (poll interval).
        """
        if bootstrap:
            _init_tq(cfg)
        else:
            _connect_existing()
        self._tq = _tq()
        self._poll_interval_s = cfg.get("get_meta_poll_interval_s", 0.5)
        self._partitions: dict[str, _PartitionRecord] = {}
        self._closed = False

    # ── (A) task-mediated ───────────────────────────────────────────────

    def register_partition(
        self,
        partition_id: str,
        fields: list[str],
        num_samples: int,
        consumer_tasks: list[str],
        grpo_group_size: int | None = None,
        enums: dict[str, list[str]] | None = None,
    ) -> None:
        # Client-side bookkeeping. TQ creates partitions implicitly on
        # first kv_batch_put; pre-registration is for our own validation
        # and the kv_clear(keys=None) recovery path.
        self._partitions[partition_id] = _PartitionRecord(
            fields=list(fields),
            num_samples=int(num_samples),
            consumer_tasks=list(consumer_tasks),
            grpo_group_size=grpo_group_size,
            enums=dict(enums) if enums else {},
        )

    def get_meta(
        self,
        partition_id: str,
        task_name: str,
        required_fields: list[str],
        batch_size: int,
        dp_rank: int | None = None,
        blocking: bool = True,
        timeout_s: float = 60.0,
    ) -> KVBatchMeta:
        client = self._tq.get_client()
        deadline = time.time() + max(0.0, timeout_s)
        sampling_config: dict[str, Any] = {}
        if dp_rank is not None:
            sampling_config["dp_rank"] = dp_rank

        while True:
            tq_meta = client.get_meta(
                data_fields=list(required_fields),
                batch_size=int(batch_size),
                partition_id=partition_id,
                task_name=task_name,
                mode="fetch",
                sampling_config=sampling_config,
            )
            if getattr(tq_meta, "size", 0) > 0:
                break
            if not blocking:
                return KVBatchMeta(
                    partition_id=partition_id,
                    task_name=task_name,
                    keys=[],
                    fields=list(required_fields),
                )
            if time.time() >= deadline:
                raise TimeoutError(
                    f"get_meta(partition={partition_id}, task={task_name}) "
                    f"timed out after {timeout_s}s"
                )
            time.sleep(self._poll_interval_s)

        keys: list[str] = client.kv_retrieve_keys(
            global_indexes=list(tq_meta.global_indexes),
            partition_id=partition_id,
        )

        # Lift sequence lengths from the rollout-side `input_lengths` tag
        # if present. Driver-side balancing (Stage 4) needs this; the
        # task-mediated path does not.
        tags = tq_meta.custom_meta or [{} for _ in keys]
        seqlens: list[int] | None = None
        if tags and any("input_lengths" in t for t in tags):
            seqlens = [int(t.get("input_lengths", 0)) for t in tags]

        return KVBatchMeta(
            partition_id=partition_id,
            task_name=task_name,
            keys=keys,
            fields=list(required_fields),
            sequence_lengths=seqlens,
        )

    def get_data(
        self,
        meta: KVBatchMeta,
        select_fields: list[str] | None = None,
    ) -> TensorDict:
        fields = select_fields if select_fields is not None else meta.fields
        if fields is None:
            raise ValueError(
                "get_data requires either select_fields or meta.fields; "
                "silently fetching all fields is forbidden (P2)."
            )
        return self.kv_batch_get(meta.keys, meta.partition_id, list(fields))

    def check_consumption_status(
        self, partition_id: str, task_names: list[str]
    ) -> bool:
        client = self._tq.get_client()
        for t in task_names:
            try:
                ok = client.check_consumption_status(
                    task_name=t, partition_id=partition_id
                )
            except Exception:
                return False
            if not ok:
                return False
        return True

    # ── (B) direct-by-key ──────────────────────────────────────────────

    def kv_batch_put(
        self,
        keys: list[str],
        partition_id: str,
        fields: TensorDict | None = None,
        tags: list[dict[str, Any]] | None = None,
    ) -> KVBatchMeta:
        if not keys:
            return KVBatchMeta(
                partition_id=partition_id, task_name=None, keys=[], fields=None
            )
        if tags is None:
            tags = [{} for _ in keys]

        wire_fields: TensorDict | None = None
        field_names: list[str] | None = None
        if fields is not None:
            wire_fields = _to_wire(fields)
            field_names = list(wire_fields.keys())

        self._tq.kv_batch_put(
            keys=list(keys),
            partition_id=partition_id,
            fields=wire_fields,
            tags=tags,
        )

        rec = self._partitions.get(partition_id)
        if rec is not None:
            rec.seen_keys.update(keys)

        return KVBatchMeta(
            partition_id=partition_id,
            task_name=None,
            keys=list(keys),
            fields=field_names,
        )

    def kv_batch_get(
        self,
        keys: list[str],
        partition_id: str,
        select_fields: list[str] | None = None,
    ) -> TensorDict:
        if not keys:
            return TensorDict({}, batch_size=(0,))
        return self._tq.kv_batch_get(
            keys=list(keys),
            partition_id=partition_id,
            select_fields=list(select_fields) if select_fields else None,
        )

    def kv_clear(self, keys: list[str] | None, partition_id: str) -> None:
        if keys is None:
            rec = self._partitions.pop(partition_id, None)
            keys = list(rec.seen_keys) if rec is not None else []
            if not keys:
                try:
                    listing = self._tq.kv_list(partition_id=partition_id)
                    keys = list(listing.get(partition_id, {}).keys())
                except Exception:
                    keys = []
        else:
            self._partitions.pop(partition_id, None)
        if keys:
            self._tq.kv_clear(keys=list(keys), partition_id=partition_id)

    # ── (C) lifecycle ──────────────────────────────────────────────────

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            self._tq.close()
        except Exception:
            pass
