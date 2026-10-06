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
"""Bootstrap provider: start the BlobDirectory and the storage units.

Called once by ``transfer_queue.init`` in the process that creates the TQ
controller. Reads ``conf.backend.NixlStore`` and writes back the directory
actor name so attaching processes can find it.

Unit placement follows the two existing NeMo-RL data planes:

- SimpleStorage style: ``num_storage_units`` total, round-robin over the
  eligible nodes (optionally filtered by ``required_node_resource``).
- PR #4465 style: ``storage_units_per_node`` on each eligible node, where
  the eligible set is ``node_ids`` (the caller resolves
  ``storage_unit_placement: all | inference | train`` to Ray node ids with
  :func:`placement_group_node_ids`, exactly like
  ``mooncake_storage_unit.start_storage_units``).

``transfer_queue.close`` does not know how to stop third-party backends,
so :func:`shutdown` is provided for the owner process to call after it.
"""

from __future__ import annotations

import os
from typing import Any, Iterable

import ray
from ray.util.placement_group import placement_group_table
from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy
from transfer_queue.storage.bootstrap.provider import StorageBootstrapProvider

from nemo_rl.data_plane.nixl.directory import BlobDirectory
from nemo_rl.data_plane.nixl.storage_unit import NixlStorageUnit
from nemo_rl.data_plane.nixl.tq.kv_client import DEFAULT_NAMESPACE, _as_dict


def _unit_name(i: int) -> str:
    return f"NixlStorageUnit#{i}"


# ----------------------------------------------------------------------------- node selection
def alive_node_ids(required_node_resource: str | None = None) -> list[str]:
    ids = sorted(
        n["NodeID"]
        for n in ray.nodes()
        if n.get("Alive", False)
        and (
            required_node_resource is None
            or n.get("Resources", {}).get(required_node_resource, 0) > 0
        )
    )
    if not ids:
        raise RuntimeError("No alive Ray nodes found. Is Ray initialized?")
    return ids


def placement_group_node_ids(placement_groups: Iterable[Any]) -> list[str]:
    """Ray node ids occupied by the given placement groups (one GCS read).

    Pass the inference and/or train clusters' placement groups to implement
    ``storage_unit_placement: all | inference | train``.
    """
    table = placement_group_table()
    return sorted(
        {
            node_id
            for pg in placement_groups
            for node_id in table[pg.id.hex()]["bundles_to_node_id"].values()
        }
    )


def unit_node_ids(cfg: dict[str, Any]) -> list[str]:
    """One node id per unit to create, from the backend config block."""
    nodes = list(cfg.get("node_ids") or []) or alive_node_ids(
        cfg.get("required_node_resource")
    )
    per_node = cfg.get("storage_units_per_node")
    if per_node:
        return [n for n in nodes for _ in range(int(per_node))]
    total = int(cfg.get("num_storage_units", 2))
    return [nodes[i % len(nodes)] for i in range(total)]


# ----------------------------------------------------------------------------- lifecycle
def _kill_named(name: str, namespace: str) -> bool:
    try:
        h = ray.get_actor(name, namespace=namespace)
    except ValueError:
        return False
    ray.kill(h, no_restart=True)
    return True


def shutdown(conf_or_backend_cfg: Any) -> None:
    """Kill the BlobDirectory and every storage unit of a NixlStore system.

    Units are named ``NixlStorageUnit#<i>``; kill consecutively until a name
    is missing, so a larger earlier system in the same namespace is swept too.
    """
    cfg = conf_or_backend_cfg
    if hasattr(cfg, "backend"):
        cfg = cfg.backend.NixlStore
    cfg = _as_dict(cfg)
    namespace = cfg.get("namespace", DEFAULT_NAMESPACE)
    i = 0
    while _kill_named(_unit_name(i), namespace):
        i += 1
    _kill_named(cfg.get("directory_name", "BlobDirectory"), namespace)


@StorageBootstrapProvider.register_provider("NixlStore")
def initialize_nixl_storage(conf: Any) -> dict[str, Any]:
    cfg = _as_dict(conf.backend.NixlStore)
    namespace = cfg.get("namespace", DEFAULT_NAMESPACE)
    dir_name = cfg.get("directory_name", "BlobDirectory")
    nixl_cfg = _as_dict(cfg.get("nixl") or {})
    store_cfg = _as_dict(cfg.get("store") or {})
    kind = store_cfg.get("kind", "unit")

    # A fresh TQ controller means a fresh data plane: stale named actors from a
    # previous system in the same namespace would otherwise block creation.
    shutdown(cfg)

    directory = BlobDirectory.options(name=dir_name, namespace=namespace).remote()
    handles: dict[str, Any] = {"BlobDirectory": directory}

    if kind == "file":
        # Actor-less store: one file per blob under store.root (Lustre / NVMe).
        root = store_cfg.get("root")
        if not root:
            raise ValueError(
                "backend.NixlStore.store.root is required for store.kind='file'"
            )
        os.makedirs(root, exist_ok=True)
    else:
        units = []
        per_node: dict[str, int] = {}
        for i, node_id in enumerate(unit_node_ids(cfg)):
            numa_index = per_node.get(
                node_id, 0
            )  # k-th unit on this node -> socket k % sockets
            per_node[node_id] = numa_index + 1
            unit = NixlStorageUnit.options(
                name=_unit_name(i),
                namespace=namespace,
                scheduling_strategy=NodeAffinitySchedulingStrategy(
                    node_id=node_id, soft=False
                ),
            ).remote(
                i,
                int(cfg.get("unit_slab_bytes", 1 << 30)),
                read_pin_s=float(cfg.get("read_pin_s", 60.0)),
                nixl_backend=nixl_cfg.get("backend_name", "UCX"),
                nixl_init_params=nixl_cfg.get("backend_init_params") or {},
                require_rdma=bool(nixl_cfg.get("require_rdma", True)),
                numa_mode=str(cfg.get("numa", "auto")),
                numa_index=numa_index,
                directory=directory,
            )
            units.append(unit)
            handles[_unit_name(i)] = unit
        infos = ray.get([u.info.remote() for u in units])
        ray.get([directory.register_unit.remote(info) for info in infos])

    conf.backend.NixlStore.directory_name = dir_name
    conf.backend.NixlStore.namespace = namespace
    return handles
