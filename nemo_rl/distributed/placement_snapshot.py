# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Combine explicit model roles with Ray's node and CPU actor inventory."""

import asyncio
from collections import Counter, defaultdict
from typing import Any

from nemo_rl.distributed.placement_report import NodePlacement, render_placement

LABEL_PREFIX = "nrl.nvidia.com/"


def build_placement_snapshot(
    nodes: list[dict[str, Any]],
    actors: list[dict[str, Any]],
    roles: dict[str, dict[str, tuple[int, ...]]],
) -> list[NodePlacement]:
    devices: dict[str, dict[int, list[str]]] = defaultdict(lambda: defaultdict(list))
    for role, placement in roles.items():
        for node_id, gpu_ids in placement.items():
            for gpu_id in gpu_ids:
                devices[node_id][gpu_id].append(role)
    cpu: dict[str, Counter[str]] = defaultdict(Counter)
    for actor in actors:
        resources = actor["required_resources"]
        if not any(
            (key == "GPU" or key.startswith("GPU_group_")) and value > 0
            for key, value in resources.items()
        ):
            cpu[actor["node_id"]][actor["name"] or actor["class_name"]] += 1
    result = []
    for node in nodes:
        if not node["Alive"]:
            continue
        node_id = node["NodeID"]
        resources = node["Resources"]
        labels = node.get("Labels", {})
        domain = labels.get(LABEL_PREFIX + "nvlink-domain") or next(
            (
                key.removeprefix("nvlink_domain_")
                for key in resources
                if key.startswith("nvlink_domain_")
            ),
            "unknown",
        )
        rank = int(
            labels.get(LABEL_PREFIX + "topo-rank", resources.get("topo_rank", -1))
        )
        capacity = int(resources.get("GPU", 0))
        inventory = labels.get(LABEL_PREFIX + "gpu-ids", "")
        gpu_ids = (
            tuple(int(gpu) for gpu in inventory.split(".") if gpu) if capacity else ()
        )
        if not inventory and capacity:
            gpu_ids = tuple(sorted(devices[node_id]))
        if len(gpu_ids) > capacity or not set(devices[node_id]).issubset(gpu_ids):
            raise ValueError(
                f"GPU inventory disagrees with placement on {node['NodeManagerHostname']}"
            )
        result.append(
            NodePlacement(
                node_id=node_id,
                hostname=node["NodeManagerHostname"],
                domain=domain,
                topo_rank=rank,
                gpu_ids=gpu_ids,
                gpu_roles={
                    gpu: tuple(assigned) for gpu, assigned in devices[node_id].items()
                },
                cpu_actors=tuple(
                    f"{name} x{count}" if count > 1 else name
                    for name, count in sorted(cpu[node_id].items())
                ),
                advertised_gpus=capacity,
            )
        )
    unknown_nodes = (set(devices) | set(cpu)) - {node.node_id for node in result}
    if unknown_nodes:
        raise ValueError(
            f"Actor placement refers to missing Ray nodes: {sorted(unknown_nodes)}"
        )
    return result


def print_actor_placement(actor_args: Any, config: Any, **render_options: Any) -> None:
    # Keep pure snapshot/renderer tests independent of Ray's optional runtime.
    import ray

    roles = {
        "P": actor_args.train_cluster.gpu_placement,
        "G": actor_args.inference_cluster.gpu_placement,
    }
    if actor_args.reference_handle is not None:
        roles["R"] = actor_args.reference_handle.worker_group.cluster.gpu_placement
    elif config.loss_fn.reference_policy_kl_penalty > 0:
        roles["R"] = actor_args.train_cluster.gpu_placement
    if actor_args.value_handle is not None:
        roles["V"] = actor_args.train_cluster.gpu_placement
    teachers = {}
    for index, (alias, group) in enumerate(
        (actor_args.teacher_worker_groups or {}).items(), 1
    ):
        label = f"T{index}"
        roles[label] = group.worker_group.cluster.gpu_placement
        aliases = tuple(
            sorted(
                key
                for key, target in (actor_args.alias_to_group_alias or {}).items()
                if target == alias
            )
        ) or (alias,)
        teachers[label] = (
            aliases,
            config.on_policy_distillation.teacher_model_by_agent_name[aliases[0]],
        )
    inventory_error = None
    try:

        async def query_actors() -> dict[Any, Any]:
            # The dashboard State API caps both query stages at 10,000 actors.
            return await ray._private.worker.global_worker.gcs_client.async_get_all_actor_info(
                job_id=ray.JobID.from_hex(ray.get_runtime_context().get_job_id()),
                actor_state_name="ALIVE",
                timeout=30,
            )

        actors = asyncio.run(query_actors())
        actor_data = [
            {
                "node_id": actor.node_id.hex(),
                "name": actor.name,
                "class_name": actor.class_name,
                "required_resources": dict(actor.required_resources),
            }
            for actor in actors.values()
        ]
    except Exception as error:
        inventory_error = str(error)
        actor_data = []
    snapshot = build_placement_snapshot(ray.nodes(), actor_data, roles)
    if inventory_error is not None:
        for node in snapshot:
            node.cpu_actors = ("inventory unavailable",)
        print(f"CPU actor inventory incomplete: {inventory_error}", flush=True)
    print(render_placement(snapshot, teachers=teachers, **render_options), flush=True)
