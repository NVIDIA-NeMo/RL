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
"""Thin wrapper around the Kubernetes API for KAI scheduler queries."""

from __future__ import annotations

from kubernetes import client, config
from kubernetes.client.exceptions import ApiException


def load_k8s_config():
    """Load in-cluster or kubeconfig."""
    try:
        config.load_incluster_config()
    except config.ConfigException:
        config.load_kube_config()


def get_queues(namespace: str | None = None) -> list[dict]:
    """Fetch all KAI scheduler queues and their resource status."""
    load_k8s_config()
    custom = client.CustomObjectsApi()
    result = custom.list_cluster_custom_object(
        group="scheduling.run.ai", version="v2", plural="queues"
    )
    queues = []
    for q in result.get("items", []):
        spec = q.get("spec", {})
        gpu = spec.get("resources", {}).get("gpu", {})
        queues.append(
            {
                "name": q["metadata"]["name"],
                "parent": spec.get("parentQueue", ""),
                "priority": spec.get("priority", ""),
                "gpu_quota": gpu.get("quota", -1),
                "gpu_limit": gpu.get("limit", -1),
                "gpu_weight": gpu.get("overQuotaWeight", 1),
                "preempt_min_runtime": spec.get("preemptMinRuntime", ""),
                "reclaim_min_runtime": spec.get("reclaimMinRuntime", ""),
            }
        )
    return queues


def get_gpu_occupancy(namespace: str = "default") -> dict:
    """Get current GPU allocation per node and per queue.

    Returns:
        {
            "nodes": [{"name": ..., "allocatable": N, "allocated": M}],
            "queues": [{"name": ..., "allocated_gpus": N}],
            "total_allocatable": N,
            "total_allocated": M,
        }
    """
    load_k8s_config()
    v1 = client.CoreV1Api()

    # Node-level GPU info.
    nodes_info = []
    total_allocatable = 0
    total_allocated = 0
    nodes = v1.list_node()
    for node in nodes.items:
        alloc = int(node.status.allocatable.get("nvidia.com/gpu", "0"))
        total_allocatable += alloc
        nodes_info.append(
            {"name": node.metadata.name, "allocatable": alloc, "allocated": 0}
        )

    # Count allocated GPUs per node from running pods.
    pods = v1.list_pod_for_all_namespaces(field_selector="status.phase=Running")
    for pod in pods.items:
        node_name = pod.spec.node_name
        for container in pod.spec.containers:
            limits = container.resources.limits or {}
            gpu_req = int(limits.get("nvidia.com/gpu", "0"))
            if gpu_req > 0:
                total_allocated += gpu_req
                for n in nodes_info:
                    if n["name"] == node_name:
                        n["allocated"] += gpu_req

    # Queue-level allocation from PodGroups.
    queue_alloc: dict[str, int] = {}
    for pod in pods.items:
        queue = (
            pod.metadata.labels.get("kai.scheduler/queue", "")
            if pod.metadata.labels
            else ""
        )
        if not queue:
            continue
        for container in pod.spec.containers:
            limits = container.resources.limits or {}
            gpu_req = int(limits.get("nvidia.com/gpu", "0"))
            queue_alloc[queue] = queue_alloc.get(queue, 0) + gpu_req

    return {
        "nodes": nodes_info,
        "queues": [
            {"name": k, "allocated_gpus": v} for k, v in sorted(queue_alloc.items())
        ],
        "total_allocatable": total_allocatable,
        "total_allocated": total_allocated,
    }


def submit_gang_rayjob(
    name: str,
    queue: str,
    image: str,
    entrypoint: str,
    num_gpus: int,
    gpus_per_worker: int = 1,
    namespace: str = "default",
    segment_size: int | None = None,
) -> str:
    """Submit a RayJob with gang scheduling via KAI.

    If segment_size is specified, creates a PodGroup with subgroups for
    topology-aware segment scheduling (equivalent to Slurm --segment=N).

    Returns the created RayJob name.
    """
    load_k8s_config()
    custom = client.CustomObjectsApi()

    num_workers = num_gpus // gpus_per_worker

    rayjob = {
        "apiVersion": "ray.io/v1",
        "kind": "RayJob",
        "metadata": {
            "name": name,
            "namespace": namespace,
            "labels": {"kai.scheduler/queue": queue},
        },
        "spec": {
            "entrypoint": entrypoint,
            "submissionMode": "HTTPMode",
            "shutdownAfterJobFinishes": True,
            "ttlSecondsAfterFinished": 60,
            "rayClusterSpec": {
                "rayVersion": "2.52.0",
                "headGroupSpec": {
                    "rayStartParams": {"object-store-memory": "200000000"},
                    "template": {
                        "spec": {
                            "schedulerName": "kai-scheduler",
                            "containers": [
                                {
                                    "name": "ray-head",
                                    "image": image,
                                    "resources": {
                                        "requests": {"cpu": "1", "memory": "4Gi"},
                                        "limits": {"cpu": "4", "memory": "16Gi"},
                                    },
                                    "ports": [
                                        {"containerPort": 6379, "name": "gcs-server"},
                                        {"containerPort": 8265, "name": "dashboard"},
                                    ],
                                    "readinessProbe": {
                                        "exec": {
                                            "command": ["ray", "health-check"],
                                        },
                                        "initialDelaySeconds": 10,
                                        "periodSeconds": 5,
                                        "timeoutSeconds": 5,
                                    },
                                }
                            ],
                        }
                    },
                },
                "workerGroupSpecs": [
                    {
                        "groupName": "gpu-workers",
                        "replicas": num_workers,
                        "minReplicas": num_workers,
                        "maxReplicas": num_workers,
                        "rayStartParams": {
                            "num-gpus": str(gpus_per_worker),
                            "object-store-memory": "200000000",
                        },
                        "template": {
                            "spec": {
                                "schedulerName": "kai-scheduler",
                                "containers": [
                                    {
                                        "name": "ray-worker",
                                        "image": image,
                                        "resources": {
                                            "requests": {
                                                "cpu": "1",
                                                "memory": "4Gi",
                                                "nvidia.com/gpu": str(gpus_per_worker),
                                            },
                                            "limits": {
                                                "cpu": "8",
                                                "memory": "32Gi",
                                                "nvidia.com/gpu": str(gpus_per_worker),
                                            },
                                        },
                                    }
                                ],
                            }
                        },
                    }
                ],
            },
        },
    }

    result = custom.create_namespaced_custom_object(
        group="ray.io",
        version="v1",
        namespace=namespace,
        plural="rayjobs",
        body=rayjob,
    )

    # If segment_size is specified, create a PodGroup with topology subgroups.
    if segment_size and segment_size < num_workers:
        import math

        num_segments = math.ceil(num_workers / segment_size)
        cluster_name = result["status"].get("rayClusterName", name)
        subgroups = []
        for i in range(num_segments):
            subgroups.append(
                {
                    "name": f"segment-{i}",
                    "minMember": min(segment_size, num_workers - i * segment_size),
                    "topologyConstraint": {
                        "topology": "cluster-topology",
                        "requiredTopologyLevel": "topology-rack",
                    },
                }
            )

        podgroup = {
            "apiVersion": "scheduling.run.ai/v2alpha2",
            "kind": "PodGroup",
            "metadata": {
                "name": f"pg-{name}",
                "namespace": namespace,
            },
            "spec": {
                "minMember": num_workers,
                "queue": queue,
                "subGroups": subgroups,
            },
        }
        try:
            custom.create_namespaced_custom_object(
                group="scheduling.run.ai",
                version="v2alpha2",
                namespace=namespace,
                plural="podgroups",
                body=podgroup,
            )
        except ApiException as e:
            if e.status != 409:
                raise

    return result["metadata"]["name"]
