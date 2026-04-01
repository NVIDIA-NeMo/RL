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
"""Unit tests for nrl_k8s.k8s_client (mocked K8s API)."""

from unittest.mock import MagicMock, patch

import pytest


@pytest.fixture(autouse=True)
def mock_k8s_config():
    """Prevent real K8s config loading in all tests."""
    with patch("nrl_k8s.k8s_client.load_k8s_config"):
        yield


class TestGetQueues:
    def test_parses_queue_fields(self):
        mock_custom = MagicMock()
        mock_custom.list_cluster_custom_object.return_value = {
            "items": [
                {
                    "metadata": {"name": "org"},
                    "spec": {
                        "resources": {
                            "gpu": {"quota": -1, "limit": -1, "overQuotaWeight": 1}
                        }
                    },
                },
                {
                    "metadata": {"name": "priority-team"},
                    "spec": {
                        "parentQueue": "org",
                        "priority": 200,
                        "preemptMinRuntime": "4h",
                        "reclaimMinRuntime": "15m",
                        "resources": {
                            "gpu": {"quota": 1, "limit": 2, "overQuotaWeight": 2}
                        },
                    },
                },
            ]
        }

        with patch(
            "nrl_k8s.k8s_client.client.CustomObjectsApi", return_value=mock_custom
        ):
            from nrl_k8s.k8s_client import get_queues

            queues = get_queues()

        assert len(queues) == 2
        assert queues[0]["name"] == "org"
        assert queues[0]["gpu_quota"] == -1
        assert queues[1]["name"] == "priority-team"
        assert queues[1]["parent"] == "org"
        assert queues[1]["priority"] == 200
        assert queues[1]["gpu_quota"] == 1
        assert queues[1]["gpu_limit"] == 2
        assert queues[1]["gpu_weight"] == 2
        assert queues[1]["preempt_min_runtime"] == "4h"
        assert queues[1]["reclaim_min_runtime"] == "15m"

    def test_empty_cluster(self):
        mock_custom = MagicMock()
        mock_custom.list_cluster_custom_object.return_value = {"items": []}

        with patch(
            "nrl_k8s.k8s_client.client.CustomObjectsApi", return_value=mock_custom
        ):
            from nrl_k8s.k8s_client import get_queues

            assert get_queues() == []


class TestGetGpuOccupancy:
    def _make_node(self, name, gpu_count):
        node = MagicMock()
        node.metadata.name = name
        node.status.allocatable = {"nvidia.com/gpu": str(gpu_count)}
        return node

    def _make_pod(self, name, node_name, gpu_count, queue=""):
        pod = MagicMock()
        pod.metadata.name = name
        pod.spec.node_name = node_name
        pod.metadata.labels = {"kai.scheduler/queue": queue} if queue else {}
        container = MagicMock()
        container.resources.limits = (
            {"nvidia.com/gpu": str(gpu_count)} if gpu_count else {}
        )
        pod.spec.containers = [container]
        return pod

    def test_counts_gpus(self):
        mock_v1 = MagicMock()
        nodes = MagicMock()
        nodes.items = [self._make_node("worker-0", 2)]
        mock_v1.list_node.return_value = nodes

        pods = MagicMock()
        pods.items = [
            self._make_pod("pod-a", "worker-0", 1, queue="priority-team"),
            self._make_pod("pod-b", "worker-0", 1, queue="community"),
        ]
        mock_v1.list_pod_for_all_namespaces.return_value = pods

        with patch("nrl_k8s.k8s_client.client.CoreV1Api", return_value=mock_v1):
            from nrl_k8s.k8s_client import get_gpu_occupancy

            result = get_gpu_occupancy()

        assert result["total_allocatable"] == 2
        assert result["total_allocated"] == 2
        assert result["nodes"][0]["allocated"] == 2
        assert len(result["queues"]) == 2

    def test_empty_cluster(self):
        mock_v1 = MagicMock()
        nodes = MagicMock()
        nodes.items = [self._make_node("worker-0", 2)]
        mock_v1.list_node.return_value = nodes
        pods = MagicMock()
        pods.items = []
        mock_v1.list_pod_for_all_namespaces.return_value = pods

        with patch("nrl_k8s.k8s_client.client.CoreV1Api", return_value=mock_v1):
            from nrl_k8s.k8s_client import get_gpu_occupancy

            result = get_gpu_occupancy()

        assert result["total_allocated"] == 0
        assert result["queues"] == []


class TestSubmitGangRayjob:
    def test_creates_rayjob(self):
        mock_custom = MagicMock()
        mock_custom.create_namespaced_custom_object.return_value = {
            "metadata": {"name": "test-job"},
            "status": {},
        }

        with patch(
            "nrl_k8s.k8s_client.client.CustomObjectsApi", return_value=mock_custom
        ):
            from nrl_k8s.k8s_client import submit_gang_rayjob

            result = submit_gang_rayjob(
                name="test-job",
                queue="priority-team",
                image="rayproject/ray:2.52.0",
                entrypoint="echo hello",
                num_gpus=2,
            )

        assert result == "test-job"
        call_args = mock_custom.create_namespaced_custom_object.call_args
        body = call_args.kwargs["body"]
        assert body["metadata"]["labels"]["kai.scheduler/queue"] == "priority-team"
        assert body["spec"]["submissionMode"] == "HTTPMode"
        workers = body["spec"]["rayClusterSpec"]["workerGroupSpecs"][0]
        assert workers["replicas"] == 2

    def test_with_segment_size(self):
        mock_custom = MagicMock()
        mock_custom.create_namespaced_custom_object.return_value = {
            "metadata": {"name": "seg-job"},
            "status": {"rayClusterName": "seg-job-abc"},
        }

        with patch(
            "nrl_k8s.k8s_client.client.CustomObjectsApi", return_value=mock_custom
        ):
            from nrl_k8s.k8s_client import submit_gang_rayjob

            submit_gang_rayjob(
                name="seg-job",
                queue="priority-team",
                image="rayproject/ray:2.52.0",
                entrypoint="echo hello",
                num_gpus=4,
                segment_size=2,
            )

        # Should have 2 calls: RayJob + PodGroup
        assert mock_custom.create_namespaced_custom_object.call_count == 2
        pg_call = mock_custom.create_namespaced_custom_object.call_args_list[1]
        pg_body = pg_call.kwargs["body"]
        assert pg_body["kind"] == "PodGroup"
        assert len(pg_body["spec"]["subGroups"]) == 2
        assert pg_body["spec"]["subGroups"][0]["minMember"] == 2
        assert (
            pg_body["spec"]["subGroups"][0]["topologyConstraint"][
                "requiredTopologyLevel"
            ]
            == "topology-rack"
        )
