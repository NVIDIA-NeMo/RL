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

"""Tests for :mod:`nrl_k8s.manifest` — RayCluster manifest builder."""

from __future__ import annotations

import pytest
from nrl_k8s.manifest import (
    build_deployment_manifest,
    build_raycluster_manifest,
    build_service_for_deployment,
)
from nrl_k8s.schema import ClusterSpec, DeploymentSpec, InfraConfig

# =============================================================================
# Fixtures
# =============================================================================


def _base_spec() -> dict:
    return {
        "headGroupSpec": {
            "template": {
                "spec": {
                    "containers": [{"name": "ray-head", "image": "registry/img:old"}],
                }
            }
        },
        "workerGroupSpecs": [
            {
                "groupName": "gpu-workers",
                "template": {
                    "spec": {
                        "containers": [
                            {"name": "ray-worker", "image": "registry/img:old"}
                        ],
                    }
                },
            }
        ],
    }


def _make_infra(**overrides) -> InfraConfig:
    payload = {"namespace": "ns-patched", "image": "registry/img:new"} | overrides
    return InfraConfig.model_validate(payload)


def _make_cluster(**overrides) -> ClusterSpec:
    payload = {"name": "rc-test", "spec": _base_spec()} | overrides
    return ClusterSpec.model_validate(payload)


# =============================================================================
# Envelope + metadata
# =============================================================================


class TestEnvelope:
    def test_apiversion_and_kind(self) -> None:
        got = build_raycluster_manifest(_make_cluster(), _make_infra())
        assert got["apiVersion"] == "ray.io/v1"
        assert got["kind"] == "RayCluster"

    def test_metadata_name_and_namespace(self) -> None:
        got = build_raycluster_manifest(_make_cluster(name="rc-x"), _make_infra())
        assert got["metadata"]["name"] == "rc-x"
        assert got["metadata"]["namespace"] == "ns-patched"

    def test_labels_merged_from_infra_and_cluster(self) -> None:
        cluster = _make_cluster(labels={"role": "training"})
        infra = _make_infra(labels={"team": "rl"})
        got = build_raycluster_manifest(cluster, infra)
        labels = got["metadata"]["labels"]
        assert labels["role"] == "training"
        assert labels["team"] == "rl"
        assert labels["app.kubernetes.io/managed-by"] == "nrl-k8s"

    def test_cluster_labels_win_on_collision(self) -> None:
        cluster = _make_cluster(labels={"team": "override"})
        infra = _make_infra(labels={"team": "rl"})
        got = build_raycluster_manifest(cluster, infra)
        assert got["metadata"]["labels"]["team"] == "override"

    def test_managed_by_label_always_present(self) -> None:
        got = build_raycluster_manifest(_make_cluster(), _make_infra())
        assert got["metadata"]["labels"]["app.kubernetes.io/managed-by"] == "nrl-k8s"


# =============================================================================
# Cross-cutting patches
# =============================================================================


class TestPatching:
    def test_image_propagates_to_every_container(self) -> None:
        got = build_raycluster_manifest(_make_cluster(), _make_infra())
        head = got["spec"]["headGroupSpec"]["template"]["spec"]["containers"][0]
        worker = got["spec"]["workerGroupSpecs"][0]["template"]["spec"]["containers"][0]
        assert head["image"] == "registry/img:new"
        assert worker["image"] == "registry/img:new"

    def test_image_pull_secrets_applied(self) -> None:
        infra = _make_infra(imagePullSecrets=["a", "b"])
        got = build_raycluster_manifest(_make_cluster(), infra)
        for pod in (
            got["spec"]["headGroupSpec"]["template"]["spec"],
            got["spec"]["workerGroupSpecs"][0]["template"]["spec"],
        ):
            assert pod["imagePullSecrets"] == [{"name": "a"}, {"name": "b"}]

    def test_image_pull_secrets_skipped_when_empty(self) -> None:
        spec = _base_spec()
        spec["headGroupSpec"]["template"]["spec"]["imagePullSecrets"] = [
            {"name": "pre"}
        ]
        got = build_raycluster_manifest(_make_cluster(spec=spec), _make_infra())
        head = got["spec"]["headGroupSpec"]["template"]["spec"]
        assert head["imagePullSecrets"] == [{"name": "pre"}]

    def test_service_account_patched_when_set(self) -> None:
        infra = _make_infra(serviceAccount="new-sa")
        got = build_raycluster_manifest(_make_cluster(), infra)
        head = got["spec"]["headGroupSpec"]["template"]["spec"]
        assert head["serviceAccountName"] == "new-sa"

    def test_service_account_untouched_when_unset(self) -> None:
        spec = _base_spec()
        spec["headGroupSpec"]["template"]["spec"]["serviceAccountName"] = "original"
        got = build_raycluster_manifest(_make_cluster(spec=spec), _make_infra())
        head = got["spec"]["headGroupSpec"]["template"]["spec"]
        assert head["serviceAccountName"] == "original"


# =============================================================================
# Immutability
# =============================================================================


class TestImmutability:
    def test_input_spec_not_mutated(self) -> None:
        """Patching writes into a deep copy so the ClusterSpec isn't mutated.

        Important when the same spec underlies multiple renders.
        """
        spec = _base_spec()
        original = spec["headGroupSpec"]["template"]["spec"]["containers"][0]["image"]
        cluster = _make_cluster(spec=spec)
        build_raycluster_manifest(cluster, _make_infra())
        assert (
            spec["headGroupSpec"]["template"]["spec"]["containers"][0]["image"]
            == original
        )

    def test_two_builds_see_same_input(self) -> None:
        """Two successive builds against the same ClusterSpec produce identical manifests.

        Verifies the first call didn't leave any mutation behind on the
        ClusterSpec's inner ``spec`` dict.
        """
        cluster = _make_cluster()
        infra = _make_infra()
        first = build_raycluster_manifest(cluster, infra)
        second = build_raycluster_manifest(cluster, infra)
        assert first == second


# =============================================================================
# Labels + annotations: merge precedence (cluster wins over infra)
# =============================================================================


class TestLabelsAnnotationsMerge:
    def test_annotations_merged_from_infra_and_cluster(self) -> None:
        cluster = _make_cluster(annotations={"kyverno.io/skip": "true"})
        infra = _make_infra(annotations={"platform/team": "rl"})
        got = build_raycluster_manifest(cluster, infra)
        assert got["metadata"]["annotations"] == {
            "kyverno.io/skip": "true",
            "platform/team": "rl",
        }

    def test_cluster_annotations_win_on_collision(self) -> None:
        cluster = _make_cluster(annotations={"team": "override"})
        infra = _make_infra(annotations={"team": "infra"})
        got = build_raycluster_manifest(cluster, infra)
        assert got["metadata"]["annotations"] == {"team": "override"}

    def test_no_annotations_key_when_empty(self) -> None:
        got = build_raycluster_manifest(_make_cluster(), _make_infra())
        assert "annotations" not in got["metadata"]


# =============================================================================
# ServiceAccount patching onto pods that already set one
# =============================================================================


class TestServiceAccountOverride:
    def test_service_account_overrides_existing_on_head_and_worker(self) -> None:
        """``infra.serviceAccount`` overwrites ``serviceAccountName`` on all pod templates.

        Applies to every pod template — head and all workers.
        """
        spec = _base_spec()
        spec["headGroupSpec"]["template"]["spec"]["serviceAccountName"] = "old-head"
        spec["workerGroupSpecs"][0]["template"]["spec"]["serviceAccountName"] = "old-wg"
        infra = _make_infra(serviceAccount="new-sa")
        got = build_raycluster_manifest(_make_cluster(spec=spec), infra)
        head = got["spec"]["headGroupSpec"]["template"]["spec"]
        worker = got["spec"]["workerGroupSpecs"][0]["template"]["spec"]
        assert head["serviceAccountName"] == "new-sa"
        assert worker["serviceAccountName"] == "new-sa"


# =============================================================================
# Segment splitting
# =============================================================================


class TestSegmentSplitting:
    def test_no_split_when_segment_size_is_none(self) -> None:
        got = build_raycluster_manifest(_make_cluster(), _make_infra())
        assert len(got["spec"]["workerGroupSpecs"]) == 1
        assert got["spec"]["workerGroupSpecs"][0]["groupName"] == "gpu-workers"

    def test_split_64_into_4_segments_of_16(self) -> None:
        spec = _base_spec()
        spec["workerGroupSpecs"][0]["replicas"] = 64
        spec["workerGroupSpecs"][0]["minReplicas"] = 64
        spec["workerGroupSpecs"][0]["maxReplicas"] = 64
        cluster = _make_cluster(spec=spec, segmentSize=16)
        got = build_raycluster_manifest(cluster, _make_infra())
        workers = got["spec"]["workerGroupSpecs"]
        assert len(workers) == 4
        for i, wg in enumerate(workers):
            assert wg["groupName"] == f"gpu-workers-segment-{i}"
            assert wg["replicas"] == 16
            assert wg["minReplicas"] == 16
            assert wg["maxReplicas"] == 16

    def test_no_split_when_replicas_leq_segment_size(self) -> None:
        spec = _base_spec()
        spec["workerGroupSpecs"][0]["replicas"] = 8
        cluster = _make_cluster(spec=spec, segmentSize=16)
        got = build_raycluster_manifest(cluster, _make_infra())
        workers = got["spec"]["workerGroupSpecs"]
        assert len(workers) == 1
        assert workers[0]["groupName"] == "gpu-workers"

    def test_indivisible_replicas_raises(self) -> None:
        spec = _base_spec()
        spec["workerGroupSpecs"][0]["replicas"] = 65
        cluster = _make_cluster(spec=spec, segmentSize=16)
        with pytest.raises(ValueError, match="not evenly divisible"):
            build_raycluster_manifest(cluster, _make_infra())

    def test_preserves_pod_template(self) -> None:
        spec = _base_spec()
        spec["workerGroupSpecs"][0]["replicas"] = 32
        spec["workerGroupSpecs"][0]["template"]["metadata"] = {
            "annotations": {"kai.scheduler/topology": "gb300-topology"}
        }
        cluster = _make_cluster(spec=spec, segmentSize=16)
        got = build_raycluster_manifest(cluster, _make_infra())
        workers = got["spec"]["workerGroupSpecs"]
        assert len(workers) == 2
        for wg in workers:
            ann = wg["template"]["metadata"]["annotations"]
            assert ann["kai.scheduler/topology"] == "gb300-topology"

    def test_does_not_mutate_input(self) -> None:
        spec = _base_spec()
        spec["workerGroupSpecs"][0]["replicas"] = 32
        cluster = _make_cluster(spec=spec, segmentSize=16)
        build_raycluster_manifest(cluster, _make_infra())
        assert len(spec["workerGroupSpecs"]) == 1
        assert spec["workerGroupSpecs"][0]["replicas"] == 32

    def test_images_patched_on_all_segments(self) -> None:
        spec = _base_spec()
        spec["workerGroupSpecs"][0]["replicas"] = 32
        cluster = _make_cluster(spec=spec, segmentSize=16)
        got = build_raycluster_manifest(cluster, _make_infra())
        for wg in got["spec"]["workerGroupSpecs"]:
            img = wg["template"]["spec"]["containers"][0]["image"]
            assert img == "registry/img:new"

    def test_dra_rewriting_on_all_segments(self) -> None:
        spec = _base_spec()
        spec["workerGroupSpecs"][0]["replicas"] = 32
        spec["workerGroupSpecs"][0]["template"]["spec"]["resourceClaims"] = [
            {
                "name": "compute-domain-channel",
                "resourceClaimTemplateName": "placeholder",
            },
        ]
        cluster = _make_cluster(spec=spec, segmentSize=16)
        got = build_raycluster_manifest(cluster, _make_infra(), role="training")
        for wg in got["spec"]["workerGroupSpecs"]:
            claims = wg["template"]["spec"]["resourceClaims"]
            assert (
                claims[0]["resourceClaimTemplateName"]
                == "compute-domain-rc-test-training"
            )


# =============================================================================
# Deployment manifest builder
# =============================================================================

_MANAGED_BY = {"app.kubernetes.io/managed-by": "nrl-k8s"}


def _base_deployment_spec() -> dict:
    return {
        "selector": {"matchLabels": {"app": "sidecar"}},
        "template": {
            "spec": {
                "containers": [
                    {
                        "name": "server",
                        "ports": [{"containerPort": 8080}],
                    }
                ],
            }
        },
    }


def _make_deployment(**overrides) -> DeploymentSpec:
    payload = {"name": "my-sidecar", "spec": _base_deployment_spec()} | overrides
    return DeploymentSpec.model_validate(payload)


class TestBuildDeploymentManifest:
    def test_basic_structure(self) -> None:
        dep = _make_deployment()
        got = build_deployment_manifest(dep, _make_infra())
        assert got["apiVersion"] == "apps/v1"
        assert got["kind"] == "Deployment"
        assert got["metadata"]["name"] == "my-sidecar"
        assert got["metadata"]["namespace"] == "ns-patched"

    def test_image_default_patched_onto_container(self) -> None:
        dep = _make_deployment()
        got = build_deployment_manifest(dep, _make_infra())
        container = got["spec"]["template"]["spec"]["containers"][0]
        assert container["image"] == "registry/img:new"

    def test_explicit_image_not_overwritten(self) -> None:
        spec = _base_deployment_spec()
        spec["template"]["spec"]["containers"][0]["image"] = "custom:v1"
        dep = _make_deployment(spec=spec)
        got = build_deployment_manifest(dep, _make_infra())
        container = got["spec"]["template"]["spec"]["containers"][0]
        assert container["image"] == "custom:v1"

    def test_image_pull_secrets_patched(self) -> None:
        dep = _make_deployment()
        infra = _make_infra(imagePullSecrets=["secret-a", "secret-b"])
        got = build_deployment_manifest(dep, infra)
        pod_spec = got["spec"]["template"]["spec"]
        assert pod_spec["imagePullSecrets"] == [
            {"name": "secret-a"},
            {"name": "secret-b"},
        ]

    def test_service_account_patched(self) -> None:
        dep = _make_deployment()
        infra = _make_infra(serviceAccount="my-sa")
        got = build_deployment_manifest(dep, infra)
        pod_spec = got["spec"]["template"]["spec"]
        assert pod_spec["serviceAccountName"] == "my-sa"

    def test_service_account_not_patched_when_none(self) -> None:
        dep = _make_deployment()
        got = build_deployment_manifest(dep, _make_infra())
        pod_spec = got["spec"]["template"]["spec"]
        assert "serviceAccountName" not in pod_spec

    def test_labels_merged(self) -> None:
        dep = _make_deployment(labels={"team": "infra"})
        infra = _make_infra(labels={"env": "test"})
        got = build_deployment_manifest(dep, infra)
        meta_labels = got["metadata"]["labels"]
        assert meta_labels["app.kubernetes.io/managed-by"] == "nrl-k8s"
        assert meta_labels["env"] == "test"
        assert meta_labels["team"] == "infra"

    def test_pod_template_labels_merged(self) -> None:
        spec = _base_deployment_spec()
        spec["template"]["metadata"] = {"labels": {"existing": "keep"}}
        dep = _make_deployment(spec=spec, labels={"dep": "v1"})
        infra = _make_infra(labels={"infra": "v2"})
        got = build_deployment_manifest(dep, infra)
        tmpl_labels = got["spec"]["template"]["metadata"]["labels"]
        assert tmpl_labels["existing"] == "keep"
        assert tmpl_labels["dep"] == "v1"
        assert tmpl_labels["infra"] == "v2"
        assert tmpl_labels["app.kubernetes.io/managed-by"] == "nrl-k8s"

    def test_existing_pod_label_wins_over_infra(self) -> None:
        spec = _base_deployment_spec()
        spec["template"]["metadata"] = {"labels": {"env": "original"}}
        dep = _make_deployment(spec=spec)
        infra = _make_infra(labels={"env": "overridden"})
        got = build_deployment_manifest(dep, infra)
        assert got["spec"]["template"]["metadata"]["labels"]["env"] == "original"

    def test_annotations_merged(self) -> None:
        dep = _make_deployment(annotations={"note": "hi"})
        infra = _make_infra(annotations={"global": "yes"})
        got = build_deployment_manifest(dep, infra)
        assert got["metadata"]["annotations"]["note"] == "hi"
        assert got["metadata"]["annotations"]["global"] == "yes"

    def test_does_not_mutate_original_spec(self) -> None:
        dep = _make_deployment()
        original_image = dep.spec["template"]["spec"]["containers"][0].get("image")
        build_deployment_manifest(dep, _make_infra())
        assert (
            dep.spec["template"]["spec"]["containers"][0].get("image") == original_image
        )


class TestBuildServiceForDeployment:
    def test_returns_none_when_no_selector(self) -> None:
        spec = _base_deployment_spec()
        del spec["selector"]
        dep = _make_deployment(spec=spec)
        assert build_service_for_deployment(dep, _make_infra()) is None

    def test_returns_none_when_no_ports(self) -> None:
        spec = _base_deployment_spec()
        del spec["template"]["spec"]["containers"][0]["ports"]
        dep = _make_deployment(spec=spec)
        assert build_service_for_deployment(dep, _make_infra()) is None

    def test_basic_service(self) -> None:
        dep = _make_deployment()
        got = build_service_for_deployment(dep, _make_infra())
        assert got is not None
        assert got["apiVersion"] == "v1"
        assert got["kind"] == "Service"
        assert got["spec"]["type"] == "ClusterIP"
        assert got["spec"]["selector"] == {"app": "sidecar"}
        assert got["metadata"]["name"] == "my-sidecar"
        assert got["metadata"]["namespace"] == "ns-patched"

    def test_ports_extracted_from_containers(self) -> None:
        spec = _base_deployment_spec()
        spec["template"]["spec"]["containers"][0]["ports"] = [
            {"containerPort": 8080, "name": "http"},
            {"containerPort": 9090},
        ]
        dep = _make_deployment(spec=spec)
        got = build_service_for_deployment(dep, _make_infra())
        ports = got["spec"]["ports"]
        assert len(ports) == 2
        assert ports[0] == {
            "port": 8080,
            "targetPort": 8080,
            "protocol": "TCP",
            "name": "http",
        }
        assert ports[1] == {
            "port": 9090,
            "targetPort": 9090,
            "protocol": "TCP",
            "name": "port-1",
        }

    def test_service_has_managed_by_label(self) -> None:
        dep = _make_deployment()
        got = build_service_for_deployment(dep, _make_infra())
        assert got["metadata"]["labels"] == _MANAGED_BY
