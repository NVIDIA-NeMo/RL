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

"""Tests for :mod:`nrl_k8s.rayjob` — RayJob manifest builder."""

from __future__ import annotations

from nrl_k8s.rayjob import DEFAULT_SUBMISSION_MODE, build_rayjob_manifest
from nrl_k8s.schema import ClusterSpec, InfraConfig


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
    payload = {"namespace": "ns", "image": "registry/img:new"} | overrides
    return InfraConfig.model_validate(payload)


def _make_cluster(**overrides) -> ClusterSpec:
    payload = {"name": "rc-test", "spec": _base_spec()} | overrides
    return ClusterSpec.model_validate(payload)


class TestEnvelope:
    def test_apiversion_and_kind(self) -> None:
        got = build_rayjob_manifest(
            _make_cluster(), _make_infra(), entrypoint="python -u foo.py"
        )
        assert got["apiVersion"] == "ray.io/v1"
        assert got["kind"] == "RayJob"

    def test_name_defaults_to_cluster_name(self) -> None:
        got = build_rayjob_manifest(
            _make_cluster(name="rc-x"), _make_infra(), entrypoint="echo hi"
        )
        assert got["metadata"]["name"] == "rc-x"

    def test_name_override_wins(self) -> None:
        got = build_rayjob_manifest(
            _make_cluster(name="rc-x"),
            _make_infra(),
            entrypoint="echo",
            name="sft-job",
        )
        assert got["metadata"]["name"] == "sft-job"

    def test_namespace_from_infra(self) -> None:
        got = build_rayjob_manifest(
            _make_cluster(),
            _make_infra(namespace="rl"),
            entrypoint="echo",
        )
        assert got["metadata"]["namespace"] == "rl"


class TestSpec:
    def test_entrypoint_and_defaults(self) -> None:
        got = build_rayjob_manifest(
            _make_cluster(), _make_infra(), entrypoint="python run.py"
        )
        spec = got["spec"]
        assert spec["entrypoint"] == "python run.py"
        assert spec["submissionMode"] == DEFAULT_SUBMISSION_MODE
        assert spec["shutdownAfterJobFinishes"] is True
        assert spec["ttlSecondsAfterFinished"] == 3600

    def test_shutdown_override(self) -> None:
        got = build_rayjob_manifest(
            _make_cluster(),
            _make_infra(),
            entrypoint="x",
            shutdown_after_finishes=False,
            ttl_seconds_after_finished=60,
        )
        assert got["spec"]["shutdownAfterJobFinishes"] is False
        assert got["spec"]["ttlSecondsAfterFinished"] == 60

    def test_ray_cluster_spec_carries_cluster_body(self) -> None:
        got = build_rayjob_manifest(
            _make_cluster(), _make_infra(), entrypoint="x"
        )
        rcs = got["spec"]["rayClusterSpec"]
        head = rcs["headGroupSpec"]["template"]["spec"]["containers"][0]
        worker = rcs["workerGroupSpecs"][0]["template"]["spec"]["containers"][0]
        # Image patch from infra should propagate inside rayClusterSpec.
        assert head["image"] == "registry/img:new"
        assert worker["image"] == "registry/img:new"

    def test_image_pull_secrets_propagate_inside_ray_cluster_spec(self) -> None:
        infra = _make_infra(imagePullSecrets=["secret-a"])
        got = build_rayjob_manifest(_make_cluster(), infra, entrypoint="x")
        head_pod = got["spec"]["rayClusterSpec"]["headGroupSpec"]["template"]["spec"]
        assert head_pod["imagePullSecrets"] == [{"name": "secret-a"}]


class TestLabels:
    def test_labels_merged_from_infra_cluster_and_extra(self) -> None:
        cluster = _make_cluster(labels={"role": "training"})
        infra = _make_infra(labels={"team": "rl"})
        got = build_rayjob_manifest(
            cluster, infra, entrypoint="x", extra_labels={"run-id": "r-1"}
        )
        labels = got["metadata"]["labels"]
        assert labels["role"] == "training"
        assert labels["team"] == "rl"
        assert labels["run-id"] == "r-1"
        assert labels["app.kubernetes.io/managed-by"] == "nrl-k8s"

    def test_extra_labels_win_on_collision(self) -> None:
        cluster = _make_cluster(labels={"team": "cluster"})
        infra = _make_infra(labels={"team": "infra"})
        got = build_rayjob_manifest(
            cluster, infra, entrypoint="x", extra_labels={"team": "extra"}
        )
        assert got["metadata"]["labels"]["team"] == "extra"

    def test_managed_by_label_always_present(self) -> None:
        got = build_rayjob_manifest(
            _make_cluster(), _make_infra(), entrypoint="x"
        )
        assert got["metadata"]["labels"]["app.kubernetes.io/managed-by"] == "nrl-k8s"


class TestImmutability:
    def test_input_spec_not_mutated(self) -> None:
        spec = _base_spec()
        original_image = spec["headGroupSpec"]["template"]["spec"]["containers"][0]["image"]
        cluster = _make_cluster(spec=spec)
        build_rayjob_manifest(cluster, _make_infra(), entrypoint="x")
        assert (
            spec["headGroupSpec"]["template"]["spec"]["containers"][0]["image"]
            == original_image
        )
