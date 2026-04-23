"""Tests for :mod:`nrl_k8s.manifest` — RayCluster manifest builder."""

from __future__ import annotations

from nrl_k8s.manifest import build_raycluster_manifest
from nrl_k8s.schema import ClusterSpec, InfraConfig

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
        """Patching writes into a deep copy so the recipe-parsed ClusterSpec
        isn't mutated between calls (important when the same spec underlies
        multiple renders).
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
        """Two successive build calls against the same ClusterSpec produce
        identical manifests — i.e. the first call didn't leave any mutation
        behind on the ClusterSpec's inner ``spec`` dict.
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
        """When ``infra.serviceAccount`` is set it overwrites any existing
        ``serviceAccountName`` on every pod template (head + all workers).
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
