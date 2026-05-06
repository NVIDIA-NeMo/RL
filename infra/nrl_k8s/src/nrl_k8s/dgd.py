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
"""DynamoGraphDeployment (DGD) ingestion for nrl-k8s.

This module owns the DGD data-flow that doesn't fit cleanly in ``manifest.py``
(RayCluster/Deployment-shaped) or ``k8s.py`` (already RayCluster-heavy):

* ``load_dgd_manifest`` resolves a path reference to a standalone DGD YAML.
* ``build_dgd_manifest`` deep-merges overrides, applies cross-cutting
  ``infra`` patches (image, imagePullSecrets, serviceAccount, labels), and
  returns a dict ready for ``apply_dgd``.
* ``resolve_dgd_name`` returns the post-override ``metadata.name`` so the
  orchestrator can stamp it into the recipe.
* ``apply_dgd`` / ``get_dgd`` / ``delete_dgd`` / ``wait_for_dgd_ready`` mirror
  the RayCluster helpers in ``k8s.py``.

The dynamo operator owns the Service for the frontend (``<dgd-name>-<service-key>``,
so ``<dgd-name>-frontend`` for the standard ``Frontend`` service key); we never
auto-create one here.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import yaml
from kubernetes import client
from kubernetes.client.exceptions import ApiException
from omegaconf import OmegaConf

from ._logging import redact
from ._retry import with_retries
from .k8s import custom_objects_api, load_kubeconfig
from .schema import DynamoGraphSpec, InfraConfig

# CRD identifiers for DynamoGraphDeployment.
DGD_GROUP = "nvidia.com"
DGD_VERSION = "v1alpha1"
DGD_PLURAL = "dynamographdeployments"
DGD_KIND = "DynamoGraphDeployment"
DGD_API_VERSION = f"{DGD_GROUP}/{DGD_VERSION}"
DGD_CRD_NAME = f"{DGD_PLURAL}.{DGD_GROUP}"

# Status enum from
# dynamo/deploy/operator/api/v1alpha1/dynamographdeployment_types.go:47-55.
_DGD_STATE_TERMINAL_GOOD = "successful"
_DGD_STATE_TERMINAL_BAD = "failed"

_MANAGED_BY_LABEL = {"app.kubernetes.io/managed-by": "nrl-k8s"}


# =============================================================================
# Manifest loading + building
# =============================================================================


def load_dgd_manifest(path: str | Path, base_dir: Path) -> dict[str, Any]:
    """Read a standalone DGD manifest from disk.

    Repo-relative paths resolve against ``base_dir``. Multi-document YAML
    files (which dynamo recipes often have — a ``DynamoGraphDeployment``
    plus a benchmark Pod) are filtered down to the first DGD doc.
    """
    p = Path(path)
    resolved = p if p.is_absolute() else (base_dir / p).resolve()
    if not resolved.is_file():
        raise FileNotFoundError(
            f"DGD manifest not found: {resolved} "
            f"(resolved from {path!r} relative to {base_dir})"
        )

    with resolved.open() as f:
        docs = list(yaml.safe_load_all(f))

    for doc in docs:
        if not isinstance(doc, dict):
            continue
        if doc.get("kind") == DGD_KIND:
            if doc.get("apiVersion") != DGD_API_VERSION:
                raise ValueError(
                    f"{resolved}: expected apiVersion={DGD_API_VERSION}, got "
                    f"{doc.get('apiVersion')!r}"
                )
            return doc

    raise ValueError(
        f"{resolved}: no document with kind={DGD_KIND} found "
        f"(saw kinds: {[d.get('kind') for d in docs if isinstance(d, dict)]})"
    )


def _walk_service_pod_specs(spec: dict[str, Any]) -> list[dict[str, Any]]:
    """Return every pod-spec inside a DGD's ``services[*].extraPodSpec``.

    Mirrors ``manifest._walk_pod_templates`` but for the DGD shape, where
    each service has its own ``extraPodSpec`` with a single ``mainContainer``.
    """
    pod_specs: list[dict[str, Any]] = []
    for svc in (spec.get("services") or {}).values():
        if not isinstance(svc, dict):
            continue
        eps = svc.get("extraPodSpec")
        if isinstance(eps, dict):
            pod_specs.append(eps)
    return pod_specs


def _patch_dgd_images(spec: dict[str, Any], image: str) -> None:
    """Default each service's ``mainContainer.image`` to ``image`` if unset.

    Mirrors ``manifest._patch_images`` semantics — services that author
    their own image win. This is a softer rule than what we apply to
    RayCluster pods (where every container gets the infra image), because
    DGDs frequently mix vLLM-runtime and frontend-runtime images per
    service and the recipe author has explicit intent there.
    """
    for eps in _walk_service_pod_specs(spec):
        main = eps.get("mainContainer")
        if isinstance(main, dict) and "image" not in main:
            main["image"] = image


def _patch_dgd_image_pull_secrets(spec: dict[str, Any], secrets: list[str]) -> None:
    if not secrets:
        return
    body = [{"name": s} for s in secrets]
    for eps in _walk_service_pod_specs(spec):
        eps.setdefault("imagePullSecrets", body)


def _patch_dgd_service_account(spec: dict[str, Any], service_account: str) -> None:
    for eps in _walk_service_pod_specs(spec):
        eps["serviceAccountName"] = service_account


def build_dgd_manifest(
    dgd: DynamoGraphSpec,
    infra: InfraConfig,
    base_dir: Path,
) -> dict[str, Any]:
    """Build the full DGD manifest ready for ``apply_dgd``.

    Steps, in order:
      1. Load the referenced manifest from disk.
      2. Deep-merge ``dgd.overrides`` onto its ``.spec``.
      3. Override ``metadata.name`` if ``dgd.name`` is set.
      4. Set ``metadata.namespace = infra.namespace``.
      5. Merge ``dgd.labels`` / ``dgd.annotations`` and the ``managed-by`` label.
      6. Patch cross-cutting fields (image / imagePullSecrets / serviceAccount)
         across every service's ``extraPodSpec``.
    """
    raw = load_dgd_manifest(dgd.manifest, base_dir)

    # Deep-merge overrides via OmegaConf so nested service-replicas / resource
    # tweaks compose cleanly with the upstream recipe.
    raw_spec = raw.get("spec") or {}
    if dgd.overrides:
        merged = OmegaConf.merge(
            OmegaConf.create(raw_spec),
            OmegaConf.create(dgd.overrides),
        )
        raw["spec"] = OmegaConf.to_container(merged, resolve=True)
    else:
        raw["spec"] = dict(raw_spec)

    metadata = raw.setdefault("metadata", {})
    if dgd.name:
        metadata["name"] = dgd.name
    if "name" not in metadata:
        raise ValueError(
            f"DGD manifest {dgd.manifest!r} has no metadata.name and "
            f"DynamoGraphSpec.name was not set."
        )
    metadata["namespace"] = infra.namespace
    metadata["labels"] = {
        **_MANAGED_BY_LABEL,
        **infra.labels,
        **(metadata.get("labels") or {}),
        **dgd.labels,
    }
    annotations = {**infra.annotations, **(metadata.get("annotations") or {}), **dgd.annotations}
    if annotations:
        metadata["annotations"] = annotations

    spec = raw["spec"]
    _patch_dgd_images(spec, infra.image)
    _patch_dgd_image_pull_secrets(spec, list(infra.imagePullSecrets))
    if infra.serviceAccount is not None:
        _patch_dgd_service_account(spec, infra.serviceAccount)

    return raw


def resolve_dgd_name(dgd: DynamoGraphSpec, base_dir: Path) -> str:
    """Return the DGD's effective ``metadata.name``.

    Used by the orchestrator to stamp the resolved name into the recipe
    before staging the working_dir, without re-doing the full manifest build.
    """
    if dgd.name:
        return dgd.name
    raw = load_dgd_manifest(dgd.manifest, base_dir)
    name = (raw.get("metadata") or {}).get("name")
    if not name:
        raise ValueError(
            f"DGD manifest {dgd.manifest!r} has no metadata.name; set "
            f"DynamoGraphSpec.name explicitly."
        )
    return name


# =============================================================================
# Kubernetes API helpers
# =============================================================================


def apply_dgd(manifest: dict[str, Any], namespace: str) -> dict[str, Any]:
    """Create-or-replace a DynamoGraphDeployment. Returns the server-side object."""
    name = manifest["metadata"]["name"]
    api = custom_objects_api()
    try:
        return with_retries(
            lambda: api.create_namespaced_custom_object(
                group=DGD_GROUP,
                version=DGD_VERSION,
                namespace=namespace,
                plural=DGD_PLURAL,
                body=manifest,
            )
        )
    except ApiException as exc:
        if exc.status == 409:
            return with_retries(
                lambda: api.patch_namespaced_custom_object(
                    group=DGD_GROUP,
                    version=DGD_VERSION,
                    namespace=namespace,
                    plural=DGD_PLURAL,
                    name=name,
                    body=manifest,
                )
            )
        exc.nrl_k8s_manifest = redact(manifest)  # type: ignore[attr-defined]
        raise


def get_dgd(name: str, namespace: str) -> dict[str, Any] | None:
    api = custom_objects_api()
    try:
        return with_retries(
            lambda: api.get_namespaced_custom_object(
                group=DGD_GROUP,
                version=DGD_VERSION,
                namespace=namespace,
                plural=DGD_PLURAL,
                name=name,
            )
        )
    except ApiException as exc:
        if exc.status == 404:
            return None
        raise


def delete_dgd(name: str, namespace: str, *, ignore_missing: bool = True) -> None:
    api = custom_objects_api()
    try:
        with_retries(
            lambda: api.delete_namespaced_custom_object(
                group=DGD_GROUP,
                version=DGD_VERSION,
                namespace=namespace,
                plural=DGD_PLURAL,
                name=name,
            )
        )
    except ApiException as exc:
        if exc.status == 404 and ignore_missing:
            return
        raise


def list_dgds(namespace: str, label_selector: str | None = None) -> list[dict]:
    api = custom_objects_api()
    resp = with_retries(
        lambda: api.list_namespaced_custom_object(
            group=DGD_GROUP,
            version=DGD_VERSION,
            namespace=namespace,
            plural=DGD_PLURAL,
            label_selector=label_selector or "",
        )
    )
    return resp.get("items", [])


def wait_for_dgd_ready(
    name: str,
    namespace: str,
    *,
    timeout_s: int = 600,
    poll_s: int = 5,
) -> None:
    """Block until ``.status.state == successful`` or time out.

    Raises ``RuntimeError`` immediately on ``failed`` so the caller doesn't
    keep waiting on a dead DGD.
    """
    deadline = time.monotonic() + timeout_s
    state: str | None = None
    while time.monotonic() < deadline:
        obj = get_dgd(name, namespace)
        state = (obj or {}).get("status", {}).get("state")
        if state == _DGD_STATE_TERMINAL_GOOD:
            return
        if state == _DGD_STATE_TERMINAL_BAD:
            conds = (obj or {}).get("status", {}).get("conditions", [])
            raise RuntimeError(
                f"DynamoGraphDeployment {name} in {namespace} reached state=failed; "
                f"conditions={conds!r}"
            )
        time.sleep(poll_s)
    raise TimeoutError(
        f"DynamoGraphDeployment {name} in {namespace} never reached state="
        f"{_DGD_STATE_TERMINAL_GOOD!r} (last seen: {state!r}) after {timeout_s}s"
    )


def wait_for_dgd_gone(
    name: str, namespace: str, *, timeout_s: int = 300, poll_s: int = 3
) -> None:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if get_dgd(name, namespace) is None:
            return
        time.sleep(poll_s)
    raise TimeoutError(
        f"DynamoGraphDeployment {name} not deleted after {timeout_s}s"
    )


# =============================================================================
# CRD precondition check
# =============================================================================


def is_dgd_crd_installed(namespace: str) -> bool:
    """True if the DGD CRD is registered on the cluster.

    Probes by listing DGDs in ``namespace`` with ``limit=1`` — the API server
    returns 404 only when the resource type is unknown (CRD missing). A 403
    means the CRD is registered but the user lacks list-RBAC; we treat that as
    "installed" and let the subsequent apply step surface any remaining
    permission issues through its clearer error path.

    Using a namespaced list (instead of reading the CRD directly) avoids
    requiring cluster-scoped RBAC that most users don't have.
    """
    api = custom_objects_api()
    try:
        api.list_namespaced_custom_object(
            group=DGD_GROUP,
            version=DGD_VERSION,
            namespace=namespace,
            plural=DGD_PLURAL,
            limit=1,
        )
        return True
    except ApiException as exc:
        if exc.status == 404:
            return False
        if exc.status == 403:
            return True
        raise
