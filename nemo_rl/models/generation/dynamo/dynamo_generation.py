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
"""Dynamo backend for nemo-rl: a thin URL forwarder to a DynamoGraphDeployment.

On Kubernetes, a ``DynamoGraphDeployment`` (DGD) owns the entire inference
stack — etcd, NATS, the dynamo frontend, and the vLLM/sglang/trtllm workers.
This class does not bring any of that up; it only resolves the cluster-internal
URL of the DGD's frontend Service so nemo-gym can dispatch rollout requests to
it over HTTP.
"""

import warnings
from typing import Any, Optional, Union

import ray

from nemo_rl.distributed.batched_data_dict import BatchedDataDict
from nemo_rl.distributed.virtual_cluster import RayVirtualCluster
from nemo_rl.models.generation.dynamo.config import DynamoConfig
from nemo_rl.models.generation.interfaces import (
    GenerationDatumSpec,
    GenerationInterface,
    GenerationOutputSpec,
)
from nemo_rl.utils.k8s import is_in_kubernetes, read_pod_namespace

DEFAULT_FRONTEND_PORT = 8000


def _derive_frontend_url_from_dgd(dynamo_cfg: dict[str, Any]) -> str:
    """Build the cluster-internal URL of the DGD's frontend Service.

    The dynamo operator names the frontend Service ``<dgd-name>-frontend``,
    so the URL is fully determined by ``dgd_name`` + namespace + port.
    """
    dgd_name = dynamo_cfg["dgd_name"]
    namespace = dynamo_cfg.get("namespace") or read_pod_namespace()
    if not namespace:
        # Falling back to "default" is almost certainly wrong, but failing
        # outright would be over-eager — the cluster might be configured
        # without serviceaccount projection.
        warnings.warn(
            "Could not determine pod namespace; falling back to 'default'. "
            "Set policy.generation.dynamo_cfg.namespace explicitly to silence this.",
            UserWarning,
            stacklevel=3,
        )
        namespace = "default"

    port = dynamo_cfg.get("frontend_port", DEFAULT_FRONTEND_PORT)
    return f"http://{dgd_name}-frontend.{namespace}.svc.cluster.local:{port}/v1"


def _resolve_frontend_url(dynamo_cfg: dict[str, Any]) -> tuple[str, bool]:
    """Resolve the frontend URL from a DynamoCfg.

    Returns ``(url, requires_k8s)``. ``requires_k8s`` is True only on the
    ``dgd_name`` path; an explicit ``frontend_url`` opts out of the
    in-pod check so the backend works against any reachable endpoint.
    """
    if "frontend_url" in dynamo_cfg:
        url = dynamo_cfg["frontend_url"]
        if not url:
            raise RuntimeError(
                "policy.generation.dynamo_cfg.frontend_url is set but empty."
            )
        return url, False

    if "dgd_name" not in dynamo_cfg:
        raise RuntimeError(
            "DynamoGeneration requires either policy.generation.dynamo_cfg.dgd_name "
            "(the metadata.name of the DynamoGraphDeployment) or "
            "policy.generation.dynamo_cfg.frontend_url (an explicit reachable URL)."
        )
    return _derive_frontend_url_from_dgd(dynamo_cfg), True


class DynamoGeneration(GenerationInterface):
    """Forwards rollout requests to a DynamoGraphDeployment frontend.

    The DGD must already exist in the cluster — this class does not create or
    wait on it. nrl-k8s is the orchestration layer that brings up the DGD and
    waits for readiness before the training entrypoint runs.
    """

    def __init__(
        self,
        cluster: Optional[RayVirtualCluster],
        config: DynamoConfig,
        name_prefix: str = "dynamo",
        workers_per_node: Optional[Union[int, list[int]]] = None,
    ):
        self.cfg = config
        dynamo_cfg = config.get("dynamo_cfg", {}) or {}
        url, requires_k8s = _resolve_frontend_url(dynamo_cfg)
        if requires_k8s and not is_in_kubernetes():
            raise RuntimeError(
                "DynamoGeneration with dgd_name requires running inside a "
                "Kubernetes pod (KUBERNETES_SERVICE_HOST is not set). "
                "Either run inside a pod, or set "
                "policy.generation.dynamo_cfg.frontend_url to a reachable URL."
            )
        self.dp_openai_server_base_urls: list[Optional[str]] = [url]
        print(f"  [Dynamo] Forwarding rollouts to {url}", flush=True)

    # ------------------------------------------------------------------
    # GenerationInterface — lifecycle
    # ------------------------------------------------------------------

    def prepare_for_generation(self, *args: Any, **kwargs: Any) -> bool:
        return True

    def finish_generation(self, *args: Any, **kwargs: Any) -> bool:
        return True

    def shutdown(self) -> bool:
        # The DGD lifecycle is owned by Kubernetes (the dynamo operator); we
        # have nothing to tear down on the nemo-rl side.
        return True

    # ------------------------------------------------------------------
    # Pickling — async rollouts ship the GenerationInterface across Ray actors
    # ------------------------------------------------------------------

    def __getstate__(self) -> dict[str, Any]:
        return {
            "cfg": self.cfg,
            "dp_openai_server_base_urls": self.dp_openai_server_base_urls,
        }

    def __setstate__(self, state: dict[str, Any]) -> None:
        self.cfg = state["cfg"]
        self.dp_openai_server_base_urls = state["dp_openai_server_base_urls"]

    def generate(
        self,
        data: BatchedDataDict["GenerationDatumSpec"],
        greedy: bool = False,
    ) -> BatchedDataDict["GenerationOutputSpec"]:
        raise NotImplementedError(
            "DynamoGeneration does not support direct generate(). "
            "Use the nemo-gym HTTP rollout path instead."
        )

    def init_collective(
        self, ip: str, port: int, world_size: int, **kwargs: Any
    ) -> list[ray.ObjectRef]:
        raise NotImplementedError(
            "DynamoGeneration does not support collective initialization "
            "(weight refit is not implemented in this phase)."
        )

    def prepare_refit_info(self, state_dict_info: dict[str, Any]) -> None:
        """No-op on the trainer side.

        With the receiver-side polling architecture, every DGD worker watches
        the MX server for new versions and refits itself — there's no
        trainer→worker RPC to forward state_dict_info on. If FP8 / assertion
        checks ever need this info, the worker can read it from the MX
        publisher's tensor descriptors at receive time.
        """
        return

    def update_weights_via_ipc_zmq(self) -> list[ray.ObjectRef]:
        raise NotImplementedError(
            "DynamoGeneration does not support IPC ZMQ weight sync — use "
            "weight_sync.method='mx' for non-colocated refit."
        )

    def update_weights_from_collective(self) -> list[ray.ObjectRef]:
        raise NotImplementedError(
            "DynamoGeneration does not support NCCL collective weight sync — "
            "use weight_sync.method='mx' for non-colocated refit."
        )

    # ------------------------------------------------------------------
    # ModelExpress v2 mid-training refit (cluster.weight_sync.method='mx')
    # ------------------------------------------------------------------

    def update_weights_via_mx(
        self,
        *,
        version: int,
        mx_config: Any,
    ) -> list[ray.ObjectRef]:
        """No-op on the trainer side; the DGD workers poll MX for new versions.

        The refit_policy_generation mx branch in nemo_rl/algorithms/grpo.py
        publishes via ``policy.stream_weights_via_mx`` and then calls this
        method on the generation interface. With the receiver-side polling
        architecture, the DGD's ``MxRefitWorkerExtension`` runs a background
        loop that watches the MX server for new versions matching its
        model_name and triggers a refit automatically when one appears —
        so no trainer→worker RPC is needed.

        Returns an empty list of ObjectRefs to satisfy the abstract method
        signature; ``ray.get([])`` is a no-op.
        """
        del version, mx_config  # unused — receiver polls
        return []
