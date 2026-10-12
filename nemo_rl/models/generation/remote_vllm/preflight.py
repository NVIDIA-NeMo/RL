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

"""Setup-time validation for a stock vLLM rollout server."""

from typing import Any

from nemo_rl.models.generation.remote_vllm.client import RemoteVllmClient
from nemo_rl.models.generation.remote_vllm.config import (
    RemoteVllmServiceConfig,
    RemoteVllmServiceInfo,
)


def preflight_remote_vllm_service(
    config: RemoteVllmServiceConfig,
) -> RemoteVllmServiceInfo:
    """Verify health, model identity, and the development control surface.

    The check is deliberately non-mutating. ``/server_info`` only exists when
    ``VLLM_SERVER_DEV_MODE=1``, which is also what exposes ``/collective_rpc``.
    """
    client = RemoteVllmClient(config)
    generation_health = client.health(timeout_s=config.connect_timeout_s)
    control_health = generation_health
    if config.control_base_url is not None:
        control_health = client.health(
            timeout_s=config.connect_timeout_s,
            control=True,
        )
    # The existing external-vLLM launcher puts a small load balancer in front
    # of each pool. Its /health response exposes the backend count. A normal
    # request proxy cannot make /collective_rpc global across independent
    # engine deployments, so fail before training rather than silently refit
    # only one replica. Native vLLM TP/PP/DP workers remain one backend here.
    if control_health is not None and isinstance(
        control_health.get("total_backends"), int
    ):
        total_backends = control_health["total_backends"]
        control_fanout = control_health.get("control_fanout") is True
        if total_backends != 1 and not control_fanout:
            raise RuntimeError(
                "External vLLM refit requires exactly one engine deployment "
                "behind the configured URL; the load balancer reports "
                f"{total_backends} independent backends. Add control-plane "
                "fan-out before using multiple replicas."
            )

    models = client.get_json(config.models_path, timeout_s=config.connect_timeout_s)
    model_entries = models.get("data")
    if not isinstance(model_entries, list):
        raise RuntimeError("External vLLM /v1/models response has no data list")
    model_ids = {
        entry.get("id")
        for entry in model_entries
        if isinstance(entry, dict) and isinstance(entry.get("id"), str)
    }
    if config.served_model_name not in model_ids:
        raise RuntimeError(
            "External vLLM model mismatch: controller expects "
            f"{config.served_model_name!r}, service reports {sorted(model_ids)!r}"
        )

    raw_server_info: Any = client.get_json(
        config.server_info_path,
        timeout_s=config.connect_timeout_s,
        control=True,
    )
    if not isinstance(raw_server_info, dict):
        raise RuntimeError("External vLLM /server_info response is not an object")

    return RemoteVllmServiceInfo(
        model=config.served_model_name,
        dev_mode_control_api=True,
        server_info=raw_server_info,
    )
