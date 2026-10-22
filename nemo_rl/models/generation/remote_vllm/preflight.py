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
from nemo_rl.models.generation.remote_vllm_config import (
    RemoteVllmServiceConfig,
    RemoteVllmServiceInfo,
)


def validate_single_backend(health: dict[str, Any] | None) -> None:
    """Reject independent replicas even when their router advertises fan-out."""
    if health is None:
        return  # Stock vLLM returns an empty successful /health response.
    if "total_backends" in health:
        total = health["total_backends"]
        if type(total) is not int or total != 1:
            raise RuntimeError(
                "External vLLM NCCL refit requires exactly one engine deployment; "
                f"/health reports total_backends={total!r}. "
                "Control-plane fan-out is not supported yet."
            )
    if "healthy_backends" in health:
        healthy = health["healthy_backends"]
        if type(healthy) is not int or healthy != 1:
            raise RuntimeError("External vLLM requires exactly one healthy backend")


def preflight_remote_vllm_service(
    config: RemoteVllmServiceConfig,
) -> RemoteVllmServiceInfo:
    """Check one vLLM 0.29.0 service and its native NCCL transfer settings."""
    client = RemoteVllmClient(config)
    validate_single_backend(client.health(timeout_s=config.connect_timeout_s))
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
    version = client.get_json("/version", timeout_s=config.connect_timeout_s).get(
        "version"
    )
    if not isinstance(version, str) or version.split("+")[0] != "0.29.0":
        raise RuntimeError(
            f"External NCCL refit requires vLLM 0.29.0; service reports {version!r}"
        )
    raw_server_info = client.get_json(
        config.server_info_path, timeout_s=config.connect_timeout_s
    )
    vllm_config = raw_server_info.get("vllm_config")
    if not isinstance(vllm_config, dict):
        raise RuntimeError("External /server_info must use config_format=json")
    transfer = vllm_config.get("weight_transfer_config")
    if not isinstance(transfer, dict) or transfer.get("backend") != "nccl":
        raise RuntimeError("External vLLM requires weight_transfer_config.backend=nccl")
    world_size = client.get_json(
        "/get_world_size", timeout_s=config.connect_timeout_s
    ).get("world_size")
    if type(world_size) is not int or world_size != config.refit.nccl_engine_world_size:
        raise RuntimeError(
            "External vLLM world size does not match refit.nccl_engine_world_size: "
            f"{world_size!r} != {config.refit.nccl_engine_world_size}"
        )
    return RemoteVllmServiceInfo(
        model=config.served_model_name,
        dev_mode_control_api=True,
        server_info=raw_server_info,
    )
