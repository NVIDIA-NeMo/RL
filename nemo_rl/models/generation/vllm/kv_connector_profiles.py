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
"""What NeMo RL has to know about a vLLM KV connector to front it.

vLLM moves the KV cache between a prefill and a decode engine through a *KV
connector* (``--kv-transfer-config``). The connector owns the handshake and the
transfer; the orchestrator in front of the engines only relays an opaque
``kv_transfer_params`` dict from the prefill response into the decode request.
The few things that *are* connector-specific, and that the frontend and the
driver therefore need, live here as one record per connector so adding a
connector is a new entry, not a code path:

* the marker a cold prefill leg carries so the engine knows a decode engine
  will pick its KV up,
* the ``kv_role`` spelling for each side,
* which per-engine network endpoints the connector needs allocated (several
  engines share a node under NeMo RL, so vLLM's "one engine per host" defaults
  collide),
* whether the connector can pull a conversation's KV back from the decode
  engine on the next turn (``bidirectional_kv_xfer``).

Everything else in ``kv_transfer_params`` is never parsed by NeMo RL.
"""

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class ConnectorProfile:
    """The connector-specific facts the disaggregation frontend and driver need."""

    name: str
    # ``kv_role`` of a prefill engine and of a decode engine.
    producer_role: str
    consumer_role: str
    # ``kv_transfer_params`` of a prefill leg whose conversation has no KV on
    # a decode engine yet. The decode leg's params are whatever the prefill
    # response returned, so they are never spelled out here.
    initial_prefill_params: dict[str, Any] = field(default_factory=dict)
    # Can the prefill engine read a previous turn's KV back from the decode
    # engine (vLLM's ``bidirectional_kv_xfer``)?
    supports_bidirectional: bool = False
    # Environment variables naming the handshake endpoint every engine must
    # bind; the driver allocates a distinct port per engine on a node and the
    # worker exports them before building its engine. None when the connector
    # has no such endpoint.
    side_channel_host_env: str | None = None
    side_channel_port_env: str | None = None


NIXL_PROFILE = ConnectorProfile(
    name="NixlConnector",
    producer_role="kv_producer",
    consumer_role="kv_consumer",
    initial_prefill_params={
        "do_remote_decode": True,
        "do_remote_prefill": False,
        "remote_engine_id": None,
        "remote_block_ids": None,
        "remote_host": None,
        "remote_port": None,
    },
    supports_bidirectional=True,
    side_channel_host_env="VLLM_NIXL_SIDE_CHANNEL_HOST",
    side_channel_port_env="VLLM_NIXL_SIDE_CHANNEL_PORT",
)

# Mooncake speaks a different handshake vocabulary (remote_bootstrap_addr,
# transfer_id) with no block ids in it, which is exactly why the frontend
# relays the prefill response's dict without reading its keys.
MOONCAKE_PROFILE = ConnectorProfile(
    name="MooncakeConnector",
    producer_role="kv_producer",
    consumer_role="kv_consumer",
    initial_prefill_params={"do_remote_decode": True, "do_remote_prefill": False},
    supports_bidirectional=False,
)

CONNECTOR_PROFILES: dict[str, ConnectorProfile] = {
    NIXL_PROFILE.name: NIXL_PROFILE,
    MOONCAKE_PROFILE.name: MOONCAKE_PROFILE,
}


def connector_profile(kv_connector: str, profile_name: str | None) -> ConnectorProfile:
    """The profile for ``kv_connector``, or the one named by ``profile_name``.

    A custom connector loaded through ``kv_connector_module_path`` has no entry
    of its own; its config names the profile whose handshake protocol it
    speaks. An unknown connector without one is refused rather than guessed:
    relaying the wrong initial marker would make every request fail on the
    prefill engine with no hint at the cause.
    """
    name = profile_name or kv_connector
    if name not in CONNECTOR_PROFILES:
        raise ValueError(
            f"No connector profile for kv_connector={kv_connector!r}"
            + (f" (connector_profile={profile_name!r})" if profile_name else "")
            + f". Known profiles: {sorted(CONNECTOR_PROFILES)}. A custom "
            "connector must name the profile whose handshake protocol it "
            "speaks via policy.generation.vllm_cfg.disagg_frontend.connector_profile."
        )
    return CONNECTOR_PROFILES[name]
