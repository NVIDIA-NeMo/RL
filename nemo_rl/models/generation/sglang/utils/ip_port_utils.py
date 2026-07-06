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

import ipaddress
import logging

import ray

from nemo_rl.distributed.virtual_cluster import (
    DEFAULT_GENERATION_PORT_RANGE_HIGH,
    DEFAULT_GENERATION_PORT_RANGE_LOW,
)
from nemo_rl.models.generation.sglang.utils.ray_utils import get_host_info

logger = logging.getLogger(__name__)


def _wrap_ipv6(host):
    """Wrap IPv6 address in [] if needed."""
    try:
        ipaddress.IPv6Address(host.strip("[]"))
        return f"[{host.strip('[]')}]"
    except ipaddress.AddressValueError:
        return host


def _format_v6_uri(addr):
    if not addr or addr.startswith("["):
        return addr
    try:
        if ipaddress.ip_address(addr).version == 6:
            return f"[{addr}]"
    except ValueError:
        pass
    return addr


def _allocate_rollout_engine_addr_and_ports_normal(
    *,
    gpus_per_node: int,
    sglang_cfg,
    local_all_engines,
    rank_offset=0,
    port_range_low: int = DEFAULT_GENERATION_PORT_RANGE_LOW,
    port_range_high: int = DEFAULT_GENERATION_PORT_RANGE_HIGH,
    node_port_cursor: dict[int, int] = {},
):
    # get ports
    # there are 4 ports we need to allocate
    # 1. server port
    # 2. nccl port
    # 3. dist_init_addr port
    # 4. other ports for dp_attention, which is of size 30 + dp_size

    sglang_dp_size = sglang_cfg["sglang_cfg"]["dp_size"]
    num_gpus_per_engine = sglang_cfg["sglang_cfg"]["sglang_server_config"][
        "num_gpus_per_engine"
    ]
    num_gpus_per_node = gpus_per_node

    _gpus_per_engine = num_gpus_per_engine
    num_engines_per_node = max(1, num_gpus_per_node // _gpus_per_engine)
    addr_and_ports: dict[int, dict] = {}

    visited_nodes = set()
    for rank, engine in local_all_engines:
        local_rank = rank - rank_offset
        node_index = local_rank // num_engines_per_node
        if node_index in visited_nodes:
            continue
        visited_nodes.add(node_index)
        # TODO: currently when restarting engines, we will set port for all engines on this node starting with this rank.
        # e.g. for 8 gpus, if we are restarting engine on gpu 3, we will set port for engine 3,4,5,6,7 on this node.
        num_engines_on_this_node = num_engines_per_node - (
            local_rank % num_engines_per_node
        )

        def get_addr_and_ports(engine, node_idx):
            # allocate from the reserved generation band, advancing a per-node
            # cursor so blocks never overlap on a given node
            start_port = node_port_cursor.get(node_idx, port_range_low)

            def port(consecutive=1):
                nonlocal start_port
                port = ray.get(
                    engine._get_current_free_port.remote(
                        port_range_low=port_range_low,
                        port_range_high=port_range_high,
                        consecutive=consecutive,
                        start_port=start_port,
                    )
                )
                start_port = port + consecutive
                node_port_cursor[node_idx] = start_port
                return port

            def addr():
                addr = ray.get(engine._get_current_node_ip.remote())
                if addr is None:
                    addr = get_host_info()[1]
                return addr

            return addr, port

        addr, port = get_addr_and_ports(engine, node_index)

        for i in range(num_engines_on_this_node):
            current_rank = rank + i
            addr_and_ports.setdefault(current_rank, {})
            addr_and_ports[current_rank]["host"] = addr()
            addr_and_ports[current_rank]["port"] = port()
            addr_and_ports[current_rank]["nccl_port"] = port()

        if _gpus_per_engine > num_gpus_per_node:
            num_node_per_engine = _gpus_per_engine // num_gpus_per_node
            if local_rank % num_node_per_engine == 0:
                dist_init_addr = f"{addr()}:{port(30 + sglang_dp_size)}"
                for i in range(num_node_per_engine):
                    addr_and_ports.setdefault(rank + i, {})
                    addr_and_ports[rank + i]["dist_init_addr"] = dist_init_addr
        else:
            for i in range(num_engines_on_this_node):
                addr_and_ports[rank + i]["dist_init_addr"] = (
                    f"{addr()}:{port(30 + sglang_dp_size)}"
                )

    for i, _ in local_all_engines:
        for key in ["port", "nccl_port", "dist_init_addr"]:
            assert key in addr_and_ports[i], f"Engine {i} {key} is not set."
        logger.info(f"Ports for engine {i}: {addr_and_ports[i]}")

    return addr_and_ports, node_port_cursor
