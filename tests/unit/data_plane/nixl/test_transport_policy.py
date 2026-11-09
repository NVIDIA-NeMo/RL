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
"""The NIXL data plane must never let NIXL's UCX backend run over TCP."""

from __future__ import annotations

import pytest

from nemo_rl.data_plane.nixl.errors import TransportPolicyError
from nemo_rl.data_plane.nixl.nixl_io import check_rdma_only

RAILS = "rdma_vf_rail0:1,rdma_vf_rail1:1,rdma_vf_rail2:1,rdma_vf_rail3:1"


@pytest.mark.parametrize(
    "env",
    [
        {"UCX_TLS": "rc,self,sm", "UCX_NET_DEVICES": RAILS},
        {"UCX_TLS": "rc_x,sm,self", "UCX_NET_DEVICES": "rdma_vf_rail0:1"},
        {"UCX_TLS": "dc_mlx5,self"},
        {"UCX_TLS": "^tcp"},
        {"UCX_TLS": "^cuda_ipc,tcp", "UCX_NET_DEVICES": "all"},
    ],
)
def test_rdma_configs_pass(env):
    assert "UCX_TLS=" in check_rdma_only(env)


@pytest.mark.parametrize(
    "env, why",
    [
        ({}, "unset"),
        ({"UCX_TLS": "tcp", "UCX_NET_DEVICES": "eth0"}, "tcp"),
        ({"UCX_TLS": "rc,tcp,self"}, "tcp"),
        ({"UCX_TLS": "all"}, "all"),
        ({"UCX_TLS": "self,sm"}, "no RDMA"),
        ({"UCX_TLS": "^cuda_ipc"}, "not tcp"),
        ({"UCX_TLS": "rc,self,sm", "UCX_NET_DEVICES": "eth0"}, "network interfaces"),
        (
            {"UCX_TLS": "rc,self,sm", "UCX_NET_DEVICES": "rdma_vf_rail0:1,eth0"},
            "network interfaces",
        ),
    ],
)
def test_tcp_capable_configs_fail(env, why):
    with pytest.raises(TransportPolicyError, match=why):
        check_rdma_only(env)
