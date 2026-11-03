# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
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

"""Refit adapts dim-0-sharded DTensors shard-locally and gathers the adapted layout."""

import os
import socket

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch import nn

pytestmark = pytest.mark.automodel

WORLD = 2


class _TransposeExpertsAdapter:
    """Mimics Muse Spark: grouped experts are transposed, one key drops rows of dim 0."""

    def convert_single_tensor_to_hf(self, fqn, tensor, **kwargs):
        if fqn == "experts":
            return [("hf.experts", tensor.transpose(-1, -2))]
        if fqn == "rows":
            return [("hf.rows", tensor[1:])]
        return [(fqn, tensor)]


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("localhost", 0))
        return sock.getsockname()[1]


def _worker(rank: int, port: int) -> None:
    from torch.distributed.device_mesh import init_device_mesh
    from torch.distributed.tensor import Shard, distribute_tensor

    from nemo_rl.models.policy.workers.automodel_policy_worker import (
        _adapt_sharded_tensor_to_hf,
        _module_params_generator,
    )

    os.environ.update(MASTER_ADDR="localhost", MASTER_PORT=str(port))
    dist.init_process_group("gloo", rank=rank, world_size=WORLD)
    try:
        mesh = init_device_mesh("cpu", (WORLD,))
        torch.manual_seed(0)
        experts = torch.randn(4, 3, 5)
        rows = torch.randn(6, 2)
        plain = torch.randn(4, 7)

        model = nn.Module()
        model.experts = nn.Parameter(distribute_tensor(experts, mesh, [Shard(0)]))
        model.rows = nn.Parameter(distribute_tensor(rows, mesh, [Shard(0)]))
        model.plain = nn.Parameter(distribute_tensor(plain, mesh, [Shard(0)]))
        model.state_dict_adapter = _TransposeExpertsAdapter()

        # Pure-view adapters that keep dim 0 take the shard-local path; one that slices
        # dim 0 must fall back to gather-then-adapt.
        assert _adapt_sharded_tensor_to_hf(model, "experts", model.experts) is not None
        assert _adapt_sharded_tensor_to_hf(model, "plain", model.plain) is not None
        assert _adapt_sharded_tensor_to_hf(model, "rows", model.rows) is None

        streamed = {
            name: tensor.clone()
            for name, tensor in _module_params_generator(model, torch.float32)
        }
        expected = {
            "hf.experts": experts.transpose(-1, -2).contiguous(),
            "hf.rows": rows[1:].contiguous(),
            "plain": plain,
        }
        assert streamed.keys() == expected.keys()
        for name, want in expected.items():
            got = streamed[name]
            assert not isinstance(got, torch.distributed.tensor.DTensor), name
            assert got.is_contiguous(), name
            torch.testing.assert_close(got, want, rtol=0, atol=0, msg=name)
    finally:
        dist.destroy_process_group()


def test_shard_local_adapt_matches_gather_then_adapt():
    mp.spawn(_worker, args=(_free_port(),), nprocs=WORLD, join=True)
