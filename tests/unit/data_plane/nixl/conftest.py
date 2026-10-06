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
"""NIXL data-plane tests.

Tests marked ``nixl`` need NIXL importable and UCX restricted to RDMA (the data
plane refuses TCP): e.g. ``UCX_TLS=rc,self,sm UCX_NET_DEVICES=<rdma dev>:1``.
They are skipped otherwise, so CPU-only runners only execute the pure tests.
Ray comes from the session fixture in ``tests/unit/conftest.py``.
"""

from __future__ import annotations

import pytest


def _nixl_ready() -> str | None:
    try:
        from nixl._api import nixl_agent  # noqa: F401
    except Exception as e:  # noqa: BLE001
        return f"nixl not importable: {e}"
    from nemo_rl.data_plane.nixl.errors import TransportPolicyError
    from nemo_rl.data_plane.nixl.nixl_io import check_rdma_only

    try:
        check_rdma_only()
    except TransportPolicyError as e:
        return f"UCX not restricted to RDMA: {e}"
    return None


def _has_cuda() -> bool:
    try:
        import torch

        return torch.cuda.is_available() and torch.cuda.device_count() > 0
    except Exception:  # noqa: BLE001
        return False


def pytest_collection_modifyitems(config, items):
    why = _nixl_ready()
    cuda = _has_cuda()
    for item in items:
        if "nixl" in item.keywords and why:
            item.add_marker(pytest.mark.skip(reason=why))
        if "gpu" in item.keywords and not cuda:
            item.add_marker(pytest.mark.skip(reason="no CUDA device"))


@pytest.fixture(scope="session")
def ray_cluster(init_ray_cluster):
    """Ray is already up (tests/unit/conftest.py); the NIXL tests only need it."""
    yield


@pytest.fixture(scope="module")
def tq_system(ray_cluster):
    """A live TransferQueue with the NixlStore backend (2 small units)."""
    import transfer_queue as tq

    import nemo_rl.data_plane.nixl.tq  # noqa: F401  registers the backend

    from tests.unit.data_plane.nixl._helpers import make_tq_conf

    conf = make_tq_conf()
    tq.init(conf)
    yield conf
    tq.close()
    nemo_rl.data_plane.nixl.tq.shutdown(conf)
