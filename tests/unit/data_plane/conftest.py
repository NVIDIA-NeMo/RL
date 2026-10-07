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
"""Shared TQ data-plane client fixtures.

Mooncake's C++ engine keeps a process-global mount registry that survives
Python-level ``close()`` (upstream ``transfer_queue.close()`` also leaves
``mooncake_master`` running). Re-initializing the client in the same
pytest worker process leaks stale segment endpoints; the next
``batch_upsert_from`` then routes to a dead endpoint from a prior init
and returns ``TRANSFER_FAIL`` (-800). Production never hits this — the
driver bootstraps once and workers attach via ``bootstrap=False``.

Session-scoping the underlying clients here mirrors production: exactly
one mooncake init per pytest worker, period. Tests must use distinct
``partition_id`` values (seqpack-eq / dynbatch-eq / nopack-eq /
smoke / smoke-backend / smoke-1d / obj-backend / mix-e2e today).
"""

from __future__ import annotations

import pytest

from nemo_rl.data_plane import build_data_plane_client

from ._rollout_shapes import mooncake_available, rdma_available


def _make_tq_cfg(backend: str) -> dict:
    return {
        "enabled": True,
        "impl": "transfer_queue",
        "backend": backend,
        "claim_meta_poll_interval_s": 0.5,
        "simple": {"storage_capacity": 1024, "num_storage_units": 1},
        "mooncake_cpu": {
            "global_segment_size": 8589934592,  # 8 GiB — sized for CI host RAM
            "local_buffer_size": 1073741824,  # 1 GiB
            # reuse_registered_buffers omitted on purpose: absent must mean on,
            # so the fixture exercises the default the same way a user config
            # that never mentions the flag does.
        },
    }


# Ray is started by the parent autouse ``init_ray_cluster`` fixture in
# ``tests/unit/conftest.py`` — no explicit init needed here.


@pytest.fixture(scope="session")
def _session_tq_client_simple():
    client = build_data_plane_client(_make_tq_cfg("simple"))
    yield client
    client.close()


@pytest.fixture(scope="session")
def _session_tq_client_mooncake_cpu():
    if not mooncake_available():
        pytest.skip(
            "mooncake not installed — skipping mooncake_cpu "
            "(set NEMO_RL_REQUIRE_MOONCAKE=1 to fail loud)"
        )
    # mooncake_cpu is RDMA-only, so it cannot run without an RDMA device. CI
    # sets NEMO_RL_REQUIRE_MOONCAKE=1 on runners that have one, which turns
    # this skip into a failure — otherwise losing the device passthrough would
    # silently drop mooncake coverage and still go green.
    if not rdma_available():
        pytest.skip(
            "no usable mlx5 RDMA device — mooncake_cpu requires RDMA "
            "(set MC_MOONCAKE_DEVICE=<dev> to override)"
        )
    client = build_data_plane_client(_make_tq_cfg("mooncake_cpu"))
    yield client
    client.close()


@pytest.fixture
def tq_client(_session_tq_client_simple):
    """One simple-backend client shared across the pytest session."""
    return _session_tq_client_simple


@pytest.fixture(
    params=["simple", "mooncake_cpu"],
    ids=["simple", "mooncake_cpu"],
)
def tq_client_backends(request):
    """Parametrized over [simple, mooncake_cpu] backends.

    Each variant returns the session-scoped client for that backend, so
    the mooncake_cpu client is initialized at most once per pytest worker
    (see module docstring).
    """
    return request.getfixturevalue(f"_session_tq_client_{request.param}")


@pytest.fixture(autouse=True)
def _isolate_codec_timer():
    """Drain the module-level codec timer around every test in this package.

    ``pack_jagged_fields`` records into a process-global timer, so a test that
    packs (the codec and column_io suites) leaves residue that the next
    ``get_step_metrics`` would report as its own ``step/codec/pack_s``.
    """
    from nemo_rl.data_plane.codec import drain_codec_ms

    drain_codec_ms()
    yield
    drain_codec_ms()


# ----------------------------------------------------------------------------- nixl backend
# Tests marked ``nixl`` need NIXL importable and UCX restricted to RDMA (the
# NIXL data plane refuses TCP), e.g. ``UCX_TLS=rc,self,sm
# UCX_NET_DEVICES=<rdma dev>:1``; tests marked ``gpu`` need a CUDA device. Both
# are skipped otherwise, so CPU-only runners execute only the pure tests.


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
    marked = [
        it
        for it in items
        if it.get_closest_marker("nixl") or it.get_closest_marker("gpu")
    ]
    if not marked:
        return
    # TransferQueue keeps one process-global client, and tq.init() reuses it.
    # Run the nixl tests before anything builds the session simple/mooncake
    # clients below, or their tq.init(NixlStore) would attach to that client.
    nixl = [it for it in items if it.get_closest_marker("nixl")]
    if nixl:
        ids = {id(it) for it in nixl}
        items[:] = nixl + [it for it in items if id(it) not in ids]
    why = _nixl_ready()
    cuda = _has_cuda()
    for item in marked:
        if item.get_closest_marker("nixl") and why:
            item.add_marker(pytest.mark.skip(reason=why))
        if item.get_closest_marker("gpu") and not cuda:
            item.add_marker(pytest.mark.skip(reason="no CUDA device"))


@pytest.fixture(scope="session")
def ray_cluster(init_ray_cluster):
    """Ray is already up (tests/unit/conftest.py); the NIXL tests only need it."""
    yield


@pytest.fixture(scope="module")
def tq_system(ray_cluster):
    """A live TransferQueue with the NixlStore backend (2 small units)."""
    import transfer_queue as tq

    import nemo_rl.data_plane.adapters.tq_nixl as tq_nixl  # registers the backend

    from ._nixl_helpers import make_tq_conf

    conf = make_tq_conf()
    tq.init(conf)
    manager = tq.get_client().storage_manager
    assert isinstance(manager, tq_nixl.NixlStorageManager), (
        f"tq.init attached to an existing {type(manager).__name__}, not NixlStore"
    )
    yield conf
    tq.close()
    tq_nixl.shutdown(conf)
