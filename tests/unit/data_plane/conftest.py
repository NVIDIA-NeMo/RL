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

from ._rollout_shapes import mooncake_available


def _make_tq_cfg(backend: str) -> dict:
    return {
        "enabled": True,
        "impl": "transfer_queue",
        "backend": backend,
        "storage_capacity": 1024,
        "num_storage_units": 1,
        "claim_meta_poll_interval_s": 0.5,
        "global_segment_size": 8589934592,  # 8 GiB — sized for CI host RAM
        "local_buffer_size": 1073741824,  # 1 GiB
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
