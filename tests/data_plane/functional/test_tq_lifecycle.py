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
"""Single-node TQ smoke — Stage 1 acceptance.

Mirrors the recipe in the integration plan §3 / Stage 1:
register → put → get_meta → get_data → check_consumption → clear.

Skipped when the ``transfer_queue`` package is not installed so CI without
the data-plane extra still passes.
"""

from __future__ import annotations

import os

import pytest
import torch
from tensordict import TensorDict

transfer_queue = pytest.importorskip("transfer_queue")  # noqa: F841

from nemo_rl.data_plane import build_data_plane_client

# ── loud-skip helpers ─────────────────────────────────────────────────────────

_REQUIRE_MOONCAKE = os.environ.get("NEMO_RL_REQUIRE_MOONCAKE") == "1"


def _mooncake_available() -> bool:
    try:
        import mooncake  # noqa: F401
    except ImportError:
        if _REQUIRE_MOONCAKE:
            raise
        return False
    return True


# ── fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture
def tq_client():
    import ray

    if not ray.is_initialized():
        ray.init(local_mode=False, include_dashboard=False)

    client = build_data_plane_client(
        {
            "enabled": True,
            "impl": "transfer_queue",
            "backend": "simple",
            "storage_capacity": 1024,
            "num_storage_units": 1,
        }
    )
    yield client
    client.close()


@pytest.fixture(
    params=["simple", "mooncake_cpu"],
    ids=["simple", "mooncake_cpu"],
)
def tq_client_backends(request):
    """Parametrized fixture over simple and mooncake_cpu backends.

    mooncake_cpu is skipped when the mooncake wheel is not installed.
    Set NEMO_RL_REQUIRE_MOONCAKE=1 to promote the skip to a loud failure.
    """
    backend = request.param
    if backend == "mooncake_cpu" and not _mooncake_available():
        pytest.skip(
            "mooncake not installed — skipping mooncake_cpu backend "
            "(set NEMO_RL_REQUIRE_MOONCAKE=1 to fail loud)"
        )

    import ray

    if not ray.is_initialized():
        ray.init(local_mode=False, include_dashboard=False)

    client = build_data_plane_client(
        {
            "enabled": True,
            "impl": "transfer_queue",
            "backend": backend,
            "storage_capacity": 1024,
            "num_storage_units": 1,
        }
    )
    yield client
    client.close()


def test_smoke_round_trip(tq_client) -> None:
    tq_client.register_partition(
        partition_id="smoke",
        fields=["x"],
        num_samples=4,
        consumer_tasks=["read"],
    )
    keys = ["a", "b", "c", "d"]
    tq_client.kv_batch_put(
        keys=keys,
        partition_id="smoke",
        fields=TensorDict({"x": torch.arange(4)}, batch_size=[4]),
    )

    meta = tq_client.get_meta(
        partition_id="smoke",
        task_name="read",
        required_fields=["x"],
        batch_size=4,
        timeout_s=30.0,
    )
    assert meta.size == 4

    data = tq_client.get_data(meta)
    # Order may differ from input — match against the meta's keys.
    expected = torch.tensor([keys.index(k) for k in meta.keys])
    assert torch.equal(data["x"], expected)

    assert tq_client.check_consumption_status("smoke", ["read"])

    tq_client.kv_clear(keys=None, partition_id="smoke")


def test_smoke_round_trip_backends(tq_client_backends) -> None:
    """Smoke round-trip parameterized over both backends.

    Covers P5 (T2-backend-bytewise-equal) — the same put/get lifecycle must
    work on simple and mooncake_cpu. mooncake_cpu is skipped when unavailable.
    """
    client = tq_client_backends
    client.register_partition(
        partition_id="smoke-backend",
        fields=["x"],
        num_samples=4,
        consumer_tasks=["read"],
    )
    keys = ["a", "b", "c", "d"]
    client.kv_batch_put(
        keys=keys,
        partition_id="smoke-backend",
        fields=TensorDict({"x": torch.arange(4)}, batch_size=[4]),
    )

    meta = client.get_meta(
        partition_id="smoke-backend",
        task_name="read",
        required_fields=["x"],
        batch_size=4,
        timeout_s=30.0,
    )
    assert meta.size == 4

    data = client.get_data(meta)
    expected = torch.tensor([keys.index(k) for k in meta.keys])
    assert torch.equal(data["x"], expected)

    client.kv_clear(keys=None, partition_id="smoke-backend")


def test_smoke_round_trip_1d_fields(tq_client) -> None:
    """A 1D (N,) tensor put into TQ must come back as (N,), not (N,1).

    Regression guard for R-C2: TQ's KVStorageManager path silently unsqueezes
    1D fields. The codec's _KV_PROMOTE_1D flag and materialize squeeze fix
    this for the mooncake_cpu backend; this test verifies simple backend does
    not introduce the regression.
    """
    n = 6
    reward = torch.arange(n, dtype=torch.float32)

    tq_client.register_partition(
        partition_id="smoke-1d",
        fields=["reward"],
        num_samples=n,
        consumer_tasks=["read"],
    )
    keys = [f"k{i}" for i in range(n)]
    tq_client.kv_batch_put(
        keys=keys,
        partition_id="smoke-1d",
        fields=TensorDict({"reward": reward}, batch_size=[n]),
    )

    meta = tq_client.get_meta(
        partition_id="smoke-1d",
        task_name="read",
        required_fields=["reward"],
        batch_size=n,
        timeout_s=30.0,
    )
    data = tq_client.get_data(meta)

    assert data["reward"].shape == reward.shape, (
        f"Expected shape {tuple(reward.shape)} for 1D field, "
        f"got {tuple(data['reward'].shape)}. "
        "TQ must not unsqueeze 1D tensors silently (R-C2)."
    )

    tq_client.kv_clear(keys=None, partition_id="smoke-1d")
