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
register → put → claim_meta → get_data → check_consumption → clear.

Skipped when the ``transfer_queue`` package is not installed so CI without
the data-plane extra still passes.
"""

from __future__ import annotations

import os

import numpy as np
import pytest
import torch
from tensordict import TensorDict

transfer_queue = pytest.importorskip("transfer_queue")  # noqa: F841

from tensordict import NonTensorStack

from nemo_rl.data_plane import build_data_plane_client, materialize
from nemo_rl.data_plane.column_io import read_columns
from nemo_rl.data_plane.interfaces import KVBatchMeta
from nemo_rl.distributed.batched_data_dict import BatchedDataDict
from nemo_rl.experience.sync_rollout_actor import kv_first_write

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

    meta = tq_client.claim_meta(
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

    meta = client.claim_meta(
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
    1D fields. The adapter's `_promote_1d_leaves` + `_from_wire` pair fix
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

    meta = tq_client.claim_meta(
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


# ── Object-field round-trip across backends ───────────────────────────────────
#
# Closes the coverage gap: prior tests exercised np.ndarray(object) only via
# the in-process codec (test_codec_object.py) or sent tensor-only fields
# through both backends (test_smoke_round_trip_backends). Sending object
# fields through mooncake_cpu was untested. This test covers that path.


def _object_payload(n: int) -> np.ndarray:
    """Heterogeneous per-row Python objects, mimicking message_log shape."""
    rows = [
        {
            "id": i,
            "text": f"sample {i} content " * (i % 5 + 1),  # variable-length strings
            "tags": [f"t{i}", f"t{i + 1}"],
        }
        for i in range(n)
    ]
    arr = np.empty(n, dtype=object)
    for i, r in enumerate(rows):
        arr[i] = r
    return arr


def test_object_round_trip_backends(tq_client_backends) -> None:
    """np.ndarray(dtype=object) put → get → decode equality, both backends.

    Mirrors the wire used by ``SyncRolloutActor.kv_first_write`` for
    ``message_log`` / ``content``: object fields ride as
    ``NonTensorStack`` leaves (TQ-native non-tensor passthrough);
    :func:`read_columns` → :func:`materialize` decodes them back to
    ``np.ndarray(dtype=object)``.
    """
    client = tq_client_backends
    n = 8
    field_name = "msg_log"
    keys = [f"obj_{i}" for i in range(n)]

    client.register_partition(
        partition_id="obj-backend",
        fields=[field_name],
        num_samples=n,
        consumer_tasks=["read"],
    )
    client.kv_batch_put(
        keys=keys,
        partition_id="obj-backend",
        fields=TensorDict(
            {field_name: NonTensorStack(*_object_payload(n).tolist())},
            batch_size=[n],
        ),
    )
    meta = KVBatchMeta(
        partition_id="obj-backend",
        task_name="read",
        keys=keys,
        fields=[field_name],
    )

    bdd = read_columns(client, meta, select_fields=[field_name])

    assert isinstance(bdd[field_name], np.ndarray)
    assert bdd[field_name].dtype == object
    assert bdd[field_name].shape == (n,)
    expected = _object_payload(n)
    for i in range(n):
        assert bdd[field_name][i] == expected[i], (
            f"row {i} mismatch: got {bdd[field_name][i]!r}, expected {expected[i]!r}"
        )

    client.kv_clear(keys=None, partition_id="obj-backend")


def test_object_and_tensor_mixed_round_trip_backends(tq_client_backends) -> None:
    """Mixed tensor + object fields in one put — exercises the actor's
    real schema (tensors + object data side-by-side).

    Regression guard: object writes coexisting with tensor writes must
    not corrupt either side. Co-fetch decodes the tensor via padding
    and the ``NonTensorStack`` leaf via :func:`materialize` in one call.
    """
    client = tq_client_backends
    n = 6
    keys = [f"mx_{i}" for i in range(n)]

    client.register_partition(
        partition_id="mix-backend",
        fields=["ids", "lens", "msg"],
        num_samples=n,
        consumer_tasks=["read"],
    )
    ids = torch.arange(n * 4, dtype=torch.long).reshape(n, 4)
    lens = torch.full((n,), 4, dtype=torch.long)
    msg = NonTensorStack(*_object_payload(n).tolist())

    client.kv_batch_put(
        keys=keys,
        partition_id="mix-backend",
        fields=TensorDict(
            {"ids": ids, "lens": lens, "msg": msg},
            batch_size=[n],
        ),
    )

    meta = KVBatchMeta(
        partition_id="mix-backend",
        task_name="read",
        keys=keys,
        fields=["ids", "lens", "msg"],
        sequence_lengths=[4] * n,
    )

    # Read all three together — tensor fields decode via padding,
    # object field decodes via NonTensorStack passthrough.
    bdd = read_columns(client, meta, select_fields=["ids", "lens", "msg"])
    assert torch.equal(bdd["ids"], ids)
    assert torch.equal(bdd["lens"], lens)
    expected = _object_payload(n)
    for i in range(n):
        assert bdd["msg"][i] == expected[i]

    # Read just the tensor.
    only_ids = read_columns(client, meta, select_fields=["ids"])
    assert torch.equal(only_ids["ids"], ids)
    assert "msg" not in only_ids

    # Read just the object.
    only_msg = read_columns(client, meta, select_fields=["msg"])
    assert isinstance(only_msg["msg"], np.ndarray)
    assert "ids" not in only_msg

    client.kv_clear(keys=None, partition_id="mix-backend")


def test_routed_experts_jagged_round_trip(tq_client) -> None:
    tq_client.register_partition(
        partition_id="r3",
        fields=[
            "input_ids",
            "input_lengths",
            "token_mask",
            "sample_mask",
            "generation_logprobs",
            "routed_experts",
        ],
        num_samples=3,
        consumer_tasks=["read"],
    )

    lengths = torch.tensor([5, 3, 4], dtype=torch.long)
    final_batch = BatchedDataDict(
        {
            "input_ids": torch.arange(3 * 5, dtype=torch.long).reshape(3, 5),
            "input_lengths": lengths,
            "token_mask": torch.ones(3, 5, dtype=torch.long),
            "sample_mask": torch.ones(3, dtype=torch.long),
            "generation_logprobs": torch.zeros(3, 5),
            "routed_experts": torch.arange(3 * 5 * 2 * 2, dtype=torch.int32).reshape(
                3, 5, 2, 2
            ),
        }
    )
    meta = kv_first_write(
        final_batch,
        uids=["a", "b", "c"],
        dp_client=tq_client,
        partition_id="r3",
        pad_to_multiple=4,
    )

    wire = tq_client.kv_batch_get(
        keys=meta.keys,
        partition_id="r3",
        select_fields=["input_ids", "routed_experts"],
    )
    materialized = materialize(
        wire,
        pad_value_dict={"input_ids": -1},
        pad_to_multiple=int((meta.extra_info or {})["pad_to_multiple"]),
    )

    assert materialized["input_ids"].shape == (3, 8)
    assert materialized["routed_experts"].shape == (3, 8, 2, 2)
    for row, seq_len in enumerate(lengths.tolist()):
        assert torch.equal(
            materialized["input_ids"][row, :seq_len],
            final_batch["input_ids"][row, :seq_len],
        )
        assert torch.equal(
            materialized["routed_experts"][row, :seq_len],
            final_batch["routed_experts"][row, :seq_len],
        )

    tq_client.kv_clear(keys=None, partition_id="r3")
