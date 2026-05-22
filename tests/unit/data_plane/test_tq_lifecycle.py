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

import numpy as np
import pytest
import torch
from tensordict import TensorDict

transfer_queue = pytest.importorskip("transfer_queue")  # noqa: F841

from nemo_rl.data_plane import build_data_plane_client
from nemo_rl.data_plane.column_io import kv_first_write, read_columns
from nemo_rl.data_plane.interfaces import KVBatchMeta
from nemo_rl.data_plane.schema import DP_TRAIN_FIELDS
from nemo_rl.distributed.batched_data_dict import BatchedDataDict

from ._rollout_shapes import mooncake_available

# ── loud-skip helpers ─────────────────────────────────────────────────────────

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
            "claim_meta_poll_interval_s": 0.5,
            "global_segment_size": 8589934592,  # 8 GiB (only read by mooncake_cpu)
            "local_buffer_size": 1073741824,  # 1 GiB (only read by mooncake_cpu)
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
    if backend == "mooncake_cpu" and not mooncake_available():
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
            "claim_meta_poll_interval_s": 0.5,
            "global_segment_size": 8589934592,  # 8 GiB
            "local_buffer_size": 1073741824,  # 1 GiB
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
    tq_client.put_samples(
        sample_ids=keys,
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
    expected = torch.tensor([keys.index(k) for k in meta.sample_ids])
    assert torch.equal(data["x"], expected)

    assert tq_client.check_consumption_status("smoke", ["read"])

    tq_client.clear_samples(sample_ids=None, partition_id="smoke")


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
    client.put_samples(
        sample_ids=keys,
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
    expected = torch.tensor([keys.index(k) for k in meta.sample_ids])
    assert torch.equal(data["x"], expected)

    client.clear_samples(sample_ids=None, partition_id="smoke-backend")


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
    tq_client.put_samples(
        sample_ids=keys,
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

    tq_client.clear_samples(sample_ids=None, partition_id="smoke-1d")


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
    ``np.ndarray(dtype=object)`` (matching ``sync_rollout_actor.py``
    line 273 / 292); the TensorDict constructor wraps them as
    ``NonTensorData`` internally. :func:`read_columns` →
    :func:`materialize` decodes them back to ``np.ndarray(dtype=object)``.
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
    client.put_samples(
        sample_ids=keys,
        partition_id="obj-backend",
        fields=TensorDict(
            {field_name: _object_payload(n)},
            batch_size=[n],
        ),
    )
    meta = KVBatchMeta(
        partition_id="obj-backend",
        task_name="read",
        sample_ids=keys,
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

    client.clear_samples(sample_ids=None, partition_id="obj-backend")


def test_object_and_tensor_mixed_round_trip_backends(tq_client_backends) -> None:
    """End-to-end mirror of ``SyncRolloutActor.kv_first_write``.

    Pins the production e2e GRPO pipeline shape on both backends:

    * ``register_partition`` declares ``DP_TRAIN_FIELDS`` (tensor-only),
      matching :meth:`TQPolicy.prepare_step`.
    * ``bulk_batch`` includes 1D + 2D tensors **and** an
      ``np.ndarray(dtype=object)`` (``content``) — the shape built by
      ``sync_rollout_actor.py`` lines 257–273.
    * ``kv_first_write`` does the put through :func:`pack_jagged_fields`.
    * ``read_columns`` fetches a mixed tensor + object subset, the same
      pattern used by ``grpo_sync.py`` lines 887–896.

    Regression guard for the data-plane wire round-trip end-to-end.
    """
    client = tq_client_backends
    n = 6
    seq_len = 4
    sample_ids = [f"sample_{i}" for i in range(n)]
    partition_id = "mix-e2e"

    # Tensor-only schema — matches `TQPolicy.prepare_step`.
    client.register_partition(
        partition_id=partition_id,
        fields=list(DP_TRAIN_FIELDS),
        num_samples=n,
        consumer_tasks=["read"],
    )

    # Production-shape `bulk_batch`: tensors + np.ndarray(dtype=object).
    input_ids = torch.arange(n * seq_len, dtype=torch.long).reshape(n, seq_len)
    input_lengths = torch.full((n,), seq_len, dtype=torch.long)
    generation_logprobs = torch.zeros(n, seq_len, dtype=torch.float)
    token_mask = torch.ones(n, seq_len, dtype=torch.float)
    sample_mask = torch.ones(n, dtype=torch.float)
    content = _object_payload(n)

    bulk_batch = BatchedDataDict(
        {
            "input_ids": input_ids,
            "input_lengths": input_lengths,
            "generation_logprobs": generation_logprobs,
            "token_mask": token_mask,
            "sample_mask": sample_mask,
            "content": content,
        }
    )

    # Production write path.
    meta = kv_first_write(
        bulk_batch,
        sample_ids=sample_ids,
        dp_client=client,
        partition_id=partition_id,
        task_name="read",
    )

    # Production read path — mixed tensor + object subset.
    bdd = read_columns(
        client, meta, select_fields=["input_ids", "input_lengths", "content"]
    )
    assert torch.equal(bdd["input_ids"], input_ids)
    assert torch.equal(bdd["input_lengths"], input_lengths)
    expected = _object_payload(n)
    for i in range(n):
        assert bdd["content"][i] == expected[i], (
            f"row {i} content mismatch: got {bdd['content'][i]!r}, "
            f"expected {expected[i]!r}"
        )

    # Tensor-only subset still works.
    only_ids = read_columns(client, meta, select_fields=["input_ids"])
    assert torch.equal(only_ids["input_ids"], input_ids)
    assert "content" not in only_ids

    # Object-only subset still works.
    only_content = read_columns(client, meta, select_fields=["content"])
    assert isinstance(only_content["content"], np.ndarray)
    assert "input_ids" not in only_content

    client.clear_samples(sample_ids=None, partition_id=partition_id)


def test_promote_1d_leaves_object_array_roundtrip() -> None:
    """``_promote_1d_leaves`` + ``_from_wire`` preserves non-tensor leaves.

    Pins the production TD shape (1D tensor + object array + 2D tensor)
    against tensordict 0.12.2 reconstruction bugs that could silently
    strip ``NonTensorStack`` / ``NonTensorData`` leaves. Symmetric to
    the documented ``.contiguous()`` bug in
    ``adapters/transfer_queue.py`` lines 558–562.
    """
    from nemo_rl.data_plane.adapters.transfer_queue import (
        _from_wire,
        _promote_1d_leaves,
    )

    arr = np.empty(4, dtype=object)
    arr[:] = [["a", "b"], ["c"], ["d", "e"], ["f"]]
    td = TensorDict(
        {
            "input_ids": torch.zeros(4, 8, dtype=torch.long),
            "input_lengths": torch.tensor([4, 3, 2, 1]),  # 1D → promoted
            "content": arr,
        },
        batch_size=[4],
    )
    promoted = _promote_1d_leaves(td)
    assert promoted["input_lengths"].shape == (4, 1)
    np.testing.assert_array_equal(promoted["content"], arr)

    restored = _from_wire(promoted)
    assert restored["input_lengths"].shape == (4,)
    np.testing.assert_array_equal(restored["content"], arr)
