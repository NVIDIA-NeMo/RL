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

import inspect
import json
from typing import Callable
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch
from tensordict import TensorDict

transfer_queue = pytest.importorskip("transfer_queue")  # noqa: F841

from nemo_rl.data_plane.column_io import kv_first_write, read_columns
from nemo_rl.data_plane.interfaces import DataPlaneClient, KVBatchMeta
from nemo_rl.data_plane.schema import DP_TRAIN_FIELDS
from nemo_rl.distributed.batched_data_dict import BatchedDataDict


def _register_partition(client: DataPlaneClient) -> None:
    client.register_partition(
        partition_id="p",
        fields=["x"],
        num_samples=1,
        consumer_tasks=["train"],
    )


def _claim_meta(client: DataPlaneClient) -> None:
    client.claim_meta(
        partition_id="p",
        task_name="train",
        required_fields=["x"],
        batch_size=1,
    )


def _get_data(client: DataPlaneClient) -> None:
    client.get_data(
        KVBatchMeta(
            partition_id="p",
            task_name="train",
            sample_ids=["sample-0"],
            fields=["x"],
        )
    )


def _check_consumption_status(client: DataPlaneClient) -> None:
    client.check_consumption_status("p", ["train"])


def _put_samples(client: DataPlaneClient) -> None:
    client.put_samples(["sample-0"], "p")


def _get_samples(client: DataPlaneClient) -> None:
    client.get_samples(["sample-0"], "p", ["x"])


def _list_sample_ids(client: DataPlaneClient) -> None:
    client.list_sample_ids("p")


def _clear_samples(client: DataPlaneClient) -> None:
    client.clear_samples(["sample-0"], "p")


_DATA_OPERATION_INVOKERS: dict[str, Callable[[DataPlaneClient], None]] = {
    "register_partition": _register_partition,
    "claim_meta": _claim_meta,
    "get_data": _get_data,
    "check_consumption_status": _check_consumption_status,
    "put_samples": _put_samples,
    "get_samples": _get_samples,
    "list_sample_ids": _list_sample_ids,
    "clear_samples": _clear_samples,
}
_LIFECYCLE_METHODS = {"save_checkpoint", "load_checkpoint", "close"}


def test_register_partition_uses_unique_schema_warmup_key(monkeypatch) -> None:
    from nemo_rl.data_plane.adapters import transfer_queue as tq_adapter

    put_calls = []
    clear_calls = []

    def fake_put(**kwargs):
        put_calls.append(kwargs)

    def fake_clear(**kwargs):
        clear_calls.append(kwargs)

    monkeypatch.setattr(tq_adapter.tq, "kv_batch_put", fake_put)
    monkeypatch.setattr(tq_adapter.tq, "kv_clear", fake_clear)
    # bootstrap=False only connects to an existing controller; stubbing that
    # lets the real __init__ run, so this test cannot drift from it.
    monkeypatch.setattr(tq_adapter, "_connect_existing", lambda: None)

    client = tq_adapter.TQDataPlaneClient(
        {
            "enabled": True,
            "impl": "transfer_queue",
            "backend": "simple",
            "claim_meta_poll_interval_s": 0.5,
        },
        bootstrap=False,
    )
    client.register_partition(
        partition_id="obj-backend",
        fields=["msg_log"],
        num_samples=8,
        consumer_tasks=["read"],
    )
    # Same fields again: already warmed, so no second put/clear.
    client.register_partition(
        partition_id="obj-backend",
        fields=["msg_log"],
        num_samples=8,
        consumer_tasks=["read"],
    )
    assert len(put_calls) == 1

    # A genuinely new field warms only that field, under a fresh key --
    # mooncake has no upsert, so a reused key would hit stale metadata.
    client.register_partition(
        partition_id="obj-backend",
        fields=["msg_log", "rewards"],
        num_samples=8,
        consumer_tasks=["read"],
    )

    assert len(put_calls) == 2
    schema_keys = [call["keys"][0] for call in put_calls]
    assert len(set(schema_keys)) == 2
    assert all(key.startswith("__schema__:obj-backend:") for key in schema_keys)
    assert [call["partition_id"] for call in put_calls] == [
        "obj-backend",
        "obj-backend",
    ]
    assert [list(call["fields"].keys()) for call in put_calls] == [
        ["msg_log"],
        ["rewards"],
    ]
    assert clear_calls == [
        {"keys": [schema_keys[0]], "partition_id": "obj-backend"},
        {"keys": [schema_keys[1]], "partition_id": "obj-backend"},
    ]


def test_data_operation_guard_covers_the_full_interface() -> None:
    public_abstract_methods = {
        name
        for name, member in inspect.getmembers(DataPlaneClient, inspect.isfunction)
        if getattr(member, "__isabstractmethod__", False)
    }
    assert set(_DATA_OPERATION_INVOKERS) == public_abstract_methods - _LIFECYCLE_METHODS


@pytest.mark.parametrize("operation_name", _DATA_OPERATION_INVOKERS)
def test_each_public_data_operation_marks_the_client_dirty(
    monkeypatch,
    operation_name: str,
) -> None:
    from nemo_rl.data_plane.adapters import transfer_queue as tq_adapter

    tq_meta = MagicMock(size=1, global_indexes=[0], custom_meta=[{}])
    tq_client = MagicMock()
    tq_client.get_meta.return_value = tq_meta
    tq_client.kv_retrieve_keys.return_value = ["sample-0"]
    tq_client.check_consumption_status.return_value = True
    monkeypatch.setattr(tq_adapter.tq, "get_client", MagicMock(return_value=tq_client))
    monkeypatch.setattr(tq_adapter.tq, "kv_batch_put", MagicMock())
    monkeypatch.setattr(
        tq_adapter.tq,
        "kv_batch_get",
        MagicMock(
            return_value=TensorDict(
                {"x": torch.tensor([1])},
                batch_size=[1],
            )
        ),
    )
    monkeypatch.setattr(
        tq_adapter.tq,
        "kv_list",
        MagicMock(return_value={"p": {"sample-0": {}}}),
    )
    monkeypatch.setattr(tq_adapter.tq, "kv_clear", MagicMock())

    client = object.__new__(tq_adapter.TQDataPlaneClient)
    client._data_operations_started = False
    client._warmed_fields = {}
    client._poll_interval_s = 0

    _DATA_OPERATION_INVOKERS[operation_name](client)

    assert client._data_operations_started


def test_checkpoint_lifecycle_forwards_to_tq(monkeypatch, tmp_path) -> None:
    from nemo_rl.data_plane.adapters import transfer_queue as tq_adapter

    connect_calls = []
    save_calls = []
    load_calls = []
    monkeypatch.setattr(
        tq_adapter,
        "_connect_existing",
        lambda: connect_calls.append(None),
    )
    monkeypatch.setattr(
        tq_adapter.tq,
        "save_checkpoint",
        lambda checkpoint_dir, *, metadata=None: save_calls.append(
            (checkpoint_dir, metadata)
        ),
    )
    monkeypatch.setattr(
        tq_adapter.tq,
        "load_checkpoint",
        lambda checkpoint_dir: load_calls.append(checkpoint_dir),
    )

    client = object.__new__(tq_adapter.TQDataPlaneClient)
    client._backend = "simple"
    client._supports_checkpointing = True
    client._data_operations_started = False
    checkpoint_dir = tmp_path / "step-7"
    checkpoint_dir.mkdir()
    (checkpoint_dir / "metadata.json").write_text(
        json.dumps({"storage_saved": True, "user_metadata": {"step": 7}})
    )
    client.save_checkpoint(checkpoint_dir, metadata={"step": 7})
    metadata = client.load_checkpoint(checkpoint_dir)

    assert connect_calls == [None, None]
    assert save_calls == [(checkpoint_dir, {"step": 7})]
    assert load_calls == [checkpoint_dir]
    assert metadata == {"step": 7}
    assert client._data_operations_started


def test_list_sample_ids_uses_tq_partition_listing(monkeypatch) -> None:
    from nemo_rl.data_plane.adapters import transfer_queue as tq_adapter

    list_call = MagicMock(
        return_value={"rollout_data": {"sample-b": {}, "sample-a": {}}}
    )
    monkeypatch.setattr(tq_adapter.tq, "kv_list", list_call)
    client = object.__new__(tq_adapter.TQDataPlaneClient)
    client._data_operations_started = False

    sample_ids = client.list_sample_ids("rollout_data")

    assert sample_ids == ["sample-a", "sample-b"]
    assert client._data_operations_started
    list_call.assert_called_once_with(partition_id="rollout_data")


def test_checkpoint_load_rejects_client_after_data_operation(
    monkeypatch,
    tmp_path,
) -> None:
    from nemo_rl.data_plane.adapters import transfer_queue as tq_adapter

    connect = MagicMock()
    load = MagicMock()
    monkeypatch.setattr(tq_adapter, "_connect_existing", connect)
    monkeypatch.setattr(tq_adapter.tq, "load_checkpoint", load)
    monkeypatch.setattr(tq_adapter.tq, "kv_batch_put", MagicMock())

    client = object.__new__(tq_adapter.TQDataPlaneClient)
    client._backend = "simple"
    client._supports_checkpointing = True
    client._data_operations_started = False
    client.put_samples(
        sample_ids=["sample-0"],
        partition_id="rollout_data",
        fields=TensorDict({"x": torch.tensor([1])}, batch_size=[1]),
    )

    with pytest.raises(RuntimeError, match="requires a clean TQ client"):
        client.load_checkpoint(tmp_path / "data-plane")

    connect.assert_not_called()
    load.assert_not_called()


def test_failed_checkpoint_load_leaves_client_in_dirty_state(
    monkeypatch,
    tmp_path,
) -> None:
    from nemo_rl.data_plane.adapters import transfer_queue as tq_adapter

    connect = MagicMock()
    load = MagicMock(side_effect=RuntimeError("injected partial restore"))
    monkeypatch.setattr(tq_adapter, "_connect_existing", connect)
    monkeypatch.setattr(tq_adapter.tq, "load_checkpoint", load)

    checkpoint_dir = tmp_path / "data-plane"
    checkpoint_dir.mkdir()
    (checkpoint_dir / "metadata.json").write_text(
        json.dumps({"storage_saved": True, "user_metadata": {"step": 7}})
    )
    client = object.__new__(tq_adapter.TQDataPlaneClient)
    client._backend = "simple"
    client._supports_checkpointing = True
    client._data_operations_started = False

    with pytest.raises(RuntimeError, match="injected partial restore"):
        client.load_checkpoint(checkpoint_dir)
    with pytest.raises(RuntimeError, match="requires a clean TQ client"):
        client.load_checkpoint(checkpoint_dir)

    connect.assert_called_once_with()
    load.assert_called_once_with(checkpoint_dir)


@pytest.mark.parametrize("operation", ["save", "load"])
def test_mooncake_checkpoint_lifecycle_fails_loudly(
    monkeypatch,
    tmp_path,
    operation: str,
) -> None:
    from nemo_rl.data_plane.adapters import transfer_queue as tq_adapter

    connect = MagicMock()
    save = MagicMock()
    load = MagicMock()
    monkeypatch.setattr(tq_adapter, "_connect_existing", connect)
    monkeypatch.setattr(tq_adapter.tq, "save_checkpoint", save)
    monkeypatch.setattr(tq_adapter.tq, "load_checkpoint", load)

    client = object.__new__(tq_adapter.TQDataPlaneClient)
    client._backend = "mooncake_cpu"
    client._supports_checkpointing = False
    client._data_operations_started = False
    checkpoint_dir = tmp_path / "step-7"

    with pytest.raises(NotImplementedError, match="mooncake_cpu"):
        if operation == "save":
            client.save_checkpoint(checkpoint_dir)
        else:
            client.load_checkpoint(checkpoint_dir)

    connect.assert_not_called()
    save.assert_not_called()
    load.assert_not_called()


# ``tq_client`` (simple) and ``tq_client_backends`` (parametrized over
# simple + mooncake_cpu) are session-scoped fixtures provided by
# ``tests/unit/data_plane/conftest.py``. See that file for the rationale.


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
    values = torch.arange(12).reshape(4, 3)
    client.put_samples(
        sample_ids=keys,
        partition_id="smoke-backend",
        fields=TensorDict({"x": values}, batch_size=[4]),
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
    expected = torch.stack([values[keys.index(k)] for k in meta.sample_ids])
    assert not data["x"].is_nested
    assert data["x"].shape == expected.shape
    assert torch.equal(data["x"], expected)

    client.clear_samples(sample_ids=None, partition_id="smoke-backend")


def test_smoke_round_trip_1d_fields(tq_client_backends) -> None:
    """A 1D (N,) tensor put into TQ must come back as (N,), not (N,1).

    Regression guard for R-C2, and the end-to-end proof of
    ``_patch_scalar_field_schema``: upstream ``extract_field_schema``
    reports a ``(1,)`` sample shape for dense 1-D fields while storage
    holds 0-d rows, so the KV path would rebuild this column as nested
    ``(1,)`` rows. The patch makes the reported shape match what is
    stored; ``simple`` never consulted the schema and is unaffected either
    way, so running both backends here pins that they agree.
    """
    n = 6
    total_reward = torch.arange(n, dtype=torch.float32)

    tq_client_backends.register_partition(
        partition_id="smoke-1d",
        fields=["total_reward"],
        num_samples=n,
        consumer_tasks=["read"],
    )
    keys = [f"k{i}" for i in range(n)]
    tq_client_backends.put_samples(
        sample_ids=keys,
        partition_id="smoke-1d",
        fields=TensorDict({"total_reward": total_reward}, batch_size=[n]),
    )

    meta = tq_client_backends.claim_meta(
        partition_id="smoke-1d",
        task_name="read",
        required_fields=["total_reward"],
        batch_size=n,
        timeout_s=30.0,
    )
    data = tq_client_backends.get_data(meta)

    assert data["total_reward"].shape == total_reward.shape, (
        f"Expected shape {tuple(total_reward.shape)} for 1D field, "
        f"got {tuple(data['total_reward'].shape)}. "
        "TQ must not unsqueeze 1D tensors silently (R-C2)."
    )

    tq_client_backends.clear_samples(sample_ids=None, partition_id="smoke-1d")


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


def test_from_wire_preserves_object_arrays_through_densify() -> None:
    """``_from_wire`` must not drop non-tensor leaves when it rebuilds.

    It only constructs a new ``TensorDict`` when it densifies a uniform
    nested field, and that rebuild is where tensordict 0.12.2 can silently
    strip ``NonTensorStack`` / ``NonTensorData`` leaves. Symmetric to the
    documented ``.contiguous()`` bug in ``adapters/transfer_queue.py``.
    """
    from nemo_rl.data_plane.adapters.transfer_queue import _from_wire

    arr = np.empty(4, dtype=object)
    arr[:] = [["a", "b"], ["c"], ["d", "e"], ["f"]]
    # Uniform rows, so _from_wire densifies and takes the rebuild path.
    nested = torch.nested.as_nested_tensor(
        [torch.zeros(8, dtype=torch.long) for _ in range(4)], layout=torch.jagged
    )
    td = TensorDict(
        {
            "input_ids": nested,
            "input_lengths": torch.tensor([4, 3, 2, 1]),
            "content": arr,
        },
        batch_size=[4],
    )

    restored = _from_wire(td)
    assert not restored["input_ids"].is_nested
    assert restored["input_ids"].shape == (4, 8)
    # Scalar column passes through untouched — no squeeze step any more.
    assert restored["input_lengths"].shape == (4,)
    np.testing.assert_array_equal(restored["content"], arr)
