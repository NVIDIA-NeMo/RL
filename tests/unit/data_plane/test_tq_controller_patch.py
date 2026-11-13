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
import pytest
import torch
from tensordict import TensorDict

pytest.importorskip("transfer_queue")
from transfer_queue.controller import FieldMeta, TransferQueueController  # noqa: E402

from nemo_rl.data_plane.adapters import tq_controller_patch as patch  # noqa: E402
from tests.unit.data_plane._rollout_shapes import register_train_partition  # noqa: E402

CONTROLLER = TransferQueueController.__ray_metadata__.modified_class
STOCK_REMOVE = FieldMeta.remove_samples


def test_installed_tq_is_the_version_the_overrides_were_copied_from():
    # fails when the TQ pin moves: re-copy the methods
    patch.assert_stock_tq_version()


def test_other_tq_version_is_rejected(monkeypatch):
    monkeypatch.setattr(patch.transfer_queue, "__version__", "0.0.0")
    with pytest.raises(RuntimeError, match="STOCK_TQ_VERSION"):
        patch.assert_stock_tq_version()


def _nested_schema(shapes):
    return {
        "dtype": "torch.float32",
        "shape": None,
        "is_nested": True,
        "is_non_tensor": False,
        "per_sample_shapes": shapes,
    }


def _nested_field(n):
    shapes = {i: (256,) if i < n - 1 else (128,) for i in range(n)}
    return FieldMeta(global_indexes=set(range(n)), **_nested_schema(shapes))


def _served_shapes(field_meta, indexes):
    return [
        field_meta.per_sample_shapes[i] if field_meta.is_nested else field_meta.shape
        for i in indexes
    ]


def test_remove_samples_never_collapses_a_nested_field():
    field_meta = _nested_field(40)

    # the odd shape is gone: stock would collapse here
    patch.remove_samples(field_meta, [39])
    assert field_meta.is_nested is True
    assert field_meta.shape is None
    assert field_meta.per_sample_shapes == {i: (256,) for i in range(39)}


def _replay(remove):
    field_meta, remaining, served = _nested_field(40), list(range(40)), []
    for new in range(40, 60, 2):
        cleared, remaining = remaining[:4], remaining[4:]
        remove(field_meta, cleared)
        field_meta.update(
            _nested_schema({new: (512,), new + 1: (256,)}), [new, new + 1]
        )
        remaining += [new, new + 1]
        served.append(
            (_served_shapes(field_meta, remaining), set(field_meta.global_indexes))
        )
    return served


def test_remove_samples_serves_the_same_shapes_as_stock():
    assert _replay(patch.remove_samples) == _replay(STOCK_REMOVE)


def test_remove_samples_resets_an_emptied_field_like_stock():
    stock, patched = _nested_field(3), _nested_field(3)
    STOCK_REMOVE(stock, [0, 1, 2])
    patch.remove_samples(patched, [0, 1, 2])
    assert patched == stock


def _schema(fields):
    return {
        f: {
            "dtype": torch.int64,
            "shape": (2,),
            "is_nested": False,
            "is_non_tensor": False,
        }
        for f in fields
    }


def test_override_matches_stock_kv_retrieve_meta():
    ctl = CONTROLLER(polling_mode=True)
    indexes = list(
        ctl.kv_retrieve_meta(["a", "b", "c"], "p", create=True).global_indexes
    )
    ctl.update_production_status(
        "p", indexes, _schema(["f0", "f1"]), {g: {} for g in indexes}
    )
    ctl.update_production_status(
        "p", indexes[:1], _schema(["f2"]), {indexes[0]: {}}
    )  # only row "a"

    for keys in (["a"], ["a", "b"], ["b", "c"], ["a", "b", "c"], ["missing"], []):
        want = CONTROLLER.kv_retrieve_meta(ctl, keys, "p")
        got = patch.kv_retrieve_meta(ctl, keys, "p")
        assert list(got.global_indexes) == list(want.global_indexes)
        assert sorted(got.field_schema) == sorted(want.field_schema)
    assert sorted(patch.kv_retrieve_meta(ctl, ["a", "b"], "p").field_schema) == [
        "f0",
        "f1",
    ]


def test_override_matches_stock_kv_retrieve_meta_with_create():
    stock, patched = CONTROLLER(polling_mode=True), CONTROLLER(polling_mode=True)
    for ctl, retrieve in (
        (stock, CONTROLLER.kv_retrieve_meta),
        (patched, patch.kv_retrieve_meta),
    ):
        indexes = list(retrieve(ctl, ["a", "b"], "p", create=True).global_indexes)
        ctl.update_production_status(
            "p", indexes, _schema(["f0"]), {g: {} for g in indexes}
        )
    for keys in (["a", "x"], ["y", "b", "z"], ["new"]):
        want = CONTROLLER.kv_retrieve_meta(stock, keys, "p", create=True)
        got = patch.kv_retrieve_meta(patched, keys, "p", create=True)
        assert list(got.global_indexes) == list(want.global_indexes)
        assert sorted(got.field_schema) == sorted(want.field_schema)
    assert (
        patched._get_partition("p").keys_mapping
        == stock._get_partition("p").keys_mapping
    )


def test_configure_controller_patch_twice_wraps_once(monkeypatch):
    monkeypatch.setattr(
        patch.interface, "TransferQueueController", TransferQueueController
    )
    patch.configure_controller_patch()
    wrapped = patch.interface.TransferQueueController
    patch.configure_controller_patch()
    assert patch.interface.TransferQueueController is wrapped
    assert wrapped.__ray_metadata__.modified_class.__mro__[1] is CONTROLLER


def test_patched_controller_installs_both_overrides_on_start(monkeypatch):
    monkeypatch.setattr(FieldMeta, "remove_samples", STOCK_REMOVE)
    patched = patch.make_patched_controller(
        TransferQueueController
    ).__ray_metadata__.modified_class
    # Ray wraps actor methods, so compare where the method comes from.
    assert patched.kv_retrieve_meta.__module__ == patch.__name__
    # Building the class patches nothing; starting the controller does.
    assert FieldMeta.remove_samples is STOCK_REMOVE

    patched(polling_mode=True)
    assert FieldMeta.remove_samples is patch.remove_samples


def test_rows_survive_a_clear_that_stock_would_collapse(tq_client_backends):
    client = tq_client_backends
    partition_id = "tq-patch-no-collapse"
    g = torch.Generator().manual_seed(0)
    rows = {
        k: torch.randint(0, 50000, (n,), generator=g)
        for k, n in (("a", 256), ("b", 256), ("c", 128), ("d", 512))
    }
    register_train_partition(
        client, num_samples=len(rows), fields=["input_ids"], partition_id=partition_id
    )

    def put(keys):
        value = torch.nested.as_nested_tensor(
            [rows[k] for k in keys], layout=torch.jagged
        )
        client.put_samples(
            sample_ids=keys,
            partition_id=partition_id,
            fields=TensorDict({"input_ids": value}, batch_size=[len(keys)]),
        )

    def assert_rows(keys):
        value = client.get_samples(
            sample_ids=keys, partition_id=partition_id, select_fields=["input_ids"]
        )["input_ids"]
        got = list(value.unbind()) if value.is_nested else list(value)
        for have, key in zip(got, keys, strict=True):
            assert torch.equal(have, rows[key]), key

    try:
        put(["a", "b", "c"])
        client.clear_samples(sample_ids=["c"], partition_id=partition_id)
        assert_rows(["a", "b"])
        client.clear_samples(sample_ids=["b"], partition_id=partition_id)
        assert_rows(["a"])
        put(["d"])
        assert_rows(["a", "d"])
    finally:
        client.clear_samples(sample_ids=["a", "d"], partition_id=partition_id)
