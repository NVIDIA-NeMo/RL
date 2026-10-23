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

pytest.importorskip("transfer_queue")
from transfer_queue.controller import FieldMeta, TransferQueueController  # noqa: E402

from nemo_rl.data_plane.adapters import tq_kv_meta_patch as patch  # noqa: E402

CONTROLLER = TransferQueueController.__ray_metadata__.modified_class


def test_installed_tq_is_the_version_the_overrides_were_copied_from():
    # fails when the TQ pin moves: re-copy the methods
    patch.assert_stock_tq_version()


def test_other_tq_version_is_rejected(monkeypatch):
    monkeypatch.setattr(patch.transfer_queue, "__version__", "0.0.0")
    with pytest.raises(RuntimeError, match="Re-copy"):
        patch.assert_stock_tq_version()


def _nested_field(n):
    return FieldMeta(
        global_indexes=set(range(n)),
        dtype="torch.float32",
        is_nested=True,
        is_non_tensor=False,
        per_sample_shapes={i: (256,) if i < n - 1 else (128,) for i in range(n)},
    )


def _served_shapes(field_meta, indexes):
    return [
        field_meta.per_sample_shapes[i] if field_meta.is_nested else field_meta.shape
        for i in indexes
    ]


def test_remove_samples_defers_the_rescan_until_an_eighth_is_gone(monkeypatch):
    monkeypatch.setattr(FieldMeta, "remove_samples", patch.remove_samples)
    monkeypatch.setattr(FieldMeta, "removed_since_scan", 0, raising=False)
    field_meta = _nested_field(40)

    field_meta.remove_samples(
        [39]
    )  # the odd shape is gone, but 1 of 39 is under an eighth
    assert field_meta.is_nested is True

    field_meta.remove_samples([0, 1, 2, 3])  # 5 of 35 reaches an eighth
    assert field_meta.is_nested is False
    assert field_meta.shape == (256,)
    assert field_meta.per_sample_shapes == {}


def test_remove_samples_override_serves_the_same_shapes_as_stock(monkeypatch):
    stock, patched = _nested_field(40), _nested_field(40)
    monkeypatch.setattr(FieldMeta, "removed_since_scan", 0, raising=False)
    remaining = list(range(40))
    while remaining:
        cleared, remaining = remaining[:4], remaining[4:]
        FieldMeta.remove_samples(stock, cleared)
        patch.remove_samples(patched, cleared)
        assert _served_shapes(patched, remaining) == _served_shapes(stock, remaining)
        assert patched.global_indexes == stock.global_indexes
    assert (patched.is_nested, patched.shape, patched.dtype) == (
        stock.is_nested,
        stock.shape,
        stock.dtype,
    )


def test_override_matches_stock_kv_retrieve_meta():
    ctl = CONTROLLER(polling_mode=True)
    schema = lambda fields: {  # noqa: E731
        f: {
            "dtype": torch.int64,
            "shape": (2,),
            "is_nested": False,
            "is_non_tensor": False,
        }
        for f in fields
    }
    indexes = list(
        ctl.kv_retrieve_meta(["a", "b", "c"], "p", create=True).global_indexes
    )
    ctl.update_production_status(
        "p", indexes, schema(["f0", "f1"]), {g: {} for g in indexes}
    )
    ctl.update_production_status(
        "p", indexes[:1], schema(["f2"]), {indexes[0]: {}}
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


def test_patched_controller_installs_both_overrides_on_start(monkeypatch):
    monkeypatch.setattr(FieldMeta, "remove_samples", FieldMeta.remove_samples)
    monkeypatch.setattr(FieldMeta, "removed_since_scan", 0, raising=False)
    patched = patch.make_patched_controller(
        TransferQueueController
    ).__ray_metadata__.modified_class
    # Ray wraps actor methods, so compare where the method comes from.
    assert patched.kv_retrieve_meta.__module__ == patch.__name__
    # Building the class patches nothing; starting the controller does.
    assert FieldMeta.remove_samples is not patch.remove_samples

    patched(polling_mode=True)
    assert FieldMeta.remove_samples is patch.remove_samples
