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
from nemo_rl.data_plane.adapters import tq_kv_meta_patch as patch  # noqa: E402
from transfer_queue.controller import TransferQueueController  # noqa: E402

CONTROLLER = TransferQueueController.__ray_metadata__.modified_class


def test_stock_source_matches_the_pinned_hash():
    patch.assert_stock_source(CONTROLLER)  # fails when the TQ pin moves: re-copy the method


def test_changed_source_is_rejected():
    class Changed:
        def kv_retrieve_meta(self, keys, partition_id, create=False):
            return None

    with pytest.raises(RuntimeError, match="changed"):
        patch.assert_stock_source(Changed)


def test_override_matches_stock_kv_retrieve_meta():
    ctl = CONTROLLER(polling_mode=True)
    schema = lambda fields: {  # noqa: E731
        f: {"dtype": torch.int64, "shape": (2,), "is_nested": False, "is_non_tensor": False} for f in fields
    }
    indexes = list(ctl.kv_retrieve_meta(["a", "b", "c"], "p", create=True).global_indexes)
    ctl.update_production_status("p", indexes, schema(["f0", "f1"]), {g: {} for g in indexes})
    ctl.update_production_status("p", indexes[:1], schema(["f2"]), {indexes[0]: {}})  # only row "a"

    for keys in (["a"], ["a", "b"], ["b", "c"], ["a", "b", "c"], ["missing"], []):
        want = CONTROLLER.kv_retrieve_meta(ctl, keys, "p")
        got = patch.kv_retrieve_meta(ctl, keys, "p")
        assert list(got.global_indexes) == list(want.global_indexes)
        assert sorted(got.field_schema) == sorted(want.field_schema)
    assert sorted(patch.kv_retrieve_meta(ctl, ["a", "b"], "p").field_schema) == ["f0", "f1"]
