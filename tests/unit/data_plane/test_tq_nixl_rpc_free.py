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
"""Hinted reads skip every RPC; stale hints fall back to the pinned path."""

from __future__ import annotations

import pytest
import torch
from tensordict import TensorDict

from tests.unit.data_plane._nixl_helpers import dense, make_tq_conf

pytestmark = pytest.mark.nixl


@pytest.fixture(scope="module")
def sys_(ray_cluster):
    import transfer_queue as tq

    import nemo_rl.data_plane.adapters.tq_nixl  # noqa: F401

    conf = make_tq_conf(num_units=2, namespace="nvdp-rpcfree")
    tq.init(conf)
    yield tq
    tq.close()
    nemo_rl.data_plane.adapters.tq_nixl.shutdown(conf)


def _client():
    from transfer_queue import interface as tq_interface

    return tq_interface._TQ_CLIENT.storage_manager.storage_client


def test_hinted_reads_are_rpc_free_and_exact(sys_):
    tq = sys_
    store = _client().store
    before = dict(store.stats)
    keys = [f"k{i}" for i in range(8)]
    data = torch.randn(8, 1000)
    tq.kv_batch_put(keys, "train", fields=TensorDict({"x": data}, batch_size=[8]))
    got = tq.kv_batch_get(keys, "train", select_fields=["x"])
    assert torch.equal(dense(got["x"]), data)
    assert store.stats["fast_reads"] > before["fast_reads"]
    assert store.stats["slow_reads"] == before["slow_reads"]
    assert (
        store.stats["unit_refresh"] == before["unit_refresh"]
    )  # puts did not refresh units
    tq.kv_clear(keys, "train")


def test_stale_hint_falls_back_and_stays_exact(sys_):
    tq = sys_
    store = _client().store
    keys = [f"s{i}" for i in range(4)]
    data = torch.arange(4 * 64, dtype=torch.float32).reshape(4, 64)
    tq.kv_batch_put(keys, "train", fields=TensorDict({"x": data}, batch_size=[4]))
    # Corrupt the cached slab offset in this client's view of the location meta by
    # reading through store.read directly with a wrong "so": the footer tag must not
    # match, and the pinned path must still return the right bytes.
    import numpy as np

    from nemo_rl.data_plane.nixl import blob_format

    metas = tq.kv_list("train") if hasattr(tq, "kv_list") else None
    del metas
    before = dict(store.stats)
    blob = next(iter(store._blob_unit))
    unit = store._blob_unit[blob]
    # Live stamp, wrong geometry: the footer read lands on bytes that are not
    # this blob's footer, so the tag check must reject it.
    stale = {"u": unit, "so": 0, "z": 4096 + 7 * 64, "g": store._gen(unit)}
    buf = np.zeros(4096 + blob_format.FOOTER_SIZE, dtype=np.uint8)
    base = store.ep.register(buf)
    try:
        store.read(
            {blob: [(0, 64, base)]},
            {blob: stale},
            (base + 4096, memoryview(buf)[4096:]),
        )
    finally:
        store.ep.deregister(base)
    assert store.stats["tag_mismatch"] > before["tag_mismatch"]
    assert (
        store.stats["slow_reads"] > before["slow_reads"]
    )  # fell back to the pinned path
    got = tq.kv_batch_get(keys, "train", select_fields=["x"])
    assert torch.equal(dense(got["x"]), data)
    tq.kv_clear(keys, "train")
