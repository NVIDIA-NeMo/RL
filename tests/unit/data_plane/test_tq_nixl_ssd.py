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
"""SSD tier end to end: puts larger than the slab demote to disk and every
value reads back exact through staging; a clear removes the files."""

from __future__ import annotations

import os
import time

import pytest
import ray
import torch
from tensordict import TensorDict

from tests.unit.data_plane._nixl_helpers import dense, make_tq_conf

pytestmark = pytest.mark.nixl

MiB = 1 << 20


@pytest.fixture(scope="module")
def ssd_sys(ray_cluster, tmp_path_factory):
    import transfer_queue as tq

    import nemo_rl.data_plane.adapters.tq_nixl  # noqa: F401

    root = str(tmp_path_factory.mktemp("ssd"))
    conf = make_tq_conf(num_units=1, slab_bytes=8 * MiB, namespace="nvdp-ssd")
    nixl = conf.backend.NixlStore
    nixl.op_timeout_s = 60.0
    # 4 MiB for puts, 4 MiB staging: a 1 MiB value per blob, three staged at once.
    nixl.ssd = {"enabled": True, "dir": root, "staging_bytes": 4 * MiB}
    tq.init(conf)
    yield tq, conf, root
    tq.close()
    nemo_rl.data_plane.adapters.tq_nixl.shutdown(conf)


def _stats(conf):
    ns = conf.backend.NixlStore.namespace
    return ray.get(ray.get_actor("NixlStorageUnit#0", namespace=ns).stats.remote())


def test_puts_beyond_the_slab_demote_and_read_back_exact(ssd_sys):
    tq, conf, root = ssd_sys
    keys = [f"d{i}" for i in range(12)]  # 12 MiB through a 4 MiB put area
    vals = [torch.randn(1, MiB // 4) for _ in keys]
    for k, v in zip(keys, vals):
        tq.kv_batch_put([k], "train", fields=TensorDict({"x": v}, batch_size=[1]))
    st = _stats(conf)
    assert st["demotions"] >= 8 and st["busy_replies"] >= 1, st
    for k, v in zip(keys, vals):  # one get per key, oldest (demoted) first
        got = tq.kv_batch_get([k], "train", select_fields=["x"])
        assert torch.equal(dense(got["x"]), v), k
    got = tq.kv_batch_get(keys[:3], "train", select_fields=["x"])  # one get, 3 staged
    assert torch.equal(dense(got["x"]), torch.cat(vals[:3]))
    st = _stats(conf)
    assert st["stages"] >= 3 and st["disk_only_blobs"] >= 1, st

    tq.kv_clear(keys, "train")
    end = time.monotonic() + 10
    while any(os.listdir(os.path.join(root, d)) for d in os.listdir(root)):
        assert time.monotonic() < end, "demoted files left after clear"
        time.sleep(0.05)
    assert _stats(conf)["blobs"] == 0
