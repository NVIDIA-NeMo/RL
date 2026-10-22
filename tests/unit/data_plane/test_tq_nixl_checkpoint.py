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
"""A2: TQ checkpoint save on N units, restore on N-1 units, byte-exact reads.

Runs the real ``tq.save_checkpoint`` / ``tq.load_checkpoint`` path, so the
controller state and our storage shards are exercised together, exactly as
NeMo-RL's TransferQueue adapter drives them.
"""

from __future__ import annotations

import os
import tempfile

import json
import uuid
from pathlib import Path

import pytest
import ray
import torch
from tensordict import NonTensorStack, TensorDict

from tests.unit.data_plane._nixl_helpers import dense, make_tq_conf

pytestmark = pytest.mark.nixl

REPO = Path(os.environ.get("NRL_NIXL_TEST_DIR", tempfile.gettempdir()))
NS = "nvdp-ckpt"
N_KEYS = 24


def _batch(seed: int) -> TensorDict:
    g = torch.Generator().manual_seed(seed)
    return TensorDict(
        {
            "input_ids": torch.randint(0, 32000, (N_KEYS, 96), generator=g),
            "logprobs": torch.randn(N_KEYS, 96, generator=g),
            "mask": torch.rand(N_KEYS, 96, generator=g) > 0.5,
            "reward": torch.randn(N_KEYS, generator=g),
            "text": NonTensorStack(
                *[f"s{seed}_{i}_{'y' * (i % 7)}" for i in range(N_KEYS)]
            ),
        },
        batch_size=[N_KEYS],
    )


def _system(num_units: int):
    import transfer_queue as tq

    import nemo_rl.data_plane.adapters.tq_nixl  # noqa: F401

    conf = make_tq_conf(
        num_units=num_units, slab_bytes=32 << 20, staging_bytes=16 << 20, namespace=NS
    )
    tq.init(conf)
    return tq, conf


def _teardown(tq, conf):
    import nemo_rl.data_plane.adapters.tq_nixl

    tq.close()
    nemo_rl.data_plane.adapters.tq_nixl.shutdown(conf)


def _check(tq, keys, expected: TensorDict):
    got = tq.kv_batch_get(
        keys, "train", select_fields=["input_ids", "logprobs", "mask", "reward", "text"]
    )
    for f in ("input_ids", "logprobs", "mask", "reward"):
        assert torch.equal(dense(got[f]), expected[f]), f
    assert list(got["text"]) == list(expected["text"])


def test_save_on_3_restore_on_2(ray_cluster, tmp_path_factory):
    ckpt = (
        REPO / ".nvdp_store" / f"ckpt-{uuid.uuid4().hex[:8]}"
    )  # Lustre: visible to every actor
    tq, conf = _system(3)
    keys = [f"k{i}" for i in range(N_KEYS)]
    expected = _batch(7)
    tq.kv_batch_put(keys, "train", fields=expected)
    # a later write-back under the same keys lands in other blobs
    adv = torch.arange(N_KEYS, dtype=torch.float32) / 3
    tq.kv_batch_put(
        keys, "train", fields=TensorDict({"advantages": adv}, batch_size=[N_KEYS])
    )
    _check(tq, keys, expected)

    tq.save_checkpoint(str(ckpt))
    meta = json.loads((ckpt / "metadata.json").read_text())
    assert meta["storage_saved"] is True
    manifest = json.loads((ckpt / "nixl_store" / "manifest.json").read_text())
    assert len(manifest["shards"]) == 3
    n_blobs = sum(len(s["blobs"]) for s in manifest["shards"])
    assert n_blobs >= 2
    _teardown(tq, conf)

    # "restart" with one unit fewer: fresh actors, fresh controller
    tq, conf = _system(2)
    tq.load_checkpoint(str(ckpt))
    _check(tq, keys, expected)
    got = tq.kv_batch_get(keys, "train", select_fields=["advantages"])
    assert torch.equal(dense(got["advantages"]), adv)

    # restored blobs are owned by the new units: clearing frees them there
    units = [ray.get_actor(f"NixlStorageUnit#{i}", namespace=NS) for i in range(2)]
    before = sum(ray.get(h.stats.remote())["blobs"] for h in units)
    assert before == n_blobs
    tq.kv_clear(keys, "train")
    import time

    deadline = (
        time.monotonic() + 10.0
    )  # clears are not awaited: units release asynchronously
    while True:
        after = sum(ray.get(h.stats.remote())["blobs"] for h in units)
        if after == 0 or time.monotonic() > deadline:
            break
        time.sleep(0.2)
    assert after == 0
    _teardown(tq, conf)


def test_restore_reads_all_shards_at_once_and_spills_what_does_not_fit(monkeypatch):
    """Each round sends every shard to its next unit before waiting on any;
    a blob its home unit has no room for moves on to the next unit."""
    from types import SimpleNamespace

    from nemo_rl.data_plane.adapters import tq_nixl_checkpoint as ck

    events: list[tuple] = []

    class Unit:
        def __init__(self, u, room):
            self.u, self.room = u, room
            self.load_shard = SimpleNamespace(remote=self._load)

        def _load(self, path, blobs, spill):
            events.append(("send", self.u, path))
            loaded, failed = [], []
            for b in blobs:
                (loaded if self.room > 0 else failed).append(b)
                self.room -= b in loaded
            return {"loaded": loaded, "failed": failed}

    units = {0: Unit(0, room=1), 1: Unit(1, room=10)}
    put = []
    store = SimpleNamespace(
        directory=SimpleNamespace(put_blobs=SimpleNamespace(remote=put.extend)),
        _blob_unit={},
    )

    def get(x):
        events.append(("wait",))
        return x

    monkeypatch.setattr(ck, "_active_units", lambda s: [0, 1])
    monkeypatch.setattr(ck, "_unit_actor", lambda s, u: units[u])
    monkeypatch.setattr(ck, "ray", SimpleNamespace(get=get))
    manifest = {
        "shards": [
            {"file": "unit-0.shard", "blobs": {"a": [0, 8, 1], "b": [8, 8, 1]}},
            {"file": "unit-1.shard", "blobs": {"c": [0, 8, 1]}},
        ]
    }
    ck._load_units(store, Path("/ckpt"), manifest)

    assert [e[0] for e in events[:2]] == ["send", "send"]  # both shards before a wait
    assert store._blob_unit == {"a": 0, "c": 1, "b": 1}
    assert sorted(put) == sorted(store._blob_unit.items())


def test_ssd_on_save_on_2_restore_on_1_spills_and_reads_back(
    ray_cluster, tmp_path_factory
):
    """SSD tier on: spilled blobs are saved from their files, and a restore the
    put area cannot hold spills the rest to disk; every value reads back exact."""
    import transfer_queue as tq

    import nemo_rl.data_plane.adapters.tq_nixl as tq_nixl

    MiB = 1 << 20
    ckpt = REPO / ".nvdp_store" / f"ckpt-{uuid.uuid4().hex[:8]}"
    ns = "nvdp-ckpt-ssd"

    def system(n):
        # 8 MiB slab per unit: 4 MiB put area + 4 MiB IO area.
        conf = make_tq_conf(num_units=n, slab_bytes=8 * MiB, namespace=ns)
        conf.backend.NixlStore.op_timeout_s = 60.0
        conf.backend.NixlStore.ssd = {
            "enabled": True,
            "dir": str(tmp_path_factory.mktemp("ssd")),
            "staging_bytes": 4 * MiB,
        }
        tq.init(conf)
        units = [ray.get_actor(f"NixlStorageUnit#{i}", namespace=ns) for i in range(n)]
        return conf, units

    def stats(units):
        return [ray.get(h.stats.remote()) for h in units]

    conf, units = system(2)
    keys = [f"d{i}" for i in range(14)]  # 14 MiB through 8 MiB of put area
    vals = [torch.randn(1, MiB // 4) for _ in keys]
    for k, v in zip(keys, vals):
        tq.kv_batch_put([k], "train", fields=TensorDict({"x": v}, batch_size=[1]))
    assert sum(s["disk_only_blobs"] for s in stats(units)) >= 4
    tq.save_checkpoint(str(ckpt))
    tq.close()
    tq_nixl.shutdown(conf)

    conf, units = system(1)
    tq.load_checkpoint(str(ckpt))
    (st,) = stats(units)
    assert st["blobs"] == 14 and st["disk_only_blobs"] >= 8, st
    for k, v in zip(keys, vals):
        got = tq.kv_batch_get([k], "train", select_fields=["x"])
        assert torch.equal(dense(got["x"]), v), k
    tq.close()
    tq_nixl.shutdown(conf)
