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
    conf.backend.NixlStore.placement = {"policy": "spread"}
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
    assert manifest["store"] == "unit" and len(manifest["shards"]) == 3
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


def test_restore_refuses_other_store_kind(ray_cluster):
    ckpt = REPO / ".nvdp_store" / f"ckpt-{uuid.uuid4().hex[:8]}"
    tq, conf = _system(1)
    tq.kv_batch_put(["a", "b"], "train", fields=_batch(1)[:2])
    tq.save_checkpoint(str(ckpt))
    _teardown(tq, conf)

    import transfer_queue as tq2

    import nemo_rl.data_plane.adapters.tq_nixl  # noqa: F401

    conf = make_tq_conf(num_units=1, namespace=NS)
    conf.backend.NixlStore.store = {
        "kind": "file",
        "root": str(REPO / ".nvdp_store" / f"fs-{uuid.uuid4().hex[:6]}"),
    }
    tq2.init(conf)
    with pytest.raises(Exception, match="store="):
        tq2.load_checkpoint(str(ckpt))
    _teardown(tq2, conf)
