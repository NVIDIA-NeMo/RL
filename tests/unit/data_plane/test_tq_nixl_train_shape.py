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
"""Stress the exact shape the SingleController train path produces.

16 groups x 16 generations, each group committed by its own put from a
different process (like RolloutManager commits), with the real field mix:
jagged int64 ids, jagged float32 logprobs, jagged bool masks, 0-dim float
scalars, a string tag column; then write-backs of new jagged columns under
the same keys (advantage stage, prev_logprobs), then DP-rank style reads
of interleaved key subsets across every blob and every unit, bit-exact.
"""

from __future__ import annotations

import random

import pytest
import ray
import torch
from tensordict import NonTensorStack, TensorDict

pytestmark = pytest.mark.nixl

GROUPS, GEN = 16, 16


def _rows(n, lo, hi, gen, dtype):
    lens = [int(gen.integers(lo, hi)) for _ in range(n)]
    if dtype is torch.bool:
        return [
            torch.from_numpy(gen.integers(0, 2, L).astype(bool)) for L in lens
        ], lens
    if dtype.is_floating_point:
        return [
            torch.from_numpy(gen.standard_normal(L).astype("float32")).to(dtype)
            for L in lens
        ], lens
    return [
        torch.from_numpy(gen.integers(0, 32000, L).astype("int64")) for L in lens
    ], lens


def _group_batch(seed: int):
    import numpy as np

    gen = np.random.default_rng(seed)
    ids, lens = _rows(GEN, 50, 900, gen, torch.int64)
    lp = [torch.from_numpy(gen.standard_normal(L).astype("float32")) for L in lens]
    mask = [torch.from_numpy(gen.integers(0, 2, L).astype(bool)) for L in lens]
    td = TensorDict(
        {
            "input_ids": torch.nested.nested_tensor(ids, layout=torch.jagged),
            "generation_logprobs": torch.nested.nested_tensor(lp, layout=torch.jagged),
            "token_mask": torch.nested.nested_tensor(mask, layout=torch.jagged),
            "input_lengths": torch.tensor(lens, dtype=torch.int64),
            "sample_mask": torch.ones(GEN, dtype=torch.float32),
            "rewards": torch.from_numpy(gen.standard_normal(GEN).astype("float32")),
            "text": NonTensorStack(
                *[f"g{seed}_r{i}_{'x' * int(gen.integers(1, 40))}" for i in range(GEN)]
            ),
        },
        batch_size=[GEN],
    )
    return td, lens


@ray.remote(num_cpus=1)
class Producer:
    def __init__(self):
        import nemo_rl.data_plane.adapters.tq_nixl  # noqa: F401
        import transfer_queue as tq

        tq.init()
        self.tq = tq

    def commit(self, g: int):
        keys = [f"g{g}_gen{i}" for i in range(GEN)]
        td, lens = _group_batch(1000 + g)
        self.tq.kv_batch_put(
            keys, "train", fields=td, tags=[{"g": g, "i": i} for i in range(GEN)]
        )
        return keys, lens


def _dense(t):
    return (
        torch.stack(list(t.unbind()))
        if isinstance(t, torch.Tensor) and t.is_nested
        else t
    )


def _rows_of(t):
    return list(t.unbind()) if isinstance(t, torch.Tensor) and t.is_nested else list(t)


@pytest.fixture(scope="module")
def many_unit_system(ray_cluster):
    import transfer_queue as tq

    import nemo_rl.data_plane.adapters.tq_nixl  # noqa: F401
    from tests.unit.data_plane._nixl_helpers import make_tq_conf

    conf = make_tq_conf(
        num_units=6,
        slab_bytes=256 << 20,
        staging_bytes=64 << 20,
        namespace="nvdp-train-shape",
    )
    conf.backend.NixlStore.placement = {"policy": "spread"}
    tq.init(conf)
    yield conf
    tq.close()
    nemo_rl.data_plane.adapters.tq_nixl.shutdown(conf)


def test_sc_train_shape_bit_exact(many_unit_system):
    import transfer_queue as tq

    producers = [Producer.remote() for _ in range(4)]
    out = ray.get([producers[g % 4].commit.remote(g) for g in range(GROUPS)])
    keys_by_group = {g: out[g][0] for g in range(GROUPS)}
    expected = {g: _group_batch(1000 + g)[0] for g in range(GROUPS)}

    # write-backs under the same keys: advantages (0-dim) + prev_logprobs (jagged)
    all_keys = [k for g in range(GROUPS) for k in keys_by_group[g]]
    adv = torch.arange(len(all_keys), dtype=torch.float32) / 7.0
    tq.kv_batch_put(
        all_keys,
        "train",
        fields=TensorDict({"advantages": adv}, batch_size=[len(all_keys)]),
    )
    lens_all = [L for g in range(GROUPS) for L in _group_batch(1000 + g)[1]]
    prev = [
        torch.full((L,), float(i), dtype=torch.float32) for i, L in enumerate(lens_all)
    ]
    tq.kv_batch_put(
        all_keys,
        "train",
        fields=TensorDict(
            {"prev_logprobs": torch.nested.nested_tensor(prev, layout=torch.jagged)},
            batch_size=[len(all_keys)],
        ),
    )

    # DP-rank style reads: 4 interleaved shards, every field, across all blobs/units
    rng = random.Random(0)
    perm = all_keys[:]
    rng.shuffle(perm)
    fields = [
        "input_ids",
        "generation_logprobs",
        "token_mask",
        "input_lengths",
        "sample_mask",
        "rewards",
        "text",
        "advantages",
        "prev_logprobs",
    ]
    for shard in range(4):
        sub = perm[shard::4]
        got = tq.kv_batch_get(sub, "train", select_fields=fields)
        idx = {k: i for i, k in enumerate(all_keys)}
        for j, k in enumerate(sub):
            g, i = int(k[1 : k.index("_")]), int(k.split("gen")[1])
            exp = expected[g]
            for f in ("input_ids", "generation_logprobs", "token_mask"):
                assert torch.equal(_rows_of(got[f])[j], exp[f].unbind()[i]), (k, f)
            assert int(_dense(got["input_lengths"])[j]) == int(exp["input_lengths"][i])
            assert float(_dense(got["sample_mask"])[j]) == 1.0
            assert torch.equal(_dense(got["rewards"])[j], exp["rewards"][i])
            assert list(got["text"])[j] == list(exp["text"])[i]
            assert float(_dense(got["advantages"])[j]) == float(adv[idx[k]])
            assert torch.equal(_rows_of(got["prev_logprobs"])[j], prev[idx[k]])
            assert not torch.isnan(_rows_of(got["generation_logprobs"])[j]).any()

    tq.kv_clear(all_keys, "train")
    for p in producers:
        ray.kill(p)
