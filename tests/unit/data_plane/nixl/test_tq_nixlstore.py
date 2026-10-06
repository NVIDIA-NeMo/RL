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
"""Use-case tests: TransferQueue with the NixlStore backend, single node.

Each test mirrors a call shape NeMo-RL makes through its DataPlaneClient:
first write of a rollout batch (jagged + dense), worker read with
select_fields, leader write-back of a new column under the same keys,
step-end clear, capacity spill, lost keys.
"""

from __future__ import annotations

import time

import pytest
import ray
import torch
from tensordict import NonTensorStack, TensorDict

pytestmark = pytest.mark.nixl


from tests.unit.data_plane.nixl._helpers import dense as _dense


def _keys(prefix: str, n: int) -> list[str]:
    return [f"{prefix}_g{i}" for i in range(n)]


def _unit_stats(conf):
    ns = conf.backend.NixlStore.namespace
    n = conf.backend.NixlStore.num_storage_units
    return [
        ray.get(ray.get_actor(f"NixlStorageUnit#{i}", namespace=ns).stats.remote())
        for i in range(n)
    ]


class TestRolloutShapes:
    def test_dense_put_get(self, tq_system):
        import transfer_queue as tq

        keys = _keys("dense", 4)
        td = TensorDict(
            {
                "input_ids": torch.arange(32).view(4, 8),
                "rewards": torch.tensor([0.1, 0.2, 0.3, 0.4]),
            },
            batch_size=[4],
        )
        meta = tq.kv_batch_put(keys, "train", fields=td)
        assert set(meta.fields) >= {"input_ids", "rewards"}
        out = tq.kv_batch_get(keys, "train", select_fields=["input_ids", "rewards"])
        assert torch.equal(_dense(out["input_ids"]), td["input_ids"])
        assert torch.allclose(_dense(out["rewards"]), td["rewards"])
        tq.kv_clear(keys, "train")

    def test_jagged_first_write_then_select(self, tq_system):
        import transfer_queue as tq

        keys = _keys("jag", 3)
        rows = [torch.arange(5), torch.arange(2), torch.arange(9)]
        lp = [torch.randn(5), torch.randn(2), torch.randn(9)]
        td = TensorDict(
            {
                "input_ids": torch.nested.nested_tensor(rows, layout=torch.jagged),
                "generation_logprobs": torch.nested.nested_tensor(
                    lp, layout=torch.jagged
                ),
                "input_lengths": torch.tensor([5, 2, 9]),
            },
            batch_size=[3],
        )
        tq.kv_batch_put(keys, "train", fields=td)
        out = tq.kv_batch_get(
            keys, "train", select_fields=["input_ids", "input_lengths"]
        )
        assert set(out.keys()) == {"input_ids", "input_lengths"}
        got = out["input_ids"]
        got_rows = got.unbind() if got.is_nested else list(got)
        for r, g in zip(rows, got_rows):
            assert torch.equal(r, g)
        assert torch.equal(_dense(out["input_lengths"]), td["input_lengths"])
        tq.kv_clear(keys, "train")

    def test_writeback_new_column_same_keys(self, tq_system):
        import transfer_queue as tq

        keys = _keys("wb", 4)
        tq.kv_batch_put(
            keys,
            "train",
            fields=TensorDict(
                {"input_ids": torch.ones(4, 3, dtype=torch.long)}, batch_size=[4]
            ),
        )
        adv = torch.tensor([1.0, -1.0, 0.5, 0.0])
        meta = tq.kv_batch_put(
            keys, "train", fields=TensorDict({"advantages": adv}, batch_size=[4])
        )
        assert "advantages" in meta.fields and "input_ids" in meta.fields
        out = tq.kv_batch_get(keys, "train", select_fields=["input_ids", "advantages"])
        assert torch.equal(_dense(out["input_ids"]), torch.ones(4, 3, dtype=torch.long))
        assert torch.allclose(_dense(out["advantages"]), adv)
        tq.kv_clear(keys, "train")

    def test_non_tensor_stack_and_tags(self, tq_system):
        import transfer_queue as tq

        keys = _keys("nts", 2)
        td = TensorDict(
            {"text": NonTensorStack("hello", "world"), "x": torch.tensor([1, 2])},
            batch_size=[2],
        )
        tq.kv_batch_put(keys, "train", fields=td, tags=[{"uid": 1}, {"uid": 2}])
        out = tq.kv_batch_get(keys, "train", select_fields=["text", "x"])
        assert list(out["text"]) == ["hello", "world"]
        assert torch.equal(_dense(out["x"]), td["x"])
        listing = tq.kv_list("train")
        if "train" in listing and isinstance(listing["train"], dict):
            listing = listing["train"]  # pinned SHA nests by partition
        assert set(keys) <= set(listing)
        tq.kv_clear(keys, "train")

    def test_partial_key_read_subset(self, tq_system):
        import transfer_queue as tq

        keys = _keys("sub", 6)
        td = TensorDict({"v": torch.arange(6, dtype=torch.float32)}, batch_size=[6])
        tq.kv_batch_put(keys, "train", fields=td)
        out = tq.kv_batch_get([keys[1], keys[4]], "train", select_fields=["v"])
        assert torch.equal(_dense(out["v"]), torch.tensor([1.0, 4.0]))
        tq.kv_clear(keys, "train")


class TestLifecycle:
    def test_clear_releases_blobs(self, tq_system):
        import transfer_queue as tq

        keys = _keys("clr", 8)
        tq.kv_batch_put(
            keys, "train", fields=TensorDict({"a": torch.randn(8, 16)}, batch_size=[8])
        )
        before = sum(s["blobs"] for s in _unit_stats(tq_system))
        assert before >= 1
        tq.kv_clear(keys, "train")
        listing = tq.kv_list("train")
        listing = (
            listing.get("train", listing) if isinstance(listing, dict) else listing
        )
        assert not any(k in listing for k in keys)
        after = sum(s["blobs"] for s in _unit_stats(tq_system))
        assert after == before - 1
        # region is reusable once the read pin window has passed
        time.sleep(float(tq_system.backend.NixlStore.read_pin_s) + 0.2)
        stats = _unit_stats(tq_system)
        assert sum(s["quarantined_bytes"] for s in stats) == 0

    def test_many_small_puts(self, tq_system):
        import transfer_queue as tq

        all_keys = []
        for i in range(50):
            k = [f"small{i}_g0"]
            tq.kv_batch_put(
                k,
                "train",
                fields=TensorDict({"t": torch.tensor([[i]])}, batch_size=[1]),
            )
            all_keys += k
        out = tq.kv_batch_get(all_keys, "train", select_fields=["t"])
        assert torch.equal(_dense(out["t"]).flatten(), torch.arange(50))
        tq.kv_clear(all_keys, "train")

    def test_spill_to_other_unit_when_full(self, tq_system):
        import transfer_queue as tq

        slab = int(tq_system.backend.NixlStore.unit_slab_bytes)
        # ~40% of one slab per put: the third put cannot fit on the first unit
        n = slab * 2 // 5 // 4
        keys = []
        for i in range(3):
            k = [f"big{i}_g0"]
            tq.kv_batch_put(
                k,
                "train",
                fields=TensorDict(
                    {"blob": torch.zeros(1, n, dtype=torch.int32)}, batch_size=[1]
                ),
            )
            keys += k
        stats = _unit_stats(tq_system)
        assert all(s["blobs"] >= 1 for s in stats), stats
        out = tq.kv_batch_get(keys, "train", select_fields=["blob"])
        assert _dense(out["blob"]).shape == (3, n)
        tq.kv_clear(keys, "train")

    def test_larger_than_staging_uses_slow_path(self, tq_system):
        import transfer_queue as tq

        staging = int(tq_system.backend.NixlStore.client_staging_bytes)
        n = staging // 4 + 1024  # int32 → just over the staging buffer
        keys = ["huge_g0"]
        t = torch.arange(n, dtype=torch.int32).view(1, n)
        tq.kv_batch_put(keys, "train", fields=TensorDict({"blob": t}, batch_size=[1]))
        out = tq.kv_batch_get(keys, "train", select_fields=["blob"])
        assert torch.equal(_dense(out["blob"]), t)
        tq.kv_clear(keys, "train")


class TestWorkersInRemoteActors:
    def test_put_in_one_actor_get_in_another(self, tq_system):
        """Producer and consumer are different processes (like rollout actor → trainer rank)."""
        import transfer_queue as tq

        @ray.remote(num_cpus=1)
        class Worker:
            def __init__(self):
                import nemo_rl.data_plane.nixl.tq  # noqa: F401
                import transfer_queue as tq

                tq.init()
                self.tq = tq

            def put(self, keys, n):
                td = TensorDict(
                    {"ids": torch.arange(len(keys) * n).view(len(keys), n)},
                    batch_size=[len(keys)],
                )
                self.tq.kv_batch_put(keys, "train", fields=td)
                return True

            def get(self, keys):
                return _dense(
                    self.tq.kv_batch_get(keys, "train", select_fields=["ids"])["ids"]
                )

        producer, consumer = Worker.remote(), Worker.remote()
        keys = _keys("xproc", 4)
        assert ray.get(producer.put.remote(keys, 5))
        got = ray.get(consumer.get.remote(keys))
        assert torch.equal(got, torch.arange(20).view(4, 5))
        # driver can read too
        drv = _dense(tq.kv_batch_get(keys, "train", select_fields=["ids"])["ids"])
        assert torch.equal(drv, got)
        tq.kv_clear(keys, "train")
        ray.kill(producer)
        ray.kill(consumer)


class TestReadPinGuard:
    def test_read_that_outlives_pin_is_rejected(self, tq_system):
        """A READ completing after its pin TTL may have seen a reused region (§8)."""
        import transfer_queue as tq

        from nemo_rl.data_plane.nixl.errors import StaleRead

        keys = _keys("pin", 2)
        tq.kv_batch_put(
            keys, "train", fields=TensorDict({"a": torch.ones(2, 4)}, batch_size=[2])
        )
        client = tq.get_client().storage_manager.storage_client
        saved = client.read_pin_s
        client.read_pin_s = 0.0  # every read now "outlives" its pin
        try:
            with pytest.raises(StaleRead):
                tq.kv_batch_get(keys, "train", select_fields=["a"])
        finally:
            client.read_pin_s = saved
        out = tq.kv_batch_get(keys, "train", select_fields=["a"])
        assert torch.equal(_dense(out["a"]), torch.ones(2, 4))
        tq.kv_clear(keys, "train")
