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
"""1 node / 4 GPU use case: GPU workers produce, others consume, driver reads back.

Mirrors NeMo-RL: each GPU rank writes its shard of a rollout (CUDA tensors)
under uid-derived keys, a different rank reads a length-balanced subset,
and the driver writes back a per-row column under the same keys.
"""

from __future__ import annotations

import pytest
import ray
import torch
from tensordict import TensorDict

pytestmark = [pytest.mark.nixl, pytest.mark.gpu]


from tests.unit.data_plane._nixl_helpers import dense as _dense


@ray.remote(num_cpus=1, num_gpus=1)
class GpuRank:
    def __init__(self, rank: int):
        import nemo_rl.data_plane.adapters.tq_nixl  # noqa: F401
        import transfer_queue as tq

        tq.init()
        self.tq = tq
        self.rank = rank
        self.dev = torch.device("cuda:0")  # Ray scoped CUDA_VISIBLE_DEVICES

    def produce(self, keys: list[str], seqlen: int):
        g = torch.Generator(device="cpu").manual_seed(self.rank)
        ids = torch.randint(0, 32000, (len(keys), seqlen), generator=g).to(self.dev)
        lp = torch.randn(len(keys), seqlen, generator=g).to(self.dev)
        td = TensorDict(
            {"input_ids": ids, "generation_logprobs": lp}, batch_size=[len(keys)]
        )
        self.tq.kv_batch_put(keys, "train", fields=td)
        return ids.cpu(), lp.cpu()

    def consume(self, keys: list[str], fields: list[str]):
        out = self.tq.kv_batch_get(keys, "train", select_fields=fields)
        # move onto this rank's GPU as a trainer would, then back for the assertion
        return {k: _dense(v).to(self.dev).cpu() for k, v in out.items()}

    def writeback(self, keys: list[str], values: torch.Tensor):
        self.tq.kv_batch_put(
            keys,
            "train",
            fields=TensorDict(
                {"prev_logprobs": values.to(self.dev)}, batch_size=[len(keys)]
            ),
        )
        return True


def test_four_gpu_ranks_round_robin(tq_system):
    import transfer_queue as tq

    n = torch.cuda.device_count()
    assert n >= 2, "test needs at least 2 GPUs"
    ranks = [GpuRank.remote(r) for r in range(n)]
    per_rank = 3
    keys = {r: [f"r{r}_g{i}" for i in range(per_rank)] for r in range(n)}
    produced = ray.get([ranks[r].produce.remote(keys[r], 16) for r in range(n)])

    # every rank reads the shard written by the next rank
    consumed = ray.get(
        [
            ranks[r].consume.remote(
                keys[(r + 1) % n], ["input_ids", "generation_logprobs"]
            )
            for r in range(n)
        ]
    )
    for r in range(n):
        ids, lp = produced[(r + 1) % n]
        assert torch.equal(consumed[r]["input_ids"], ids)
        assert torch.allclose(consumed[r]["generation_logprobs"], lp)

    # leader write-back of a new column, then driver reads both
    all_keys = [k for r in range(n) for k in keys[r]]
    vals = torch.arange(len(all_keys), dtype=torch.float32)
    assert ray.get(ranks[0].writeback.remote(all_keys, vals))
    out = tq.kv_batch_get(
        all_keys, "train", select_fields=["prev_logprobs", "input_ids"]
    )
    assert torch.allclose(_dense(out["prev_logprobs"]), vals)
    assert _dense(out["input_ids"]).shape == (len(all_keys), 16)

    tq.kv_clear(all_keys, "train")
    for a in ranks:
        ray.kill(a)
