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
"""Zero-copy put + buffer pool: every put path, oversize buffers, concurrency.

One blob mixes: small values packed into a pool buffer, values in the zero-copy
window sent straight from the caller's tensor memory, values above the window
copied with torch's parallel copy, and a pickled object. Staging is 16 MiB so
the blobs and reads need oversize pool buffers. Then 8 threads put and get at
once. Everything must come back byte-exact, and the pool must reuse its oversize
buffers rather than register one per call.
"""

from __future__ import annotations

import threading

import pytest
import torch
from tensordict import NonTensorStack, TensorDict

from tests.unit.data_plane.nixl._helpers import dense, make_tq_conf

pytestmark = pytest.mark.nixl

ROWS = 4


def _batch(seed: int) -> TensorDict:
    g = torch.Generator().manual_seed(seed)
    return TensorDict(
        {
            "small": torch.randint(
                0, 1 << 30, (ROWS, 1024), generator=g
            ),  # 8 KiB/row: packed
            "zc": torch.randn(ROWS, 512 * 1024, generator=g),  # 2 MiB/row: zero-copy
            "big": torch.randint(
                0, 255, (ROWS, 48 << 20), generator=g, dtype=torch.uint8
            ),  # 48 MiB/row: parallel copy
            "routes": torch.randint(
                -1, 128, (ROWS, 4096, 56, 6), generator=g, dtype=torch.int16
            ),  # 2.6 MiB/row
            "text": NonTensorStack(*[f"s{seed}-{i}" * 50 for i in range(ROWS)]),
        },
        batch_size=[ROWS],
    )


FIELDS = ["small", "zc", "big", "routes", "text"]


@pytest.fixture(scope="module")
def zc_system(ray_cluster):
    import transfer_queue as tq

    import nemo_rl.data_plane.nixl.tq  # noqa: F401

    conf = make_tq_conf(
        num_units=2, slab_bytes=4 << 30, staging_bytes=16 << 20, namespace="nvdp-zc"
    )
    nc = conf.backend.NixlStore
    nc.zero_copy_min_bytes = 1 << 20
    nc.zero_copy_max_bytes = 32 << 20
    nc.parallel_copy_min_bytes = 16 << 20
    nc.client_staging_buffers = 2
    nc.client_pool_max_bytes = 2 << 30
    tq.init(conf)
    yield tq, conf
    tq.close()
    nemo_rl.data_plane.nixl.tq.shutdown(conf)


def _check(tq, keys, exp):
    got = tq.kv_batch_get(keys, "train", select_fields=FIELDS)
    for f in ("small", "zc", "big", "routes"):
        assert torch.equal(dense(got[f]), exp[f]), f
    assert list(got["text"]) == list(exp["text"])


def test_mixed_paths_byte_exact(zc_system):
    tq, _ = zc_system
    for seed in range(3):
        keys = [f"m{seed}-{i}" for i in range(ROWS)]
        exp = _batch(seed)
        tq.kv_batch_put(keys, "train", fields=exp)
        _check(tq, keys, exp)
        tq.kv_clear(keys, "train")


def test_caller_mutation_after_put_is_not_visible(zc_system):
    """Zero-copy must finish reading caller memory before put returns."""
    tq, _ = zc_system
    keys = [f"mut-{i}" for i in range(ROWS)]
    exp = _batch(99)
    src = exp.clone()
    tq.kv_batch_put(keys, "train", fields=src)
    src["zc"].zero_()
    src["big"].zero_()
    _check(tq, keys, exp)
    tq.kv_clear(keys, "train")


def test_concurrent_threads_byte_exact(zc_system):
    tq, _ = zc_system
    errors: list[BaseException] = []

    def worker(t: int) -> None:
        try:
            for r in range(3):
                keys = [f"t{t}-{r}-{i}" for i in range(ROWS)]
                exp = _batch(1000 + 10 * t + r)
                tq.kv_batch_put(keys, "train", fields=exp)
                _check(tq, keys, exp)
                tq.kv_clear(keys, "train")
        except BaseException as e:  # noqa: BLE001 - surfaced below
            errors.append(e)

    threads = [threading.Thread(target=worker, args=(t,)) for t in range(8)]
    for th in threads:
        th.start()
    for th in threads:
        th.join()
    assert not errors, errors[0]


def test_pool_reuses_oversize_buffers(zc_system):
    from transfer_queue import interface as tq_interface

    tq, _ = zc_system
    client = tq_interface._TQ_CLIENT.storage_manager.storage_client
    before = dict(client.pool.stats)
    for seed in range(4):
        keys = [f"p{seed}-{i}" for i in range(ROWS)]
        tq.kv_batch_put(keys, "train", fields=_batch(seed))
        tq.kv_batch_get(keys, "train", select_fields=FIELDS)
        tq.kv_clear(keys, "train")
    after = client.pool.stats
    assert after["oversize_hits"] > before["oversize_hits"]
    assert after["oversize_registers"] - before["oversize_registers"] <= 2
