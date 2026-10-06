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
"""FileStore on Lustre: same TQ API, no storage-unit actors, one file per blob."""

from __future__ import annotations

import os
import tempfile

import uuid
from pathlib import Path

import pytest
import ray
import torch
from tensordict import TensorDict

from tests.unit.data_plane.nixl._helpers import dense

pytestmark = pytest.mark.nixl

REPO = Path(
    os.environ.get("NRL_NIXL_TEST_DIR", tempfile.gettempdir())
)  # shared by all actors on the node


@pytest.fixture(scope="module")
def file_system(ray_cluster):
    import transfer_queue as tq

    import nemo_rl.data_plane.nixl.tq  # noqa: F401
    from tests.unit.data_plane.nixl._helpers import make_tq_conf

    root = REPO / ".nvdp_store" / f"test-{uuid.uuid4().hex[:8]}"
    conf = make_tq_conf(namespace="nvdp-file")
    conf.backend.NixlStore.store = {"kind": "file", "root": str(root)}
    tq.init(conf)
    yield conf, root
    tq.close()
    nemo_rl.data_plane.nixl.tq.shutdown(conf)
    for p in root.glob("*"):
        p.unlink()
    root.rmdir()


def _blobs(root: Path) -> list[Path]:
    return sorted(root.glob("*.blob"))


def test_no_units_started(file_system):
    conf, _ = file_system
    with pytest.raises(ValueError):
        ray.get_actor("NixlStorageUnit#0", namespace=conf.backend.NixlStore.namespace)


def test_put_get_clear_round_trip_on_lustre(file_system):
    import transfer_queue as tq

    conf, root = file_system
    keys = [f"f_g{i}" for i in range(4)]
    td = TensorDict(
        {
            "input_ids": torch.nested.nested_tensor(
                [torch.arange(n) for n in (3, 5, 2, 7)], layout=torch.jagged
            ),
            "rewards": torch.tensor([0.1, 0.2, 0.3, 0.4]),
            "text": ["a", "b", "c", "d"],
        },
        batch_size=[4],
    )
    tq.kv_batch_put(keys, "train", fields=td)
    files = _blobs(root)
    assert len(files) == 1 and files[0].stat().st_size > 0  # one blob = one file

    out = tq.kv_batch_get(keys, "train", select_fields=["input_ids", "rewards", "text"])
    got = out["input_ids"]
    rows = got.unbind() if got.is_nested else list(got)
    for n, r in zip((3, 5, 2, 7), rows):
        assert torch.equal(r, torch.arange(n))
    assert torch.allclose(dense(out["rewards"]), td["rewards"])
    assert list(out["text"]) == ["a", "b", "c", "d"]

    # write-back under the same keys → a second file; partial clear keeps both
    tq.kv_batch_put(
        keys, "train", fields=TensorDict({"adv": torch.ones(4)}, batch_size=[4])
    )
    assert len(_blobs(root)) == 2
    tq.kv_clear(keys[:2], "train")
    assert len(_blobs(root)) == 2  # blobs still referenced by the other two keys
    tq.kv_clear(keys[2:], "train")
    assert _blobs(root) == []  # last reference deletes the file


def test_cross_process_read(file_system):
    import transfer_queue as tq

    @ray.remote(num_cpus=1)
    class Reader:
        def __init__(self):
            import nemo_rl.data_plane.nixl.tq  # noqa: F401
            import transfer_queue as tq

            tq.init()
            self.tq = tq

        def get(self, keys):
            out = self.tq.kv_batch_get(keys, "train", select_fields=["v"])["v"]
            return torch.stack(list(out.unbind())) if out.is_nested else out

    keys = [f"x_g{i}" for i in range(3)]
    v = torch.arange(9, dtype=torch.float32).view(3, 3)
    tq.kv_batch_put(keys, "train", fields=TensorDict({"v": v}, batch_size=[3]))
    r = Reader.remote()
    assert torch.equal(ray.get(r.get.remote(keys)), v)
    ray.kill(r)
    tq.kv_clear(keys, "train")


def test_missing_file_is_lost_keys(file_system):
    import transfer_queue as tq

    from nemo_rl.data_plane.nixl.errors import DataPlaneLostKeys

    conf, root = file_system
    keys = ["m_g0"]
    tq.kv_batch_put(
        keys, "train", fields=TensorDict({"v": torch.ones(1, 2)}, batch_size=[1])
    )
    for f in _blobs(root):
        os.remove(f)
    # evict any cached fd in this process so the read must reopen the file
    tq.get_client().storage_manager.storage_client.store.close()
    with pytest.raises(DataPlaneLostKeys):
        tq.kv_batch_get(keys, "train", select_fields=["v"])
    tq.kv_clear(keys, "train")
