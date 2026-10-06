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
"""Unit death → DataPlaneLostKeys for the keys it held (Q14)."""

from __future__ import annotations

import time

import pytest
import ray
import torch
from tensordict import TensorDict

pytestmark = pytest.mark.nixl


@pytest.fixture(scope="module")
def one_unit_system(ray_cluster):
    import transfer_queue as tq

    import nemo_rl.data_plane.nixl.tq  # noqa: F401
    from tests.unit.data_plane.nixl._helpers import make_tq_conf

    conf = make_tq_conf(num_units=1, namespace="nvdp-failure")
    tq.init(conf)
    yield conf
    tq.close()
    nemo_rl.data_plane.nixl.tq.shutdown(conf)


def test_lost_keys_when_unit_dies(one_unit_system):
    import transfer_queue as tq

    from nemo_rl.data_plane.nixl.errors import DataPlaneLostKeys

    keys = ["lost_g0", "lost_g1"]
    tq.kv_batch_put(
        keys, "p", fields=TensorDict({"a": torch.ones(2, 4)}, batch_size=[2])
    )
    ns = one_unit_system.backend.NixlStore.namespace
    ray.kill(ray.get_actor("NixlStorageUnit#0", namespace=ns), no_restart=True)
    time.sleep(1.0)
    with pytest.raises(DataPlaneLostKeys) as ei:
        tq.kv_batch_get(keys, "p", select_fields=["a"])
    err = ei.value
    assert err.fields() == ["a"]
    assert len(err.global_indexes()) == 2
    # storage-level keys map back to the caller's sample ids through TQ
    mapped = tq.get_client().kv_retrieve_keys(err.global_indexes(), "p")
    if not isinstance(mapped, list):  # older TQ returns a BatchMeta-like object
        mapped = list(getattr(mapped, "keys", mapped))
    assert sorted(mapped) == keys
    # clearing lost keys must not raise
    tq.kv_clear(keys, "p")
