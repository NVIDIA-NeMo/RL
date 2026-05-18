# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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
"""2-node Slurm smoke — verifies controller-actor placement and ZMQ.

Driver registers a partition, a producer Ray actor on a different node
puts data, the driver fetches and validates. Run via ``RL/ray.sub`` over
2 nodes (mirrors ``rl-arena/launch/run_arena.sh``).

Skipped automatically when:
  * ``transfer_queue`` is not installed, or
  * the test is invoked on a single-node Ray cluster.
"""

from __future__ import annotations

import pytest
import torch
from tensordict import TensorDict

transfer_queue = pytest.importorskip("transfer_queue")  # noqa: F841


def _ray_node_count() -> int:
    import ray

    if not ray.is_initialized():
        return 0
    return len([n for n in ray.nodes() if n.get("Alive", False)])


@pytest.mark.skipif(_ray_node_count() < 2, reason="requires a multi-node Ray cluster")
def test_multinode_round_trip() -> None:
    import ray

    from nemo_rl.data_plane import build_data_plane_client

    driver = build_data_plane_client(
        {
            "enabled": True,
            "impl": "transfer_queue",
            "backend": "simple",
            "storage_capacity": 1024,
            "num_storage_units": 2,
        }
    )

    try:
        driver.register_partition(
            partition_id="mn",
            fields=["x"],
            num_samples=4,
            consumer_tasks=["read"],
        )

        @ray.remote(num_cpus=1)
        def produce(keys: list[str]) -> None:
            from nemo_rl.data_plane import build_data_plane_client

            actor_client = build_data_plane_client(
                {"enabled": True, "impl": "transfer_queue", "backend": "simple"}
            )
            try:
                actor_client.kv_batch_put(
                    sample_ids=keys,
                    partition_id="mn",
                    fields=TensorDict(
                        {"x": torch.arange(len(keys))}, batch_size=[len(keys)]
                    ),
                )
            finally:
                actor_client.close()

        ray.get(produce.remote(["a", "b", "c", "d"]))

        meta = driver.claim_meta(
            partition_id="mn",
            task_name="read",
            required_fields=["x"],
            batch_size=4,
            timeout_s=60.0,
        )
        assert meta.size == 4
        data = driver.get_data(meta)
        assert int(data["x"].sum()) == 0 + 1 + 2 + 3
    finally:
        driver.kv_clear(sample_ids=None, partition_id="mn")
        driver.close()
