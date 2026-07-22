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

from unittest.mock import MagicMock, patch

import pytest
import ray

from nemo_rl.models.generation.trtllm.trtllm_generation import TrtllmGeneration


def _generation_config(tp_size: int, *, colocated: bool = False) -> dict:
    return {
        "backend": "trtllm",
        "colocated": {"enabled": colocated},
        "trtllm_cfg": {
            "tensor_parallel_size": tp_size,
            "pipeline_parallel_size": 1,
        },
    }


def test_init_cluster_placement_groups_uses_unified_pg_for_cross_node_tp():
    cluster = MagicMock(num_gpus_per_node=4)
    config = _generation_config(tp_size=8)

    TrtllmGeneration.init_cluster_placement_groups(cluster, config)

    cluster._init_placement_groups.assert_called_once_with(
        strategy="PACK",
        use_unified_pg=True,
    )


def test_init_cluster_placement_groups_uses_per_node_pgs_for_node_local_tp():
    cluster = MagicMock(num_gpus_per_node=4)
    config = _generation_config(tp_size=4)

    TrtllmGeneration.init_cluster_placement_groups(cluster, config)

    cluster._init_placement_groups.assert_called_once_with(
        strategy="PACK",
        use_unified_pg=False,
    )


def test_init_cluster_placement_groups_rejects_colocated_cross_node_tp():
    cluster = MagicMock(num_gpus_per_node=4)
    config = _generation_config(tp_size=8, colocated=True)

    with pytest.raises(AssertionError, match="only supported for non-colocated"):
        TrtllmGeneration.init_cluster_placement_groups(cluster, config)

    cluster._init_placement_groups.assert_not_called()


def test_cross_node_tp_replicas_use_unified_placement_group():
    generation = TrtllmGeneration.__new__(TrtllmGeneration)
    generation.model_parallel_size = 8
    generation.worker_group = MagicMock()

    unified_pg = MagicMock()
    bundle_to_node = {i: f"node-{i // 4}" for i in range(16)}
    cluster = MagicMock()
    cluster.get_placement_groups.return_value = [unified_pg]
    cluster._sorted_bundle_indices = list(range(16))

    with patch.object(
        ray.util,
        "placement_group_table",
        return_value={"bundles_to_node_id": bundle_to_node},
    ):
        result = generation._get_tied_worker_bundle_indices(cluster)

    assert result == [
        (0, list(range(8))),
        (0, list(range(8, 16))),
    ]
