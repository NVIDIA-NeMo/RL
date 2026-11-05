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

from unittest.mock import MagicMock

import pytest

from nemo_rl.models.generation.megatron import MegatronGeneration


def _placement_policy_config(
    *,
    tp: int = 1,
    pp: int = 1,
    cp: int = 1,
    ep: int = 1,
    etp: int = 1,
    colocated: bool = False,
    mcore_overrides: dict | None = None,
) -> dict:
    """Minimal PolicyConfig slice consumed by init_cluster_placement_groups."""
    return {
        "megatron_cfg": {
            "tensor_model_parallel_size": tp,
            "pipeline_model_parallel_size": pp,
            "context_parallel_size": cp,
            "expert_model_parallel_size": ep,
            "expert_tensor_parallel_size": etp,
        },
        "generation": {
            "colocated": {"enabled": colocated},
            "mcore_generation_config": mcore_overrides or {},
        },
    }


@pytest.mark.parametrize(
    "config_kwargs,expected_strategy,expected_unified",
    [
        # cross-node span via TP alone -> one unified PG
        (dict(tp=8), "PACK", True),
        # PP is excluded from the NVLink span: TP*CP=4 fits a node even
        # though the full TP*PP*CP instance would not
        (dict(tp=2, pp=2, cp=2), "PACK", False),
        # node-local span at the == boundary -> per-node PGs
        (dict(tp=4), "PACK", False),
        # cross-node MoE expert group (ETP*EP > TP*CP) -> one unified PG:
        # the NVLS dispatcher needs the ep_group fully NVLink-connected
        (dict(tp=2, ep=8), "PACK", True),
        # non-colocated generation parallelism overrides the training values
        # (mirrors MegatronGeneration's megatron_cfg merge)
        (
            dict(tp=8, mcore_overrides={"tensor_model_parallel_size": 2}),
            "PACK",
            False,
        ),
        # colocated reuses the training layout (overrides do not apply):
        # no PACK strategy, span from the training config incl. its EP
        (dict(tp=2, ep=8, colocated=True), None, True),
    ],
)
def test_megatron_init_cluster_placement_groups(
    config_kwargs, expected_strategy, expected_unified
):
    """The NVLink-domain span is max(TP*CP, ETP*EP) of the operative config."""
    cluster = MagicMock(num_gpus_per_node=4)

    MegatronGeneration.init_cluster_placement_groups(
        cluster, _placement_policy_config(**config_kwargs)
    )

    cluster._init_placement_groups.assert_called_once_with(
        strategy=expected_strategy,
        use_unified_pg=expected_unified,
    )
