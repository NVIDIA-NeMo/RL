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

import pytest

from nemo_rl.models.generation.dynamo.role_plan import DisaggRolePlan


def _plan() -> DisaggRolePlan:
    return DisaggRolePlan.from_role_counts(
        decode_workers=2,
        decode_world_size=2,
        prefill_workers=1,
        prefill_world_size=4,
        decode_cfg={"tensor_model_parallel_size": 2},
        prefill_cfg={"tensor_model_parallel_size": 4},
    )


def test_engines_are_ordered_decode_first_with_stable_names() -> None:
    plan = _plan()
    assert [spec.name for spec in plan.engines] == [
        "decode-0",
        "decode-1",
        "prefill-0",
    ]
    assert [spec.component for spec in plan.engines] == [
        "backend",
        "backend",
        "prefill",
    ]
    assert plan.engines[2].engine_cfg == {"tensor_model_parallel_size": 4}


def test_rank_offsets_are_cumulative_over_unequal_engines() -> None:
    plan = _plan()
    assert plan.rank_offsets(train_world_size=8) == [8, 10, 12]
    assert plan.inference_world_size == 8
    assert plan.max_engine_world_size == 4


def test_expected_components_count_engines_per_dynamo_component() -> None:
    assert _plan().expected_components == {"backend": 2, "prefill": 1}


def test_gpu_budget_must_match_exactly() -> None:
    plan = _plan()
    plan.assert_fits(8)
    with pytest.raises(ValueError, match="allocated=7, expected=8"):
        plan.assert_fits(7)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"decode_workers": 0},
        {"prefill_workers": 0},
        {"decode_world_size": 0},
    ],
)
def test_role_counts_and_sizes_must_be_positive(kwargs) -> None:
    base = dict(
        decode_workers=1, decode_world_size=1, prefill_workers=1, prefill_world_size=1
    )
    base.update(kwargs)
    with pytest.raises(ValueError, match="must be >= 1"):
        DisaggRolePlan.from_role_counts(**base)
