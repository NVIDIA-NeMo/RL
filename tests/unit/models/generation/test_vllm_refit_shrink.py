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

from unittest.mock import MagicMock, call

import pytest

from nemo_rl.models.generation.vllm.vllm_generation import VllmGeneration


def _make_generation(async_engine: bool = True) -> VllmGeneration:
    generation = VllmGeneration.__new__(VllmGeneration)
    generation.cfg = {"vllm_cfg": {"async_engine": async_engine}}
    generation.model_parallel_size = 2
    generation._refit_train_world_size = 16
    generation._active_refit_instance_ids = list(range(8))
    generation.worker_group = MagicMock()
    generation.worker_group.get_dp_leader_worker_idx.side_effect = lambda value: value
    generation.worker_group.run_single_worker_single_data.side_effect = (
        lambda method_name, worker_idx, **kwargs: (method_name, worker_idx, kwargs)
    )
    return generation


def test_adjust_refit_comm_group_excludes_instance_tp_ranks() -> None:
    generation = _make_generation()

    exclude_ranks, new_world_size, futures = generation.adjust_refit_comm_group(3)

    assert exclude_ranks == [22, 23]
    assert new_world_size == 30
    assert generation._active_refit_instance_ids == [0, 1, 2, 4, 5, 6, 7]
    assert [future[1] for future in futures] == [0, 1, 2, 4, 5, 6, 7]
    assert generation.worker_group.run_single_worker_single_data.call_args_list == [
        call(
            "adjust_refit_comm_group_async",
            worker_idx=instance_id,
            exclude_ranks=[22, 23],
        )
        for instance_id in [0, 1, 2, 4, 5, 6, 7]
    ]


def test_adjust_refit_comm_group_uses_compacted_ranks_after_prior_shrink() -> None:
    generation = _make_generation()
    generation.adjust_refit_comm_group(3)
    generation.worker_group.reset_mock()
    generation.worker_group.get_dp_leader_worker_idx.side_effect = lambda value: value

    exclude_ranks, new_world_size, _ = generation.adjust_refit_comm_group(5)

    assert exclude_ranks == [24, 25]
    assert new_world_size == 28
    assert generation._active_refit_instance_ids == [0, 1, 2, 4, 6, 7]


def test_collective_weight_update_skips_faulty_instance() -> None:
    generation = _make_generation(async_engine=False)
    generation._active_refit_instance_ids = [0, 1, 2, 4, 5, 6, 7]

    futures = generation.update_weights_from_collective()

    assert [future[1] for future in futures] == [0, 1, 2, 4, 5, 6, 7]
    generation.worker_group.run_single_worker_single_data.assert_has_calls(
        [
            call("update_weights_from_collective", worker_idx=instance_id)
            for instance_id in [0, 1, 2, 4, 5, 6, 7]
        ]
    )


def test_adjust_refit_comm_group_rejects_inactive_instance() -> None:
    generation = _make_generation()
    generation._active_refit_instance_ids.remove(3)

    with pytest.raises(ValueError, match="not in the active refit set"):
        generation.adjust_refit_comm_group(3)
