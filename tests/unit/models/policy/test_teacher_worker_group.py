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

from unittest.mock import MagicMock

import torch

from nemo_rl.distributed.batched_data_dict import BatchedDataDict


def test_teacher_resource_config_defaults():
    from nemo_rl.algorithms.opd import TeacherResourceConfig

    res = TeacherResourceConfig(tensor_model_parallel_size=4)
    assert res.tensor_model_parallel_size == 4
    assert res.pipeline_model_parallel_size == 1
    assert res.gpus_per_node == 8
    assert res.precision == "bf16"


def test_create_teacher_configs_homogeneous():
    from nemo_rl.models.policy.teacher_worker_group import (
        create_teacher_configs_from_opd_config,
    )

    configs = create_teacher_configs_from_opd_config(
        {
            "teacher_model_by_agent_name": {"math": "/ckpt/math", "code": "/ckpt/code"},
            "non_colocated_teachers": {
                "default_teacher_cfg": {"tensor_model_parallel_size": 4}
            },
        }
    )
    assert len(configs) == 2
    assert all(c.tensor_model_parallel_size == 4 for c in configs)


def test_create_teacher_configs_heterogeneous_override():
    from nemo_rl.models.policy.teacher_worker_group import (
        create_teacher_configs_from_opd_config,
    )

    configs = create_teacher_configs_from_opd_config(
        {
            "teacher_model_by_agent_name": {"math": "/ckpt/math", "code": "/ckpt/code"},
            "non_colocated_teachers": {
                "default_teacher_cfg": {"tensor_model_parallel_size": 4},
                "teacher_overrides": {"code": {"tensor_model_parallel_size": 8}},
            },
        }
    )
    code_cfg = [c for c in configs if c.alias == "code"][0]
    assert code_cfg.tensor_model_parallel_size == 8


def test_create_teacher_configs_resolves_parallelism_from_megatron_overrides():
    from nemo_rl.models.policy.teacher_worker_group import (
        create_teacher_configs_from_opd_config,
    )

    config = create_teacher_configs_from_opd_config(
        {
            "teacher_model_by_agent_name": {"large": "/ckpt/large"},
            "non_colocated_teachers": {
                "default_teacher_cfg": {
                    "tensor_model_parallel_size": 1,
                    "pipeline_model_parallel_size": 1,
                    "context_parallel_size": 1,
                    "expert_model_parallel_size": 1,
                },
                "teacher_overrides": {
                    "large": {
                        "megatron_cfg_overrides": {
                            "tensor_model_parallel_size": 2,
                            "pipeline_model_parallel_size": 3,
                            "context_parallel_size": 4,
                            "expert_model_parallel_size": 5,
                        }
                    }
                },
            },
        }
    )[0]

    assert (
        config.tensor_model_parallel_size,
        config.pipeline_model_parallel_size,
        config.context_parallel_size,
        config.expert_model_parallel_size,
    ) == (2, 3, 4, 5)


def test_create_teacher_configs_deduplicates():
    from nemo_rl.models.policy.teacher_worker_group import (
        create_teacher_configs_from_opd_config,
    )

    configs = create_teacher_configs_from_opd_config(
        {
            "teacher_model_by_agent_name": {
                "math": "/shared",
                "code": "/shared",
                "rlhf": "/rlhf",
            },
            "deduplicate_shared_teacher_checkpoints": True,
            "non_colocated_teachers": {
                "default_teacher_cfg": {"tensor_model_parallel_size": 2}
            },
        }
    )
    assert len(configs) == 2


def test_get_logprobs_on_support_routes_shards_and_preserves_worker_order():
    from nemo_rl.models.policy.teacher_worker_group import TeacherWorkerGroup

    teacher = object.__new__(TeacherWorkerGroup)
    teacher.use_sequence_packing = False
    teacher.cfg = {"megatron_cfg": {"context_parallel_size": 1}}
    teacher._micro_batch_size = 3
    teacher.sharding_annotations = MagicMock()
    teacher.sharding_annotations.get_axis_size.return_value = 2
    teacher.worker_group = MagicMock()
    teacher.worker_group.run_all_workers_sharded_data.return_value = ["f0", "f1"]
    first = torch.tensor([[[1.0, 2.0]], [[3.0, 4.0]]])
    second = torch.tensor([[[5.0, 6.0]], [[7.0, 8.0]]])
    teacher.worker_group.get_all_worker_results.return_value = [
        {"support_logprobs": first},
        {"support_logprobs": second},
    ]
    data = BatchedDataDict(
        {
            "input_ids": torch.arange(4).unsqueeze(1),
            "topk_indices": torch.zeros(4, 1, 2, dtype=torch.long),
        }
    )

    result = teacher.get_logprobs_on_support(data)

    torch.testing.assert_close(
        result["support_logprobs"], torch.cat((first, second), dim=0)
    )
    teacher.worker_group.run_all_workers_sharded_data.assert_called_once()
    call = teacher.worker_group.run_all_workers_sharded_data.call_args
    assert call.args == ("get_logprobs_on_support",)
    assert len(call.kwargs["data"]) == 2
    assert call.kwargs["common_kwargs"] == {"micro_batch_size": 3}


def test_get_topk_logprobs_routes_shards_and_preserves_worker_order():
    from nemo_rl.models.policy.teacher_worker_group import TeacherWorkerGroup

    teacher = object.__new__(TeacherWorkerGroup)
    teacher.use_sequence_packing = False
    teacher.cfg = {"megatron_cfg": {"context_parallel_size": 1}}
    teacher._micro_batch_size = 3
    teacher.sharding_annotations = MagicMock()
    teacher.sharding_annotations.get_axis_size.return_value = 2
    teacher.worker_group = MagicMock()
    teacher.worker_group.run_all_workers_sharded_data.return_value = ["f0", "f1"]
    first_indices = torch.tensor([[[1, 2]], [[3, 4]]])
    second_indices = torch.tensor([[[5, 6]], [[7, 8]]])
    first_logprobs = torch.tensor([[[-1.0, -2.0]], [[-3.0, -4.0]]])
    second_logprobs = torch.tensor([[[-5.0, -6.0]], [[-7.0, -8.0]]])
    first_targets = torch.tensor([[0.0], [-1.0]])
    second_targets = torch.tensor([[-2.0], [-3.0]])
    teacher.worker_group.get_all_worker_results.return_value = [
        {
            "logprobs": first_targets,
            "topk_indices": first_indices,
            "topk_logprobs": first_logprobs,
        },
        {
            "logprobs": second_targets,
            "topk_indices": second_indices,
            "topk_logprobs": second_logprobs,
        },
    ]
    data = BatchedDataDict({"input_ids": torch.arange(4).unsqueeze(1)})

    result = teacher.get_topk_logprobs(data, k=2)

    torch.testing.assert_close(
        result["topk_indices"], torch.cat((first_indices, second_indices), dim=0)
    )
    torch.testing.assert_close(
        result["topk_logprobs"], torch.cat((first_logprobs, second_logprobs), dim=0)
    )
    torch.testing.assert_close(
        result["reference_logprobs"],
        torch.cat((first_targets, second_targets), dim=0),
    )
    call = teacher.worker_group.run_all_workers_sharded_data.call_args
    assert call.args == ("get_topk_logits",)
    assert len(call.kwargs["data"]) == 2
    assert call.kwargs["common_kwargs"] == {
        "k": 2,
        "micro_batch_size": 3,
        "return_logprobs": True,
    }


def test_get_topk_logprobs_packs_and_restores_original_order():
    from nemo_rl.models.policy.teacher_worker_group import TeacherWorkerGroup

    teacher = object.__new__(TeacherWorkerGroup)
    teacher.use_sequence_packing = True
    teacher.sequence_packing_args = {
        "algorithm": "first_fit_decreasing",
        "input_key": "input_ids",
        "input_lengths_key": "input_lengths",
        "sequence_length_pad_multiple": 4,
    }
    teacher.cfg = {
        "megatron_cfg": {"context_parallel_size": 2},
        "sequence_packing": {"logprob_mb_tokens": 128},
    }
    teacher._micro_batch_size = 3
    teacher.sharding_annotations = MagicMock()
    teacher.sharding_annotations.get_axis_size.return_value = 1
    teacher.worker_group = MagicMock()
    teacher.worker_group.run_all_workers_sharded_data.return_value = ["future"]
    sorted_indices = torch.tensor([[[10, 11]], [[20, 21]]])
    sorted_support = torch.tensor([[[-1.0, -2.0]], [[-3.0, -4.0]]])
    sorted_targets = torch.tensor([[-0.1], [-0.2]])
    teacher.worker_group.get_all_worker_results.return_value = [
        {
            "logprobs": sorted_targets,
            "topk_indices": sorted_indices,
            "topk_logprobs": sorted_support,
        }
    ]
    data = MagicMock()
    data.shard_by_batch_size.return_value = (["packed_shard"], [1, 0])

    result = teacher.get_topk_logprobs(data, k=2)

    torch.testing.assert_close(result["topk_indices"], sorted_indices.flip(0))
    torch.testing.assert_close(result["topk_logprobs"], sorted_support.flip(0))
    torch.testing.assert_close(result["reference_logprobs"], sorted_targets.flip(0))
    assert teacher.sequence_packing_args["max_tokens_per_microbatch"] == 128
    data.shard_by_batch_size.assert_called_once_with(
        1,
        batch_size=None,
        sequence_packing_args=teacher.sequence_packing_args,
    )
