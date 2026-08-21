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

import pytest
import torch

from nemo_rl.distributed.batched_data_dict import BatchedDataDict


def test_teacher_resource_config_defaults():
    from nemo_rl.algorithms.opd import TeacherResourceConfig

    res = TeacherResourceConfig(tensor_model_parallel_size=4)
    assert res.tensor_model_parallel_size == 4
    assert res.pipeline_model_parallel_size == 1
    assert res.expert_tensor_parallel_size == 1
    assert res.gpus_per_node == 8
    assert res.precision == "bfloat16"


def test_teacher_resource_config_normalizes_legacy_bf16_precision():
    from nemo_rl.algorithms.opd import TeacherResourceConfig

    assert TeacherResourceConfig(precision="bf16").precision == "bfloat16"


def test_apply_teacher_resource_config_sets_precision_and_warns_on_etp_change():
    from nemo_rl.models.policy.teacher_worker_group import (
        TeacherConfig,
        _apply_teacher_resource_config,
    )

    cfg = {
        "precision": "float32",
        "megatron_cfg": {
            "enabled": True,
            "tensor_model_parallel_size": 8,
            "pipeline_model_parallel_size": 2,
            "context_parallel_size": 2,
            "expert_tensor_parallel_size": 8,
            "expert_model_parallel_size": 4,
        },
    }
    teacher_cfg = TeacherConfig(
        alias="large",
        model_name="/ckpt/large",
        tensor_model_parallel_size=4,
        pipeline_model_parallel_size=1,
        context_parallel_size=1,
        expert_tensor_parallel_size=1,
        expert_model_parallel_size=2,
        num_nodes=1,
        gpus_per_node=8,
        precision="bfloat16",
        micro_batch_size=1,
        megatron_cfg_overrides={},
    )

    with pytest.warns(UserWarning, match="independently of the policy value 8"):
        _apply_teacher_resource_config(cfg, teacher_cfg)

    assert cfg["precision"] == "bfloat16"
    assert cfg["megatron_cfg"] == {
        "enabled": True,
        "tensor_model_parallel_size": 4,
        "pipeline_model_parallel_size": 1,
        "context_parallel_size": 1,
        "expert_tensor_parallel_size": 1,
        "expert_model_parallel_size": 2,
    }


def test_create_teacher_configs_homogeneous():
    from nemo_rl.algorithms.opd import OnPolicyDistillationConfig, _opd_cfg
    from nemo_rl.models.policy.teacher_worker_group import (
        create_teacher_configs_from_opd_config,
    )

    config = OnPolicyDistillationConfig(
        enabled=True,
        teacher_model_by_agent_name={"math": "/ckpt/math", "code": "/ckpt/code"},
        non_colocated_teachers={
            "default_teacher_cfg": {"tensor_model_parallel_size": 4}
        },
    )
    configs = create_teacher_configs_from_opd_config(
        _opd_cfg({"on_policy_distillation": config})
    )
    assert len(configs) == 2
    assert all(c.tensor_model_parallel_size == 4 for c in configs)


def test_create_teacher_configs_sparse_override_preserves_default_resources():
    from nemo_rl.algorithms.opd import OnPolicyDistillationConfig, _opd_cfg
    from nemo_rl.models.policy.teacher_worker_group import (
        create_teacher_configs_from_opd_config,
    )

    config = OnPolicyDistillationConfig(
        enabled=True,
        teacher_model_by_agent_name={"big": "/ckpt/big"},
        non_colocated_teachers={
            "default_teacher_cfg": {
                "tensor_model_parallel_size": 8,
                "pipeline_model_parallel_size": 3,
                "context_parallel_size": 2,
                "expert_tensor_parallel_size": 5,
                "expert_model_parallel_size": 4,
                "num_nodes": 6,
                "gpus_per_node": 7,
                "micro_batch_size": 9,
                "megatron_cfg_overrides": {
                    "moe_token_dispatcher_type": "flex",
                    "moe_flex_dispatcher_backend": "deepep",
                },
            },
            "teacher_overrides": {
                "big": {
                    "num_nodes": 10,
                    "megatron_cfg_overrides": {"sequence_parallel": True},
                }
            },
        },
    )
    opd_cfg = _opd_cfg({"on_policy_distillation": config})

    assert opd_cfg["non_colocated_teachers"]["teacher_overrides"]["big"] == {
        "num_nodes": 10,
        "megatron_cfg_overrides": {"sequence_parallel": True},
    }
    resolved = create_teacher_configs_from_opd_config(opd_cfg)[0]
    assert resolved.tensor_model_parallel_size == 8
    assert resolved.pipeline_model_parallel_size == 3
    assert resolved.context_parallel_size == 2
    assert resolved.expert_tensor_parallel_size == 5
    assert resolved.expert_model_parallel_size == 4
    assert resolved.num_nodes == 10
    assert resolved.gpus_per_node == 7
    assert resolved.micro_batch_size == 9
    assert resolved.megatron_cfg_overrides == {
        "moe_token_dispatcher_type": "flex",
        "moe_flex_dispatcher_backend": "deepep",
        "sequence_parallel": True,
    }


def test_alias_field_overrides_same_default_megatron_override():
    from nemo_rl.algorithms.opd import OnPolicyDistillationConfig, _opd_cfg
    from nemo_rl.models.policy.teacher_worker_group import (
        create_teacher_configs_from_opd_config,
    )

    config = OnPolicyDistillationConfig(
        enabled=True,
        teacher_model_by_agent_name={"large": "/ckpt/large"},
        non_colocated_teachers={
            "default_teacher_cfg": {
                "context_parallel_size": 1,
                "megatron_cfg_overrides": {"context_parallel_size": 2},
            },
            "teacher_overrides": {
                "large": {
                    "context_parallel_size": 4,
                    "megatron_cfg_overrides": {"sequence_parallel": True},
                }
            },
        },
    )

    resolved = create_teacher_configs_from_opd_config(
        _opd_cfg({"on_policy_distillation": config})
    )[0]

    assert resolved.context_parallel_size == 4
    assert resolved.megatron_cfg_overrides == {"sequence_parallel": True}


def test_create_teacher_configs_heterogeneous_override():
    from nemo_rl.algorithms.opd import OnPolicyDistillationConfig, _opd_cfg
    from nemo_rl.models.policy.teacher_worker_group import (
        create_teacher_configs_from_opd_config,
    )

    config = OnPolicyDistillationConfig(
        enabled=True,
        teacher_model_by_agent_name={"math": "/ckpt/math", "code": "/ckpt/code"},
        non_colocated_teachers={
            "default_teacher_cfg": {"tensor_model_parallel_size": 4},
            "teacher_overrides": {"code": {"tensor_model_parallel_size": 8}},
        },
    )
    configs = create_teacher_configs_from_opd_config(
        _opd_cfg({"on_policy_distillation": config})
    )
    code_cfg = [c for c in configs if c.alias == "code"][0]
    assert code_cfg.tensor_model_parallel_size == 8


def test_create_teacher_configs_resolves_parallelism_from_megatron_overrides():
    from nemo_rl.algorithms.opd import OnPolicyDistillationConfig, _opd_cfg
    from nemo_rl.models.policy.teacher_worker_group import (
        create_teacher_configs_from_opd_config,
    )

    config = OnPolicyDistillationConfig(
        enabled=True,
        teacher_model_by_agent_name={"large": "/ckpt/large"},
        non_colocated_teachers={
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
                        "expert_tensor_parallel_size": 6,
                        "expert_model_parallel_size": 5,
                    }
                }
            },
        },
    )
    resolved = create_teacher_configs_from_opd_config(
        _opd_cfg({"on_policy_distillation": config})
    )[0]

    assert (
        resolved.tensor_model_parallel_size,
        resolved.pipeline_model_parallel_size,
        resolved.context_parallel_size,
        resolved.expert_tensor_parallel_size,
        resolved.expert_model_parallel_size,
    ) == (2, 3, 4, 6, 5)


def test_create_teacher_configs_deduplicates():
    from nemo_rl.algorithms.opd import OnPolicyDistillationConfig, _opd_cfg
    from nemo_rl.models.policy.teacher_worker_group import (
        create_teacher_configs_from_opd_config,
    )

    config = OnPolicyDistillationConfig(
        enabled=True,
        teacher_model_by_agent_name={
            "math": "/shared",
            "code": "/shared",
            "rlhf": "/rlhf",
        },
        deduplicate_shared_teacher_checkpoints=True,
        non_colocated_teachers={
            "default_teacher_cfg": {"tensor_model_parallel_size": 2}
        },
    )
    configs = create_teacher_configs_from_opd_config(
        _opd_cfg({"on_policy_distillation": config})
    )
    assert len(configs) == 2


def test_create_teacher_configs_rejects_conflicting_deduplicated_aliases():
    from nemo_rl.algorithms.opd import OnPolicyDistillationConfig, _opd_cfg
    from nemo_rl.models.policy.teacher_worker_group import (
        create_teacher_configs_from_opd_config,
    )

    config = OnPolicyDistillationConfig(
        enabled=True,
        teacher_model_by_agent_name={"math": "/shared", "code": "/shared"},
        deduplicate_shared_teacher_checkpoints=True,
        non_colocated_teachers={
            "default_teacher_cfg": {"tensor_model_parallel_size": 2},
            "teacher_overrides": {"code": {"num_nodes": 2}},
        },
    )

    with pytest.raises(ValueError, match="code.*math.*shared"):
        create_teacher_configs_from_opd_config(
            _opd_cfg({"on_policy_distillation": config})
        )


def test_create_teacher_configs_keeps_shared_checkpoint_aliases_without_dedup():
    from nemo_rl.algorithms.opd import OnPolicyDistillationConfig, _opd_cfg
    from nemo_rl.models.policy.teacher_worker_group import (
        create_teacher_configs_from_opd_config,
    )

    config = OnPolicyDistillationConfig(
        enabled=True,
        teacher_model_by_agent_name={"math": "/shared", "code": "/shared"},
        deduplicate_shared_teacher_checkpoints=False,
    )

    configs = create_teacher_configs_from_opd_config(
        _opd_cfg({"on_policy_distillation": config})
    )

    assert [config.alias for config in configs] == ["math", "code"]


def test_teacher_worker_group_rejects_invalid_expert_parallel_grid():
    from nemo_rl.models.policy.teacher_worker_group import (
        TeacherConfig,
        TeacherWorkerGroup,
    )

    teacher_cfg = TeacherConfig(
        alias="large",
        model_name="/ckpt/large",
        tensor_model_parallel_size=1,
        pipeline_model_parallel_size=1,
        context_parallel_size=1,
        expert_tensor_parallel_size=8,
        expert_model_parallel_size=4,
        num_nodes=2,
        gpus_per_node=8,
        precision="bf16",
        micro_batch_size=1,
        megatron_cfg_overrides={},
    )
    cluster = MagicMock()
    cluster.world_size.return_value = 16

    with pytest.raises(ValueError, match=r"ETP\(8\) \* EP\(4\) \* PP\(1\) = 32"):
        TeacherWorkerGroup(
            teacher_cfg=teacher_cfg,
            cluster=cluster,
            policy_config={"megatron_cfg": {"enabled": True}},
            tokenizer=object(),
        )


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
    from nemo_rl.algorithms.opd_packed import OPD_TEACHER_TOPK_PACKED_KEY
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
    sorted_targets = [torch.tensor([-0.1]), torch.tensor([-0.2])]
    sorted_entries = [
        {
            "seq_len": 1,
            "topk": 2,
            "topk_indices": sorted_indices[i],
            "topk_logprobs": sorted_support[i],
            "target_logprobs": sorted_targets[i],
        }
        for i in range(2)
    ]
    teacher.worker_group.get_all_worker_results.return_value = [
        {
            "per_sample_refs": sorted_entries,
            "unpacked_seq_length": 2,
        }
    ]
    data = MagicMock()
    data.size = 2
    data.__getitem__.side_effect = lambda key: (
        torch.zeros(2, 2, dtype=torch.int64)
        if key == "input_ids"
        else torch.ones(2, dtype=torch.int64)
    )
    data.shard_by_batch_size.return_value = (["packed_shard"], [1, 0])

    result = teacher.get_topk_logprobs(data, k=2)

    packed = result[OPD_TEACHER_TOPK_PACKED_KEY]
    torch.testing.assert_close(packed[0]["topk_indices"], sorted_indices[1])
    torch.testing.assert_close(packed[1]["topk_indices"], sorted_indices[0])
    torch.testing.assert_close(
        result["reference_logprobs"],
        torch.tensor([[0.0, -0.2], [0.0, -0.1]]),
    )
    assert teacher.sequence_packing_args["max_tokens_per_microbatch"] == 128
    data.shard_by_batch_size.assert_called_once_with(
        1,
        batch_size=None,
        sequence_packing_args=teacher.sequence_packing_args,
    )
