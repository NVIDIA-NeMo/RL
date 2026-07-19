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

from unittest.mock import MagicMock, patch

from nemo_rl.models.policy import (
    DraftConfig,
    DraftConfigDisabled,
    DynamicBatchingConfigDisabled,
    SequencePackingConfigDisabled,
)


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


@patch("nemo_rl.distributed.worker_groups.RayWorkerGroup")
@patch("nemo_rl.distributed.worker_groups.RayWorkerBuilder")
def test_teacher_replaces_enabled_draft_with_disabled_config(
    mock_worker_builder, mock_worker_group
):
    from nemo_rl.models.policy.teacher_worker_group import (
        TeacherConfig,
        TeacherWorkerGroup,
    )

    teacher_cfg = TeacherConfig(
        alias="teacher",
        model_name="teacher-model",
        tensor_model_parallel_size=1,
        pipeline_model_parallel_size=1,
        context_parallel_size=1,
        expert_model_parallel_size=1,
        num_nodes=1,
        gpus_per_node=1,
        precision="bf16",
        micro_batch_size=1,
        megatron_cfg_overrides={},
    )
    cluster = MagicMock()
    cluster.world_size.return_value = 1
    policy_config = {
        "megatron_cfg": {"enabled": True},
        "dtensor_cfg": {"enabled": False},
        "draft": DraftConfig(model_name="student-draft"),
        "sequence_packing": SequencePackingConfigDisabled(),
        "dynamic_batching": DynamicBatchingConfigDisabled(),
    }

    teacher = TeacherWorkerGroup(
        teacher_cfg=teacher_cfg,
        cluster=cluster,
        policy_config=policy_config,
        tokenizer=MagicMock(),
    )

    worker_config = mock_worker_builder.call_args.args[1]
    assert isinstance(worker_config["draft"], DraftConfigDisabled)
    assert policy_config["draft"].enabled is True
    assert teacher.cfg is worker_config
    mock_worker_group.assert_called_once()
