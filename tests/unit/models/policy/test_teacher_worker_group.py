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

import pytest


def test_teacher_config_defaults():
    from nemo_rl.models.policy.teacher_worker_group import TeacherConfig
    cfg = TeacherConfig(alias="math", model_name="/ckpt/math", tensor_model_parallel_size=4)
    assert cfg["alias"] == "math"
    assert cfg["tensor_model_parallel_size"] == 4


def test_create_teacher_configs_homogeneous():
    from nemo_rl.models.policy.teacher_worker_group import create_teacher_configs_from_opd_config
    configs = create_teacher_configs_from_opd_config({
        "teacher_model_by_agent_name": {"math": "/ckpt/math", "code": "/ckpt/code"},
        "non_colocated_teachers": {"default_teacher_cfg": {"tensor_model_parallel_size": 4}},
    })
    assert len(configs) == 2
    assert all(c["tensor_model_parallel_size"] == 4 for c in configs)


def test_create_teacher_configs_heterogeneous_override():
    from nemo_rl.models.policy.teacher_worker_group import create_teacher_configs_from_opd_config
    configs = create_teacher_configs_from_opd_config({
        "teacher_model_by_agent_name": {"math": "/ckpt/math", "code": "/ckpt/code"},
        "non_colocated_teachers": {
            "default_teacher_cfg": {"tensor_model_parallel_size": 4},
            "teacher_overrides": {"code": {"tensor_model_parallel_size": 8}},
        },
    })
    code_cfg = [c for c in configs if c["alias"] == "code"][0]
    assert code_cfg["tensor_model_parallel_size"] == 8


def test_create_teacher_configs_deduplicates():
    from nemo_rl.models.policy.teacher_worker_group import create_teacher_configs_from_opd_config
    configs = create_teacher_configs_from_opd_config({
        "teacher_model_by_agent_name": {"math": "/shared", "code": "/shared", "rlhf": "/rlhf"},
        "deduplicate_shared_teacher_checkpoints": True,
        "non_colocated_teachers": {"default_teacher_cfg": {"tensor_model_parallel_size": 2}},
    })
    assert len(configs) == 2
