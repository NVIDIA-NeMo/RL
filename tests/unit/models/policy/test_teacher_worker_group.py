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

from copy import deepcopy
from typing import Any
from unittest.mock import MagicMock

import pytest
import torch

from nemo_rl.distributed import worker_groups
from nemo_rl.distributed.batched_data_dict import BatchedDataDict
from nemo_rl.models.megatron.router_replay import router_replay_enabled
from nemo_rl.models.policy.teacher_worker_group import (
    TeacherWorkerGroup,
    create_teacher_configs_from_opd_config,
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


def test_partial_override_does_not_clobber_defaults():
    """A per-alias block setting one key must keep every default_teacher_cfg value."""
    from nemo_rl.models.policy.teacher_worker_group import (
        create_teacher_configs_from_opd_config,
    )

    configs = create_teacher_configs_from_opd_config(
        {
            "teacher_model_by_agent_name": {"general": "/ckpt/general"},
            "non_colocated_teachers": {
                "default_teacher_cfg": {
                    "tensor_model_parallel_size": 8,
                    "gpus_per_node": 4,
                    "num_nodes": 2,
                },
                # partial block: only a megatron override, no resource fields
                "teacher_overrides": {"general": {"some_megatron_knob": True}},
            },
        }
    )
    cfg = configs[0]
    assert cfg.gpus_per_node == 4  # not the schema default 8
    assert cfg.num_nodes == 2
    assert cfg.tensor_model_parallel_size == 8
    assert cfg.megatron_cfg_overrides["some_megatron_knob"] is True


def test_typed_override_blocks_round_trip_without_schema_fill():
    """The override blocks are typed as partial models (all fields None); after
    the ``model_dump(exclude_none=True)`` in ``_opd_cfg``, unset fields must not
    reappear as schema defaults and clobber default_teacher_cfg in the merge."""
    from nemo_rl.algorithms.opd import OnPolicyDistillationConfig, _opd_cfg
    from nemo_rl.models.policy.teacher_worker_group import (
        create_teacher_configs_from_opd_config,
    )

    opd = OnPolicyDistillationConfig(
        enabled=True,
        teacher_model_by_agent_name={"general": "/ckpt/general"},
        non_colocated_teachers={
            "enabled": True,
            "default_teacher_cfg": {"gpus_per_node": 4, "num_nodes": 2},
            "teacher_overrides": {"general": {"micro_batch_size": 1}},
        },
    )
    (cfg,) = create_teacher_configs_from_opd_config(
        _opd_cfg({"on_policy_distillation": opd})
    )
    assert cfg.gpus_per_node == 4  # not the schema default 8
    assert cfg.num_nodes == 2
    assert cfg.micro_batch_size == 1


def test_provider_override_allowlist_is_explicit_keys_only():
    """Only explicitly-set teacher override keys may reach the model provider;
    architecture keys inherited from the student config must be blocked, while
    student configs (no allowlist) keep the status-quo behavior."""
    from nemo_rl.models.policy import provider_override_allowed
    from nemo_rl.models.policy.teacher_worker_group import (
        create_teacher_configs_from_opd_config,
    )

    (teacher_cfg,) = create_teacher_configs_from_opd_config(
        {
            "teacher_model_by_agent_name": {"general": "/ckpt/general"},
            "non_colocated_teachers": {
                "default_teacher_cfg": {"gpus_per_node": 4},
                "teacher_overrides": {"general": {"mtp_num_layers": 1}},
            },
        }
    )
    allowlist = sorted(teacher_cfg.megatron_cfg_overrides.keys())
    assert allowlist == ["mtp_num_layers"]

    # teacher megatron_cfg: cloned student keys + explicit override + allowlist
    teacher_megatron_cfg = {
        "mtp_num_layers": 1,  # explicit for this teacher
        "radio_force_cpe_eval_mode": True,  # inherited from the VLM student
        "freeze_vision_model": False,  # inherited from the VLM student
        "_provider_override_allowlist": allowlist,
    }
    assert provider_override_allowed(teacher_megatron_cfg, "mtp_num_layers")
    assert not provider_override_allowed(
        teacher_megatron_cfg, "radio_force_cpe_eval_mode"
    )
    assert not provider_override_allowed(teacher_megatron_cfg, "freeze_vision_model")

    # student configs carry no allowlist: every key applies as before
    student_megatron_cfg = {"radio_force_cpe_eval_mode": True}
    assert provider_override_allowed(student_megatron_cfg, "radio_force_cpe_eval_mode")


@pytest.fixture
def student_policy_config() -> dict[str, Any]:
    return {
        "model_name": "/student",
        "router_replay": {
            "enabled": True,
            "transport": "ray",
            "_store_run_instance_id": "student-run",
        },
        "megatron_cfg": {"enabled": True},
        "sequence_packing": {
            "enabled": False,
            "algorithm": "modified_first_fit_decreasing",
            "logprob_mb_tokens": 16,
        },
        "dynamic_batching": {"enabled": False},
    }


@pytest.fixture
def mock_ray_worker_group(monkeypatch: pytest.MonkeyPatch) -> MagicMock:
    mock_group = MagicMock()
    monkeypatch.setattr(worker_groups, "RayWorkerGroup", mock_group)
    return mock_group


def _make_teacher_group(policy_config: dict[str, Any]) -> TeacherWorkerGroup:
    (teacher_cfg,) = create_teacher_configs_from_opd_config(
        {
            "teacher_model_by_agent_name": {"teacher": "/teacher"},
            "non_colocated_teachers": {
                "default_teacher_cfg": {"gpus_per_node": 2, "micro_batch_size": 1}
            },
        }
    )
    cluster = MagicMock()
    cluster.world_size.return_value = 2
    return TeacherWorkerGroup(teacher_cfg, cluster, policy_config, MagicMock())


@pytest.mark.parametrize(
    "replay_config",
    [
        None,
        {"enabled": False},
        {"enabled": True, "transport": "inline"},
        {
            "enabled": True,
            "transport": "ray",
            "_store_run_instance_id": "student-run",
        },
    ],
    ids=["absent", "disabled", "inline", "ray"],
)
def test_teacher_disables_student_router_replay(
    student_policy_config: dict[str, Any],
    mock_ray_worker_group: MagicMock,
    replay_config: dict[str, Any] | None,
) -> None:
    if replay_config is None:
        del student_policy_config["router_replay"]
    else:
        student_policy_config["router_replay"] = replay_config
    original_policy_config = deepcopy(student_policy_config)

    teacher = _make_teacher_group(student_policy_config)

    # Check the config actually sent to MegatronPolicyWorker, not just the
    # group wrapper: this controls both the worker guard and MCore routers.
    worker_builder = mock_ray_worker_group.call_args.args[1]
    worker_config = worker_builder.args[0]
    assert worker_config is teacher.cfg
    assert not router_replay_enabled(worker_config)
    assert student_policy_config == original_policy_config


@pytest.mark.parametrize("use_sequence_packing", [False, True])
def test_teacher_logprobs_explicitly_skip_router_replay(
    student_policy_config: dict[str, Any],
    mock_ray_worker_group: MagicMock,
    use_sequence_packing: bool,
) -> None:
    student_policy_config["sequence_packing"]["enabled"] = use_sequence_packing
    teacher = _make_teacher_group(student_policy_config)
    mock_group = mock_ray_worker_group.return_value

    def fake_get_results(_futures: Any) -> list[BatchedDataDict]:
        call = mock_group.run_all_workers_sharded_data.call_args
        assert call.args == ("get_logprobs",)
        assert call.kwargs["common_kwargs"] == {
            "micro_batch_size": 1,
            "require_router_replay": False,
        }
        # The collector supplies tokens/lengths, not student expert routes.
        shards = call.kwargs["data"]
        assert all("routed_experts" not in shard for shard in shards)
        return [
            BatchedDataDict(logprobs=shard["input_ids"].float()) for shard in shards
        ]

    mock_group.get_all_worker_results.side_effect = fake_get_results
    data = BatchedDataDict(
        input_ids=torch.tensor([[1, 2, 0, 0], [3, 4, 5, 6]]),
        input_lengths=torch.tensor([2, 4]),
    )

    result = teacher.get_logprobs(data)

    # Preserve the teacher-logprob result contract, including unpacking order.
    assert set(result) == {"reference_logprobs"}
    torch.testing.assert_close(result["reference_logprobs"], data["input_ids"].float())
