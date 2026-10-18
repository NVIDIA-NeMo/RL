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


# ── fp32 LM head: explicit per teacher ─────────────────────────────────────


def test_teacher_fp32_lm_head_is_an_explicit_typed_field():
    import pydantic

    from nemo_rl.algorithms.opd import TeacherResourceConfig, TeacherResourceOverrides

    assert TeacherResourceConfig().fp32_lm_head is False
    assert TeacherResourceConfig(fp32_lm_head="tf32").fp32_lm_head == "tf32"
    assert TeacherResourceOverrides().fp32_lm_head is None
    with pytest.raises(pydantic.ValidationError):
        TeacherResourceConfig(fp32_lm_head="fp16")
    # A copy inside megatron_cfg_overrides would sidestep the student/teacher
    # match check, so the typed field is the only place to set it.
    for schema in (TeacherResourceConfig, TeacherResourceOverrides):
        with pytest.raises(
            pydantic.ValidationError, match="not inside megatron_cfg_overrides"
        ):
            schema(megatron_cfg_overrides={"fp32_lm_head": "tf32"})


def test_teacher_fp32_lm_head_survives_a_partial_override_in_the_parsed_config():
    """default_teacher_cfg sets it; an override that does not mention it keeps it."""
    from nemo_rl.algorithms.opd import OnPolicyDistillationConfig, _opd_cfg

    opd = OnPolicyDistillationConfig(
        enabled=True,
        teacher_model_by_agent_name={"general": "/ckpt/general", "code": "/ckpt/code"},
        non_colocated_teachers={
            "enabled": True,
            "default_teacher_cfg": {"gpus_per_node": 4, "fp32_lm_head": "tf32"},
            "teacher_overrides": {"code": {"micro_batch_size": 1}},
        },
    )

    configs = {
        config.alias: config
        for config in create_teacher_configs_from_opd_config(
            _opd_cfg({"on_policy_distillation": opd})
        )
    }

    assert configs["general"].fp32_lm_head == "tf32"
    assert configs["code"].fp32_lm_head == "tf32"
    assert configs["code"].micro_batch_size == 1
    assert "fp32_lm_head" not in configs["code"].megatron_cfg_overrides


def _make_fp32_teacher_group(
    policy_config: dict[str, Any], teacher_fp32_lm_head: Any
) -> TeacherWorkerGroup:
    (teacher_cfg,) = create_teacher_configs_from_opd_config(
        {
            "teacher_model_by_agent_name": {"teacher": "/teacher"},
            "non_colocated_teachers": {
                "default_teacher_cfg": {
                    "gpus_per_node": 2,
                    "micro_batch_size": 1,
                    "fp32_lm_head": teacher_fp32_lm_head,
                }
            },
        }
    )
    cluster = MagicMock()
    cluster.world_size.return_value = 2
    return TeacherWorkerGroup(teacher_cfg, cluster, policy_config, MagicMock())


@pytest.mark.parametrize("teacher_value", [False, "tf32"])
def test_teacher_never_inherits_the_student_fp32_lm_head(
    student_policy_config: dict[str, Any],
    mock_ray_worker_group: MagicMock,
    teacher_value: Any,
) -> None:
    """The worker config carries the teacher's own setting, not the student's."""
    student_policy_config["megatron_cfg"]["fp32_lm_head"] = "tf32"

    teacher = _make_fp32_teacher_group(student_policy_config, teacher_value)

    worker_config = mock_ray_worker_group.call_args.args[1].args[0]
    assert worker_config is teacher.cfg
    assert worker_config["megatron_cfg"]["fp32_lm_head"] == teacher_value
    # The student's own config is left alone.
    assert student_policy_config["megatron_cfg"]["fp32_lm_head"] == "tf32"


def test_teacher_strict_fp32_lm_head_is_not_implemented(
    student_policy_config: dict[str, Any], mock_ray_worker_group: MagicMock
) -> None:
    with pytest.raises(NotImplementedError, match="not implemented for MOPD teachers"):
        _make_fp32_teacher_group(student_policy_config, True)
    mock_ray_worker_group.assert_not_called()


class _RecordingOutputLayer(torch.nn.Module):
    """Stand-in output layer recording each GEMM's operand dtypes and TF32 flag."""

    def __init__(self) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(
            torch.ones(4, 2, dtype=torch.bfloat16), requires_grad=False
        )
        self.calls: list[tuple[torch.dtype, torch.dtype, bool]] = []

    def forward(self, input_, *args, weight=None, **kwargs):
        w = weight if weight is not None else self.weight
        self.calls.append(
            (input_.dtype, w.dtype, torch.backends.cuda.matmul.allow_tf32)
        )
        return input_ @ w.t()


@pytest.mark.mcore
@pytest.mark.parametrize("mode", [False, "tf32"])
def test_teacher_worker_config_applies_its_own_fp32_lm_head(
    student_policy_config: dict[str, Any],
    mock_ray_worker_group: MagicMock,
    mode: Any,
) -> None:
    """A teacher matching the student passes the worker check and gets its head.

    Runs MegatronPolicyWorker.__init__'s fp32-head steps on the worker config
    TeacherWorkerGroup builds: validate_fp32_lm_head_config, which checks the
    teacher's setting against the student's vLLM env var carried over in the
    copied config, then apply_fp32_lm_head(use_tf32=...) when enabled.
    """
    from types import SimpleNamespace

    from nemo_rl.models.megatron.setup import (
        apply_fp32_lm_head,
        validate_fp32_lm_head_config,
    )

    student_policy_config["megatron_cfg"]["fp32_lm_head"] = mode
    student_policy_config["generation"] = {
        "backend": "vllm",
        "vllm_cfg": {"env_vars": {"NRL_VLLM_FP32_LM_HEAD": "1"} if mode else {}},
    }
    cfg = _make_fp32_teacher_group(student_policy_config, mode).cfg

    validate_fp32_lm_head_config(cfg)
    layer = _RecordingOutputLayer()
    fp32_lm_head = cfg["megatron_cfg"]["fp32_lm_head"]
    if fp32_lm_head:
        chunk = SimpleNamespace(
            module=SimpleNamespace(output_layer=layer, post_process=True)
        )
        apply_fp32_lm_head([chunk], use_tf32=(fp32_lm_head == "tf32"))
    tf32_before = torch.backends.cuda.matmul.allow_tf32

    logits = layer(torch.ones(3, 2, dtype=torch.bfloat16))

    if mode:
        # Upcast operands, GEMM on TF32 tensor cores, fp32 logits.
        assert logits.dtype == torch.float32
        assert layer.calls == [(torch.float32, torch.float32, True)]
    else:
        assert logits.dtype == torch.bfloat16
        assert layer.calls == [(torch.bfloat16, torch.bfloat16, tf32_before)]
    assert torch.backends.cuda.matmul.allow_tf32 == tf32_before
