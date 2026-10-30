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

from nemo_rl.data_plane import KVBatchMeta
from nemo_rl.data_plane.schema import (
    GLOBAL_FORWARD_PAD_SEQLEN,
    MICRO_BATCH_INDICES,
    MICRO_BATCH_LENGTHS,
    OPD_FULL_HIDDEN_STATES_FIELD,
    OPD_FULL_TEACHER_INDEX_FIELD,
    TEACHER_LP_FIELDS,
    W2S_ARM_FIELDS,
    W2S_PROXY_FIELDS,
    W2S_PROXY_SUPPORT_IDS_FIELD,
    W2S_PROXY_TARGET_LOGPROBS_FIELD,
)
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
                "default_teacher_cfg": {
                    "tensor_model_parallel_size": 4,
                    "gpus_per_node": 4,
                },
                "teacher_overrides": {"code": {"tensor_model_parallel_size": 8}},
            },
        }
    )
    code_cfg = [c for c in configs if c.alias == "code"][0]
    assert code_cfg.tensor_model_parallel_size == 8
    # A partial per-alias block keeps every other default_teacher_cfg value.
    assert code_cfg.gpus_per_node == 4


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


@pytest.mark.parametrize("student_mtp_layers", [0, 5])
@pytest.mark.parametrize(
    "teacher_overrides,expected_mtp_layers",
    [
        ({}, 0),
        ({"mtp_num_layers": 0}, 0),
        ({"mtp_num_layers": 2}, 2),
        ({"freeze_vision_model": False}, 0),
    ],
)
def test_teacher_worker_group_isolates_student_inference_settings(
    monkeypatch, student_mtp_layers, teacher_overrides, expected_mtp_layers
):
    """Scoring disables router replay and defaults MTP off independently of training."""
    import nemo_rl.distributed.worker_groups as worker_groups
    from nemo_rl.models.policy.teacher_worker_group import (
        TeacherConfig,
        TeacherWorkerGroup,
    )

    captured = {}

    class FakeWorkerBuilder:
        def __init__(self, worker_path, cfg, **kwargs):
            del worker_path, kwargs
            captured["cfg"] = cfg

    class FakeWorkerGroup:
        def __init__(self, cluster, worker_builder, **kwargs):
            del cluster, worker_builder, kwargs

    monkeypatch.setattr(worker_groups, "RayWorkerBuilder", FakeWorkerBuilder)
    monkeypatch.setattr(worker_groups, "RayWorkerGroup", FakeWorkerGroup)
    cluster = MagicMock()
    cluster.world_size.return_value = 1
    policy_config = {
        "model_name": "/ckpt/student",
        "megatron_cfg": {
            "enabled": True,
            "mtp_num_layers": student_mtp_layers,
            "mtp_use_repeated_layer": True,
            "freeze_vision_model": True,
        },
        "automodel_cfg": {"enabled": False},
        "sequence_packing": {"enabled": False},
        "dynamic_batching": {"enabled": False},
        "router_replay": {"enabled": True},
    }
    teacher_config = TeacherConfig(
        alias="teacher",
        model_name="/ckpt/teacher",
        tensor_model_parallel_size=1,
        pipeline_model_parallel_size=1,
        context_parallel_size=1,
        expert_model_parallel_size=1,
        num_nodes=1,
        gpus_per_node=1,
        precision="bf16",
        micro_batch_size=1,
        megatron_cfg_overrides=teacher_overrides,
    )

    teacher = TeacherWorkerGroup(
        teacher_config,
        cluster,
        policy_config,
        MagicMock(),
        teacher_index=0,
    )

    assert captured["cfg"]["router_replay"]["enabled"] is False
    assert teacher.cfg["router_replay"]["enabled"] is False
    assert policy_config["router_replay"]["enabled"] is True
    megatron_cfg = captured["cfg"]["megatron_cfg"]
    assert megatron_cfg["mtp_num_layers"] == expected_mtp_layers
    assert megatron_cfg["_provider_override_allowlist"] == sorted(
        {"mtp_num_layers", *teacher_overrides}
    )
    assert policy_config["megatron_cfg"]["mtp_num_layers"] == student_mtp_layers
    assert teacher_config.megatron_cfg_overrides == teacher_overrides


def _build_teacher_cfg(monkeypatch, policy_config, megatron_cfg_overrides):
    """Build a TeacherWorkerGroup with fake Ray builders; return the teacher cfg."""
    import nemo_rl.distributed.worker_groups as worker_groups
    from nemo_rl.models.policy.teacher_worker_group import (
        TeacherConfig,
        TeacherWorkerGroup,
    )

    class FakeWorkerBuilder:
        def __init__(self, worker_path, cfg, **kwargs):
            del worker_path, cfg, kwargs

    class FakeWorkerGroup:
        def __init__(self, cluster, worker_builder, **kwargs):
            del cluster, worker_builder, kwargs

    monkeypatch.setattr(worker_groups, "RayWorkerBuilder", FakeWorkerBuilder)
    monkeypatch.setattr(worker_groups, "RayWorkerGroup", FakeWorkerGroup)
    cluster = MagicMock()
    cluster.world_size.return_value = 1
    teacher_config = TeacherConfig(
        alias="teacher",
        model_name="/ckpt/teacher",
        tensor_model_parallel_size=1,
        pipeline_model_parallel_size=1,
        context_parallel_size=1,
        expert_model_parallel_size=1,
        num_nodes=1,
        gpus_per_node=1,
        precision="bf16",
        micro_batch_size=1,
        megatron_cfg_overrides=megatron_cfg_overrides,
    )
    teacher = TeacherWorkerGroup(
        teacher_config, cluster, policy_config, MagicMock(), teacher_index=0
    )
    return teacher.cfg


def _student_policy_config_with_manual_gc():
    return {
        "model_name": "/ckpt/student",
        "megatron_cfg": {
            "enabled": True,
            "manual_gc": True,
            "manual_gc_interval": 10,
        },
        "automodel_cfg": {"enabled": False},
        "sequence_packing": {"enabled": False},
        "dynamic_batching": {"enabled": False},
    }


def test_teacher_worker_group_turns_off_the_student_manual_gc(monkeypatch):
    """A teacher never trains, so it would never collect with GC disabled."""
    policy_config = _student_policy_config_with_manual_gc()

    cfg = _build_teacher_cfg(monkeypatch, policy_config, megatron_cfg_overrides={})

    assert cfg["megatron_cfg"]["manual_gc"] is False
    # The student's own config is untouched.
    assert policy_config["megatron_cfg"]["manual_gc"] is True


def test_teacher_worker_group_rejects_manual_gc_on_a_teacher(monkeypatch):
    with pytest.raises(ValueError, match="manual_gc is not supported on teachers"):
        _build_teacher_cfg(
            monkeypatch,
            _student_policy_config_with_manual_gc(),
            megatron_cfg_overrides={"manual_gc": True},
        )


def test_teacher_worker_group_drops_the_student_pretrained_checkpoint(monkeypatch):
    """Left in, the teacher would load the student's weights and distill itself.

    Resume keeps student weights out of the shared config by passing them as a
    ``weights_path`` argument; this key arrives through the config instead.
    """
    import nemo_rl.distributed.worker_groups as worker_groups
    from nemo_rl.models.policy.teacher_worker_group import (
        TeacherConfig,
        TeacherWorkerGroup,
    )

    captured = {}

    class FakeWorkerBuilder:
        def __init__(self, worker_path, cfg, **kwargs):
            del worker_path, kwargs
            captured["cfg"] = cfg

    class FakeWorkerGroup:
        def __init__(self, cluster, worker_builder, **kwargs):
            del cluster, worker_builder, kwargs

    monkeypatch.setattr(worker_groups, "RayWorkerBuilder", FakeWorkerBuilder)
    monkeypatch.setattr(worker_groups, "RayWorkerGroup", FakeWorkerGroup)
    cluster = MagicMock()
    cluster.world_size.return_value = 1
    policy_config = {
        "model_name": "/ckpt/student",
        "pretrained_checkpoint": {"format": "megatron_bridge", "path": "/ckpt/sft"},
        "megatron_cfg": {"enabled": True},
        "automodel_cfg": {"enabled": False},
        "sequence_packing": {"enabled": False},
        "dynamic_batching": {"enabled": False},
    }
    teacher_config = TeacherConfig(
        alias="teacher",
        model_name="/ckpt/teacher",
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

    teacher = TeacherWorkerGroup(
        teacher_config,
        cluster,
        policy_config,
        MagicMock(),
        teacher_index=0,
    )

    assert "pretrained_checkpoint" not in captured["cfg"]
    assert "pretrained_checkpoint" not in teacher.cfg
    # The student's own config is left alone.
    assert policy_config["pretrained_checkpoint"]["path"] == "/ckpt/sft"


def _disable_opd_full(teacher) -> None:
    """Set the opd_full attributes to the state __init__ gives them when off.

    These doubles are built with ``object.__new__``, so every attribute the
    dispatch path reads has to be set here by hand.
    """
    teacher._opd_full_payload = None
    teacher._opd_full_payload_dtype = "bfloat16"
    teacher._opd_full_payload_field = None
    teacher._opd_full_teacher_index_field = None
    teacher.teacher_index = 0


@pytest.mark.parametrize(
    "media_shapes,keep_packed",
    [
        (None, True),
        ([[], []], False),
        ([[[0, 3, 2, 2]], [[0, 3, 2, 2]]], False),
        ([[[1, 3, 2, 2]], []], True),
        ([[], [[1, 3, 2, 2]]], True),
        ([[], None], True),
        ([[], "invalid"], True),
        ([[], [[0, -1]]], True),
    ],
)
@pytest.mark.parametrize("multimodal", [False, True])
def test_get_logprobs_from_meta_dispatches_tq_shards_to_teacher_workers(
    multimodal, media_shapes, keep_packed
):
    """TeacherWorkerGroup sends metadata, not token tensors, to each DP rank."""
    from nemo_rl.models.policy.teacher_worker_group import TeacherWorkerGroup

    class Sharding:
        def get_axis_size(self, axis):
            assert axis == "data_parallel"
            return 2

    worker_group = MagicMock()
    worker_group.run_all_workers_sharded_data.return_value = "futures"
    teacher = object.__new__(TeacherWorkerGroup)
    _disable_opd_full(teacher)
    teacher.alias = "teacher"
    teacher.use_sequence_packing = False
    teacher.use_dynamic_batches = False
    teacher.sequence_length_pad_multiple = 2
    teacher.sharding_annotations = Sharding()
    teacher.worker_group = worker_group
    teacher._micro_batch_size = 1
    meta = KVBatchMeta(
        partition_id="rollout_data",
        task_name="train",
        sample_ids=["a", "b"],
        fields=["input_ids", "input_lengths", "content", "generation_logprobs"]
        + (
            ["pixel_values", "imgs_sizes", "num_frames", "token_type_ids"]
            if multimodal
            else []
        ),
        sequence_lengths=[3, 5],
        tags=[{"row": "a"}, {"row": "b"}],
    )
    if multimodal and media_shapes is not None:
        for tag, shapes in zip(meta.tags, media_shapes):
            for field in ("pixel_values", "imgs_sizes", "num_frames"):
                tag[f"{field}__row_shapes"] = (
                    {"shapes": shapes} if shapes is not None else {}
                )

    teacher.get_logprobs_from_meta(meta)

    call = worker_group.run_all_workers_sharded_data.call_args
    kwargs = call.kwargs
    assert call.args == ("get_teacher_logprobs_presharded",)
    assert [shard.sample_ids for shard in kwargs["meta"]] == [["a"], ["b"]]
    expected_fields = list(TEACHER_LP_FIELDS) + (
        (["imgs_sizes", "num_frames", "pixel_values"] if keep_packed else [])
        + ["token_type_ids"]
        if multimodal
        else []
    )
    assert all(shard.fields == expected_fields for shard in kwargs["meta"])
    assert [shard.tags for shard in kwargs["meta"]] == [[tag] for tag in meta.tags]
    assert all(
        shard.extra_info[GLOBAL_FORWARD_PAD_SEQLEN] == 6 for shard in kwargs["meta"]
    )
    worker_group.get_all_worker_results.assert_called_once_with("futures")


def test_get_logprobs_from_meta_builds_global_dynamic_batch_plan(monkeypatch):
    """Teacher presharding uses logprob tokens and ships one balanced global plan."""
    import nemo_rl.models.policy.teacher_worker_group as teacher_module
    from nemo_rl.models.policy.teacher_worker_group import TeacherWorkerGroup

    class Sharding:
        def get_axis_size(self, axis):
            assert axis == "data_parallel"
            return 2

    captured = {}
    real_shard_meta_for_dp = teacher_module.shard_meta_for_dp

    def capture_plan(meta, **kwargs):
        captured.update(kwargs)
        captured["meta"] = meta
        return real_shard_meta_for_dp(meta, **kwargs)

    monkeypatch.setattr(teacher_module, "shard_meta_for_dp", capture_plan)
    worker_group = MagicMock()
    teacher = object.__new__(TeacherWorkerGroup)
    _disable_opd_full(teacher)
    teacher.alias = "teacher"
    teacher.use_sequence_packing = False
    teacher.use_dynamic_batches = True
    teacher.sequence_length_pad_multiple = 1
    teacher.dynamic_batching_args = {
        "input_key": "input_ids",
        "input_lengths_key": "input_lengths",
        "sequence_length_round": 64,
        "max_tokens_per_microbatch": 999,
    }
    teacher.cfg = {
        "dynamic_batching": {
            "enabled": True,
            "train_mb_tokens": 999,
            "logprob_mb_tokens": 256,
            "sequence_length_round": 64,
        }
    }
    teacher.sharding_annotations = Sharding()
    teacher.worker_group = worker_group
    teacher._micro_batch_size = 1
    meta = KVBatchMeta(
        partition_id="rollout_data",
        task_name="train",
        sample_ids=["a", "b"],
        fields=["input_ids", "input_lengths"],
        sequence_lengths=[100, 60],
    )

    teacher.get_logprobs_from_meta(meta)

    assert captured["sequence_packing_args"] is None
    assert captured["dynamic_batching_args"]["max_tokens_per_microbatch"] == 256
    assert captured["meta"].extra_info[GLOBAL_FORWARD_PAD_SEQLEN] == 128
    shards = worker_group.run_all_workers_sharded_data.call_args.kwargs["meta"]
    assert sorted(sample_id for shard in shards for sample_id in shard.sample_ids) == [
        "a",
        "b",
    ]
    assert all(MICRO_BATCH_INDICES in shard.extra_info for shard in shards)
    assert all(MICRO_BATCH_LENGTHS in shard.extra_info for shard in shards)
    assert (
        len(
            {
                len(microbatch_lengths)
                for shard in shards
                for microbatch_lengths in shard.extra_info[MICRO_BATCH_LENGTHS]
            }
        )
        == 1
    )
    assert all(
        microbatch_length <= 128
        for shard in shards
        for microbatch_lengths in shard.extra_info[MICRO_BATCH_LENGTHS]
        for microbatch_length in microbatch_lengths
    )


def test_get_logprobs_from_meta_builds_global_sequence_packing_plan(monkeypatch):
    """Teacher packing plans use logprob tokens and skip global pad rounding."""
    import nemo_rl.models.policy.teacher_worker_group as teacher_module
    from nemo_rl.models.policy.teacher_worker_group import TeacherWorkerGroup

    class Sharding:
        def get_axis_size(self, axis):
            assert axis == "data_parallel"
            return 2

    captured = {}
    real_shard_meta_for_dp = teacher_module.shard_meta_for_dp

    def capture_plan(meta, **kwargs):
        captured.update(kwargs)
        captured["meta"] = meta
        return real_shard_meta_for_dp(meta, **kwargs)

    monkeypatch.setattr(teacher_module, "shard_meta_for_dp", capture_plan)
    worker_group = MagicMock()
    teacher = object.__new__(TeacherWorkerGroup)
    _disable_opd_full(teacher)
    teacher.alias = "teacher"
    teacher.use_sequence_packing = True
    teacher.use_dynamic_batches = False
    teacher.sequence_length_pad_multiple = 16
    teacher.sequence_packing_args = {
        "algorithm": "modified_first_fit_decreasing",
        "input_key": "input_ids",
        "input_lengths_key": "input_lengths",
        "sequence_length_pad_multiple": 16,
    }
    teacher.cfg = {
        "sequence_packing": {
            "enabled": True,
            "train_mb_tokens": 999,
            "logprob_mb_tokens": 16,
        }
    }
    teacher.sharding_annotations = Sharding()
    teacher.worker_group = worker_group
    teacher._micro_batch_size = 1
    meta = KVBatchMeta(
        partition_id="rollout_data",
        task_name="train",
        sample_ids=["a", "b", "c", "d"],
        fields=["input_ids", "input_lengths"],
        sequence_lengths=[8, 1, 7, 2],
    )

    teacher.get_logprobs_from_meta(meta)

    assert captured["dynamic_batching_args"] is None
    assert captured["sequence_packing_args"]["max_tokens_per_microbatch"] == 16
    assert captured["meta"].extra_info[GLOBAL_FORWARD_PAD_SEQLEN] == 8
    shards = worker_group.run_all_workers_sharded_data.call_args.kwargs["meta"]
    assert sorted(sample_id for shard in shards for sample_id in shard.sample_ids) == [
        "a",
        "b",
        "c",
        "d",
    ]
    assert all(MICRO_BATCH_INDICES in shard.extra_info for shard in shards)
    assert all(MICRO_BATCH_LENGTHS in shard.extra_info for shard in shards)
    assert (
        len(
            {
                len(microbatch_lengths)
                for shard in shards
                for microbatch_lengths in shard.extra_info[MICRO_BATCH_LENGTHS]
            }
        )
        == 1
    )


def test_teacher_worker_presharded_entrypoint_writes_teacher_tq_field():
    """The worker consumes its TQ shard and writes only the teacher delta."""
    from nemo_rl.data_plane.worker_mixin import TQWorkerMixin

    class Worker(TQWorkerMixin):
        cfg = {"sequence_packing": {"enabled": False}}

        def __init__(self):
            self.written = None
            self.received_batching_metadata = None

        def _fetch(self, meta):
            del meta
            return BatchedDataDict(
                {
                    "input_ids": torch.ones(1, 3, dtype=torch.long),
                    "input_lengths": torch.tensor([3]),
                }
            )

        def get_logprobs(self, data, micro_batch_size=None):
            del micro_batch_size
            self.received_batching_metadata = (
                data.micro_batch_indices,
                data.micro_batch_lengths,
            )
            return BatchedDataDict({"logprobs": torch.full((1, 3), 0.25)})

        def _write_back_result_field(self, meta, result, *, result_key, tq_field):
            self.written = (meta, result[result_key], tq_field)

    meta = KVBatchMeta(
        partition_id="rollout_data",
        task_name="teacher_lp:teacher",
        sample_ids=["a"],
        fields=["input_ids", "input_lengths"],
        sequence_lengths=[3],
        extra_info={
            MICRO_BATCH_INDICES: [[0]],
            MICRO_BATCH_LENGTHS: [3],
        },
    )
    worker = Worker()

    worker.get_teacher_logprobs_presharded(meta)

    assert worker.written is not None
    assert worker.written[0] is meta
    assert torch.allclose(worker.written[1], torch.full((1, 3), 0.25))
    assert worker.written[2] == "teacher_reference_logprobs"
    assert worker.received_batching_metadata == ([[0]], [3])


def test_teacher_worker_rejects_local_dynamic_batch_planning():
    """A missing driver plan fails instead of independently repacking each DP rank."""
    from nemo_rl.data_plane.worker_mixin import TQWorkerMixin

    class Worker(TQWorkerMixin):
        cfg = {
            "sequence_packing": {"enabled": False},
            "dynamic_batching": {
                "enabled": True,
                "sequence_length_round": 1,
                "train_mb_tokens": 8,
            },
        }

        def _fetch(self, meta):
            del meta
            return BatchedDataDict(
                {
                    "input_ids": torch.ones(1, 3, dtype=torch.long),
                    "input_lengths": torch.tensor([3]),
                }
            )

    meta = KVBatchMeta(
        partition_id="rollout_data",
        task_name="teacher_lp:teacher",
        sample_ids=["a"],
        fields=["input_ids", "input_lengths"],
        sequence_lengths=[3],
    )

    with pytest.raises(RuntimeError, match="driver-provided global"):
        Worker().get_teacher_logprobs_presharded(meta)


def test_get_logprobs_from_meta_forwards_the_opd_full_payload_request():
    """With full-vocabulary MOPD on, every DP rank is told what to emit and where."""
    from nemo_rl.models.policy.teacher_worker_group import TeacherWorkerGroup

    class Sharding:
        def get_axis_size(self, axis):
            assert axis == "data_parallel"
            return 1

    worker_group = MagicMock()
    worker_group.run_all_workers_sharded_data.return_value = "futures"
    teacher = object.__new__(TeacherWorkerGroup)
    teacher._opd_full_payload = "hidden_states"
    teacher._opd_full_payload_dtype = "float16"
    teacher._opd_full_payload_field = OPD_FULL_HIDDEN_STATES_FIELD
    teacher._opd_full_teacher_index_field = OPD_FULL_TEACHER_INDEX_FIELD
    # Not 0: a hard-coded or defaulted index would still look right at 0.
    teacher.teacher_index = 2
    teacher.alias = "teacher"
    teacher.use_sequence_packing = False
    teacher.use_dynamic_batches = False
    teacher.sequence_length_pad_multiple = 1
    teacher.sharding_annotations = Sharding()
    teacher.worker_group = worker_group
    teacher._micro_batch_size = 2
    meta = KVBatchMeta(
        partition_id="rollout_data",
        task_name="train",
        sample_ids=["a"],
        fields=["input_ids", "input_lengths"],
        sequence_lengths=[3],
    )

    teacher.get_logprobs_from_meta(meta)

    common_kwargs = worker_group.run_all_workers_sharded_data.call_args.kwargs[
        "common_kwargs"
    ]
    assert common_kwargs == {
        "micro_batch_size": 2,
        "opd_full_payload": "hidden_states",
        "opd_full_payload_dtype": "float16",
        "opd_full_payload_field": OPD_FULL_HIDDEN_STATES_FIELD,
        # Multi-teacher routing: this group's own index rides every request so
        # the student can pick the matching LM head back on the train side.
        "opd_full_teacher_index": 2,
        "opd_full_teacher_index_field": OPD_FULL_TEACHER_INDEX_FIELD,
    }


def _full_payload_worker_class():
    from nemo_rl.data_plane.worker_mixin import TQWorkerMixin

    class Worker(TQWorkerMixin):
        cfg = {"sequence_packing": {"enabled": False}}

        def __init__(self, payload):
            self.payload = payload
            self.logprob_call = None
            self.full_payload_call = None
            self.stage_local_writes = []
            self.written = None

        def _fetch(self, meta):
            del meta
            return BatchedDataDict(
                {
                    "input_ids": torch.ones(1, 3, dtype=torch.long),
                    "input_lengths": torch.tensor([3]),
                }
            )

        def get_logprobs(self, data, micro_batch_size=None):
            self.logprob_call = (data, micro_batch_size)
            return BatchedDataDict({"logprobs": torch.full((1, 3), 0.25)})

        def get_logprobs_with_full_payload(
            self, *, data, payload, payload_dtype, micro_batch_size=None
        ):
            del data
            self.full_payload_call = (payload, payload_dtype, micro_batch_size)
            return BatchedDataDict(
                {
                    "logprobs": torch.full((1, 3), 0.5),
                    "teacher_full_payload": self.payload,
                }
            )

        def _write_back_stage_local(self, meta, fields):
            self.stage_local_writes.append((meta, fields))

        def _write_back_result_field(self, meta, result, *, result_key, tq_field):
            self.written = (meta, result[result_key], tq_field)

    return Worker


def _presharded_meta():
    return KVBatchMeta(
        partition_id="rollout_data",
        task_name="teacher_lp:teacher",
        sample_ids=["a"],
        fields=["input_ids", "input_lengths"],
        sequence_lengths=[3],
        extra_info={MICRO_BATCH_INDICES: [[0]], MICRO_BATCH_LENGTHS: [3]},
    )


def test_teacher_worker_presharded_entrypoint_writes_the_full_payload_from_its_stage():
    """The payload goes through the stage-local writer; logprobs keep their path."""
    payload = torch.randn(1, 3, 4)
    worker = _full_payload_worker_class()(payload)
    meta = _presharded_meta()

    worker.get_teacher_logprobs_presharded(
        meta,
        micro_batch_size=1,
        opd_full_payload="hidden_states",
        # Not the config default: the driver's value must win.
        opd_full_payload_dtype="float16",
        opd_full_payload_field=OPD_FULL_HIDDEN_STATES_FIELD,
    )

    # The single forward replaced get_logprobs; it received the payload request.
    assert worker.logprob_call is None
    assert worker.full_payload_call == ("hidden_states", "float16", 1)
    # Sampled-token logprobs still land in the teacher column.
    assert worker.written is not None
    assert torch.allclose(worker.written[1], torch.full((1, 3), 0.5))
    assert worker.written[2] == "teacher_reference_logprobs"
    # The payload is written from this stage under the configured column.
    assert len(worker.stage_local_writes) == 1
    write_meta, fields = worker.stage_local_writes[0]
    assert write_meta is meta
    assert list(fields) == [OPD_FULL_HIDDEN_STATES_FIELD]
    assert torch.equal(fields[OPD_FULL_HIDDEN_STATES_FIELD], payload)


def test_teacher_worker_presharded_entrypoint_tags_every_row_with_its_teacher():
    """The routing column is written from the same stage as the payload.

    One ``TeacherWorkerGroup`` per checkpoint, so the tag is a constant
    broadcast over this call's batch dim -- but it has to be *this* group's
    index, in the payload's own stage-local write, or the student projects the
    rows through some other teacher's LM head.
    """
    payload = torch.randn(2, 3, 4)
    worker = _full_payload_worker_class()(payload)

    worker.get_teacher_logprobs_presharded(
        _presharded_meta(),
        opd_full_payload="hidden_states",
        opd_full_payload_dtype="bfloat16",
        opd_full_payload_field=OPD_FULL_HIDDEN_STATES_FIELD,
        # Not 0: a defaulted or dropped index still looks right at 0.
        opd_full_teacher_index=2,
        opd_full_teacher_index_field=OPD_FULL_TEACHER_INDEX_FIELD,
    )

    assert len(worker.stage_local_writes) == 1
    _, fields = worker.stage_local_writes[0]
    assert list(fields) == [OPD_FULL_HIDDEN_STATES_FIELD, OPD_FULL_TEACHER_INDEX_FIELD]
    tags = fields[OPD_FULL_TEACHER_INDEX_FIELD]
    # One int per sample, not per token: [B], never [B, S].
    assert tags.shape == (payload.shape[0],)
    assert tags.dtype == torch.int64
    assert torch.equal(tags, torch.tensor([2, 2]))


def test_teacher_worker_presharded_entrypoint_rejects_an_index_column_without_an_index():
    """A configured routing column with no index would tag every row as teacher 0.

    Guarded on every rank before the forward runs, so nothing is computed or
    written.
    """
    worker = _full_payload_worker_class()(torch.randn(2, 3, 4))

    with pytest.raises(ValueError, match="requires opd_full_teacher_index"):
        worker.get_teacher_logprobs_presharded(
            _presharded_meta(),
            opd_full_payload="hidden_states",
            opd_full_payload_dtype="bfloat16",
            opd_full_payload_field=OPD_FULL_HIDDEN_STATES_FIELD,
            opd_full_teacher_index=None,
            opd_full_teacher_index_field=OPD_FULL_TEACHER_INDEX_FIELD,
        )

    assert worker.full_payload_call is None
    assert worker.stage_local_writes == []


def test_teacher_worker_presharded_entrypoint_skips_the_payload_off_the_last_stage():
    """A stage that produced no payload must not write an empty column."""
    worker = _full_payload_worker_class()(None)

    worker.get_teacher_logprobs_presharded(
        _presharded_meta(),
        opd_full_payload="hidden_states",
        opd_full_payload_dtype="bfloat16",
        opd_full_payload_field=OPD_FULL_HIDDEN_STATES_FIELD,
    )

    assert worker.stage_local_writes == []
    assert worker.written is not None


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
    logged = opd.model_dump()["non_colocated_teachers"]
    assert logged["default_teacher_cfg"]["precision"] == cfg.precision == "bf16"
    assert (
        logged["default_teacher_cfg"]["tensor_model_parallel_size"]
        == cfg.tensor_model_parallel_size
        == 1
    )


def test_teacher_megatron_override_dictionary_replaces_default():
    from nemo_rl.algorithms.opd import OnPolicyDistillationConfig, _opd_cfg
    from nemo_rl.models.policy.teacher_worker_group import (
        create_teacher_configs_from_opd_config,
    )

    opd = OnPolicyDistillationConfig(
        teacher_model_by_agent_name={"teacher": "/ckpt/teacher"},
        non_colocated_teachers={
            "default_teacher_cfg": {
                "megatron_cfg_overrides": {
                    "mtp_num_layers": 5,
                    "mtp_detach_heads": True,
                },
                "freeze_vision_model": True,
                "freeze_language_model": True,
            },
            "teacher_overrides": {
                "teacher": {
                    "megatron_cfg_overrides": {"mtp_num_layers": 0},
                    "freeze_vision_model": False,
                }
            },
        },
    )
    (cfg,) = create_teacher_configs_from_opd_config(
        _opd_cfg({"on_policy_distillation": opd})
    )
    assert cfg.megatron_cfg_overrides == {
        "mtp_num_layers": 0,
        "freeze_vision_model": False,
        "freeze_language_model": True,
    }


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


# ── W2S-OPD ────────────────────────────────────────────────────────────────


def _w2s_dispatch_teacher():
    from nemo_rl.models.policy.teacher_worker_group import TeacherWorkerGroup

    class Sharding:
        def get_axis_size(self, axis):
            assert axis == "data_parallel"
            return 1

    teacher = object.__new__(TeacherWorkerGroup)
    _disable_opd_full(teacher)
    teacher._opd_full_payload = "hidden_states"
    teacher._opd_full_payload_field = OPD_FULL_HIDDEN_STATES_FIELD
    teacher.alias = "teacher"
    teacher.use_sequence_packing = False
    teacher.use_dynamic_batches = False
    teacher.sequence_length_pad_multiple = 1
    teacher.sharding_annotations = Sharding()
    teacher.worker_group = MagicMock()
    teacher._micro_batch_size = 1
    return teacher


def _w2s_meta():
    return KVBatchMeta(
        partition_id="rollout_data",
        task_name="train",
        sample_ids=["a"],
        fields=["input_ids", "input_lengths"],
        sequence_lengths=[3],
    )


def test_w2s_arm_pass_writes_its_hidden_states_to_the_arm_column():
    teacher = _w2s_dispatch_teacher()

    teacher.get_logprobs_from_meta(_w2s_meta(), payload_field="arm_column")

    kwargs = teacher.worker_group.run_all_workers_sharded_data.call_args.kwargs
    assert kwargs["common_kwargs"]["opd_full_payload_field"] == "arm_column"
    assert "w2s_proxy" not in kwargs["common_kwargs"]


def test_w2s_proxy_pass_forwards_the_arm_rows_to_every_worker():
    teacher = _w2s_dispatch_teacher()
    arm_fields = W2S_ARM_FIELDS

    teacher.get_w2s_proxy_from_meta(
        _w2s_meta(), arm_sample_ids={"a": "a__arms"}, arm_fields=arm_fields
    )

    kwargs = teacher.worker_group.run_all_workers_sharded_data.call_args.kwargs
    assert kwargs["common_kwargs"]["w2s_proxy"] == {
        "arm_sample_ids": {"a": "a__arms"},
        "arm_fields": arm_fields,
    }
    # The default payload column is untouched: the proxy pass writes the support.
    assert kwargs["common_kwargs"]["opd_full_payload_field"] == (
        OPD_FULL_HIDDEN_STATES_FIELD
    )
    with pytest.raises(ValueError, match="lack arm rows"):
        teacher.get_w2s_proxy_from_meta(
            _w2s_meta(), arm_sample_ids={}, arm_fields=arm_fields
        )


@pytest.mark.parametrize("writer", [True, False])
def test_teacher_worker_presharded_entrypoint_writes_the_w2s_proxy_support(writer):
    """Only the rank holding the proxy support writes it; logprobs keep their path."""
    support = torch.zeros(1, 3, 4, dtype=torch.int32)
    target = torch.full((1, 3, 4), -1.0)
    proxy_calls = []

    class Worker(_full_payload_worker_class()):
        def get_w2s_proxy_logprobs(self, *, data, fetch_arm_rows, micro_batch_size):
            del data
            proxy_calls.append((fetch_arm_rows, micro_batch_size))
            result = {"logprobs": torch.full((1, 3), 0.75)}
            if writer:
                result[W2S_PROXY_SUPPORT_IDS_FIELD] = support
                result[W2S_PROXY_TARGET_LOGPROBS_FIELD] = target
            return BatchedDataDict(result)

    worker = Worker(None)
    worker.get_teacher_logprobs_presharded(
        _presharded_meta(),
        micro_batch_size=1,
        opd_full_payload="hidden_states",
        opd_full_payload_dtype="bfloat16",
        opd_full_payload_field=OPD_FULL_HIDDEN_STATES_FIELD,
        w2s_proxy={"arm_sample_ids": {"a": "a__arms"}, "arm_fields": ("x", "y")},
    )

    assert len(proxy_calls) == 1 and callable(proxy_calls[0][0])
    assert worker.full_payload_call is None and worker.logprob_call is None
    assert worker.written[2] == "teacher_reference_logprobs"
    assert torch.allclose(worker.written[1], torch.full((1, 3), 0.75))
    if writer:
        assert len(worker.stage_local_writes) == 1
        _, fields = worker.stage_local_writes[0]
        assert list(fields) == list(W2S_PROXY_FIELDS)
    else:
        assert worker.stage_local_writes == []


def test_fetch_w2s_arm_rows_reads_each_rows_temporary_arm_row(monkeypatch):
    import nemo_rl.data_plane.worker_mixin as worker_mixin_module
    from nemo_rl.data_plane.worker_mixin import TQWorkerMixin

    requested = {}

    class Client:
        def get_samples(self, *, sample_ids, partition_id, select_fields):
            requested.update(
                sample_ids=sample_ids,
                partition_id=partition_id,
                select_fields=select_fields,
            )
            return "td"

    anchor = [torch.ones(3, 2), torch.ones(2, 2)]
    positive = [torch.ones(3, 5), torch.ones(2, 5)]

    def materialize(td, **kwargs):
        assert td == "td" and kwargs["layout"] == "jagged"
        return {
            "anchor": torch.nested.nested_tensor(anchor, layout=torch.jagged),
            "positive": torch.nested.nested_tensor(positive, layout=torch.jagged),
        }

    monkeypatch.setattr(worker_mixin_module, "_materialize_fetched", materialize)
    worker = object.__new__(TQWorkerMixin)
    worker._dp_client = Client()
    meta = KVBatchMeta(
        partition_id="rollout_data",
        task_name="teacher_lp:negative",
        sample_ids=["b", "a"],
        fields=["input_ids", "input_lengths"],
        sequence_lengths=[3, 2],
    )

    rows = worker._fetch_w2s_arm_rows(
        meta,
        {
            "arm_sample_ids": {"a": "a__arms", "b": "b__arms"},
            "arm_fields": ("anchor", "positive"),
        },
    )

    # Arm rows come back in the canonical rows' order, one (anchor, positive) each.
    assert requested == {
        "sample_ids": ["b__arms", "a__arms"],
        "partition_id": "rollout_data",
        "select_fields": ["anchor", "positive"],
    }
    assert [tuple(t.shape for t in row) for row in rows] == [
        ((3, 2), (3, 5)),
        ((2, 2), (2, 5)),
    ]
    with pytest.raises(ValueError, match="exactly two arm columns"):
        worker._fetch_w2s_arm_rows(
            meta, {"arm_sample_ids": {}, "arm_fields": ("anchor",)}
        )
