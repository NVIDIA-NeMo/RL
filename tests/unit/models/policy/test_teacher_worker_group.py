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


def test_teacher_worker_group_disables_student_router_replay(monkeypatch):
    """Frozen teachers do not require rollout-to-training route consistency."""
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
        "megatron_cfg": {"enabled": True},
        "dtensor_cfg": {"enabled": False},
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
        megatron_cfg_overrides={},
    )

    teacher = TeacherWorkerGroup(
        teacher_config,
        cluster,
        policy_config,
        MagicMock(),
    )

    assert captured["cfg"]["router_replay"]["enabled"] is False
    assert teacher.cfg["router_replay"]["enabled"] is False
    assert policy_config["router_replay"]["enabled"] is True


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
        "dtensor_cfg": {"enabled": False},
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


def test_get_logprobs_from_meta_dispatches_tq_shards_to_teacher_workers():
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
        fields=["input_ids", "input_lengths"],
        sequence_lengths=[3, 5],
    )

    teacher.get_logprobs_from_meta(meta)

    call = worker_group.run_all_workers_sharded_data.call_args
    kwargs = call.kwargs
    assert call.args == ("get_teacher_logprobs_presharded",)
    assert [shard.sample_ids for shard in kwargs["meta"]] == [["a"], ["b"]]
    assert all(shard.fields == list(TEACHER_LP_FIELDS) for shard in kwargs["meta"])
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
