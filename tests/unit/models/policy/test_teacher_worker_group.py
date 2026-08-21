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

from nemo_rl.data_plane import KVBatchMeta
from nemo_rl.data_plane.schema import GLOBAL_FORWARD_PAD_SEQLEN
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
    teacher.alias = "teacher"
    teacher.use_sequence_packing = False
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
    assert all(
        shard.extra_info[GLOBAL_FORWARD_PAD_SEQLEN] == 6 for shard in kwargs["meta"]
    )
    worker_group.get_all_worker_results.assert_called_once_with("futures")


def test_teacher_worker_presharded_entrypoint_writes_teacher_tq_field():
    """The worker consumes its TQ shard and writes only the teacher delta."""
    from nemo_rl.data_plane.worker_mixin import TQWorkerMixin

    class Worker(TQWorkerMixin):
        cfg = {"sequence_packing": {"enabled": False}}

        def __init__(self):
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
            del data, micro_batch_size
            return BatchedDataDict({"logprobs": torch.full((1, 3), 0.25)})

        def _write_back_result_field(self, meta, result, *, result_key, tq_field):
            self.written = (meta, result[result_key], tq_field)

    meta = KVBatchMeta(
        partition_id="rollout_data",
        task_name="teacher_lp:teacher",
        sample_ids=["a"],
        fields=["input_ids", "input_lengths"],
        sequence_lengths=[3],
    )
    worker = Worker()

    worker.get_teacher_logprobs_presharded(meta)

    assert worker.written is not None
    assert worker.written[0] is meta
    assert torch.allclose(worker.written[1], torch.full((1, 3), 0.25))
    assert worker.written[2] == "teacher_reference_logprobs"
