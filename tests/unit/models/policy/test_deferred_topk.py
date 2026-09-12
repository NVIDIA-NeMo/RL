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
from typing import Any, Optional
from unittest.mock import MagicMock

import pytest
import ray
import torch

from nemo_rl.distributed.batched_data_dict import BatchedDataDict, SlicedDataDict
from nemo_rl.models.policy.deferred import (
    DeferredTopkWorkerResult,
    attach_deferred_topk_logits,
)
from nemo_rl.models.policy.lm_policy import Policy
from nemo_rl.models.policy.workers.base_policy_worker import AbstractPolicyWorker


def test_attach_deferred_topk_logits() -> None:
    data = BatchedDataDict({"input_ids": torch.ones(2, 4, dtype=torch.long)})
    teacher_result = {
        "topk_logits": torch.randn(2, 4, 3),
        "topk_indices": torch.randint(0, 10, (2, 4, 3)),
    }

    attach_deferred_topk_logits(data, teacher_result)

    assert data["teacher_topk_logits"] is teacher_result["topk_logits"]
    assert data["teacher_topk_indices"] is teacher_result["topk_indices"]


@pytest.mark.parametrize(
    ("logits_shape", "indices_shape", "error"),
    [
        ((2, 4, 3), (2, 4, 2), "shape mismatch"),
        ((1, 4, 3), (1, 4, 3), "different batch sizes"),
    ],
)
def test_attach_deferred_topk_logits_validates_shapes(
    logits_shape: tuple[int, ...],
    indices_shape: tuple[int, ...],
    error: str,
) -> None:
    data = BatchedDataDict({"input_ids": torch.ones(2, 4, dtype=torch.long)})
    teacher_result = {
        "topk_logits": torch.randn(logits_shape),
        "topk_indices": torch.randint(0, 10, indices_shape),
    }

    with pytest.raises(ValueError, match=error):
        attach_deferred_topk_logits(data, teacher_result)


def test_attach_deferred_topk_logits_rejects_materialized_fields() -> None:
    data = BatchedDataDict(
        {
            "input_ids": torch.ones(2, 4, dtype=torch.long),
            "teacher_topk_logits": torch.randn(2, 4, 3),
        }
    )
    teacher_result = {
        "topk_logits": torch.randn(2, 4, 3),
        "topk_indices": torch.randint(0, 10, (2, 4, 3)),
    }

    with pytest.raises(ValueError, match="cannot be combined"):
        attach_deferred_topk_logits(data, teacher_result)


def _make_policy_for_deferred_check(
    *,
    dp_size: int = 1,
    dynamic: bool = False,
    sequence_packing: bool = False,
    token_budget: int = 128,
) -> Policy:
    policy = object.__new__(Policy)
    policy.sharding_annotations = MagicMock()
    policy.sharding_annotations.get_axis_size.return_value = dp_size
    policy.use_dynamic_batches = dynamic
    policy.use_sequence_packing = sequence_packing
    policy.sequence_packing_args = {
        "input_key": "input_ids",
        "input_lengths_key": "input_lengths",
        "algorithm": "modified_first_fit_decreasing",
        "sequence_length_pad_multiple": 8,
        "max_tokens_per_microbatch": token_budget,
    }
    policy.cfg = {
        "sequence_packing": {
            "train_mb_tokens": token_budget,
            "logprob_mb_tokens": token_budget,
        }
    }
    return policy


def test_deferred_topk_requires_matching_dp_layout() -> None:
    data = BatchedDataDict({"input_ids": torch.ones(4, 8, dtype=torch.long)})
    student = _make_policy_for_deferred_check(dp_size=2)
    teacher = _make_policy_for_deferred_check(dp_size=2)

    assert student.can_consume_deferred_topk_from(teacher, data, batch_size=4)

    teacher.sharding_annotations.get_axis_size.return_value = 1
    assert not student.can_consume_deferred_topk_from(teacher, data, batch_size=4)
    teacher.sharding_annotations.get_axis_size.return_value = 2
    teacher.use_dynamic_batches = True
    assert not student.can_consume_deferred_topk_from(teacher, data, batch_size=4)


def test_deferred_topk_requires_matching_sequence_packing() -> None:
    data = BatchedDataDict({"input_ids": torch.ones(4, 8, dtype=torch.long)})
    student = _make_policy_for_deferred_check(sequence_packing=True)
    teacher = _make_policy_for_deferred_check(sequence_packing=True)

    assert student.can_consume_deferred_topk_from(teacher, data, batch_size=4)

    teacher.cfg["sequence_packing"]["logprob_mb_tokens"] = 256
    assert not student.can_consume_deferred_topk_from(teacher, data, batch_size=4)


def test_get_topk_logits_deferred_keeps_worker_results_as_refs() -> None:
    policy = _make_policy_for_deferred_check(dp_size=2)
    policy.worker_group = MagicMock()
    future_bundle = MagicMock()
    policy.worker_group.run_all_workers_sharded_data.return_value = future_bundle
    refs = [ray.put({"topk_logits": 0}), ray.put({"topk_logits": 1})]
    policy.worker_group.get_all_worker_results.return_value = [
        DeferredTopkWorkerResult(payload_ref=ref) for ref in refs
    ]
    shards = [
        SlicedDataDict({"input_ids": torch.ones(2, 8, dtype=torch.long)}),
        SlicedDataDict({"input_ids": torch.ones(2, 8, dtype=torch.long)}),
    ]
    policy._shard_for_logprob = MagicMock(  # type: ignore[method-assign]
        return_value=(shards, [2, 0, 3, 1])
    )
    data = BatchedDataDict({"input_ids": torch.ones(4, 8, dtype=torch.long)})

    result = policy.get_topk_logits_deferred(data, k=3)

    assert result.refs == refs
    assert result.global_indices_per_dp == [[2, 0], [3, 1]]
    policy.worker_group.get_all_worker_results.assert_called_once_with(future_bundle)


def test_policy_worker_defers_topk_payload_with_ray_put(monkeypatch) -> None:
    class TestPolicyWorker(AbstractPolicyWorker):
        def get_topk_logits(
            self,
            *,
            data: BatchedDataDict[Any],
            k: int,
            micro_batch_size: Optional[int] = None,
        ) -> BatchedDataDict[Any]:
            del k, micro_batch_size
            return data

    data = BatchedDataDict({"input_ids": torch.ones(2, 4, dtype=torch.long)})
    payload_ref = MagicMock(spec=ray.ObjectRef)
    put = MagicMock(return_value=payload_ref)
    monkeypatch.setattr(ray, "put", put)

    result = TestPolicyWorker().get_topk_logits_deferred(data=data, k=3)

    assert result.payload_ref is payload_ref
    put.assert_called_once_with(data)
