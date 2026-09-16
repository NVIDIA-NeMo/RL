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

from unittest.mock import MagicMock

import pytest
import torch

from nemo_rl.data.packed_rollouts import PACKED_ATTENTION_SEGMENT_LENGTHS
from nemo_rl.distributed.batched_data_dict import BatchedDataDict
from nemo_rl.models.policy.lm_policy import Policy


@pytest.mark.parametrize(
    "megatron,packing,draft,error",
    [
        (False, True, False, "Megatron training backend"),
        (True, False, False, "sequence_packing.enabled=true"),
        (True, True, True, "policy.draft.enabled=true"),
        (True, True, False, None),
    ],
)
def test_tree_backend_guard(megatron, packing, draft, error):
    policy = Policy.__new__(Policy)
    policy.cfg = {
        "megatron_cfg": {"enabled": megatron},
        "sequence_packing": {"enabled": packing},
        "draft": {"enabled": draft},
    }
    if error is None:
        policy._validate_tree_execution()
    else:
        with pytest.raises(ValueError, match=error):
            policy._validate_tree_execution()


def _logical_rollout_batch() -> BatchedDataDict:
    return BatchedDataDict(
        {
            "input_ids": torch.tensor(
                [
                    [10, 11, 12, 20, 21],
                    [30, 31, 32, 33, 0],
                ]
            ),
            "input_lengths": torch.tensor([5, 4]),
            "token_mask": torch.tensor(
                [
                    [0, 1, 1, 0, 1],
                    [0, 1, 1, 1, 0],
                ]
            ),
            "sample_mask": torch.tensor([1.0, 0.5]),
            PACKED_ATTENTION_SEGMENT_LENGTHS: [[3, 2], [4]],
        }
    )


def _policy_shell() -> Policy:
    policy = Policy.__new__(Policy)
    policy.worker_group = MagicMock()
    policy.flops_tracker = None
    policy.debug_payload_metrics = False
    policy.cfg = {
        "train_global_batch_size": 2,
        "train_micro_batch_size": 1,
        "megatron_cfg": {"enabled": True},
    }
    return policy


def test_get_logprobs_reassembles_independent_call_rows() -> None:
    policy = _policy_shell()
    policy._shard_for_logprob = MagicMock(return_value=(["shard"], None))
    policy.worker_group.run_all_workers_sharded_data.return_value = "future"
    policy.worker_group.get_all_worker_results.return_value = [
        {
            "logprobs": torch.tensor(
                [
                    [0.0, 1.0, 2.0, 0.0],
                    [0.0, 3.0, 0.0, 0.0],
                    [0.0, 4.0, 5.0, 6.0],
                ]
            )
        }
    ]

    result = policy.get_logprobs(_logical_rollout_batch())

    expanded = policy._shard_for_logprob.call_args.args[0]
    assert expanded.size == 3
    assert PACKED_ATTENTION_SEGMENT_LENGTHS not in expanded
    assert policy._shard_for_logprob.call_args.kwargs == {"allow_uneven_shards": True}
    assert torch.equal(
        result["logprobs"],
        torch.tensor(
            [
                [0.0, 1.0, 2.0, 0.0, 3.0],
                [0.0, 4.0, 5.0, 6.0, 0.0],
            ]
        ),
    )


def test_get_reference_logprobs_reassembles_independent_call_rows() -> None:
    policy = _policy_shell()
    policy._shard_for_logprob = MagicMock(return_value=(["shard"], None))
    policy.worker_group.run_all_workers_sharded_data.return_value = "future"
    policy.worker_group.get_all_worker_results.return_value = [
        {
            "reference_logprobs": torch.tensor(
                [
                    [0.0, 1.0, 2.0, 0.0],
                    [0.0, 3.0, 0.0, 0.0],
                    [0.0, 4.0, 5.0, 6.0],
                ]
            )
        }
    ]

    result = policy.get_reference_policy_logprobs(_logical_rollout_batch())

    assert torch.equal(
        result["reference_logprobs"],
        torch.tensor(
            [
                [0.0, 1.0, 2.0, 0.0, 3.0],
                [0.0, 4.0, 5.0, 6.0, 0.0],
            ]
        ),
    )


def test_kv_calibration_expands_independent_call_rows() -> None:
    policy = _policy_shell()
    policy._shard_for_logprob = MagicMock(return_value=(["shard"], None))
    policy.worker_group.run_all_workers_sharded_data.return_value = "future"
    policy.worker_group.get_all_worker_results.return_value = [{"layers": {}}]

    assert policy.calibrate_qkv_fp8_scales(_logical_rollout_batch()) == {"layers": {}}

    expanded = policy._shard_for_logprob.call_args.args[0]
    assert expanded.size == 3
    assert policy._shard_for_logprob.call_args.kwargs == {"allow_uneven_shards": True}


def test_train_dispatches_all_calls_as_one_update() -> None:
    policy = _policy_shell()
    policy._shard_for_train = MagicMock(return_value=["shard"])
    policy.worker_group.run_all_workers_sharded_data.return_value = "future"
    policy.worker_group.get_all_worker_results.return_value = [
        {
            "global_loss": torch.tensor(1.0),
            "grad_norm": torch.tensor(2.0),
            "all_mb_metrics": {},
        }
    ]
    loss_fn = MagicMock()

    policy.train(_logical_rollout_batch(), loss_fn)

    expanded = policy._shard_for_train.call_args.args[0]
    assert expanded.size == 3
    assert policy._shard_for_train.call_args.args[1] is None
    assert policy._shard_for_train.call_args.kwargs == {"allow_uneven_shards": True}
    common_kwargs = policy.worker_group.run_all_workers_sharded_data.call_args.kwargs[
        "common_kwargs"
    ]
    assert common_kwargs == {
        "loss_fn": loss_fn,
        "eval_mode": False,
        "gbs": 3,
        "mbs": 1,
        "check_dim_skip_keys": None,
        "scheduler_step_increment": 2,
    }
