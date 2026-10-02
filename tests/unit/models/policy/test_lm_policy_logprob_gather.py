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

from types import SimpleNamespace

import pytest
import torch

from nemo_rl.distributed.batched_data_dict import BatchedDataDict
from nemo_rl.models.policy.lm_policy import Policy


class _WorkerGroup:
    """Records how Policy gathers results; returns one batch per DP shard."""

    def __init__(self, results):
        self.results = results
        self.gather_kwargs = []

    def run_all_workers_sharded_data(self, method_name, **kwargs):
        return "futures"

    def get_all_worker_results(self, futures, **kwargs):
        assert futures == "futures"
        self.gather_kwargs.append(kwargs)
        return self.results

    def shutdown(self, **_kwargs):
        return True


def _policy(worker_group, dp_size=2):
    policy = Policy.__new__(Policy)
    policy.worker_group = worker_group
    policy.sharding_annotations = SimpleNamespace(get_axis_size=lambda _axis: dp_size)
    policy.use_dynamic_batches = False
    policy.use_sequence_packing = False
    policy.debug_payload_metrics = False
    return policy


def _batch():
    return BatchedDataDict(
        {
            "input_ids": torch.zeros(2, 3, dtype=torch.long),
            "input_lengths": torch.tensor([3, 3]),
        }
    )


@pytest.mark.parametrize(
    ("method", "output_key"),
    [
        ("get_logprobs", "logprobs"),
        ("get_reference_policy_logprobs", "reference_logprobs"),
    ],
)
def test_logprob_gathers_fetch_only_returned_workers(method, output_key):
    """CP/TP/PP replicas hold identical [B, S] copies; only DP leaders are fetched."""
    worker_group = _WorkerGroup(
        [
            BatchedDataDict({output_key: torch.zeros(1, 3)}),
            BatchedDataDict({output_key: torch.ones(1, 3)}),
        ]
    )

    result = getattr(_policy(worker_group), method)(_batch())

    assert worker_group.gather_kwargs == [{"fetch_returned_only": True}]
    torch.testing.assert_close(result[output_key], torch.tensor([[0.0] * 3, [1.0] * 3]))


def test_train_gather_keeps_fetching_every_worker():
    """Small per-rank results keep the default gather, so any rank's error surfaces."""
    worker_group = _WorkerGroup(
        [{"global_loss": 0.0, "grad_norm": 0.0, "all_mb_metrics": {}}] * 2
    )
    policy = _policy(worker_group)
    policy.cfg = {"train_global_batch_size": 2, "train_micro_batch_size": 1}
    policy.flops_tracker = None

    policy.train(_batch(), loss_fn=None)

    assert worker_group.gather_kwargs == [{}]
