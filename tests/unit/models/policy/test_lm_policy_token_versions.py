# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
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
"""Driver dispatch and abort semantics for a multi-version optimizer step."""

from types import SimpleNamespace

import pytest
import torch

from nemo_rl.distributed.batched_data_dict import BatchedDataDict
from nemo_rl.models.policy.lm_policy import Policy


@pytest.mark.parametrize("fail_version", [None, 1])
def test_versions_share_one_optimizer_step_and_deduplicate_tp_metrics(
    monkeypatch, fail_version
):
    events = []

    class Workers:
        def run_all_workers_single_data(self, method, **kwargs):
            events.append(method)
            if method == "finish_train_step_presharded":
                return [
                    {
                        "is_replica_leader": rank % 2 == 0,
                        "global_loss": 3.0,
                        "grad_norm": 2.0,
                        "all_mb_metrics": {"tokens": [rank], "num_valid_samples": [3]},
                    }
                    for rank in range(4)
                ]
            return []

        def run_all_workers_sharded_data(
            self, method, *, data, common_kwargs, **kwargs
        ):
            assert method == "train_microbatch"
            version = common_kwargs["version_point"]
            events.append(version)
            assert len(data) == 2
            if version == fail_version:
                raise ValueError("pass failed")
            return SimpleNamespace(results=[None, None])

        def get_all_worker_results(self, bundle):
            return bundle.results

    monkeypatch.setattr("nemo_rl.models.policy.lm_policy.ray.get", lambda refs: refs)
    policy = SimpleNamespace(
        cfg={"train_global_batch_size": 4, "train_micro_batch_size": 1},
        data_parallel_size=2,
        flops_tracker=None,
        worker_group=Workers(),
        _shard_for_train=lambda data, batch_size: data.shard_by_batch_size(
            2, batch_size=batch_size
        ),
    )
    policy._aggregate_train_results = lambda results: Policy._aggregate_train_results(
        policy, results
    )
    data = BatchedDataDict(
        {
            "input_ids": torch.ones(4, 8, dtype=torch.long),
            "input_lengths": torch.full((4,), 8),
            "token_mask": torch.tensor([[0, 0, 1, 1, 1, 1, 1, 1]] * 4),
            "generation_versions": torch.tensor([[0, 0, 1, 1, 2, 2, 2, 2]] * 4),
            "sample_mask": torch.ones(4),
        }
    )
    if fail_version is not None:
        with pytest.raises(ValueError, match="pass failed"):
            Policy.train_token_versions(policy, data, None, [0, 1])
        assert events == ["begin_train_step_presharded", 0, 1, "abort_train_step"]
    else:
        result = Policy.train_token_versions(policy, data, None, [0, 1])
        assert events == [
            "begin_train_step_presharded",
            0,
            1,
            "finish_train_step_presharded",
        ]
        assert result["all_mb_metrics"]["tokens"] == [0, 2]
        assert result["all_mb_metrics"]["num_valid_samples"] == [4.0]


def test_multimodal_batch_is_rejected_before_opening_a_step():
    policy = SimpleNamespace(
        cfg={"train_global_batch_size": 1, "train_micro_batch_size": 1}
    )
    data = SimpleNamespace(get_multimodal_dict=lambda: {"images": object()})
    with pytest.raises(ValueError, match="text-only"):
        Policy.train_token_versions(policy, data, None, [0])
