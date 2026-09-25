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

from nemo_rl.models.policy.lm_policy import Policy


def _policy(results, *, tracker=True):
    policy = Policy.__new__(Policy)
    policy.cfg = {"train_global_batch_size": 8, "train_micro_batch_size": 2}
    shard = {"input_lengths": torch.tensor([7, 11])}
    policy._shard_for_train = MagicMock(return_value=[shard, shard])
    policy._report_sharded_payload = MagicMock()
    policy.flops_tracker = MagicMock(total_flops=999.0) if tracker else None
    policy.worker_group = MagicMock()
    policy.worker_group.cluster.world_size.return_value = 4
    policy.worker_group.get_all_worker_results.return_value = results
    return policy


def _result(local_flops):
    return {
        "global_loss": 1.0,
        "grad_norm": 0.5,
        "all_mb_metrics": {"loss": [0.1]},
        "gpu_name": "NVIDIA H100 80GB HBM3",
        "model_dtype": torch.bfloat16,
        "local_flops": local_flops,
        "train_elapsed_seconds": 1.25,
    }


@pytest.mark.parametrize("tracker", [False, True])
def test_train_sums_dp_work_and_preserves_legacy_timer(tracker):
    policy = _policy([_result(11), _result(17)], tracker=tracker)
    metrics = policy.train(MagicMock(), loss_fn=MagicMock())
    assert metrics["total_flops"] == 28
    assert metrics["flops_from_bridge"] == 1.0
    assert metrics["num_ranks"] == 4
    # Two DP results represent four H100s, not just two GPUs.
    assert metrics["theoretical_tflops"] == pytest.approx(3958)
    assert metrics["all_mb_metrics"]["loss"] == [0.1, 0.1]
    if tracker:
        assert "train_elapsed_seconds" not in metrics
    else:
        assert metrics["train_elapsed_seconds"] == 1.25


def test_train_uses_complete_fallback_for_unsupported_shard():
    policy = _policy([_result(11), _result(None)])
    with pytest.warns(UserWarning, match="unsupported"):
        metrics = policy.train(MagicMock(), loss_fn=MagicMock())
    assert metrics["total_flops"] == 999.0
    assert metrics["flops_from_bridge"] == 0.0
    assert metrics["theoretical_tflops"] == pytest.approx(3958)


def test_train_omits_flops_when_no_calculator_supports_the_model():
    policy = _policy([_result(None), _result(None)], tracker=False)
    with pytest.warns(UserWarning, match="MFU omitted"):
        metrics = policy.train(MagicMock(), loss_fn=MagicMock())
    assert "total_flops" not in metrics
    assert "theoretical_tflops" not in metrics
