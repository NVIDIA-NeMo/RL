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
from unittest.mock import Mock

import pytest
import torch

from nemo_rl.algorithms.loss import ClippedPGLossConfig, ClippedPGLossFn


@pytest.mark.mcore
@pytest.mark.parametrize("is_last_stage", [False, True])
@pytest.mark.parametrize("eval_mode", [False, True])
def test_survivor_normalization_reduces_dp_counts_on_every_pipeline_stage(
    monkeypatch, is_last_stage, eval_mode
):
    # Optional Megatron imports are available only in the mcore test environment.
    from nemo_rl.models.policy.workers import megatron_policy_worker as worker_module

    group = object()
    monkeypatch.setattr(
        worker_module.parallel_state, "get_data_parallel_group", lambda: group
    )
    monkeypatch.setattr(
        worker_module.parallel_state,
        "is_pipeline_last_stage",
        lambda **_: is_last_stage,
    )
    monkeypatch.setattr(
        worker_module.parallel_state,
        "get_pipeline_model_parallel_world_size",
        lambda: 2,
    )
    monkeypatch.setattr(
        worker_module.parallel_state, "get_pipeline_model_parallel_last_rank", lambda: 7
    )
    pp_group = object()
    monkeypatch.setattr(
        worker_module.parallel_state,
        "get_pipeline_model_parallel_group",
        lambda: pp_group,
    )
    metrics = [
        {
            "seq_logprob_error_valid_seqs": 1.0,
            "seq_logprob_error_valid_tokens": 3.0,
            "loss": -0.3,
            "is_oob_ratio": 0.0,
            "num_masked_seqs_by_logprob_error": 1.0,
        },
        {
            "seq_logprob_error_valid_seqs": 1.0,
            "seq_logprob_error_valid_tokens": 1.0,
            "loss": -0.1,
            "is_oob_ratio": 0.25,
            "num_masked_seqs_by_logprob_error": 0.0,
        },
    ]

    def reduce_counts(counts, **kwargs):
        assert is_last_stage
        assert kwargs["group"] is group
        torch.testing.assert_close(
            counts, torch.tensor([2.0, 4.0], dtype=torch.float64)
        )
        # Another DP rank contributes one surviving sequence with two tokens.
        counts.add_(torch.tensor([1.0, 2.0]))

    def broadcast_counts(counts, **kwargs):
        assert kwargs == {"src": 7, "group": pp_group}
        if is_last_stage:
            assert counts.tolist() == [3.0, 6.0]
        else:
            assert counts.tolist() == [0.0, 0.0]
            counts.copy_(torch.tensor([3.0, 6.0]))

    monkeypatch.setattr(torch.distributed, "all_reduce", reduce_counts)
    monkeypatch.setattr(torch.distributed, "broadcast", broadcast_counts)
    synchronize = Mock()
    monkeypatch.setattr(torch.cuda, "synchronize", synchronize)
    model = Mock()
    loss_fn = ClippedPGLossFn(
        ClippedPGLossConfig(
            force_on_policy_ratio=True,
            reference_policy_kl_penalty=0,
            use_importance_sampling_correction=True,
            truncated_importance_sampling_type="seq-mask-tis",
            truncated_importance_sampling_ratio_min=0.999,
            truncated_importance_sampling_ratio=1.002,
        ),
        seq_logprob_error_threshold=2.0,
    )
    result, sequences, tokens = (
        worker_module.MegatronPolicyWorkerImpl._normalize_in_loss_seq_filter(
            SimpleNamespace(model=model),
            loss_fn,
            metrics if is_last_stage else [],
            global_valid_seqs=torch.tensor(4.0),
            global_valid_toks=torch.tensor(9.0),
            eval_mode=eval_mode,
        )
    )
    assert sequences.item() == 3
    assert tokens.item() == 6
    if is_last_stage:
        assert sum(m["loss"] for m in result) == pytest.approx(-0.6)
        assert sum(m["is_oob_ratio"] for m in result) == pytest.approx(1 / 3)
        assert sum(m["num_masked_seqs_by_logprob_error"] for m in result) == 1
    else:
        assert result == []
    if eval_mode:
        model.scale_gradients.assert_not_called()
        synchronize.assert_not_called()
    else:
        synchronize.assert_called_once()
        model.scale_gradients.assert_called_once_with(1.5)


@pytest.mark.mcore
def test_empty_survivor_batch_fails_before_gradient_scaling(monkeypatch):
    # Optional Megatron imports are available only in the mcore test environment.
    from nemo_rl.models.policy.workers import megatron_policy_worker as worker_module

    monkeypatch.setattr(
        worker_module.parallel_state, "get_data_parallel_group", lambda: None
    )
    monkeypatch.setattr(
        worker_module.parallel_state, "is_pipeline_last_stage", lambda **_: True
    )
    monkeypatch.setattr(
        worker_module.parallel_state,
        "get_pipeline_model_parallel_world_size",
        lambda: 1,
    )
    monkeypatch.setattr(torch.distributed, "all_reduce", lambda *_a, **_kw: None)
    worker = SimpleNamespace(model=Mock(), optimizer=Mock(), scheduler=Mock())
    with pytest.raises(RuntimeError, match="No valid response tokens"):
        worker_module.MegatronPolicyWorkerImpl._normalize_in_loss_seq_filter(
            worker,
            Mock(),
            [
                {
                    "seq_logprob_error_valid_seqs": 0.0,
                    "seq_logprob_error_valid_tokens": 0.0,
                }
            ],
            global_valid_seqs=torch.tensor(2.0),
            global_valid_toks=torch.tensor(8.0),
            eval_mode=False,
        )
    worker.model.scale_gradients.assert_not_called()
    worker.optimizer.step.assert_not_called()
    worker.scheduler.step.assert_not_called()
