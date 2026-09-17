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
@pytest.mark.parametrize("sample_weight", [1.0, 1 / 16])
def test_survivor_normalization_reduces_dp_counts_on_every_pipeline_stage(
    monkeypatch: pytest.MonkeyPatch,
    is_last_stage: bool,
    eval_mode: bool,
    sample_weight: float,
) -> None:
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

    # Scaling every sample weight preserves normalized losses and gradients,
    # even when the surviving token and sequence weights are less than one.
    for metric in metrics:
        for key in (
            "seq_logprob_error_valid_seqs",
            "seq_logprob_error_valid_tokens",
            "num_masked_seqs_by_logprob_error",
        ):
            metric[key] *= sample_weight

    def reduce_counts(counts, **kwargs):
        assert is_last_stage
        assert kwargs["group"] is group
        torch.testing.assert_close(
            counts, torch.tensor([2.0, 4.0], dtype=torch.float64) * sample_weight
        )
        # Another DP rank contributes one surviving sequence with two tokens.
        counts.add_(torch.tensor([1.0, 2.0]) * sample_weight)

    def broadcast_counts(counts, **kwargs):
        assert kwargs == {"src": 7, "group": pp_group}
        if is_last_stage:
            assert counts.tolist() == [3.0 * sample_weight, 6.0 * sample_weight]
        else:
            assert counts.tolist() == [0.0, 0.0]
            counts.copy_(torch.tensor([3.0, 6.0]) * sample_weight)

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
            global_valid_seqs=torch.tensor(4.0 * sample_weight),
            global_valid_toks=torch.tensor(9.0 * sample_weight),
            eval_mode=eval_mode,
        )
    )
    assert sequences.item() == 3 * sample_weight
    assert tokens.item() == 6 * sample_weight
    if is_last_stage:
        assert sum(m["loss"] for m in result) == pytest.approx(-0.6)
        assert sum(m["is_oob_ratio"] for m in result) == pytest.approx(1 / 3)
        assert (
            sum(m["num_masked_seqs_by_logprob_error"] for m in result) == sample_weight
        )
    else:
        assert result == []
    if eval_mode:
        model.scale_gradients.assert_not_called()
        synchronize.assert_not_called()
    else:
        synchronize.assert_called_once()
        model.scale_gradients.assert_called_once_with(1.5)


@pytest.mark.mcore
@pytest.mark.parametrize("eval_mode", [False, True])
def test_empty_survivor_batch_never_scales_gradients(
    monkeypatch: pytest.MonkeyPatch, eval_mode: bool
) -> None:
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
    metrics = [
        {
            "seq_logprob_error_valid_seqs": 0.0,
            "seq_logprob_error_valid_tokens": 0.0,
            "loss": 0.0,
        }
    ]
    loss_fn = ClippedPGLossFn(
        ClippedPGLossConfig(force_on_policy_ratio=True), seq_logprob_error_threshold=2.0
    )
    kwargs = dict(
        global_valid_seqs=torch.tensor(2.0),
        global_valid_toks=torch.tensor(8.0),
        eval_mode=eval_mode,
    )
    normalize = worker_module.MegatronPolicyWorkerImpl._normalize_in_loss_seq_filter
    if eval_mode:
        result, sequences, tokens = normalize(worker, loss_fn, metrics, **kwargs)
        assert sequences.item() == tokens.item() == 0
        assert result[0]["loss"] == 0.0
    else:
        with pytest.raises(RuntimeError, match="No valid response tokens"):
            normalize(worker, loss_fn, metrics, **kwargs)
    worker.model.scale_gradients.assert_not_called()
    worker.optimizer.step.assert_not_called()
    worker.scheduler.step.assert_not_called()


@pytest.mark.mcore
@pytest.mark.parametrize("sample_weight", [0.25, 0.5])
@pytest.mark.parametrize("kl_penalty", [0.0, 0.1])
def test_fractional_survivor_gradients_match_prefiltering(
    monkeypatch: pytest.MonkeyPatch, sample_weight: float, kl_penalty: float
) -> None:
    """Post-backward normalization preserves a fractional surviving token."""
    # Optional Megatron imports are available only in the mcore test environment.
    from nemo_rl.models.policy.workers import megatron_policy_worker as worker_module

    monkeypatch.setattr(
        worker_module.parallel_state, "is_pipeline_last_stage", lambda **_: True
    )
    monkeypatch.setattr(
        worker_module.parallel_state, "get_data_parallel_group", lambda: None
    )
    monkeypatch.setattr(
        worker_module.parallel_state,
        "get_pipeline_model_parallel_world_size",
        lambda: 1,
    )
    monkeypatch.setattr(torch.distributed, "all_reduce", lambda *_a, **_kw: None)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: None)
    cfg = ClippedPGLossConfig(
        force_on_policy_ratio=True, reference_policy_kl_penalty=kl_penalty
    )
    data = {
        "token_mask": torch.tensor([[0.0, 1.0], [0.0, 1.0]]),
        "sample_mask": torch.tensor([sample_weight, 1.0]),
        "generation_logprobs": torch.tensor([[0.0, -1.0], [0.0, -1.0]]),
        "reference_policy_logprobs": torch.full((2, 2), -1.2),
        "advantages": torch.ones(2, 2),
    }
    # The second row is rejected; the first leaves less than one token of weight.
    expected_lp = torch.tensor([[-1.0], [-4.0]], requires_grad=True)
    actual_lp = expected_lp.detach().clone().requires_grad_()
    expected_data = {**data, "sample_mask": torch.tensor([sample_weight, 0.0])}
    surviving_weight = torch.tensor(sample_weight)
    original_weight = data["sample_mask"].sum()
    expected_loss, _ = ClippedPGLossFn(cfg)(
        expected_lp, expected_data, surviving_weight, surviving_weight
    )
    expected_loss.backward()
    loss_fn = ClippedPGLossFn(cfg, seq_logprob_error_threshold=1.5)
    loss, metrics = loss_fn(actual_lp, data, original_weight, original_weight)
    loss.backward()
    model = Mock()
    model.scale_gradients.side_effect = lambda factor: actual_lp.grad.mul_(factor)
    normalized, sequences, tokens = (
        worker_module.MegatronPolicyWorkerImpl._normalize_in_loss_seq_filter(
            SimpleNamespace(model=model),
            loss_fn,
            [metrics],
            global_valid_seqs=original_weight,
            global_valid_toks=original_weight,
            eval_mode=False,
        )
    )
    assert sequences.item() == tokens.item() == sample_weight
    torch.testing.assert_close(actual_lp.grad, expected_lp.grad)
    assert normalized[0]["loss"] == pytest.approx(expected_loss.item())
