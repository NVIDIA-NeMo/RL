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
from unittest.mock import MagicMock

import pytest
import torch
from pydantic import ValidationError

from nemo_rl.algorithms import opd_diagnostics as diagnostics
from nemo_rl.algorithms.opd import OnPolicyDistillationConfig


def _master_config(**overrides):
    return SimpleNamespace(
        on_policy_distillation=OnPolicyDistillationConfig(enabled=True, **overrides)
    )


def _payload_inputs():
    return {
        "step": 0,
        "num_generations_per_prompt": 2,
        "input_ids": torch.tensor([[10, 11, 12, 0], [20, 21, 0, 0]]),
        "teacher_logprobs": torch.tensor(
            [[0.0, -0.2, -0.4, 0.0], [0.0, -0.3, 0.0, 0.0]]
        ),
        "prev_logprobs": torch.tensor([[0.0, -0.7, -0.1, 0.0], [0.0, -0.8, 0.0, 0.0]]),
        "generation_logprobs": torch.tensor(
            [[0.0, -0.6, -0.2, 0.0], [0.0, -0.7, 0.0, 0.0]]
        ),
        "token_mask": torch.tensor(
            [[False, True, True, False], [False, True, False, False]]
        ),
        "sample_mask": torch.tensor([1.0, 0.0]),
        "rewards": torch.tensor([1.0, 0.0]),
        "input_lengths": torch.tensor([3, 2]),
        "repeated_batch": {
            "agent_ref": [{"name": "a"}, {"name": "b"}],
            "task_name": ["task-a", "task-b"],
        },
    }


def test_diagnostic_periods_are_one_based():
    config = _master_config(
        log_sample_stats=True,
        sample_stats_log_period=2,
        log_token_stats=True,
        token_stats_log_period=3,
        log_topk_stats=True,
        topk_stats_log_period=5,
    )

    assert not diagnostics._should_log_opd_sample_stats(config, 0)
    assert diagnostics._should_log_opd_sample_stats(config, 1)
    assert diagnostics._should_log_opd_token_stats(config, 2)
    assert diagnostics._should_log_opd_topk_stats(config, 4)


@pytest.mark.parametrize(
    "field,value",
    [
        ("sample_stats_log_period", 0),
        ("token_stats_log_period", 0),
        ("topk_stats_log_period", 0),
        ("topk_stats_k", 0),
        ("topk_stats_mode", "invalid"),
        ("sample_response_max_tokens", 0),
        ("topk_stats_max_tokens", 0),
    ],
)
def test_diagnostic_config_rejects_invalid_values(field, value):
    with pytest.raises(ValidationError):
        OnPolicyDistillationConfig(**{field: value})


def test_sample_and_token_payloads_preserve_raw_teacher_student_gaps():
    inputs = _payload_inputs()
    sample, sample_metrics = diagnostics._build_opd_sample_stats_log_data(
        tokenizer=None,
        log_sample_responses=False,
        sample_response_max_tokens=None,
        **inputs,
    )
    token, token_metrics = diagnostics._build_opd_token_stats_payload(**inputs)

    assert sample["step"] == [1, 1]
    assert sample["agent_ref"] == inputs["repeated_batch"]["agent_ref"]
    assert sample["teacher_student_logprob_gap_mean"] == pytest.approx([0.1, 0.5])
    assert sample_metrics["on_policy_distillation/sample_stats/logged_samples"] == 2
    assert token["format_version"] == 1
    assert token["step"] == 1
    torch.testing.assert_close(
        token["teacher_student_logprob_gap"], torch.tensor([0.5, -0.3, 0.5])
    )
    assert token_metrics["on_policy_distillation/token_stats/logged_tokens"] == 3


def test_payload_builders_reject_missing_or_misaligned_logprobs():
    inputs = _payload_inputs()
    inputs["teacher_logprobs"] = torch.zeros(2, 3)

    with pytest.raises(ValueError, match="Teacher logprobs shape"):
        diagnostics._build_opd_token_stats_payload(**inputs)


def test_next_token_topk_alignment_and_full_vocab_terms():
    next_token_ids = torch.tensor([[[5, 7], [6, 8], [9, 1]]])
    aligned = diagnostics._align_next_token_topk_to_input_positions(
        next_token_ids, target_seq_len=4, fill_value=-1
    )
    torch.testing.assert_close(
        aligned,
        torch.tensor([[[-1, -1], [5, 7], [6, 8], [9, 1]]]),
    )

    # Non-zero normalizer so the test exercises logits - V_logsumexp.
    shift = 1.5
    student_logits = torch.log(torch.tensor([[0.6, 0.2]])) + shift
    teacher_logits = torch.log(torch.tensor([[0.5, 0.3]])) + shift
    terms = diagnostics._compute_topk_full_vocab_terms_chunked(
        student_ids=torch.tensor([[1, 2]]),
        teacher_ids=torch.tensor([[1, 3]]),
        student_logits=student_logits,
        teacher_logits=teacher_logits,
        student_V_logsumexp=torch.tensor([shift]),
        teacher_V_logsumexp=torch.tensor([shift]),
    )

    assert terms["student_topk_head_prob_mass"].item() == pytest.approx(0.8)
    assert terms["teacher_topk_head_prob_mass"].item() == pytest.approx(0.8)
    assert terms["student_topk_mass_in_teacher_topk"].item() == pytest.approx(0.6)
    assert terms["teacher_topk_mass_in_student_topk"].item() == pytest.approx(0.5)


def test_opd_seq_error_logging_fields_match_seq_logprob_error_masking():
    """The per-sample OPD payload fields agree with the masking helper."""
    from nemo_rl.algorithms.grpo import compute_and_apply_seq_logprob_error_masking
    from nemo_rl.distributed.batched_data_dict import BatchedDataDict

    generation_logprobs = torch.zeros(4, 6)
    prev_logprobs = torch.zeros(4, 6)
    prev_logprobs[1, :] = -0.1
    prev_logprobs[2, :] = -0.5
    # Position 0 is never a response token; both helpers must ignore it.
    prev_logprobs[:, 0] = -5.0
    token_mask = torch.ones(4, 6)
    token_mask[:, 1] = 0.0
    token_mask[3] = 0.0
    train_data = BatchedDataDict(
        {
            "token_mask": token_mask,
            "sample_mask": torch.ones(4),
            "prev_logprobs": prev_logprobs,
            "generation_logprobs": generation_logprobs,
        }
    )
    rewards = torch.tensor([1.0, 0.0, 1.0, 0.0])
    pre_mask = train_data["sample_mask"].clone()

    legacy = compute_and_apply_seq_logprob_error_masking(
        train_data, rewards, seq_logprob_error_threshold=1.2
    )
    seq_error, masked = diagnostics._opd_seq_error_logging_fields(
        train_data, pre_mask, have_real_prev_logprobs=True
    )

    torch.testing.assert_close(seq_error[:3], torch.exp(torch.tensor([0.0, 0.1, 0.5])))
    assert seq_error[3] == 0.0
    assert seq_error[:3].max().item() == pytest.approx(
        legacy["max_seq_mult_prob_error"]
    )
    assert seq_error[:3].mean().item() == pytest.approx(
        legacy["mean_seq_mult_prob_error"]
    )
    assert seq_error[:3].min().item() == pytest.approx(
        legacy["min_seq_mult_prob_error"]
    )
    assert masked.tolist() == [False, False, True, False]
    assert int(masked.sum()) == legacy["num_masked_seqs"]
    assert torch.equal(masked, pre_mask.bool() & ~train_data["sample_mask"].bool())

    nan_error, no_mask = diagnostics._opd_seq_error_logging_fields(
        train_data, pre_mask, have_real_prev_logprobs=False
    )
    assert torch.isnan(nan_error).all()
    assert not no_mask.any()


@pytest.mark.parametrize(
    "opd_config",
    [
        None,
        {
            "enabled": False,
            "log_sample_stats": True,
            "log_token_stats": True,
            "log_topk_stats": True,
        },
        {"enabled": True},
    ],
)
def test_collect_opd_diagnostic_payloads_is_noop_unless_logging_requested(
    opd_config,
):
    from nemo_rl.distributed.batched_data_dict import BatchedDataDict

    master_config = SimpleNamespace(
        on_policy_distillation=(
            None if opd_config is None else OnPolicyDistillationConfig(**opd_config)
        ),
        grpo=SimpleNamespace(num_generations_per_prompt=2),
    )
    policy = MagicMock()
    trajectory_collector = MagicMock()
    train_data = BatchedDataDict(
        {
            "input_ids": torch.zeros(2, 4, dtype=torch.long),
            "token_mask": torch.ones(2, 4),
            "sample_mask": torch.ones(2),
            # These deliberately fail tensor math if the no-op gate regresses.
            "prev_logprobs": MagicMock(),
            "generation_logprobs": MagicMock(),
        }
    )

    payloads, metrics = diagnostics._collect_opd_diagnostic_payloads(
        master_config=master_config,
        step=0,
        tokenizer=None,
        policy=policy,
        trajectory_collector=trajectory_collector,
        train_data=train_data,
        repeated_batch=BatchedDataDict({"agent_ref": [{"name": "a"}, {"name": "b"}]}),
        rewards=torch.ones(2),
        input_lengths=torch.tensor([4, 4]),
        teacher_logprobs=torch.zeros(2, 4),
        fused_student_topk=None,
        pre_seq_error_sample_loss_mask=torch.ones(2),
        have_real_prev_logprobs=True,
        timer=MagicMock(),
    )

    assert payloads == {}
    assert metrics == {}
    policy.get_topk_logits.assert_not_called()
    trajectory_collector.compute_teacher_topk.remote.assert_not_called()
