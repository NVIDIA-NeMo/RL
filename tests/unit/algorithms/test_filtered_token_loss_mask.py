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

import math
from unittest.mock import patch

import pytest
import torch

from nemo_rl.algorithms.advantage_estimator import (
    AdvEstimatorConfig,
    GAEConfig,
    GeneralizedAdvantageEstimator,
    OPDAdvantageEstimator,
    ReinforcePlusPlusAdvantageEstimator,
)
from nemo_rl.algorithms.logits_sampling_utils import TrainingSamplingParams
from nemo_rl.algorithms.loss.loss_functions import ClippedPGLossConfig, ClippedPGLossFn
from nemo_rl.algorithms.loss.loss_input import prepare_loss_input
from nemo_rl.distributed.batched_data_dict import BatchedDataDict


@pytest.mark.parametrize("sequence_level", [False, True])
@pytest.mark.parametrize("metrics_level", ["full", "minimal"])
@pytest.mark.parametrize("invalid_source", ["current", "previous", "generation", "all"])
def test_invalid_policy_tokens_match_explicit_actor_exclusion(
    sequence_level, invalid_source, metrics_level
):
    fn = ClippedPGLossFn(
        ClippedPGLossConfig(
            reference_policy_kl_penalty=0.0,
            metrics_level=metrics_level,
            sequence_level_importance_ratios=sequence_level,
            token_level_loss=not sequence_level,
            use_importance_sampling_correction=True,
        )
    )

    def run(explicit_mask):
        logits = torch.tensor(
            [[[0.0, 0.0, -5.0], [0.0, -5.0, 0.0], [0.0, 0.0, 0.0]]], requires_grad=True
        )
        if invalid_source != "current":
            logits = torch.tensor(
                [[[0.0, -5.0, 0.0], [0.0, -5.0, 0.0], [0.0, 0.0, 0.0]]],
                requires_grad=True,
            )
        previous = torch.tensor([[0.0, -math.log(2), -math.log(2)]])
        generation = previous.clone()
        if invalid_source in ("previous", "all"):
            previous[0, 1:] = -torch.inf if invalid_source == "all" else previous[0, 1:]
            previous[0, 1] = -torch.inf
        if invalid_source == "generation":
            generation[0, 1] = -torch.inf
        mask = torch.tensor([[0.0, 1.0, 1.0]])
        if explicit_mask:
            mask[0, 1] = 0
            if invalid_source == "all":
                mask[0, 2] = 0
        data = BatchedDataDict(
            {
                "input_ids": torch.tensor([[0, 2, 2]]),
                "token_mask": mask,
                "sample_mask": torch.ones(1),
                "advantages": torch.ones(1, 3),
                "prev_logprobs": previous,
                "generation_logprobs": generation,
            }
        )
        with patch.object(torch.Tensor, "cuda", lambda self, *args, **kwargs: self):
            inputs, data = prepare_loss_input(
                logits, data, fn, sampling_params=TrainingSamplingParams(top_k=2)
            )
        # Preserve the global rollout-batch denominator in both comparisons.
        loss, metrics = fn(
            data=data,
            global_valid_seqs=torch.tensor(1.0),
            global_valid_toks=torch.tensor(2.0),
            **inputs,
        )
        loss.backward()
        assert torch.isfinite(loss)
        assert torch.isfinite(logits.grad).all()
        excluded = metrics.pop("policy_support_excluded_tokens")
        assert excluded == (
            0 if explicit_mask else (2 if invalid_source == "all" else 1)
        )
        return loss.detach(), logits.grad, metrics

    actual = run(False)
    expected = run(True)
    for a, b in zip(actual, expected):
        torch.testing.assert_close(a, b, rtol=0, atol=0)
    if invalid_source != "all":
        assert actual[1][0, 1].abs().sum() > 0
    else:
        assert actual[1].count_nonzero() == 0


@pytest.mark.parametrize("sequence_level", [False, True])
@pytest.mark.parametrize("metrics_level", ["full", "minimal"])
@pytest.mark.parametrize("on_policy_kl", [False, True])
def test_actor_support_exclusion_preserves_unfiltered_reference_kl(
    sequence_level, on_policy_kl, metrics_level
):
    fn = ClippedPGLossFn(
        ClippedPGLossConfig(
            reference_policy_kl_penalty=0.1,
            metrics_level=metrics_level,
            use_on_policy_kl_approximation=on_policy_kl,
            sequence_level_importance_ratios=sequence_level,
            token_level_loss=not sequence_level,
        )
    )

    def run(excluded):
        unfiltered = torch.tensor([[-0.8, -0.7]], requires_grad=True)
        current = torch.tensor(
            [[-torch.inf if excluded else -0.6, -0.6]], requires_grad=True
        )
        data = BatchedDataDict(
            {
                "token_mask": torch.tensor([[0.0, 1.0, 1.0]]),
                "sample_mask": torch.ones(1),
                "advantages": torch.zeros(1, 3),
                "prev_logprobs": torch.tensor([[0.0, -0.6, -0.6]]),
                "generation_logprobs": torch.tensor([[0.0, -0.6, -0.6]]),
                "reference_policy_logprobs": torch.tensor([[0.0, -1.1, -0.9]]),
                "curr_logprobs_unfiltered": unfiltered,
            }
        )
        loss, _ = fn(
            next_token_logprobs=current,
            policy_support_mask=~torch.isneginf(current),
            data=data,
            global_valid_seqs=torch.tensor(1.0),
            global_valid_toks=torch.tensor(2.0),
        )
        loss.backward()
        return loss.detach(), unfiltered.grad

    for actual, expected in zip(run(True), run(False)):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_packed_support_mismatch_retains_valid_token_gradient(tmp_path):
    from nemo_rl.algorithms.loss.loss_input import prepare_packed_loss_input

    initialized_here = not torch.distributed.is_initialized()
    if initialized_here:
        torch.distributed.init_process_group(
            "nccl", init_method=f"file://{tmp_path}/pg", rank=0, world_size=1
        )
    try:
        fn = ClippedPGLossFn(
            ClippedPGLossConfig(
                reference_policy_kl_penalty=0.0,
                sequence_level_importance_ratios=True,
                token_level_loss=False,
            )
        )
        logits = torch.tensor(
            [[[0.0, 0.0, -5.0], [0.0, -5.0, 0.0], [0.0, 0.0, 0.0]]],
            device="cuda",
            requires_grad=True,
        )
        data = BatchedDataDict(
            {
                "input_ids": torch.tensor([[0, 2, 2]], device="cuda"),
                "token_mask": torch.tensor([[0.0, 1.0, 1.0]], device="cuda"),
                "sample_mask": torch.ones(1, device="cuda"),
                "advantages": torch.ones(1, 3, device="cuda"),
                "prev_logprobs": torch.tensor(
                    [[0.0, -math.log(2), -math.log(2)]], device="cuda"
                ),
                "generation_logprobs": torch.tensor(
                    [[0.0, -math.log(2), -math.log(2)]], device="cuda"
                ),
            }
        )
        cu = torch.tensor([0, 3], dtype=torch.int32, device="cuda")
        inputs, data = prepare_packed_loss_input(
            logits,
            data,
            fn,
            cu,
            cu,
            vocab_parallel_rank=0,
            vocab_parallel_group=torch.distributed.group.WORLD,
            sampling_params=TrainingSamplingParams(top_k=2),
        )
        assert torch.isneginf(inputs["next_token_logprobs"][0, 0])
        loss, _ = fn(
            data=data,
            global_valid_seqs=torch.tensor(1.0, device="cuda"),
            global_valid_toks=torch.tensor(2.0, device="cuda"),
            **inputs,
        )
        loss.backward()
        torch.testing.assert_close(
            logits.grad[0, 1],
            torch.tensor([0.5, 0.0, -0.5], device="cuda"),
            rtol=0,
            atol=0,
        )
        assert logits.grad[0, 0].count_nonzero() == 0
    finally:
        if initialized_here:
            torch.distributed.destroy_process_group()


@pytest.mark.automodel
def test_automodel_prev_logprobs_keep_support_only_on_valid_tokens():
    from nemo_rl.models.automodel.data import ProcessedInputs
    from nemo_rl.models.automodel.train import LogprobsPostProcessor

    processor = LogprobsPostProcessor(
        cfg={"logprob_chunk_size": None},
        sampling_params=TrainingSamplingParams(top_k=1),
    )
    # Token 0 is the only token in the top-1 support at every position.
    logits = torch.tensor([5.0, 0.0, 0.0]).expand(2, 4, 3).clone()
    input_ids = torch.tensor([[0, 1, 0, 1], [0, 0, 1, 1]])
    data = BatchedDataDict(
        {
            "input_lengths": torch.tensor([4, 3]),
            "token_mask": torch.tensor([[0.0, 0.0, 1.0, 1.0], [0.0, 1.0, 1.0, 0.0]]),
            "sample_mask": torch.ones(2),
        }
    )

    logprobs, updated_token_mask = processor(
        logits,
        data,
        ProcessedInputs(input_ids=input_ids, seq_len=4),
        original_batch_size=2,
        original_seq_len=4,
        cp_sharder=None,
    )

    # Prompt (row 0, pos 1) and padding (row 1, pos 3) are zeroed instead of
    # carrying -inf/NaN; valid out-of-support tokens keep -inf for the loss.
    expected = torch.tensor([[0.0, 0.0, 0.0, -torch.inf], [0.0, 0.0, -torch.inf, 0.0]])
    torch.testing.assert_close(updated_token_mask, data["token_mask"])
    torch.testing.assert_close(logprobs, expected, rtol=0, atol=0)


def _kl_in_reward_loss_config():
    return ClippedPGLossConfig(
        use_kl_in_reward=True,
        reference_policy_kl_penalty=0.1,
        reference_policy_kl_type="k3",
    )


@pytest.mark.parametrize("estimator", ["reinforce_plus_plus", "gae"])
def test_kl_in_reward_ignores_tokens_outside_policy_support(estimator):
    if estimator == "gae":
        est = GeneralizedAdvantageEstimator(GAEConfig(), _kl_in_reward_loss_config())
    else:
        est = ReinforcePlusPlusAdvantageEstimator(
            AdvEstimatorConfig.model_construct(minus_baseline=True),
            _kl_in_reward_loss_config(),
        )
    reference = torch.tensor([[-1.0, -0.5, -0.7], [-0.2, -0.9, -0.3]])

    def run(policy):
        return est.compute_advantage(
            prompt_ids=torch.tensor([[0], [0]]),
            rewards=torch.tensor([0.0, 1.0]),
            mask=torch.ones(2, 3),
            values=torch.zeros(2, 3),
            logprobs_policy=policy,
            logprobs_reference=reference,
        )

    policy = torch.tensor([[-0.8, -torch.inf, -0.6], [-0.4, -0.9, -0.1]])
    actual = run(policy)
    # An excluded token contributes no KL penalty, same as a zero-KL token.
    expected = run(torch.where(torch.isinf(policy), reference, policy))
    if estimator == "gae":  # GAE returns (advantages, returns).
        actual, expected = torch.stack(actual), torch.stack(expected)
    assert torch.isfinite(actual).all()
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_opd_advantage_ignores_tokens_outside_student_support():
    est = OPDAdvantageEstimator(AdvEstimatorConfig(name="opd"), ClippedPGLossConfig())
    advantages = est.compute_advantage(
        prompt_ids=None,
        rewards=None,
        mask=torch.tensor([[0.0, 1.0, 1.0]]),
        teacher_logprobs=torch.tensor([[0.0, -0.5, -0.2]]),
        prev_logprobs=torch.tensor([[0.0, -torch.inf, -0.7]]),
    )

    torch.testing.assert_close(advantages, torch.tensor([[0.0, 0.0, 0.5]]))
    assert all(math.isfinite(v) for v in est.last_metrics.values())


@pytest.mark.parametrize("alpha", [1.0, 0.4])
@pytest.mark.parametrize("subtract_baseline", [False, True])
@pytest.mark.parametrize("all_excluded", [False, True])
def test_opd_support_exclusion_with_proximal_teacher_and_baseline(
    alpha: float, subtract_baseline: bool, all_excluded: bool
) -> None:
    config = AdvEstimatorConfig(
        name="opd",
        proximal_teacher_alpha=alpha,
        subtract_global_baseline=subtract_baseline,
    )
    actual_estimator = OPDAdvantageEstimator(config, ClippedPGLossConfig())
    control_estimator = OPDAdvantageEstimator(config, ClippedPGLossConfig())
    teacher = torch.tensor([[-0.2, -0.3, -0.4]])
    student = torch.tensor([[-torch.inf, -0.8, -0.6]])
    if all_excluded:
        student.fill_(-torch.inf)
    valid = torch.isfinite(student)
    actual = actual_estimator.compute_advantage(
        None,
        None,
        torch.ones_like(student),
        teacher_logprobs=teacher,
        prev_logprobs=student,
    )
    expected = control_estimator.compute_advantage(
        None,
        None,
        valid.float(),
        teacher_logprobs=teacher,
        prev_logprobs=torch.where(valid, student, teacher),
    )
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert actual_estimator.last_metrics == control_estimator.last_metrics
    assert all(math.isfinite(value) for value in actual_estimator.last_metrics.values())


@pytest.mark.mcore
@pytest.mark.parametrize("top_k", [None, 1])
def test_megatron_prev_logprobs_preserve_shared_mask(monkeypatch, top_k):
    # Import the optional Megatron stack only for its marked backend test.
    from nemo_rl.models.megatron import train as module

    monkeypatch.setattr(module, "get_tensor_model_parallel_group", lambda: None)
    monkeypatch.setattr(module, "get_tensor_model_parallel_rank", lambda: 0)
    monkeypatch.setattr(
        module,
        "from_parallel_logits_to_logprobs",
        lambda *args, **kwargs: torch.tensor([[-torch.inf, -0.2, -torch.inf]]),
    )
    data = BatchedDataDict(
        {
            "input_ids": torch.tensor([[0, 1, 2, 0]]),
            "token_mask": torch.tensor([[0, 1, 1, 0]]),
            "sample_mask": torch.ones(1),
        }
    )
    processor = module.LogprobsPostProcessor(
        {"sequence_packing": {"enabled": False}},
        sampling_params=TrainingSamplingParams(top_k=top_k),
    )
    _, result = processor(data, data["input_ids"], None, 3)(torch.zeros(1, 4, 3))
    assert torch.isneginf(result["logprobs"][0, 1])
    if top_k is not None:
        torch.testing.assert_close(result["token_mask"], data["token_mask"][:, :3])
    else:
        assert "token_mask" not in result


@pytest.mark.parametrize("driver", ["grpo", "grpo_sync", "ppo"])
@pytest.mark.parametrize("filtering_on", [False, True])
@pytest.mark.parametrize("bad_logprob", [-torch.inf, torch.inf, torch.nan])
def test_sequence_rejection_ignores_only_filtered_negative_infinity(
    filtering_on, bad_logprob, driver
):
    from nemo_rl.algorithms.grpo import compute_and_apply_seq_logprob_error_masking

    token_mask = torch.tensor([[0.0, 1.0, 1.0, 1.0, 1.0]])
    data = BatchedDataDict(
        {
            "token_mask": token_mask.clone(),
            "sample_mask": torch.ones(1),
            "prev_logprobs": torch.tensor([[0.0, -0.4, -0.9, bad_logprob, -0.5]]),
            "generation_logprobs": torch.tensor([[0.0, -0.3, -0.9, -2.0, -0.5]]),
        }
    )
    if driver == "grpo":
        metrics = compute_and_apply_seq_logprob_error_masking(
            data, torch.ones(1), 2.0, filtering_on=filtering_on
        )
        masked = metrics["num_masked_seqs"]
    elif driver == "grpo_sync":
        from nemo_rl.algorithms.grpo_sync import _compute_seq_logprob_error_metrics

        data["sample_mask"], metrics = _compute_seq_logprob_error_metrics(
            **data,
            rewards=torch.ones(1),
            seq_logprob_error_threshold=2.0,
            filtering_on=filtering_on,
        )
        masked = metrics["num_masked_seqs_by_logprob_error"]
    else:
        from nemo_rl.algorithms.ppo import _apply_ppo_seq_logprob_error_masking

        if not (filtering_on and bad_logprob == -torch.inf):
            with pytest.raises(RuntimeError, match="PPO has no valid response tokens"):
                _apply_ppo_seq_logprob_error_masking(
                    data, torch.ones(1), 2.0, filtering_on=filtering_on
                )
            assert data["sample_mask"].item() == 0
            torch.testing.assert_close(data["token_mask"], token_mask)
            return
        _, metrics = _apply_ppo_seq_logprob_error_masking(
            data, torch.ones(1), 2.0, filtering_on=filtering_on
        )
        masked = metrics["num_masked_seqs_by_logprob_error"]
    retained = filtering_on and bad_logprob == -torch.inf
    assert data["sample_mask"].item() == int(retained)
    assert masked == int(not retained)
    torch.testing.assert_close(data["token_mask"], token_mask)
    if retained:
        assert metrics["mean_seq_mult_prob_error"] == pytest.approx(
            (math.exp(0.1) + 2) / 3
        )


@pytest.mark.parametrize("sequence_level", [False, True])
def test_loss_sequence_rejection_retains_supported_tokens_and_reference_kl(
    sequence_level,
):
    def run(check_sequence_error):
        fn = ClippedPGLossFn(
            ClippedPGLossConfig(
                reference_policy_kl_penalty=0.1,
                force_on_policy_ratio=True,
                seq_logprob_error_in_loss=check_sequence_error,
            ),
            seq_logprob_error_threshold=2.0,
        )
        current = torch.tensor([[-0.4, -0.9, -torch.inf, -0.5]], requires_grad=True)
        unfiltered = torch.tensor([[-0.4, -0.9, -2.0, -0.5]], requires_grad=True)
        data = BatchedDataDict(
            {
                "token_mask": torch.tensor([[0.0, 1.0, 1.0, 1.0, 1.0]]),
                "sample_mask": torch.ones(1),
                "advantages": torch.ones(1, 5),
                "generation_logprobs": torch.tensor([[0.0, -0.3, -0.9, -2.0, -0.5]]),
                "reference_policy_logprobs": torch.tensor(
                    [[0.0, -0.6, -1.0, -2.5, -0.7]]
                ),
                "curr_logprobs_unfiltered": unfiltered,
            }
        )
        loss, metrics = fn(
            current,
            data,
            torch.tensor(1.0),
            torch.tensor(4.0),
            policy_support_mask=~torch.isneginf(current),
        )
        loss.backward()
        assert metrics["policy_support_excluded_tokens"] == 1
        assert metrics["num_valid_samples"] == 1
        assert unfiltered.grad[0, 2] != 0
        if check_sequence_error:
            assert metrics["num_masked_seqs_by_logprob_error"] == 0
        return loss.detach(), current.grad, unfiltered.grad

    for actual, expected in zip(run(True), run(False)):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.parametrize("sequence_level", [False, True])
@pytest.mark.parametrize("metrics_level", ["full", "minimal"])
def test_fully_unsupported_response_matches_explicit_sample_exclusion(
    sequence_level, metrics_level
):
    fn = ClippedPGLossFn(
        ClippedPGLossConfig(
            reference_policy_kl_penalty=0.0,
            sequence_level_importance_ratios=sequence_level,
            token_level_loss=not sequence_level,
            metrics_level=metrics_level,
        )
    )

    def run(explicit):
        current = torch.tensor(
            [[-torch.inf, -torch.inf], [-0.5, -0.6]], requires_grad=True
        )
        data = BatchedDataDict(
            {
                "token_mask": torch.tensor([[0.0, 1.0, 1.0], [0.0, 1.0, 1.0]]),
                "sample_mask": torch.tensor([0.0 if explicit else 1.0, 1.0]),
                "advantages": torch.ones(2, 3),
                "prev_logprobs": torch.tensor([[0.0, -0.5, -0.6], [0.0, -0.5, -0.6]]),
                "generation_logprobs": torch.tensor(
                    [[0.0, -0.5, -0.6], [0.0, -0.5, -0.6]]
                ),
            }
        )
        loss, metrics = fn(
            current,
            data,
            torch.tensor(2.0),
            torch.tensor(4.0),
            policy_support_mask=~torch.isneginf(current),
        )
        loss.backward()
        assert metrics.pop("policy_support_excluded_tokens") == (0 if explicit else 2)
        return loss.detach(), current.grad, metrics

    actual, expected = run(False), run(True)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
