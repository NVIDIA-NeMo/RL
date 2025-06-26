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
import pytest
import torch

from nemo_rl.algorithms.loss_functions import (
    ClippedPGLossFn,
    DPOLossFn,
    NLLLoss,
)
from nemo_rl.algorithms.utils import masked_mean
from nemo_rl.distributed.batched_data_dict import BatchedDataDict


def setup_dpo_loss_test_data(vocab_size=16, batch_size=1):
    seq_len = 4
    data = {
        "input_ids": torch.arange(vocab_size / 2)
        .reshape(2 * batch_size, 4)
        .to(torch.int64)
        .to("cuda"),
        "token_mask": torch.tensor([[0, 0, 1, 1], [0, 0, 1, 1]]).to("cuda"),
        "sample_mask": torch.tensor([1, 1]).to("cuda"),
        "reference_policy_logprobs": torch.zeros((2 * batch_size, seq_len)).to("cuda"),
    }

    next_token_logits = torch.zeros((2 * batch_size, seq_len, vocab_size)).to("cuda")
    return data, next_token_logits


def test_nll_loss():
    if not torch.cuda.is_available():
        pytest.skip("No GPU available")

    loss_fn = NLLLoss()

    vocab_size = 8
    data = {
        "input_ids": torch.arange(vocab_size / 2)
        .unsqueeze(0)
        .to(torch.int64)
        .to("cuda"),
        "token_mask": torch.tensor([[0, 0, 1, 1]]).to("cuda"),
        "sample_mask": torch.tensor([1]).to("cuda"),
        "num_valid_tokens_in_batch": torch.tensor([2]),
    }

    ### assume we predict the correct token with high probability
    next_token_logits = (
        torch.tensor(
            [
                [0, 999.0, 0, 0, 0, 0, 0, 0],
                [0, 0, 999.0, 0, 0, 0, 0, 0],
                [0, 0, 0, 999.0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0.0, 0, 0, 0],  ## unused because we don't have a label
            ]
        )
        .unsqueeze(0)
        .to("cuda")
    )
    loss, metrics_dict = loss_fn(
        next_token_logits,
        data,
        global_valid_seqs=torch.sum(data["sample_mask"]),
        global_valid_toks=torch.sum(
            data["token_mask"] * data["sample_mask"].unsqueeze(-1)
        ),
    )
    torch.testing.assert_close(loss.cpu(), torch.tensor(0.0))
    # Check the metrics dictionary contains the expected values
    assert metrics_dict["num_unmasked_tokens"] == 2

    ## now assume we predict the incorrect token with high probability
    next_token_logits = (
        torch.tensor(
            [
                [999.0, 0, 0, 0, 0, 0, 0, 0],
                [0, 999.0, 0, 0, 0, 0, 0, 0],
                [0, 0, 999.0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0],
            ]
        )
        .unsqueeze(0)
        .to("cuda")
    )
    loss, metrics_dict = loss_fn(
        next_token_logits,
        data,
        global_valid_seqs=torch.sum(data["sample_mask"]),
        global_valid_toks=torch.sum(
            data["token_mask"] * data["sample_mask"].unsqueeze(-1)
        ),
    )
    ## loss per token is 999, and we have two unmasked tokens
    ## NLLLoss averages the loss over unmasked tokens
    torch.testing.assert_close(loss.cpu(), torch.tensor(999.0))
    assert metrics_dict["num_unmasked_tokens"] == 2


def test_dpo_loss():
    if not torch.cuda.is_available():
        pytest.skip("No GPU available")

    vocab_size = 16
    batch_size = 1
    num_unmasked_tokens = 2
    data, next_token_logits = setup_dpo_loss_test_data(
        vocab_size=vocab_size,
        batch_size=batch_size,
    )
    loss_fn = DPOLossFn(
        cfg={
            "reference_policy_kl_penalty": 0.0,
            "preference_loss_weight": 1.0,
            "sft_loss_weight": 0.0,
            "preference_average_log_probs": False,
            "sft_average_log_probs": False,
        }
    )

    loss, metrics_dict = loss_fn(
        next_token_logits,
        data,
        global_valid_seqs=torch.sum(data["sample_mask"]),
        global_valid_toks=torch.sum(
            data["sample_mask"].unsqueeze(-1) * data["token_mask"]
        ),
    )

    ## chosen and rejected errors are the same, so difference between them is 0
    assert torch.isclose(loss.cpu(), -torch.nn.functional.logsigmoid(torch.tensor(0.0)))

    loss_fn_with_sft = DPOLossFn(
        cfg={
            "reference_policy_kl_penalty": 0.0,
            "preference_loss_weight": 1.0,
            "sft_loss_weight": 0.5,
            "preference_average_log_probs": False,
            "sft_average_log_probs": False,
        }
    )

    expected_sft_loss = (
        -(
            torch.nn.functional.log_softmax(torch.tensor([[0.0] * vocab_size]), dim=-1)[
                :, 0
            ].sum()
        )
        * num_unmasked_tokens
        * batch_size
    )
    expected_preference_loss = -torch.nn.functional.logsigmoid(torch.tensor(0.0))
    assert torch.isclose(
        loss_fn_with_sft(
            next_token_logits,
            data,
            global_valid_seqs=torch.sum(data["sample_mask"]),
            global_valid_toks=torch.sum(
                data["sample_mask"].unsqueeze(-1) * data["token_mask"]
            ),
        )[0].cpu(),
        0.5 * expected_sft_loss + expected_preference_loss,
    )


def test_dpo_loss_varying_sequence_lengths():
    """Test DPO loss with varying sequence lengths and preference_average_log_probs=True."""
    if not torch.cuda.is_available():
        pytest.skip("No GPU available")

    # Create DPO loss function with preference_average_log_probs=True
    dpo_loss_fn_no_avg = DPOLossFn(
        {
            "reference_policy_kl_penalty": 0.1,
            "preference_loss_weight": 1.0,
            "sft_loss_weight": 0.5,
            "preference_average_log_probs": False,
            "sft_average_log_probs": False,
        }
    )
    dpo_loss_fn_avg = DPOLossFn(
        {
            "reference_policy_kl_penalty": 0.1,
            "preference_loss_weight": 1.0,
            "sft_loss_weight": 0.5,
            "preference_average_log_probs": True,
            "sft_average_log_probs": True,
        }
    )

    # Create test data with varying sequence lengths
    # Batch size 4 (2 pairs of chosen/rejected)
    # Sequence lengths: [3, 5, 4, 6]
    batch_size = 4
    max_seq_len = 6
    vocab_size = 10

    # Create input_ids with varying lengths
    input_ids = torch.zeros((batch_size, max_seq_len), dtype=torch.long).to("cuda")
    input_ids[0, :3] = torch.arange(3)  # length 3
    input_ids[1, :5] = torch.arange(5)  # length 5
    input_ids[2, :4] = torch.arange(4)  # length 4
    input_ids[3, :6] = torch.arange(6)  # length 6

    # Create token masks based on sequence lengths
    token_mask = torch.zeros((batch_size, max_seq_len)).to("cuda")
    token_mask[0, :3] = 1.0
    token_mask[1, :5] = 1.0
    token_mask[2, :4] = 1.0
    token_mask[3, :6] = 1.0

    # Create sample mask (all valid)
    sample_mask = torch.ones(batch_size).to("cuda")

    # Create reference policy logprobs
    # Make chosen responses have slightly higher logprobs than rejected
    reference_policy_logprobs = torch.zeros((batch_size, max_seq_len)).to("cuda")
    # Create next token logits
    next_token_logits = torch.zeros((batch_size, max_seq_len, vocab_size)).to("cuda")

    # Create batched data dictionary
    data = BatchedDataDict(
        {
            "input_ids": input_ids,
            "reference_policy_logprobs": reference_policy_logprobs,
            "token_mask": token_mask,
            "sample_mask": sample_mask,
        }
    )

    # Compute loss
    loss, metrics = dpo_loss_fn_no_avg(
        next_token_logits,
        data,
        global_valid_seqs=torch.sum(sample_mask),
        global_valid_toks=torch.sum(sample_mask.unsqueeze(-1) * token_mask),
    )
    loss_avg, metrics_avg = dpo_loss_fn_avg(
        next_token_logits,
        data,
        global_valid_seqs=torch.sum(sample_mask),
        global_valid_toks=torch.sum(sample_mask.unsqueeze(-1) * token_mask),
    )

    num_unmasked_tokens = token_mask[:, 1:][::2].sum().item()
    logprobs = torch.nn.functional.log_softmax(next_token_logits[:, 1:], dim=-1)
    token_logprobs = logprobs.gather(
        dim=-1, index=input_ids[:, 1:].unsqueeze(-1)
    ).squeeze(-1)
    expected_per_token_sft_loss = -(token_logprobs[::2] * token_mask[:, 1:][::2])
    ## sum across tokens in an example, average across examples
    expected_sft_loss_no_avg = expected_per_token_sft_loss.sum(-1).mean()
    ## average across tokens in an example, then average across examples
    expected_sft_loss_avg = expected_per_token_sft_loss.sum() / num_unmasked_tokens

    assert torch.isclose(torch.tensor(metrics["sft_loss"]), expected_sft_loss_no_avg)
    assert torch.isclose(torch.tensor(metrics_avg["sft_loss"]), expected_sft_loss_avg)


def test_dpo_sft_matches_nll_loss():
    """Test that DPO SFT loss matches NLL loss when preference_loss_weight=0."""
    if not torch.cuda.is_available():
        pytest.skip("No GPU available")

    # Setup test data
    vocab_size = 8
    batch_size = 2
    dpo_data = {
        "input_ids": torch.randint(0, vocab_size, (batch_size * 2, 5))
        .to(torch.int64)
        .to("cuda"),
        "token_mask": torch.tensor(
            [[0, 0, 1, 1, 0], [0, 0, 1, 1, 1], [0, 1, 1, 1, 1], [0, 1, 1, 1, 0]]
        ).to("cuda"),
        "sample_mask": torch.tensor([1, 1, 1, 1]).to("cuda"),
        "reference_policy_logprobs": torch.randn((4, 5)).to("cuda"),
    }

    ## when computing the sft loss in DPO, we only use the chosen samples
    sft_data = {
        "input_ids": dpo_data["input_ids"][::2],
        "token_mask": dpo_data["token_mask"][::2],
        "sample_mask": dpo_data["sample_mask"][::2],
    }

    # Create next token logits that will give non-zero loss
    ## * 2 for chosen/rejected
    next_token_logits = torch.randn((batch_size * 2, 5, vocab_size)).to("cuda")

    # Compute NLL loss
    nll_loss_fn = NLLLoss()
    nll_loss, nll_metrics = nll_loss_fn(
        next_token_logits[::2],
        sft_data,
        global_valid_seqs=None,
        global_valid_toks=torch.sum(
            sft_data["sample_mask"].unsqueeze(-1) * torch.sum(sft_data["token_mask"])
        ),
    )

    # Compute DPO loss with preference_loss_weight=0
    dpo_loss_fn = DPOLossFn(
        cfg={
            "reference_policy_kl_penalty": 0.0,
            "preference_loss_weight": 0.0,  # Disable preference loss
            "sft_loss_weight": 1.0,  # Only use SFT loss
            "preference_average_log_probs": False,
            "sft_average_log_probs": False,
        }
    )
    dpo_loss, dpo_metrics = dpo_loss_fn(
        next_token_logits,
        dpo_data,
        global_valid_seqs=torch.sum(dpo_data["sample_mask"]),
        global_valid_toks=torch.sum(
            dpo_data["sample_mask"].unsqueeze(-1) * dpo_data["token_mask"]
        ),
    )

    # Verify losses match
    ## since DPO SFT loss just sums across tokens in a batch and then averages over the batch,
    ## we need to re-normalize by multiplying by the batch size and dividing by the total number
    ## of unmasked chosen tokens
    scaled_dpo_loss = (
        dpo_loss
        * (torch.sum(sft_data["sample_mask"]))
        / torch.sum(
            sft_data["sample_mask"].unsqueeze(-1) * torch.sum(sft_data["token_mask"])
        )
    )
    torch.testing.assert_close(scaled_dpo_loss, nll_loss)


def _setup_clipped_pg_test_data(batch_size=1, seq_len=4, vocab_size=8, device="cuda"):
    """Sets up basic mock data structure. Tests should fill values."""
    input_ids = torch.randint(  # Input IDs only needed if original loss fn used
        0, vocab_size, (batch_size, seq_len), dtype=torch.int64, device=device
    )
    # Default mask: Mask first token [[0, 1, 1, 1]]
    token_mask = torch.ones((batch_size, seq_len), dtype=torch.int64, device=device)
    token_mask[:, 0] = 0
    # sample_mask needs shape [B]
    sample_mask = torch.ones(batch_size, dtype=torch.int64, device=device)

    # Simple default values, tests overwrite these
    advantages = torch.zeros((batch_size, seq_len), device=device)
    prev_logprobs = torch.zeros((batch_size, seq_len), device=device)
    reference_policy_logprobs = torch.zeros((batch_size, seq_len), device=device)
    generation_logprobs = torch.zeros((batch_size, seq_len), device=device)

    data = BatchedDataDict(
        {
            "input_ids": input_ids,  # Include for completeness
            "token_mask": token_mask,
            "sample_mask": sample_mask,
            "advantages": advantages,
            "prev_logprobs": prev_logprobs,
            "reference_policy_logprobs": reference_policy_logprobs,
            "generation_logprobs": generation_logprobs,
        }
    )
    # Return seq_len and vocab_size needed by tests
    return data, seq_len, vocab_size


# Helper to create logits that yield specific target log probs after log_softmax
def _create_exact_logits(target_curr_lp_masked, input_ids, seq_len, vocab_size, device):
    """Constructs logits such that log_softmax results in target_curr_lp_masked."""
    dummy_logits = torch.full(
        (1, seq_len, vocab_size), -100.0, device=device
    )  # Start very low

    # Loss fn uses logits[:, :-1] and gathers based on next_tokens = input_ids[:, 1:]
    # We need to set logits for indices i=0..S-2 of the sliced logits tensor.
    # These correspond to target logprobs at indices 0..S-2 of target_curr_lp_masked.
    num_effective_pos = target_curr_lp_masked.shape[1]
    for i in range(num_effective_pos):
        logit_idx = i  # Index in the sliced logits tensor (dummy_logits[:, 0:S-1, :])
        data_idx = i + 1  # Index in the original input_ids to find the target token

        target_token_id = input_ids[0, data_idx].item()
        # Keep target_lp as a 0-dim tensor for torch ops
        target_lp = target_curr_lp_masked[0, i]

        # Handle target_lp = 0 case separately
        if torch.isclose(target_lp, torch.tensor(0.0, device=device)):
            dummy_logits[0, logit_idx, target_token_id] = 100.0  # Large positive logit
        elif target_lp < 0:
            # Set target token logit to 0
            dummy_logits[0, logit_idx, target_token_id] = 0.0
            # Set one distractor token logit using the formula
            distractor_token_id = (target_token_id + 1) % vocab_size
            # Ensure distractor isn't same as target if vocab_size=1 (edge case)
            if distractor_token_id == target_token_id:
                distractor_token_id = (target_token_id + 2) % vocab_size
            distractor_logit = torch.log(torch.exp(-target_lp) - 1.0)
            dummy_logits[0, logit_idx, distractor_token_id] = distractor_logit
        else:  # target_lp > 0 is not supported by this method
            raise ValueError(
                "Target log probability must be negative or zero for this construction"
            )
    return dummy_logits


# Simplified PPO Clipping Test using original Loss
def test_clipped_pg_loss_ppo_clipping():
    """Tests PPO clipping calculations directly."""
    if not torch.cuda.is_available():
        pytest.skip("No GPU available")

    device = "cuda"
    data, seq_len, vocab_size = _setup_clipped_pg_test_data(device=device)

    ratio_clip = 0.2
    cfg = {
        "ratio_clip_min": ratio_clip,
        "ratio_clip_max": ratio_clip,
        "ratio_clip_c": None,
        "reference_policy_kl_penalty": 0.0,  # Disable KL
        "disable_ppo_ratio": False,
        "use_on_policy_kl_approximation": False,
        "use_importance_sampling_correction": False,
        "token_level_loss": True,
    }
    loss_fn = ClippedPGLossFn(cfg)

    adv_masked = torch.tensor([[1.0, -1.0, 2.0]], device=device)
    # Use non-zero prev_lp to allow ratios > 1 with valid curr_lp <= 0
    prev_lp_masked = torch.tensor([[-1.0, -1.0, -1.0]], device=device)
    # Target Curr logprobs (masked pos 1, 2, 3) - design for clipping
    # Target ratios: 0.5 (<0.8), 1.0 (in [0.8, 1.2]), 1.5 (>1.2)
    # Curr = log(Ratio) + Prev
    curr_lp_masked = torch.tensor(
        [[-1.69315, -1.0, -0.59453]], device=device
    )  # approx log(0.5)-1, log(1)-1, log(1.5)-1

    # Fill full tensors (only need first dim for B=1)
    data["advantages"][0, 1:] = adv_masked
    data["prev_logprobs"][0, 1:] = prev_lp_masked

    # --- Hand Calculation ---
    ratios = torch.exp(curr_lp_masked - prev_lp_masked)  # approx [0.5, 1.0, 1.5]
    assert torch.allclose(
        ratios, torch.tensor([[0.5, 1.0, 1.5]], device=device), rtol=1e-3
    )

    ratios_clamped = torch.clamp(
        ratios, 1.0 - ratio_clip, 1.0 + ratio_clip
    )  # [0.8, 1.0, 1.2]
    assert torch.allclose(
        ratios_clamped, torch.tensor([[0.8, 1.0, 1.2]], device=device), rtol=1e-3
    )

    loss1 = -adv_masked * ratios  # approx -[1*0.5, -1*1.0, 2*1.5] = [-0.5, 1.0, -3.0]
    assert torch.allclose(
        loss1, torch.tensor([[-0.5, 1.0, -3.0]], device=device), rtol=1e-3
    )

    loss2 = -adv_masked * ratios_clamped  # -[1*0.8, -1*1.0, 2*1.2] = [-0.8, 1.0, -2.4]
    assert torch.allclose(
        loss2, torch.tensor([[-0.8, 1.0, -2.4]], device=device), rtol=1e-3
    )

    max_loss = torch.maximum(loss1, loss2)  # approx [-0.5, 1.0, -2.4]
    assert torch.allclose(
        max_loss, torch.tensor([[-0.5, 1.0, -2.4]], device=device), rtol=1e-3
    )

    expected_loss = torch.mean(
        max_loss
    )  # approx (-0.5 + 1.0 - 2.4) / 3 = -1.9 / 3 = -0.6333
    assert torch.allclose(
        expected_loss, torch.tensor(-0.6333, device=device), rtol=1e-3
    )

    input_ids = data["input_ids"]
    dummy_logits = _create_exact_logits(
        curr_lp_masked, input_ids, seq_len, vocab_size, device
    )

    actual_loss, _ = loss_fn(
        dummy_logits,
        data,
        global_valid_seqs=torch.sum(data["sample_mask"]),
        global_valid_toks=torch.sum(data["sample_mask"] * data["token_mask"]),
    )
    torch.testing.assert_close(actual_loss, expected_loss)


# Simplified REINFORCE Test using original Loss
def test_clipped_pg_loss_reinforce_mode():
    """Tests REINFORCE mode calculations directly."""
    if not torch.cuda.is_available():
        pytest.skip("No GPU available")

    device = "cuda"
    data, seq_len, vocab_size = _setup_clipped_pg_test_data(device=device)

    cfg = {
        "disable_ppo_ratio": True,
        "reference_policy_kl_penalty": 0.0,
        "ratio_clip_min": 0.0,  # Placeholder, ignored
        "ratio_clip_max": 0.0,  # Placeholder, ignored
        "ratio_clip_c": None,
        "use_on_policy_kl_approximation": False,
        "use_importance_sampling_correction": False,
        "token_level_loss": True,
    }
    loss_fn = ClippedPGLossFn(cfg)

    adv_masked = torch.tensor([[1.0, -1.0, 2.0]], device=device)
    curr_lp_masked = torch.tensor([[-0.5, -1.0, -1.5]], device=device)

    data["advantages"][0, 1:] = adv_masked
    data["_test_curr_logprobs"] = curr_lp_masked
    data["prev_logprobs"][0, 1:] = torch.zeros_like(curr_lp_masked)

    # --- Hand Calculation ---
    expected_loss_per_token = -adv_masked * curr_lp_masked  # [0.5, -1.0, 3.0]
    assert torch.allclose(
        expected_loss_per_token,
        torch.tensor([[0.5, -1.0, 3.0]], device=device),
        rtol=1e-3,
    )

    expected_loss = torch.mean(expected_loss_per_token)  # 2.5 / 3 = 0.8333
    assert torch.allclose(expected_loss, torch.tensor(0.8333, device=device), rtol=1e-3)

    input_ids = data["input_ids"]
    dummy_logits = _create_exact_logits(
        curr_lp_masked, input_ids, seq_len, vocab_size, device
    )

    actual_loss, _ = loss_fn(
        dummy_logits,
        data,
        global_valid_seqs=torch.sum(data["sample_mask"]),
        global_valid_toks=torch.sum(
            data["sample_mask"].unsqueeze(-1) * data["token_mask"]
        ),
    )
    torch.testing.assert_close(actual_loss, expected_loss)


# Simplified KL Penalty Test using original Loss
def test_clipped_pg_loss_kl_penalty():
    """Tests KL penalty calculations directly."""
    if not torch.cuda.is_available():
        pytest.skip("No GPU available")

    device = "cuda"
    data, seq_len, vocab_size = _setup_clipped_pg_test_data(device=device)

    # --- Test Setup ---
    kl_beta = 0.1
    cfg = {
        "reference_policy_kl_penalty": kl_beta,
        "ratio_clip_min": 0.2,
        "ratio_clip_max": 0.2,
        "ratio_clip_c": None,
        "disable_ppo_ratio": False,
        "use_on_policy_kl_approximation": False,
        "use_importance_sampling_correction": False,
        "token_level_loss": True,
    }
    loss_fn = ClippedPGLossFn(cfg)

    adv_masked = torch.tensor([[0.0, 0.0, 0.0]], device=device)
    curr_lp_masked = torch.tensor([[0.0, -1.0, -2.0]], device=device)
    ref_lp_masked = torch.tensor([[-1.0, -1.0, -1.0]], device=device)
    prev_lp_masked = torch.tensor([[0.0, 0.0, 0.0]], device=device)

    data["advantages"][0, 1:] = adv_masked
    data["reference_policy_logprobs"][0, 1:] = ref_lp_masked
    data["prev_logprobs"][0, 1:] = prev_lp_masked
    data["_test_curr_logprobs"] = curr_lp_masked

    # --- Hand Calculation ---
    # Actor loss is 0. Total loss = kl_beta * mean(kl_term)
    # kl_term = exp(ref - curr) - (ref - curr) - 1
    r = ref_lp_masked - curr_lp_masked  # [-1.0, 0.0, 1.0]
    assert torch.allclose(r, torch.tensor([[-1.0, 0.0, 1.0]], device=device), rtol=1e-3)

    kl_term_per_token = torch.exp(r) - r - 1  # [0.368, 0.0, 0.718]
    assert torch.allclose(
        kl_term_per_token, torch.tensor([[0.368, 0.0, 0.718]], device=device), rtol=1e-3
    )

    expected_kl_mean = torch.mean(kl_term_per_token)  # 0.362
    assert torch.allclose(
        expected_kl_mean, torch.tensor(0.362, device=device), rtol=1e-3
    )

    expected_loss = kl_beta * expected_kl_mean  # 0.0362
    assert torch.allclose(expected_loss, torch.tensor(0.0362, device=device), rtol=1e-3)

    input_ids = data["input_ids"]
    dummy_logits = _create_exact_logits(
        curr_lp_masked, input_ids, seq_len, vocab_size, device
    )

    actual_loss, _ = loss_fn(
        dummy_logits,
        data,
        global_valid_seqs=torch.sum(data["sample_mask"]),
        global_valid_toks=torch.sum(
            data["sample_mask"].unsqueeze(-1) * data["token_mask"]
        ),
    )
    torch.testing.assert_close(actual_loss, expected_loss)


# Masking tests - Should work with original Loss Fn if needed, but less critical
def test_clipped_pg_loss_masking():
    """Tests the effect of token_mask and sample_mask."""
    if not torch.cuda.is_available():
        pytest.skip("No GPU available")

    batch_size = 2
    seq_len = 4
    device = "cuda"
    # Use original loss function for masking tests, as it involves interactions
    # that the Testable class might obscure slightly.
    data, seq_len, vocab_size = _setup_clipped_pg_test_data(
        batch_size=batch_size, seq_len=seq_len, device=device
    )
    # Need some realistic-ish logits and logprobs for masking test
    dummy_logits = torch.randn(batch_size, seq_len, vocab_size, device=device)
    # Ensure logprobs used by the loss fn make sense relative to advantages
    data["prev_logprobs"] = torch.randn_like(data["prev_logprobs"]) * 0.1
    data["reference_policy_logprobs"] = (
        torch.randn_like(data["reference_policy_logprobs"]) * 0.1
    )
    # Make advantages non-zero
    data["advantages"] = torch.randn_like(data["advantages"]) + 1.0

    cfg = {
        "ratio_clip_min": 0.2,
        "ratio_clip_max": 0.2,
        "ratio_clip_c": None,
        "reference_policy_kl_penalty": 0.1,
        "disable_ppo_ratio": False,
        "use_on_policy_kl_approximation": False,
        "use_importance_sampling_correction": False,
        "token_level_loss": True,
    }
    loss_fn = ClippedPGLossFn(cfg)  # Use original loss fn

    # --- Test 1: Token Mask ---
    # Default mask: [[0, 1, 1, 1], [0, 1, 1, 1]] -> 3 tokens per sample
    loss_default, _ = loss_fn(
        dummy_logits,
        data,
        global_valid_seqs=torch.sum(data["sample_mask"]),
        global_valid_toks=torch.sum(
            data["sample_mask"].unsqueeze(-1) * data["token_mask"]
        ),
    )

    # Modify token_mask for batch item 0 to mask one more token (pos 1)
    data_mod_token = data.copy()
    data_mod_token["token_mask"] = data["token_mask"].clone()
    data_mod_token["token_mask"][0, 1] = (
        0  # New mask: [[0, 0, 1, 1], [0, 1, 1, 1]] -> 2 tokens sample 0, 3 tokens sample 1
    )

    loss_token_masked, _ = loss_fn(
        dummy_logits,
        data_mod_token,
        global_valid_seqs=torch.sum(data_mod_token["sample_mask"]),
        global_valid_toks=torch.sum(
            data_mod_token["sample_mask"].unsqueeze(-1) * data_mod_token["token_mask"]
        ),
    )
    # Loss should change if a potentially contributing token is masked
    assert not torch.isclose(loss_default, loss_token_masked, atol=1e-4), (
        "Token mask did not change loss as expected"
    )

    # --- Test 2: Sample Mask ---
    data_mod_sample = data.copy()
    data_mod_sample["sample_mask"] = torch.tensor(
        [1, 0], dtype=torch.int64, device=device
    )  # Ignore item 1

    loss_sample_masked, _ = loss_fn(
        dummy_logits,
        data_mod_sample,
        global_valid_seqs=torch.sum(data_mod_sample["sample_mask"]),
        global_valid_toks=torch.sum(
            data_mod_sample["sample_mask"].unsqueeze(-1) * data_mod_sample["token_mask"]
        ),
    )

    # Manually create data dict for only batch 0
    data_only_b0_dict = {}
    for key, value in data.items():
        if isinstance(value, torch.Tensor):
            if key == "sample_mask":
                data_only_b0_dict[key] = value[0:1]
            else:
                data_only_b0_dict[key] = value[0:1]
        else:
            data_only_b0_dict[key] = value
    data_only_b0 = BatchedDataDict(data_only_b0_dict)

    logits_only_b0 = dummy_logits[0:1]
    loss_only_b0, _ = loss_fn(
        logits_only_b0,
        data_only_b0,
        global_valid_seqs=torch.sum(data_only_b0["sample_mask"]),
        global_valid_toks=torch.sum(
            data_only_b0["sample_mask"].unsqueeze(-1) * data_only_b0["token_mask"]
        ),
    )

    torch.testing.assert_close(loss_sample_masked, loss_only_b0)


def test_clipped_pg_loss_zero_mask():
    """Tests the case where the combined mask sum is zero."""
    if not torch.cuda.is_available():
        pytest.skip("No GPU available")

    device = "cuda"
    data, seq_len, vocab_size = _setup_clipped_pg_test_data(device=device)
    # Need dummy logits
    dummy_logits = torch.randn(1, seq_len, vocab_size, device=device)

    cfg = {
        "ratio_clip_min": 0.2,
        "ratio_clip_max": 0.2,
        "ratio_clip_c": None,
        "reference_policy_kl_penalty": 0.1,
        "disable_ppo_ratio": False,
        "use_on_policy_kl_approximation": False,
        "use_importance_sampling_correction": False,
        "token_level_loss": True,
    }
    loss_fn = ClippedPGLossFn(cfg)  # Use original loss fn

    # Set token mask to all zeros
    data["token_mask"] = torch.zeros_like(data["token_mask"])

    loss, _ = loss_fn(
        dummy_logits,
        data,
        global_valid_seqs=torch.sum(data["sample_mask"]),
        global_valid_toks=torch.sum(
            data["sample_mask"].unsqueeze(-1) * data["token_mask"]
        ),
    )

    # Loss should be exactly zero
    torch.testing.assert_close(loss, torch.tensor(0.0, device=device))


def test_clipped_pg_loss_on_policy_kl_importance_sampling():
    """Tests PPO loss with KL penalty and importance sampling enabled."""
    if not torch.cuda.is_available():
        pytest.skip("No GPU available")

    device = "cuda"
    data, seq_len, vocab_size = _setup_clipped_pg_test_data(device=device)

    ratio_clip = 0.2
    kl_beta = 0.1

    cfg = {
        "ratio_clip_min": ratio_clip,
        "ratio_clip_max": ratio_clip,
        "ratio_clip_c": None,
        "reference_policy_kl_penalty": kl_beta,
        "disable_ppo_ratio": False,
        "use_on_policy_kl_approximation": True,
        "use_importance_sampling_correction": True,
        "token_level_loss": True,
    }
    loss_fn = ClippedPGLossFn(cfg)

    adv_masked = torch.tensor([[1.0, -1.0, 2.0]], device=device)
    prev_lp_masked = torch.tensor([[-1.0, -1.0, -1.0]], device=device)
    curr_lp_masked = torch.tensor(
        [[-1.69315, -1.0, -0.59453]], device=device
    )  # approx log(0.5)-1, log(1)-1, log(1.5)-1

    ref_lp_masked = torch.tensor([[-1.0, -1.0, -1.0]], device=device)

    # For Importance Sampling
    gen_lp_masked = torch.tensor([[-0.5, -1.5, -0.8]], device=device)

    # Fill full tensors
    data["advantages"][0, 1:] = adv_masked
    data["prev_logprobs"][0, 1:] = prev_lp_masked
    data["generation_logprobs"][0, 1:] = gen_lp_masked
    data["reference_policy_logprobs"][0, 1:] = ref_lp_masked

    # --- Hand Calculation ---
    # Actor Loss Calculation
    actor_importance_weights = torch.exp(
        prev_lp_masked - gen_lp_masked
    )  # exp([-1 - (-0.5), -1 - (-1.5), -1 - (-0.8)]) = [0.6065, 1.6487, 0.8187]
    assert torch.allclose(
        actor_importance_weights,
        torch.tensor([[0.6065, 1.6487, 0.8187]], device=device),
        rtol=1e-3,
    )

    ratios = torch.exp(curr_lp_masked - prev_lp_masked)  # [0.5, 1.0, 1.5]
    assert torch.allclose(
        ratios, torch.tensor([[0.5, 1.0, 1.5]], device=device), rtol=1e-3
    )

    ratios_clamped = torch.clamp(
        ratios, 1.0 - ratio_clip, 1.0 + ratio_clip
    )  # [0.8, 1.0, 1.2]
    assert torch.allclose(
        ratios_clamped, torch.tensor([[0.8, 1.0, 1.2]], device=device), rtol=1e-3
    )

    loss1 = -adv_masked * ratios  # [-0.5, 1.0, -3.0]
    assert torch.allclose(
        loss1, torch.tensor([[-0.5, 1.0, -3.0]], device=device), rtol=1e-3
    )

    loss2 = -adv_masked * ratios_clamped  # [-0.8, 1.0, -2.4]
    assert torch.allclose(
        loss2, torch.tensor([[-0.8, 1.0, -2.4]], device=device), rtol=1e-3
    )

    max_loss = torch.maximum(loss1, loss2)  # [-0.5, 1.0, -2.4]
    assert torch.allclose(
        max_loss, torch.tensor([[-0.5, 1.0, -2.4]], device=device), rtol=1e-3
    )

    importance_weighted_max_loss = (
        actor_importance_weights * max_loss
    )  # [0.6065*(-0.5), 1.6487*1.0, 0.8187*(-2.4)] = [-0.30325, 1.6487, -1.96488]
    assert torch.allclose(
        importance_weighted_max_loss,
        torch.tensor([[-0.30325, 1.6487, -1.96488]], device=device),
        rtol=1e-3,
    )

    expected_actor_loss = torch.mean(importance_weighted_max_loss)  # -0.2065
    assert torch.allclose(
        expected_actor_loss, torch.tensor(-0.2065, device=device), rtol=1e-3
    )

    # KL Loss Calculation
    kl_importance_weights = torch.exp(
        curr_lp_masked - gen_lp_masked
    )  # exp([-1.69315 - (-0.5), -1 - (-1.5), -0.59453 - (-0.8)]) = [0.3033, 1.6487, 1.2281]
    assert torch.allclose(
        kl_importance_weights,
        torch.tensor([[0.3033, 1.6487, 1.2281]], device=device),
        rtol=1e-3,
    )

    r = (
        ref_lp_masked - curr_lp_masked
    )  # [-1.0 - (-1.69315), -1.0 - (-1.0), -1.0 - (-0.59453)] = [0.69315, 0.0, -0.40547]
    assert torch.allclose(
        r, torch.tensor([[0.69315, 0.0, -0.40547]], device=device), rtol=1e-3
    )

    kl_term_per_token = (
        torch.exp(r) - r - 1
    )  # [exp(0.69315)-0.69315-1, exp(0)-0-1, exp(-0.40547)-(-0.40547)-1] = [0.3069, 0.0, 0.0721]
    assert torch.allclose(
        kl_term_per_token,
        torch.tensor([[0.3069, 0.0, 0.0721]], device=device),
        rtol=1e-3,
    )
    # Apply importance weights to KL loss
    # kl_term = importance_weights * kl_beta * kl_indiv
    importance_weighted_kl_term_per_token = (
        kl_importance_weights * kl_term_per_token
    )  # [0.3033*0.3069, 1.6487*0.0, 1.2281*0.0721] = [0.09308, 0.0, 0.08855]
    assert torch.allclose(
        importance_weighted_kl_term_per_token,
        torch.tensor([[0.09308, 0.0, 0.08855]], device=device),
        rtol=1e-3,
    )

    expected_kl_mean = torch.mean(
        importance_weighted_kl_term_per_token
    )  # mean([0.09308, 0.0, 0.08855]) = 0.060543
    expected_kl_loss = kl_beta * expected_kl_mean  # 0.1 * 0.060543 = 0.0060543

    expected_total_loss = (
        expected_actor_loss + expected_kl_loss
    )  # -0.2065 + 0.0060543 = -0.2004457

    input_ids = data["input_ids"]
    dummy_logits = _create_exact_logits(
        curr_lp_masked, input_ids, seq_len, vocab_size, device
    )

    actual_loss, _ = loss_fn(
        dummy_logits,
        data,
        global_valid_seqs=torch.sum(data["sample_mask"]),
        global_valid_toks=torch.sum(data["sample_mask"] * data["token_mask"]),
    )
    torch.testing.assert_close(actual_loss, expected_total_loss, atol=1e-4, rtol=1e-3)


def test_masked_mean_all_zeros():
    """Test masked_mean function with all zeros mask."""
    values = torch.tensor([1.0, 2.0, 3.0, 4.0])
    mask = torch.zeros_like(values)

    # All zeros mask should return 0
    result = masked_mean(values, mask)
    print(result)
    torch.testing.assert_allclose(result, torch.tensor(0.0))

    # With check_zero_mask=False
    mask[0] = 1
    result = masked_mean(values, mask)
    torch.testing.assert_allclose(result, torch.tensor(1.0))

    # Case 2: dim is not None
    values = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    mask = torch.zeros_like(values)
    result = masked_mean(values, mask, dim=1)
    torch.testing.assert_allclose(result, torch.tensor([0.0, 0.0]))


def test_clipped_pg_loss_dual_clip():
    """
    Tests dual clipping in PPO loss function.

    Dual clipping prevents excessive policy updates when dealing with:
    1. Strongly negative advantages
    2. Very large probability ratios (when curr_logprobs >> prev_logprobs)

    This test verifies that when advantages are negative, ratio_clip_c serves as an upper
    bound on the loss.
    """
    if not torch.cuda.is_available():
        pytest.skip("No GPU available")

    device = "cuda"
    data, seq_len, vocab_size = _setup_clipped_pg_test_data(device=device)

    ratio_clip = 0.2
    ratio_clip_c = 3.0
    cfg = {
        "ratio_clip_min": ratio_clip,
        "ratio_clip_max": ratio_clip,
        "ratio_clip_c": ratio_clip_c,
        "reference_policy_kl_penalty": 0.0,  # Disable KL
        "disable_ppo_ratio": False,
        "use_on_policy_kl_approximation": False,
        "use_importance_sampling_correction": False,
        "token_level_loss": True,
    }
    loss_fn = ClippedPGLossFn(cfg)

    # Create test data with a mix of advantages: positive, slightly negative, strongly negative
    adv_masked = torch.tensor([[1.0, -1.0, -4.0]], device=device)

    # Set up target logprobs to test various probability ratios
    prev_lp_masked = torch.tensor([[-1.0, -1.0, -3.0]], device=device)
    curr_lp_masked = torch.tensor(
        [[-1.69315, -1.0, -0.69741]], device=device
    )  # approx log(0.5)-1, log(1)-1, log(10)-3

    ratios = torch.exp(curr_lp_masked - prev_lp_masked)  # approx [0.5, 1.0, 1.5]
    assert torch.allclose(
        ratios, torch.tensor([[0.5, 1.0, 10.0]], device=device), rtol=1e-3
    )

    # Fill full tensors (only need first dim for B=1)
    data["advantages"][0, 1:] = adv_masked
    data["prev_logprobs"][0, 1:] = prev_lp_masked

    # --- Hand Calculation ---
    # Actor Loss Calculation
    ratios_clamped = torch.clamp(
        ratios, 1.0 - ratio_clip, 1.0 + ratio_clip
    )  # [0.8, 1.0, 1.2]
    assert torch.allclose(
        ratios_clamped, torch.tensor([[0.8, 1.0, 1.2]], device=device), rtol=1e-3
    )

    # Standard PPO clipping
    loss1 = -adv_masked * ratios  # -[1*0.5, -1*1.0, -4*10.] = [-0.5, 1.0, 40.]
    assert torch.allclose(
        loss1, torch.tensor([[-0.5, 1.0, 40.0]], device=device), rtol=1e-3
    )

    loss2 = -adv_masked * ratios_clamped  # -[1*0.8, -1*1.0, -4*1.2] = [-0.8, 1.0, 4.8]
    assert torch.allclose(
        loss2, torch.tensor([[-0.8, 1.0, 4.8]], device=device), rtol=1e-3
    )

    max_loss = torch.maximum(loss1, loss2)  # [-0.5, 1.0, 40.]
    assert torch.allclose(
        max_loss, torch.tensor([[-0.5, 1.0, 40.0]], device=device), rtol=1e-3
    )

    # Dual clipping
    loss3 = -adv_masked * ratio_clip_c  # -[1*3.0, -1*3.0, -4*3.0] = [-3.0, 3.0, 12.0]
    assert torch.allclose(
        loss3, torch.tensor([[-3.0, 3.0, 12.0]], device=device), rtol=1e-3
    )
    min_loss = torch.minimum(max_loss, loss3)  # [-3.0, 1.0, 12.0]
    assert torch.allclose(
        min_loss, torch.tensor([[-3.0, 1.0, 12.0]], device=device), rtol=1e-3
    )

    # For negative advantages, dual clipping reduces the loss from 40.0 to 12.0
    clip_loss = torch.where(adv_masked < 0, min_loss, max_loss)  # [-0.5, 1.0, 12.0]
    assert torch.allclose(
        clip_loss, torch.tensor([[-0.5, 1.0, 12.0]], device=device), rtol=1e-3
    ), f"clip_loss is {clip_loss}, expected [[-0.5, 1.0, 12.0]]"

    expected_loss = torch.mean(clip_loss)  # (-0.5 + 1.0 + 12.0) / 3 = 12.5 / 3 = 4.1667
    assert torch.allclose(expected_loss, torch.tensor(4.1667, device=device), rtol=1e-3)

    input_ids = data["input_ids"]
    dummy_logits = _create_exact_logits(
        curr_lp_masked, input_ids, seq_len, vocab_size, device
    )

    actual_loss, _ = loss_fn(
        dummy_logits,
        data,
        global_valid_seqs=torch.sum(data["sample_mask"]),
        global_valid_toks=torch.sum(
            data["sample_mask"].unsqueeze(-1) * data["token_mask"]
        ),
    )
    torch.testing.assert_close(actual_loss, expected_loss)


def test_clipped_pg_loss_entropy():
    """Tests approximate entropy calculation in ClippedPGLossFn."""
    if not torch.cuda.is_available():
        pytest.skip("No GPU available")

    device = "cuda"
    data, seq_len, vocab_size = _setup_clipped_pg_test_data(device=device)

    cfg = {
        "ratio_clip_min": 0.2,
        "ratio_clip_max": 0.2,
        "ratio_clip_c": None,
        "reference_policy_kl_penalty": 0.0,  # Disable KL for simplicity
        "disable_ppo_ratio": False,
        "use_on_policy_kl_approximation": False,
        "use_importance_sampling_correction": False,  # This flag does not affect entropy calculation
        "token_level_loss": True,
    }
    loss_fn = ClippedPGLossFn(cfg)

    # Log probs for 3 tokens (default token_mask is [0, 1, 1, 1], so 3 unmasked after slicing)
    # curr_lp_masked: log probabilities from the current policy (model output)
    # gen_lp_masked: log probabilities from the generation policy (from data)
    curr_lp_masked = torch.tensor([[-0.5, -1.0, -1.5]], device=device)
    gen_lp_masked = torch.tensor([[-0.6, -1.1, -1.6]], device=device)

    # prev_lp_masked is needed for actor loss but not directly for this entropy formula
    prev_lp_masked = torch.tensor([[-0.4, -0.9, -1.4]], device=device)

    data["prev_logprobs"][0, 1:] = prev_lp_masked
    data["generation_logprobs"][0, 1:] = gen_lp_masked
    # _create_exact_logits needs input_ids
    data["input_ids"] = torch.randint(0, vocab_size, (1, seq_len), device=device)

    # seq_entropy_approx = -masked_mean(torch.exp(curr_logprobs - generation_logprobs) * curr_logprobs, mask)
    # curr_lp_masked represents curr_logprobs for the hand calculation.
    # gen_lp_masked represents generation_logprobs.
    importance_weight_factor = torch.exp(curr_lp_masked - gen_lp_masked)
    entropy_terms = importance_weight_factor * curr_lp_masked
    expected_entropy = -torch.mean(
        entropy_terms
    )  # torch.mean because default mask applies to these 3 terms

    dummy_logits = _create_exact_logits(
        curr_lp_masked, data["input_ids"], seq_len, vocab_size, device
    )
    _, metrics = loss_fn(
        dummy_logits,
        data,
        global_valid_seqs=torch.sum(data["sample_mask"]),
        global_valid_toks=torch.sum(data["sample_mask"] * data["token_mask"]),
    )

    torch.testing.assert_close(
        torch.tensor(metrics["approx_entropy"], device=device),
        expected_entropy,
        rtol=1e-3,
        atol=1e-5,
    )
