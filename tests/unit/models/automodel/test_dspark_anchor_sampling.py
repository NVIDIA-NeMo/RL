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
import torch

from nemo_rl.models.automodel.draft.common import (
    build_anchor_candidate_mask,
    sample_anchor_positions,
)


def _rl_loss_mask(seq_len: int, response_start: int, response_len: int) -> torch.Tensor:
    """RL-style token mask: 1 on response tokens, 0 on prompt/padding."""
    mask = torch.zeros(1, seq_len)
    mask[0, response_start : response_start + response_len] = 1.0
    return mask


def test_prompt_boundary_anchor_is_candidate():
    """The final prompt token anchors the first response token (where
    inference-time drafting starts) even though its own loss mask is 0."""
    seq_len, response_start = 16, 8
    valid = build_anchor_candidate_mask(
        seq_len=seq_len, loss_mask=_rl_loss_mask(seq_len, response_start, 6)
    )
    assert valid[0, response_start - 1].item() is True
    # Prompt-interior anchors (first target still in the prompt) stay excluded.
    assert not valid[0, : response_start - 1].any().item()


def test_one_token_response_still_supervised():
    seq_len, response_start = 16, 8
    loss_mask = _rl_loss_mask(seq_len, response_start, 1)
    valid = build_anchor_candidate_mask(seq_len=seq_len, loss_mask=loss_mask)
    assert valid.sum().item() == 1
    assert valid[0, response_start - 1].item() is True

    anchors, keep_mask = sample_anchor_positions(
        seq_len=seq_len, loss_mask=loss_mask, num_anchors=4, device=torch.device("cpu")
    )
    assert keep_mask.sum().item() == 1
    assert anchors[0, 0].item() == response_start - 1


def test_candidate_count_matches_response_length():
    """Every response token gets exactly one anchor (the position before it)."""
    seq_len, response_start, response_len = 32, 10, 7
    valid = build_anchor_candidate_mask(
        seq_len=seq_len,
        loss_mask=_rl_loss_mask(seq_len, response_start, response_len),
    )
    assert valid.sum().item() == response_len
    expected = set(range(response_start - 1, response_start + response_len - 1))
    assert {i for i in range(seq_len - 1) if valid[0, i]} == expected


def test_zero_valid_tokens_yield_no_anchors():
    seq_len = 16
    loss_mask = torch.zeros(1, seq_len)
    anchors, keep_mask = sample_anchor_positions(
        seq_len=seq_len, loss_mask=loss_mask, num_anchors=4, device=torch.device("cpu")
    )
    assert keep_mask.sum().item() == 0
    assert anchors.shape == (1, 4)


def test_anchor_sampling_is_pure_function_of_generator_seed():
    """The draft is replicated across TP peers; anchor draws must depend only
    on the explicit generator seed, never on each rank's ambient RNG state."""
    import torch

    from nemo_rl.models.automodel.draft.common import sample_anchor_positions

    loss_mask = torch.ones(2, 33)

    def draw(seed, perturb_global_rng):
        if perturb_global_rng:
            torch.manual_seed(12345)
            torch.rand(100)  # desynchronize the ambient RNG stream
        else:
            torch.manual_seed(999)
        gen = torch.Generator().manual_seed(seed)
        return sample_anchor_positions(
            seq_len=33,
            loss_mask=loss_mask,
            num_anchors=8,
            device=torch.device("cpu"),
            generator=gen,
        )

    a1, k1 = draw(7, perturb_global_rng=False)
    a2, k2 = draw(7, perturb_global_rng=True)
    assert torch.equal(a1, a2) and torch.equal(k1, k2)

    a3, _ = draw(8, perturb_global_rng=False)
    assert not torch.equal(a1, a3)


def test_anchor_sampling_seed_varies_over_rank_step_and_microbatch():
    from nemo_rl.models.automodel.draft.integration import anchor_sampling_seed

    base = anchor_sampling_seed(0, 1, 1)
    assert anchor_sampling_seed(0, 1, 1) == base  # deterministic
    seeds = {
        anchor_sampling_seed(r, s, m)
        for r in range(4)
        for s in range(1, 4)
        for m in range(1, 4)
    }
    assert len(seeds) == 4 * 3 * 3  # distinct across every counter
    assert all(0 <= s < 2**63 for s in seeds)
