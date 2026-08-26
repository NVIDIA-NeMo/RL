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

"""Tests for the masked-diffusion block-denoising sampler."""

import pytest
import torch

from gdpo.denoise import (
    block_denoise,
    build_canvas,
    get_num_transfer_tokens,
    unpack_generations,
)

MASK_ID = 99
PAD_ID = 0
VOCAB = 16


def constant_logits_fn(favored: int, vocab: int = VOCAB):
    """A stub model that always prefers a single token id."""

    def logits_fn(input_ids, attention_mask):
        logits = torch.zeros(*input_ids.shape, vocab)
        logits[..., favored] = 10.0
        return logits

    return logits_fn


def positional_logits_fn(vocab: int = VOCAB):
    """A stub model that predicts ``position % vocab`` at every position."""

    def logits_fn(input_ids, attention_mask):
        seq_len = input_ids.shape[1]
        logits = torch.zeros(*input_ids.shape, vocab)
        preferred = torch.arange(seq_len) % vocab
        logits.scatter_(
            -1, preferred.view(1, seq_len, 1).expand(input_ids.shape[0], -1, -1), 10.0
        )
        return logits

    return logits_fn


class TestGetNumTransferTokens:
    def test_budgets_sum_to_the_masked_count(self):
        mask = torch.tensor([[True] * 7 + [False] * 3, [True] * 4 + [False] * 6])
        budgets = get_num_transfer_tokens(mask, steps=3)
        assert budgets.shape == (2, 3)
        assert budgets.sum(dim=1).tolist() == [7, 4]

    def test_remainder_goes_to_the_earliest_steps(self):
        mask = torch.tensor([[True] * 7])
        budgets = get_num_transfer_tokens(mask, steps=3)
        assert budgets[0].tolist() == [3, 2, 2]

    def test_all_masked_rows_split_evenly_when_divisible(self):
        mask = torch.tensor([[True] * 8])
        budgets = get_num_transfer_tokens(mask, steps=4)
        assert budgets[0].tolist() == [2, 2, 2, 2]

    def test_a_fully_unmasked_row_gets_a_zero_budget(self):
        mask = torch.zeros(1, 5, dtype=torch.bool)
        budgets = get_num_transfer_tokens(mask, steps=3)
        assert budgets[0].tolist() == [0, 0, 0]

    def test_more_steps_than_masked_tokens_still_sums_correctly(self):
        mask = torch.tensor([[True, True, False, False]])
        budgets = get_num_transfer_tokens(mask, steps=5)
        assert budgets.sum().item() == 2
        assert budgets[0].tolist() == [1, 1, 0, 0, 0]

    def test_the_dtype_is_int64(self):
        mask = torch.tensor([[True, False]])
        assert get_num_transfer_tokens(mask, steps=2).dtype == torch.int64


class TestBuildCanvas:
    def test_right_padded_prompts_are_left_aligned(self):
        input_ids = torch.tensor([[5, 6, 7, PAD_ID], [8, 9, PAD_ID, PAD_ID]])
        lengths = torch.tensor([3, 2])
        canvas, _ = build_canvas(
            input_ids, lengths, gen_length=2, mask_id=MASK_ID, pad_id=PAD_ID
        )
        assert canvas[0].tolist() == [PAD_ID, 5, 6, 7, MASK_ID, MASK_ID]
        assert canvas[1].tolist() == [PAD_ID, PAD_ID, 8, 9, MASK_ID, MASK_ID]

    def test_the_attention_mask_excludes_only_the_left_padding(self):
        input_ids = torch.tensor([[5, 6, 7, PAD_ID], [8, 9, PAD_ID, PAD_ID]])
        lengths = torch.tensor([3, 2])
        _, attention_mask = build_canvas(
            input_ids, lengths, gen_length=2, mask_id=MASK_ID, pad_id=PAD_ID
        )
        assert attention_mask[0].tolist() == [False, True, True, True, True, True]
        assert attention_mask[1].tolist() == [False, False, True, True, True, True]

    def test_the_generation_region_starts_fully_masked(self):
        input_ids = torch.tensor([[5, 6]])
        canvas, _ = build_canvas(
            input_ids, torch.tensor([2]), gen_length=4, mask_id=MASK_ID, pad_id=PAD_ID
        )
        assert (canvas[:, 2:] == MASK_ID).all()

    def test_a_full_length_prompt_needs_no_realignment(self):
        input_ids = torch.tensor([[5, 6, 7]])
        canvas, attention_mask = build_canvas(
            input_ids, torch.tensor([3]), gen_length=1, mask_id=MASK_ID, pad_id=PAD_ID
        )
        assert canvas[0, :3].tolist() == [5, 6, 7]
        assert attention_mask.all()

    def test_the_canvas_width_is_prompt_plus_generation(self):
        canvas, attention_mask = build_canvas(
            torch.zeros(3, 5, dtype=torch.long),
            torch.tensor([5, 5, 5]),
            gen_length=7,
            mask_id=MASK_ID,
            pad_id=PAD_ID,
        )
        assert canvas.shape == (3, 12)
        assert attention_mask.shape == (3, 12)

    def test_the_prompt_token_order_is_preserved(self):
        input_ids = torch.tensor([[11, 12, 13, 14, PAD_ID]])
        canvas, _ = build_canvas(
            input_ids, torch.tensor([4]), gen_length=1, mask_id=MASK_ID, pad_id=PAD_ID
        )
        assert canvas[0, 1:5].tolist() == [11, 12, 13, 14]


class TestBlockDenoise:
    @pytest.fixture
    def canvas(self):
        input_ids = torch.tensor([[5, 6, 7, PAD_ID], [8, 9, PAD_ID, PAD_ID]])
        return build_canvas(
            input_ids,
            torch.tensor([3, 2]),
            gen_length=4,
            mask_id=MASK_ID,
            pad_id=PAD_ID,
        )

    def test_no_masked_position_survives(self, canvas):
        x, attention_mask = canvas
        out = block_denoise(
            constant_logits_fn(3),
            x,
            attention_mask,
            gen_start=4,
            mask_id=MASK_ID,
            steps=4,
            block_length=2,
        )
        assert not (out == MASK_ID).any()

    def test_the_prompt_region_is_untouched(self, canvas):
        x, attention_mask = canvas
        out = block_denoise(
            constant_logits_fn(3),
            x,
            attention_mask,
            gen_start=4,
            mask_id=MASK_ID,
            steps=4,
            block_length=2,
        )
        assert torch.equal(out[:, :4], x[:, :4])

    def test_the_input_canvas_is_not_mutated(self, canvas):
        x, attention_mask = canvas
        before = x.clone()
        block_denoise(
            constant_logits_fn(3),
            x,
            attention_mask,
            gen_start=4,
            mask_id=MASK_ID,
            steps=4,
            block_length=2,
        )
        assert torch.equal(x, before)

    def test_greedy_decoding_writes_the_favored_token(self, canvas):
        x, attention_mask = canvas
        out = block_denoise(
            constant_logits_fn(3),
            x,
            attention_mask,
            gen_start=4,
            mask_id=MASK_ID,
            steps=4,
            block_length=2,
        )
        assert out[:, 4:].tolist() == [[3, 3, 3, 3], [3, 3, 3, 3]]

    def test_blocks_are_denoised_left_to_right(self, canvas):
        x, attention_mask = canvas
        seen = []

        def recording_fn(input_ids, mask):
            seen.append(input_ids.clone())
            return constant_logits_fn(3)(input_ids, mask)

        block_denoise(
            recording_fn,
            x,
            attention_mask,
            gen_start=4,
            mask_id=MASK_ID,
            steps=4,
            block_length=2,
        )
        # Two steps per block: the first block's steps must still see the
        # second block fully masked.
        assert len(seen) == 4
        assert (seen[0][:, 6:] == MASK_ID).all()
        assert (seen[1][:, 6:] == MASK_ID).all()
        # By the second block's first step, the first block is fully written.
        assert not (seen[2][:, 4:6] == MASK_ID).any()

    def test_the_forward_count_matches_the_step_budget(self, canvas):
        x, attention_mask = canvas
        calls = []

        def counting_fn(input_ids, mask):
            calls.append(input_ids.shape[0])
            return constant_logits_fn(3)(input_ids, mask)

        block_denoise(
            counting_fn,
            x,
            attention_mask,
            gen_start=4,
            mask_id=MASK_ID,
            steps=8,
            block_length=2,
        )
        assert len(calls) == 8

    def test_guidance_doubles_the_forward_batch(self, canvas):
        x, attention_mask = canvas
        batch_sizes = []

        def counting_fn(input_ids, mask):
            batch_sizes.append(input_ids.shape[0])
            return constant_logits_fn(3)(input_ids, mask)

        block_denoise(
            counting_fn,
            x,
            attention_mask,
            gen_start=4,
            mask_id=MASK_ID,
            steps=2,
            block_length=4,
            cfg_scale=1.0,
        )
        assert batch_sizes == [4, 4]

    def test_guidance_masks_the_prompt_in_the_unconditional_half(self, canvas):
        x, attention_mask = canvas
        seen = []

        def recording_fn(input_ids, mask):
            seen.append(input_ids.clone())
            return constant_logits_fn(3)(input_ids, mask)

        block_denoise(
            recording_fn,
            x,
            attention_mask,
            gen_start=4,
            mask_id=MASK_ID,
            steps=1,
            block_length=4,
            cfg_scale=1.0,
        )
        conditional, unconditional = torch.chunk(seen[0], 2, dim=0)
        assert torch.equal(conditional, x)
        assert (unconditional[:, :4] == MASK_ID).all()

    def test_a_non_dividing_block_length_is_rejected(self, canvas):
        x, attention_mask = canvas
        with pytest.raises(ValueError, match="must divide the generation width"):
            block_denoise(
                constant_logits_fn(3),
                x,
                attention_mask,
                gen_start=4,
                mask_id=MASK_ID,
                steps=4,
                block_length=3,
            )

    def test_sampling_is_reproducible_for_a_fixed_seed(self, canvas):
        x, attention_mask = canvas

        def noisy_fn(input_ids, mask):
            return (
                torch.arange(VOCAB, dtype=torch.float32).expand(*input_ids.shape, VOCAB)
                * 0.1
            )

        outs = []
        for _ in range(2):
            generator = torch.Generator().manual_seed(1234)
            outs.append(
                block_denoise(
                    noisy_fn,
                    x,
                    attention_mask,
                    gen_start=4,
                    mask_id=MASK_ID,
                    steps=4,
                    block_length=2,
                    temperature=1.0,
                    generator=generator,
                )
            )
        assert torch.equal(outs[0], outs[1])

    def test_different_seeds_can_diverge(self, canvas):
        x, attention_mask = canvas

        def flat_fn(input_ids, mask):
            return torch.zeros(*input_ids.shape, VOCAB)

        outs = []
        for seed in (1, 2):
            generator = torch.Generator().manual_seed(seed)
            outs.append(
                block_denoise(
                    flat_fn,
                    x,
                    attention_mask,
                    gen_start=4,
                    mask_id=MASK_ID,
                    steps=4,
                    block_length=2,
                    temperature=1.0,
                    generator=generator,
                )
            )
        assert not torch.equal(outs[0], outs[1])

    def test_top_k_excludes_filtered_tokens(self, canvas):
        x, attention_mask = canvas

        def ramp_fn(input_ids, mask):
            return torch.arange(VOCAB, dtype=torch.float32).expand(
                *input_ids.shape, VOCAB
            )

        generator = torch.Generator().manual_seed(0)
        out = block_denoise(
            ramp_fn,
            x,
            attention_mask,
            gen_start=4,
            mask_id=MASK_ID,
            steps=4,
            block_length=2,
            temperature=1.0,
            top_k=2,
            generator=generator,
        )
        # Only the two highest-scoring ids survive the filter.
        assert set(out[:, 4:].flatten().tolist()) <= {VOCAB - 1, VOCAB - 2}

    def test_a_single_block_covers_the_whole_generation_region(self, canvas):
        x, attention_mask = canvas
        out = block_denoise(
            constant_logits_fn(7),
            x,
            attention_mask,
            gen_start=4,
            mask_id=MASK_ID,
            steps=1,
            block_length=4,
        )
        assert out[:, 4:].tolist() == [[7] * 4, [7] * 4]

    def test_more_steps_than_positions_still_fills_the_block(self, canvas):
        x, attention_mask = canvas
        out = block_denoise(
            constant_logits_fn(3),
            x,
            attention_mask,
            gen_start=4,
            mask_id=MASK_ID,
            steps=64,
            block_length=4,
        )
        assert not (out == MASK_ID).any()

    def test_position_dependent_predictions_land_at_their_positions(self, canvas):
        x, attention_mask = canvas
        out = block_denoise(
            positional_logits_fn(),
            x,
            attention_mask,
            gen_start=4,
            mask_id=MASK_ID,
            steps=4,
            block_length=2,
        )
        assert out[0, 4:].tolist() == [4, 5, 6, 7]


class TestUnpackGenerations:
    def test_the_response_stops_at_the_first_stop_token(self):
        canvas = torch.tensor([[PAD_ID, 5, 6, 1, 2, 3, 4]])
        out = unpack_generations(
            canvas,
            torch.tensor([2]),
            gen_start=3,
            eos_token_ids=[3],
            pad_id=PAD_ID,
        )
        assert out["generation_lengths"].tolist() == [3]
        assert out["output_ids"][0, :5].tolist() == [5, 6, 1, 2, 3]

    def test_tokens_after_the_stop_token_are_padded_away(self):
        canvas = torch.tensor([[PAD_ID, 5, 6, 1, 3, 8, 9]])
        out = unpack_generations(
            canvas, torch.tensor([2]), gen_start=3, eos_token_ids=[3], pad_id=PAD_ID
        )
        assert out["output_ids"][0].tolist() == [5, 6, 1, 3, PAD_ID, PAD_ID, PAD_ID]

    def test_a_row_without_a_stop_token_is_truncated(self):
        canvas = torch.tensor([[PAD_ID, 5, 6, 1, 2, 4, 7]])
        out = unpack_generations(
            canvas, torch.tensor([2]), gen_start=3, eos_token_ids=[3], pad_id=PAD_ID
        )
        assert out["truncated"].tolist() == [True]
        assert out["generation_lengths"].tolist() == [4]

    def test_a_row_with_a_stop_token_is_not_truncated(self):
        canvas = torch.tensor([[PAD_ID, 5, 6, 3, 2, 4, 7]])
        out = unpack_generations(
            canvas, torch.tensor([2]), gen_start=3, eos_token_ids=[3], pad_id=PAD_ID
        )
        assert out["truncated"].tolist() == [False]

    def test_any_of_several_stop_tokens_terminates(self):
        canvas = torch.tensor([[PAD_ID, 5, 6, 1, 4, 2, 7]])
        out = unpack_generations(
            canvas, torch.tensor([2]), gen_start=3, eos_token_ids=[3, 4], pad_id=PAD_ID
        )
        assert out["generation_lengths"].tolist() == [2]

    def test_unpadded_lengths_are_prompt_plus_generation(self):
        canvas = torch.tensor(
            [[PAD_ID, 5, 6, 1, 3, 8, 9], [PAD_ID, PAD_ID, 7, 3, 8, 9, 9]]
        )
        out = unpack_generations(
            canvas,
            torch.tensor([2, 1]),
            gen_start=3,
            eos_token_ids=[3],
            pad_id=PAD_ID,
        )
        assert out["generation_lengths"].tolist() == [2, 1]
        assert out["unpadded_sequence_lengths"].tolist() == [4, 2]

    def test_rows_of_different_prompt_lengths_are_left_justified(self):
        canvas = torch.tensor(
            [[PAD_ID, 5, 6, 1, 3, 8, 9], [PAD_ID, PAD_ID, 7, 2, 3, 9, 9]]
        )
        out = unpack_generations(
            canvas,
            torch.tensor([2, 1]),
            gen_start=3,
            eos_token_ids=[3],
            pad_id=PAD_ID,
        )
        assert out["output_ids"][0, :4].tolist() == [5, 6, 1, 3]
        assert out["output_ids"][1, :3].tolist() == [7, 2, 3]

    def test_output_width_matches_the_canvas_width(self):
        canvas = torch.full((3, 9), 5)
        out = unpack_generations(
            canvas,
            torch.tensor([4, 3, 2]),
            gen_start=4,
            eos_token_ids=[3],
            pad_id=PAD_ID,
        )
        assert out["output_ids"].shape == (3, 9)

    def test_a_stop_token_at_the_first_generated_position(self):
        canvas = torch.tensor([[PAD_ID, 5, 6, 3, 8, 9, 9]])
        out = unpack_generations(
            canvas, torch.tensor([2]), gen_start=3, eos_token_ids=[3], pad_id=PAD_ID
        )
        assert out["generation_lengths"].tolist() == [1]
        assert out["output_ids"][0, :3].tolist() == [5, 6, 3]

    def test_round_trips_a_prompt_through_build_and_unpack(self):
        input_ids = torch.tensor([[5, 6, 7, PAD_ID], [8, 9, PAD_ID, PAD_ID]])
        lengths = torch.tensor([3, 2])
        canvas, attention_mask = build_canvas(
            input_ids, lengths, gen_length=4, mask_id=MASK_ID, pad_id=PAD_ID
        )
        denoised = block_denoise(
            constant_logits_fn(3),
            canvas,
            attention_mask,
            gen_start=4,
            mask_id=MASK_ID,
            steps=4,
            block_length=2,
        )
        out = unpack_generations(
            denoised,
            lengths,
            gen_start=4,
            eos_token_ids=[11],
            pad_id=PAD_ID,
        )
        # No stop token was generated, so every row keeps all four tokens.
        assert out["generation_lengths"].tolist() == [4, 4]
        assert out["output_ids"][0, :7].tolist() == [5, 6, 7, 3, 3, 3, 3]
        assert out["output_ids"][1, :6].tolist() == [8, 9, 3, 3, 3, 3]
