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

"""CPU unit tests for the BlockJustGRPO block-reveal builder primitives.

Pure-CPU (no GPU / model / Ray). Validates that per reveal level the view
reveals the first ``l*k`` block tokens and the harvest masks partition the
response exactly once, that scatter round-trips noisy positions back to
[N, S], the loss view carries scattered advantages, and the training schedule
emits exactly the per-level views -- across reveal widths k in {1, 2, 3, 4}.
"""
from __future__ import annotations

import torch

from nemo_rl.algorithms.block_just_grpo_logprobs import (
    BlockJustGRPORevealSchedule,
    build_block_reveal_base,
    make_reveal_level_view,
    scatter_block_reveal_logprobs,
)
from nemo_rl.distributed.batched_data_dict import BatchedDataDict

MASK = 100
PAD = 0
BLOCK = 4
SCORE = ("diffu_grpo_score_mask",)
LOSS = ("diffu_grpo_score_mask", "diffu_grpo_loss_mask")


def _make_data() -> BatchedDataDict:
    S, N = 16, 2
    input_ids = torch.arange(1, S + 1).unsqueeze(0).repeat(N, 1).long()
    token_mask = torch.zeros(N, S)
    input_lengths = torch.zeros(N, dtype=torch.long)
    token_mask[0, 3:14] = 1.0
    input_lengths[0] = 14
    token_mask[1, 3:8] = 1.0
    input_lengths[1] = 8
    return BatchedDataDict(
        {
            "input_ids": input_ids,
            "input_lengths": input_lengths,
            "token_mask": token_mask,
            "sample_mask": torch.ones(N),
        }
    )


def test_reveal_pattern_and_harvest_partition():
    data = _make_data()
    base, N, num_levels = build_block_reveal_base(
        data, mask_token_id=MASK, pad_token_id=PAD, block_size=BLOCK, include_loss=False
    )
    assert N == 2 and num_levels == 4
    noisy_len = int(base["diffu_grpo_noisy_lengths"][0].item())
    resp = base["diffu_grpo_response_lengths"]
    cover = torch.zeros(N, noisy_len)
    for j in range(num_levels):
        view = make_reveal_level_view(base, j, BLOCK, SCORE)
        target = view["diffu_grpo_target_ids"]
        for s in range(N):
            for rel in range(int(resp[s].item())):
                off = rel % BLOCK
                if off < j:
                    assert int(view["input_ids"][s, rel].item()) == int(target[s, rel].item())
                else:
                    assert int(view["input_ids"][s, rel].item()) == MASK
        cover += view["block_reveal_harvest_mask"][:, :noisy_len]
    for s in range(N):
        r = int(resp[s].item())
        assert torch.all(cover[s, :r] == 1.0) and torch.all(cover[s, r:] == 0.0)


def test_reveal_k_levels_partition_and_scatter():
    """k tokens revealed/harvested per level: correct level count, the harvest
    windows still partition the response exactly once, and scatter round-trips."""
    data = _make_data()
    for k, expect_levels in ((1, 4), (2, 2), (3, 2), (4, 1)):
        base, N, num_levels = build_block_reveal_base(
            data,
            mask_token_id=MASK,
            pad_token_id=PAD,
            block_size=BLOCK,
            include_loss=False,
            reveal_tokens_per_level=k,
        )
        assert num_levels == expect_levels, (k, num_levels)
        noisy_len = int(base["diffu_grpo_noisy_lengths"][0].item())
        noisy_offset = int(base["diffu_grpo_noisy_response_offsets"][0].item())
        resp = base["diffu_grpo_response_lengths"]
        cover = torch.zeros(N, noisy_len)
        out = torch.zeros(N, data["input_ids"].shape[1])
        for level in range(num_levels):
            view = make_reveal_level_view(base, level, BLOCK, SCORE, k)
            target = view["diffu_grpo_target_ids"]
            for s in range(N):
                for rel in range(int(resp[s].item())):
                    off = rel % BLOCK
                    if off < level * k:
                        assert int(view["input_ids"][s, rel].item()) == int(
                            target[s, rel].item()
                        )
                    else:
                        assert int(view["input_ids"][s, rel].item()) == MASK
            cover += view["block_reveal_harvest_mask"][:, :noisy_len]
            starts = view["diffu_grpo_completion_starts"]
            flat = torch.zeros(N, noisy_len)
            for s in range(N):
                flat[s] = torch.arange(noisy_len) + int(starts[s].item()) + 1.0
            out += scatter_block_reveal_logprobs(
                flat_logprobs=flat,
                harvest_mask=view["block_reveal_harvest_mask"],
                sample_index=view["block_reveal_sample_index"],
                completion_starts=starts,
                noisy_response_offset=noisy_offset,
                original_seq_len=data["input_ids"].shape[1],
                num_samples=N,
            )
        for s in range(N):
            r = int(resp[s].item())
            assert torch.all(cover[s, :r] == 1.0) and torch.all(cover[s, r:] == 0.0)
        for s, (a, b) in {0: (3, 14), 1: (3, 8)}.items():
            assert torch.allclose(out[s, a:b], torch.arange(a, b) + 1.0)
            m = torch.ones(out.shape[1], dtype=torch.bool)
            m[a:b] = False
            assert torch.all(out[s, m] == 0.0)


def test_scatter_roundtrip():
    data = _make_data()
    base, N, num_levels = build_block_reveal_base(
        data, mask_token_id=MASK, pad_token_id=PAD, block_size=BLOCK, include_loss=False
    )
    noisy_len = int(base["diffu_grpo_noisy_lengths"][0].item())
    noisy_offset = int(base["diffu_grpo_noisy_response_offsets"][0].item())
    out = torch.zeros(N, data["input_ids"].shape[1])
    for j in range(num_levels):
        view = make_reveal_level_view(base, j, BLOCK, SCORE)
        starts = view["diffu_grpo_completion_starts"]
        # encode each noisy position as (mapped original position + 1)
        flat = torch.zeros(N, noisy_len)
        for s in range(N):
            flat[s] = torch.arange(noisy_len) + int(starts[s].item()) + 1.0
        out += scatter_block_reveal_logprobs(
            flat_logprobs=flat,
            harvest_mask=view["block_reveal_harvest_mask"],
            sample_index=view["block_reveal_sample_index"],
            completion_starts=starts,
            noisy_response_offset=noisy_offset,
            original_seq_len=data["input_ids"].shape[1],
            num_samples=N,
        )
    for s, (a, b) in {0: (3, 14), 1: (3, 8)}.items():
        assert torch.allclose(out[s, a:b], torch.arange(a, b) + 1.0)
        m = torch.ones(out.shape[1], dtype=torch.bool)
        m[a:b] = False
        assert torch.all(out[s, m] == 0.0)


def test_loss_view():
    data = _make_data()
    N, S = data["input_ids"].shape
    data["advantages"] = torch.arange(N * S, dtype=torch.float32).reshape(N, S)
    data["prev_logprobs"] = torch.zeros(N, S)
    data["generation_logprobs"] = torch.zeros(N, S)
    base, _, num_levels = build_block_reveal_base(
        data, mask_token_id=MASK, pad_token_id=PAD, block_size=BLOCK, include_loss=True
    )
    noisy_len = int(base["diffu_grpo_noisy_lengths"][0].item())
    for j in range(num_levels):
        view = make_reveal_level_view(base, j, BLOCK, LOSS)
        assert torch.allclose(view["diffu_grpo_loss_mask"], view["block_reveal_harvest_mask"])
        starts = view["diffu_grpo_completion_starts"]
        for s in range(N):
            start = int(starts[s].item())
            harvested = torch.nonzero(
                view["block_reveal_harvest_mask"][s, :noisy_len] > 0.5, as_tuple=False
            ).flatten().tolist()
            for rel in harvested:
                assert float(view["advantages"][s, rel].item()) == float(
                    data["advantages"][s, start + rel].item()
                )


def test_reveal_schedule():
    """Schedule microbatches == the per-level views concatenated (training path)."""
    data = _make_data()
    N, S = data["input_ids"].shape
    data["advantages"] = torch.arange(N * S, dtype=torch.float32).reshape(N, S)
    data["prev_logprobs"] = torch.zeros(N, S)
    data["generation_logprobs"] = torch.zeros(N, S)
    base, n, num_levels = build_block_reveal_base(
        data, mask_token_id=MASK, pad_token_id=PAD, block_size=BLOCK, include_loss=True
    )
    schedule = BlockJustGRPORevealSchedule(base).configure(
        num_levels=num_levels, block_size=BLOCK, harvest_keys=LOSS
    )
    mbs = 1  # N == 2; two sample microbatches per reveal level
    assert schedule.size == num_levels * n
    mbs_list = list(schedule.make_microbatch_iterator(mbs))
    assert len(mbs_list) == schedule.size // mbs
    for key in ("input_ids", "diffu_grpo_loss_mask"):
        got = torch.cat([mb[key] for mb in mbs_list], dim=0)
        ref = torch.cat(
            [make_reveal_level_view(base, j, BLOCK, LOSS)[key] for j in range(num_levels)],
            dim=0,
        )
        assert torch.equal(got, ref), key

