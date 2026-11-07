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
"""Compare block execution against independently executed serial canvases."""

import copy

import pytest
import torch
from just_grpo.algorithms.block_just_grpo import (
    BlockJustGRPOSchedule,
)
from just_grpo.config import ScheduleConfig
from just_grpo.diffusion.block_layout import prepare_block_inputs
from torch import nn

from nemo_rl.distributed.batched_data_dict import BatchedDataDict


def block_mask(padding_mask):
    """Independent dense equal-half mask with explicit loops over blocks."""
    batch, _, _, length = padding_mask.shape
    width = length // 2
    mask = torch.eye(length, dtype=torch.bool)[None].repeat(batch, 1, 1)
    for row in range(batch):
        valid = ~padding_mask[row, 0, 0]
        for start in range(0, width, 4):
            mask[row, start : start + 4, start : start + 4] = True
            mask[row, start : start + 4, width : width + start] = True
        mask[row, width:, width:] = torch.ones(width, width, dtype=torch.bool).tril()
        mask[row] &= valid[:, None] & valid[None, :]
        mask[row] |= torch.diag(~valid)
    return mask[:, None]


class TinyDiffusion(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(32, 8)
        self.projections = nn.ModuleList([nn.Linear(8, 24) for _ in range(2)])
        self.head = nn.Linear(8, 32)

    def forward(self, data):
        if "masked_indices" in data:
            noisy = torch.where(data["masked_indices"], 31, data["input_ids"])
            ids, positions, padding = prepare_block_inputs(
                noisy,
                data["clean_input_ids"],
                prompt_lengths=data["prompt_lengths"],
                noisy_valid_lengths=data["noisy_valid_lengths"],
                clean_lengths=data["clean_lengths"],
                block_size=4,
                mask_token_id=31,
            )
            attention = block_mask(padding)
            width = data["input_ids"].shape[1]
        else:
            ids, positions, attention = (
                data["input_ids"],
                data["position_ids"],
                data["attention_mask"],
            )
            width = ids.shape[1]
        x = self.embedding(ids)
        x = x + positions[..., None] / 100
        for layer in self.projections:
            q, k, v = layer(x).chunk(3, dim=-1)
            x = (
                x
                + torch.nn.functional.scaled_dot_product_attention(
                    q[:, None], k[:, None], v[:, None], attn_mask=attention
                )[:, 0]
            )
        if "masked_indices" in data:
            offsets = data["response_offsets"][:, None] + torch.arange(width)[None]
            x = x.gather(
                1,
                offsets.clamp_max(x.shape[1] - 1)[..., None].expand(
                    -1, -1, x.shape[-1]
                ),
            )
        return (
            self.head(x)
            .log_softmax(-1)
            .gather(-1, data["target_ids"][..., None])
            .squeeze(-1)
        )


def batch():
    ids = torch.tensor(
        [
            [31, 31, 1, 2, 3, 4, 5, 6, 7, 8, 0, 0],
            [31, 31, 31, 31, 31, 2, 3, 4, 9, 8, 7, 0],
        ]
    )
    return BatchedDataDict(
        input_ids=ids,
        input_lengths=torch.tensor([10, 11]),
        token_mask=torch.tensor(
            [[0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0]]
        ),
        sample_mask=torch.ones(2),
    )


def serial(model, data, block, k):
    values = []
    for row in range(2):
        start = int(data["token_mask"][row].nonzero()[0])
        length = int(data["input_lengths"][row])
        out = []
        for token in range(start, length):
            block_start = start + (token - start) // block * block
            reveal = ((token - start) % block) // k * k
            clean = data["input_ids"][row, : block_start + reveal]
            ids = torch.cat([clean, torch.full((block - reveal,), 31)])[None]
            n = ids.shape[1]
            attention = torch.ones(n, n, dtype=torch.bool).tril()
            attention[block_start:, :] = True
            target = torch.zeros_like(ids)
            target[0, token] = data["input_ids"][row, token]
            traj = BatchedDataDict(
                input_ids=ids,
                target_ids=target,
                position_ids=torch.arange(n)[None],
                attention_mask=attention[None, None],
            )
            out.append(model(traj)[0, token])
        values.append(
            torch.nn.functional.pad(
                torch.stack(out), (start, data["input_ids"].shape[1] - length)
            )
        )
    return torch.stack(values)


@pytest.mark.parametrize("k", [1, 2, 3, 4])
def test_logprobs_and_gradients_match_serial(k):
    torch.manual_seed(1)
    data = batch()
    model = TinyDiffusion()
    independent = copy.deepcopy(model)
    schedule = BlockJustGRPOSchedule(
        data,
        ScheduleConfig(mask_token_id=31, block_size=4, reveal_tokens_per_step=k),
        pad_token_id=0,
    )
    accumulated = sum(
        schedule.scatter_logprobs(t, model(t)) for t in schedule.return_schedule(2)
    )
    expected = serial(independent, data, 4, k)
    torch.testing.assert_close(accumulated, expected, atol=1e-6, rtol=1e-5)
    accumulated.sum().backward()
    expected.sum().backward()
    for actual, reference in zip(model.parameters(), independent.parameters()):
        torch.testing.assert_close(actual.grad, reference.grad, atol=1e-5, rtol=1e-4)


@pytest.mark.parametrize("k", [1, 2, 3, 4])
def test_each_token_harvested_once_and_tail_stays_masked(k):
    data = batch()
    schedule = BlockJustGRPOSchedule(
        data,
        ScheduleConfig(mask_token_id=31, block_size=4, reveal_tokens_per_step=k),
        pad_token_id=0,
    )
    coverage = torch.zeros_like(data["input_ids"])
    for t in schedule.return_schedule(2):
        coverage += schedule.scatter_logprobs(t, torch.ones_like(t["input_ids"]))
        assert t["masked_indices"][0, 6:8].all()
        assert t["masked_indices"][1, 3].all()
        torch.testing.assert_close(t["clean_input_ids"], data["input_ids"])
    torch.testing.assert_close(coverage, data["token_mask"])


def test_reject_disjoint_responses():
    data = batch()
    data["token_mask"][0, 6] = 0
    with pytest.raises(ValueError, match="contiguous"):
        BlockJustGRPOSchedule(
            data,
            ScheduleConfig(mask_token_id=31, block_size=4, reveal_tokens_per_step=1),
            pad_token_id=0,
        )


def test_empty_and_excluded_samples():
    data = batch()
    data["sample_mask"][:] = 0
    schedule = BlockJustGRPOSchedule(
        data,
        ScheduleConfig(mask_token_id=31, block_size=4, reveal_tokens_per_step=1),
        pad_token_id=0,
    )
    assert schedule.num_steps == 4
    assert sum(int(t["token_mask"].sum()) for t in schedule.return_schedule(1)) == 0
    for t in schedule.return_schedule(1):
        assert not t["masked_indices"].any()


@pytest.mark.parametrize("prompt_length", [1, 3, 4, 5, 15, 16, 17])
@pytest.mark.parametrize("reveal", [1, 2, 4])
def test_variable_prompts_match_serial_without_prefix(prompt_length, reveal):
    torch.manual_seed(3)
    data = BatchedDataDict(
        input_ids=torch.randint(1, 30, (2, prompt_length + 6)),
        input_lengths=torch.tensor(
            [prompt_length + 6, prompt_length + 5], dtype=torch.int32
        ),
        token_mask=torch.zeros(2, prompt_length + 6, dtype=torch.long),
        sample_mask=torch.ones(2),
    )
    data["token_mask"][:, prompt_length:] = 1
    data["token_mask"][1, -1] = 0
    model = TinyDiffusion()
    independent = copy.deepcopy(model)
    schedule = BlockJustGRPOSchedule(
        data,
        ScheduleConfig(mask_token_id=31, block_size=4, reveal_tokens_per_step=reveal),
        pad_token_id=0,
    )
    actual = sum(schedule.scatter_logprobs(t, model(t)) for t in schedule.iter_levels())
    expected = serial(independent, data, 4, reveal)
    torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-5)
    actual.sum().backward()
    expected.sum().backward()
    for a, b in zip(model.parameters(), independent.parameters()):
        torch.testing.assert_close(a.grad, b.grad, atol=1e-5, rtol=1e-4)
