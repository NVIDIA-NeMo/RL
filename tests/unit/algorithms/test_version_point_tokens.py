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
"""Tokenwise VPG planning, gradient and optimizer equivalence."""

import copy

import pytest
import torch

from nemo_rl.algorithms.version_point_tokens import (
    generated_token_versions,
    plan_token_version_passes,
    token_version_candidates,
    token_version_pass_data,
)
from nemo_rl.distributed.batched_data_dict import BatchedDataDict
from nemo_rl.models.megatron.version_snapshots import VersionSnapshots


def test_generated_token_versions_follow_the_engine_epoch_boundaries():
    # Prompt of 4 tokens (positions 0-3). Epoch 5 from the start; a refit while the
    # request had 3 generated tokens appends (4 + 3 - 1, 6): the forward at position
    # 6 and later runs at 6, so generated token 3 (position 7) is the first at 6.
    assert generated_token_versions([(0, 5), (6, 6)], 4, 6) == [5, 5, 5, 6, 6, 6]
    # Refit before the first token (request waiting): every token at the new epoch.
    assert generated_token_versions([(0, 5), (3, 6)], 4, 2) == [6, 6]
    # Two refits with no token in between: the later one wins.
    assert generated_token_versions([(0, 1), (5, 2), (5, 3)], 4, 3) == [1, 1, 3]
    assert generated_token_versions(None, 4, 3) is None
    assert generated_token_versions([], 4, 3) is None
    with pytest.raises(ValueError):
        generated_token_versions([(0, 1), (5, 2), (4, 3)], 4, 3)


VOCAB, DIM = 11, 6


class _CausalLM(torch.nn.Module):
    """Embedding -> causal prefix mean -> tanh -> vocab, params viewing one flat buffer."""

    def __init__(self):
        super().__init__()
        torch.manual_seed(0)
        self.embed = torch.nn.Parameter(torch.randn(VOCAB, DIM, dtype=torch.float32))
        self.proj = torch.nn.Parameter(torch.randn(DIM, VOCAB, dtype=torch.float32))
        params = [self.embed, self.proj]
        self.flat = torch.cat([p.detach().reshape(-1) for p in params])
        offset = 0
        for p in params:
            p.data = self.flat[offset : offset + p.numel()].view_as(p)
            offset += p.numel()

    def token_logprobs(self, ids):
        """log p(ids[i] | ids[:i]) for i >= 1, shape [len - 1]."""
        h = self.embed[ids]
        h = torch.cumsum(h, 0) / torch.arange(1, len(ids) + 1, dtype=h.dtype)[:, None]
        logits = torch.tanh(h) @ self.proj
        return torch.log_softmax(logits[:-1], -1).gather(1, ids[1:, None])[:, 0]


def _pg_loss(model, ids, length, token_mask, advantage):
    """Un-normalized token-level policy-gradient loss of one row (ratio 1)."""
    logprobs = model.token_logprobs(ids[:length])
    return -(advantage * logprobs * token_mask[1:length]).sum()


def _batch():
    torch.manual_seed(12)
    stamps = torch.tensor(
        [
            [0, 0, 1, 1, 2, 2, 2, 2],
            [0, 0, 2, 2, 2, 0, 0, 0],
            [0, 0, 1, 1, 1, 1, 0, 0],
            [0, 0, 3, 3, 3, 3, 3, 3],
        ]
    )
    return BatchedDataDict(
        {
            "input_ids": torch.randint(VOCAB, (4, 8)),
            "input_lengths": torch.tensor([8, 5, 6, 8]),
            "generation_versions": stamps,
            "token_mask": (stamps > 0).float(),
            "sample_mask": torch.tensor([1.0, 0.5, 1.5, 0.0]),
            "advantages": torch.tensor([0.7, -1.3, 0.4, 2.0]),
        }
    )


def test_candidates_ignore_masked_rows_and_fall_back_on_unknown_or_evicted_tokens():
    data = _batch()
    assert token_version_candidates(data, lambda v: v in (0, 1))[0] == [0, 1]
    assert token_version_candidates(data, lambda v: v == 0)[0] == []
    data["generation_versions"][0, 3] = 0
    versions, metrics = token_version_candidates(data, lambda v: True)
    assert not versions and metrics["version_point_unstamped_tokens"] == 1
    del data["generation_versions"]
    assert not token_version_candidates(data, lambda v: True)[0]


@pytest.mark.parametrize("row_multiple", [1, 2, 4, 8])
def test_passes_partition_weighted_loss_tokens_and_leave_original_unchanged(
    row_multiple,
):
    data = _batch()
    before = copy.deepcopy(data)
    passes = plan_token_version_passes(data, [0, 1], row_multiple=row_multiple)
    observed = torch.zeros_like(data["token_mask"])
    count = 0.0
    sequences = 0.0
    for plan in passes:
        sub = token_version_pass_data(data, plan)
        assert len(plan.rows) % row_multiple == 0
        assert sub["sample_mask"][plan.num_real_rows :].sum() == 0
        assert sub["token_mask"][plan.num_real_rows :].sum() == 0
        assert sub["input_ids"].shape[1] == max(plan.lengths)
        for i, row in enumerate(plan.rows[: plan.num_real_rows]):
            width = sub["token_mask"].shape[1]
            observed[row, :width] += sub["token_mask"][i]
        count += (sub["token_mask"] * sub["sample_mask"][:, None]).sum()
        sequences += sub["version_point_sequence_count"].sum()
    expected = data["token_mask"] * (data["sample_mask"][:, None] > 0)
    torch.testing.assert_close(observed, expected, rtol=0, atol=0)
    torch.testing.assert_close(
        count, (data["token_mask"] * data["sample_mask"][:, None]).sum()
    )
    torch.testing.assert_close(sequences, data["sample_mask"].sum())
    for key in data:
        torch.testing.assert_close(data[key], before[key], rtol=0, atol=0)


@pytest.mark.parametrize("row_multiple", [1, 4])
def test_grouped_gradients_and_current_weight_adam_step_match_per_token_oracle(
    row_multiple,
):
    data = _batch()
    model = _CausalLM()
    snapshots = VersionSnapshots(
        local_shards=lambda: [model.flat],
        sync=lambda: None,
        keep_versions=3,
        invalidate_weight_caches=lambda: None,
    )
    states = {}
    for version in range(3):
        with torch.no_grad():
            model.flat.add_(torch.randn_like(model.flat) * 0.1)
        snapshots.snapshot(version)
        states[version] = model.flat.clone()
    current = model.flat.clone()
    denominator = (data["token_mask"][:, 1:] * data["sample_mask"][:, None]).sum()
    model.zero_grad()
    passes = plan_token_version_passes(data, [0, 1], row_multiple=row_multiple)
    for plan in passes:
        sub = token_version_pass_data(data, plan)
        with snapshots.swapped_in(plan.version):
            loss = (
                sum(
                    _pg_loss(
                        model,
                        sub["input_ids"][i],
                        int(length),
                        sub["token_mask"][i],
                        sub["advantages"][i] * sub["sample_mask"][i],
                    )
                    for i, length in enumerate(sub["input_lengths"])
                )
                / denominator
            )
            loss.backward()
        assert torch.equal(model.flat, current)
    grouped = [p.grad.clone() for p in model.parameters()]
    reference = [torch.zeros_like(p) for p in model.parameters()]
    for row in range(data.size):
        if data["sample_mask"][row] == 0:
            continue
        for pos in range(1, int(data["input_lengths"][row])):
            if data["token_mask"][row, pos] == 0:
                continue
            fresh = _CausalLM()
            with torch.no_grad():
                version = int(data["generation_versions"][row, pos]) - 1
                fresh.flat.copy_(states[version])
            one = torch.zeros(8)
            one[pos] = 1
            (
                _pg_loss(
                    fresh,
                    data["input_ids"][row],
                    int(data["input_lengths"][row]),
                    one,
                    data["advantages"][row] * data["sample_mask"][row],
                )
                / denominator
            ).backward()
            for grad, param in zip(reference, fresh.parameters(), strict=True):
                grad.add_(param.grad)
    for actual, expected in zip(grouped, reference, strict=True):
        torch.testing.assert_close(actual, expected, rtol=2e-5, atol=1e-7)
    fresh = _CausalLM()
    with torch.no_grad():
        fresh.flat.copy_(current)
    for param, grad in zip(fresh.parameters(), reference, strict=True):
        param.grad = grad
    torch.optim.AdamW(model.parameters(), lr=0.01).step()
    torch.optim.AdamW(fresh.parameters(), lr=0.01).step()
    torch.testing.assert_close(model.flat, fresh.flat, rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("row_multiple", [0, -1])
def test_invalid_row_multiple_fails(row_multiple):
    with pytest.raises(ValueError, match="row_multiple"):
        plan_token_version_passes(_batch(), [0], row_multiple=row_multiple)
