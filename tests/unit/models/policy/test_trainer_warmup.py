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
"""Synthetic batch of the trainer warm-up (CPU only)."""

import pytest
import torch

from nemo_rl.models.policy.trainer_warmup import (
    build_warmup_batch,
    may_draw_token_id,
    warmup_seqlens,
)

TRAIN_FIELDS = (
    "input_ids",
    "input_lengths",
    "generation_logprobs",
    "prev_logprobs",
    "advantages",
    "token_mask",
    "sample_mask",
)


def _packed_cfg(tokens: int = 131072, max_len: int = 131072, pad: int = 128) -> dict:
    return {
        "max_total_sequence_length": max_len,
        "make_sequence_length_divisible_by": pad,
        "sequence_packing": {
            "enabled": True,
            "train_mb_tokens": tokens,
            "algorithm": "modified_first_fit_decreasing",
        },
    }


def _unpacked_cfg(dynamic_tokens: int | None) -> dict:
    return {
        "max_total_sequence_length": 4096,
        "make_sequence_length_divisible_by": 1,
        "dynamic_batching": {
            "enabled": dynamic_tokens is not None,
            "train_mb_tokens": dynamic_tokens,
        },
    }


@pytest.mark.parametrize(
    ("cfg", "expected"),
    [
        (_packed_cfg(), [65536, 65536, 65536, 32768]),
        (_packed_cfg(tokens=1000, max_len=300, pad=64), [256, 256, 256, 192]),
        (_unpacked_cfg(dynamic_tokens=2048), [1024, 1024, 1024, 512]),
        (_unpacked_cfg(dynamic_tokens=None), [2048, 2048, 2048, 1024]),
    ],
    ids=["packed", "clamped-and-padded", "dynamic-batching", "plain"],
)
def test_default_lengths_are_three_halves_and_a_quarter_of_the_budget(cfg, expected):
    assert warmup_seqlens(cfg) == expected


def test_an_override_replaces_the_default_and_is_checked():
    assert warmup_seqlens(_packed_cfg(), "1024, 512") == [1024, 512]
    for override in ("1024,999999", "1", "512,abc", ","):
        with pytest.raises(ValueError, match="NRL_TRAINER_WARMUP_SEQLENS"):
            warmup_seqlens(_packed_cfg(), override)


def test_the_batch_holds_exactly_the_train_fields_in_rollout_dtypes():
    batch = build_warmup_batch(
        [200, 100], train_fields=TRAIN_FIELDS, pad_to_multiple=128
    )

    assert set(batch.keys()) == set(TRAIN_FIELDS)
    assert batch["input_ids"].shape == (2, 256)
    assert batch["input_ids"].dtype == torch.int64
    assert batch["input_lengths"].dtype == torch.int32
    assert batch["token_mask"].dtype == torch.int64
    for name in ("generation_logprobs", "prev_logprobs", "advantages", "sample_mask"):
        assert batch[name].dtype == torch.float32
    assert batch["input_lengths"].tolist() == [200, 100]


def test_prompt_and_padding_are_masked_and_advantages_follow_the_mask():
    batch = build_warmup_batch([64, 64], train_fields=TRAIN_FIELDS, pad_to_multiple=128)
    token_mask = batch["token_mask"][0]

    assert token_mask[:8].sum() == 0
    assert token_mask[8:64].sum() == 56
    assert token_mask[64:].sum() == 0
    assert batch["input_ids"][0, 64:].sum() == 0
    assert torch.equal(batch["advantages"][0], token_mask.to(torch.float32))
    assert torch.equal(batch["advantages"][1], -token_mask.to(torch.float32))


def test_every_rank_builds_the_same_batch_from_the_synthetic_id_range():
    first = build_warmup_batch([300, 200], train_fields=TRAIN_FIELDS)
    second = build_warmup_batch([300, 200], train_fields=TRAIN_FIELDS)

    assert torch.equal(first["input_ids"], second["input_ids"])
    drawn = first["input_ids"][0, :300].tolist() + first["input_ids"][1, :200].tolist()
    assert all(may_draw_token_id(token_id) for token_id in drawn)
    assert not may_draw_token_id(99)
    assert not may_draw_token_id(1000)


def test_an_unknown_train_field_is_rejected():
    with pytest.raises(ValueError, match="teacher_logprobs"):
        build_warmup_batch([16], train_fields=("input_ids", "teacher_logprobs"))


def test_the_default_batch_packs_into_two_large_microbatches_of_different_sizes():
    cfg = _packed_cfg()
    batch = build_warmup_batch(
        warmup_seqlens(cfg), train_fields=TRAIN_FIELDS, pad_to_multiple=128
    )

    packed, _ = batch.shard_by_batch_size(
        shards=1,
        batch_size=None,
        sequence_packing_args={
            "algorithm": "modified_first_fit_decreasing",
            "input_key": "input_ids",
            "input_lengths_key": "input_lengths",
            "sequence_length_pad_multiple": 128,
            "max_tokens_per_microbatch": 131072,
        },
    )

    assert sorted(packed[0].micro_batch_lengths[0]) == [98304, 131072]
