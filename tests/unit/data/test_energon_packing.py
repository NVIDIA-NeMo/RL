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

from __future__ import annotations

import pytest
import torch

pytest.importorskip("megatron.energon")
pytest.importorskip("megatron.core")

pytestmark = pytest.mark.mcore

from nemo_rl.data.energon.multimodal.packing import (  # noqa: E402
    pack_selected_samples,
    prepare_packed_sft_batch,
    select_samples_to_pack,
)
from nemo_rl.data.energon.multimodal.types import EncodedSFTSample  # noqa: E402
from nemo_rl.data.multimodal_utils import PackedTensor  # noqa: E402
from nemo_rl.data.packing import PackingAlgorithm, get_packer  # noqa: E402


class _Tokenizer:
    pad_token_id = 0


def _sample(
    key: str,
    length: int,
    *,
    group: str = "text",
    packing_cost: int | None = None,
) -> EncodedSFTSample:
    user_length = max(1, length - 2)
    return EncodedSFTSample(
        __key__=key,
        __restore_key__=(key,),
        message_log=[
            {"role": "user", "token_ids": torch.arange(1, user_length + 1)},
            {
                "role": "assistant",
                "token_ids": torch.arange(user_length + 1, length + 1),
            },
        ],
        length=length,
        packing_cost=length if packing_cost is None else packing_cost,
        loss_multiplier=1.0,
        group_key=(group,),
        sample_key=key,
    )


@pytest.mark.parametrize("algorithm", list(PackingAlgorithm))
def test_selection_uses_aligned_costs_and_keeps_groups_separate(
    algorithm: PackingAlgorithm,
) -> None:
    samples = [
        _sample("s0", 5),
        _sample("s1", 3),
        _sample("s2", 3, group="image"),
    ]

    selected = select_samples_to_pack(
        samples,
        packer=get_packer(algorithm, 12),
        sequence_length_pad_multiple=4,
    )

    assert [{sample.sample_key for sample in pack} for pack in selected] == [
        {"s0", "s1"},
        {"s2"},
    ]


def test_preparation_builds_model_ready_pack_and_jagged_boundaries() -> None:
    packed = pack_selected_samples(
        [_sample("s0", 5), _sample("s1", 3)],
        pack_capacity=12,
        sequence_length_pad_multiple=4,
    )

    second_pack = pack_selected_samples(
        [_sample("s2", 4)],
        pack_capacity=12,
        sequence_length_pad_multiple=4,
    )
    prepared = prepare_packed_sft_batch(
        [packed, second_pack], tokenizer=_Tokenizer(), only_unmask_final=False
    )

    assert prepared["input_ids"][0].tolist() == [1, 2, 3, 4, 5, 0, 0, 0, 1, 2, 3, 0]
    assert prepared["token_mask"][0].tolist() == [0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 0]
    assert prepared["input_lengths"].tolist() == [12, 12]
    assert prepared["source_ids"] == [["s0", "s1"], ["s2"]]
    assert isinstance(prepared["cu_seqlens"], PackedTensor)
    assert isinstance(prepared["cu_seqlens_padded"], PackedTensor)
    first = prepared.slice(0, 1)
    assert first["cu_seqlens"].as_tensor().tolist() == [0, 5, 8]
    assert first["cu_seqlens_padded"].as_tensor().tolist() == [0, 8, 12]
    sliced = prepared.slice(1, 2)
    assert sliced["cu_seqlens"].as_tensor().tolist() == [0, 4]
    assert sliced["cu_seqlens_padded"].as_tensor().tolist() == [0, 12]


def test_preparation_backfills_multimodal_token_fields() -> None:
    text_sample = _sample("text", 4)
    multimodal_sample = _sample("image", 4)
    for message in multimodal_sample.message_log:
        message["mm_token_type_ids"] = torch.ones_like(message["token_ids"])
    packed = pack_selected_samples(
        [text_sample, multimodal_sample],
        pack_capacity=12,
        sequence_length_pad_multiple=1,
    )

    prepared = prepare_packed_sft_batch(
        [packed], tokenizer=_Tokenizer(), only_unmask_final=False
    )

    assert prepared["mm_token_type_ids"].tolist() == [
        [0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0]
    ]


def test_physical_pack_rejects_incompatible_or_over_capacity_sources() -> None:
    with pytest.raises(ValueError, match="compatible sources"):
        pack_selected_samples(
            [_sample("s0", 3), _sample("s1", 3, group="image")],
            pack_capacity=8,
            sequence_length_pad_multiple=1,
        )
    with pytest.raises(ValueError, match="exceed the pack capacity"):
        pack_selected_samples(
            [_sample("s0", 5), _sample("s1", 4)],
            pack_capacity=8,
            sequence_length_pad_multiple=1,
        )
