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

from unittest.mock import MagicMock

import pytest
import torch

from nemo_rl.data.collate_fn import preference_collate_fn, rl_collate_fn
from nemo_rl.data.interfaces import DatumSpec
from nemo_rl.distributed.batched_data_dict import BatchedDataDict


def _packed_datum(idx: int, cu_seqlens: list[int] | None = None) -> DatumSpec:
    pack_length = 4
    if cu_seqlens is None:
        cu_seqlens = [0, 2, pack_length]
    first_token = idx * 10
    return DatumSpec(
        message_log=[],
        input_ids=torch.arange(
            first_token, first_token + pack_length, dtype=torch.int64
        ),
        target_ids=torch.arange(
            first_token + 100,
            first_token + 100 + pack_length,
            dtype=torch.int64,
        ),
        token_mask=torch.tensor([1.0, 0.0, 1.0, float(idx % 2)], dtype=torch.float32),
        position_ids=torch.tensor([0, 1, 0, 1], dtype=torch.int64),
        packed_cu_seqlens=torch.tensor(cu_seqlens, dtype=torch.int32),
        packed_max_seqlen=max(b - a for a, b in zip(cu_seqlens, cu_seqlens[1:])),
        packed_context_parallel_size=1,
        length=pack_length,
        extra_env_info=None,
        loss_multiplier=1.0,
        idx=idx,
        task_name="megatron_sft_packed",
    )


def test_rl_collate_fn_preserves_packed_tensors() -> None:
    data = [_packed_datum(0), _packed_datum(1)]

    batch = rl_collate_fn(data)

    assert torch.equal(
        batch["input_ids"],
        torch.tensor([[0, 1, 2, 3], [10, 11, 12, 13]], dtype=torch.int64),
    )
    assert torch.equal(
        batch["target_ids"],
        torch.tensor([[100, 101, 102, 103], [110, 111, 112, 113]], dtype=torch.int64),
    )
    assert torch.equal(
        batch["token_mask"],
        torch.tensor([[1.0, 0.0, 1.0, 0.0], [1.0, 0.0, 1.0, 1.0]]),
    )
    assert torch.equal(
        batch["position_ids"],
        torch.tensor([[0, 1, 0, 1], [0, 1, 0, 1]], dtype=torch.int64),
    )


def test_rl_collate_fn_preserves_ragged_packed_metadata() -> None:
    data = [_packed_datum(0), _packed_datum(1, [0, 1, 3, 4])]

    batch = rl_collate_fn(data)

    assert torch.equal(
        batch["packed_cu_seqlens"],
        torch.tensor([[0, 2, 4, -1], [0, 1, 3, 4]], dtype=torch.int32),
    )
    assert torch.equal(batch["packed_cu_seqlens_lengths"], torch.tensor([3, 4]))
    assert torch.equal(batch["packed_max_seqlen"], torch.tensor([2, 2]))
    assert torch.equal(batch["input_lengths"], torch.tensor([4, 4]))
    assert torch.equal(batch["sample_mask"], torch.tensor([1.0, 1.0]))
    assert batch["idx"] == [0, 1]
    assert batch["task_name"] == ["megatron_sft_packed", "megatron_sft_packed"]


def test_rl_collate_fn_rejects_mixed_packed_and_non_packed_rows() -> None:
    regular = DatumSpec(
        message_log=[],
        length=1,
        loss_multiplier=1.0,
        extra_env_info=None,
        idx=1,
    )

    with pytest.raises(ValueError, match="all packed or all non-packed"):
        rl_collate_fn([_packed_datum(0), regular])


def test_rl_collate_fn_rejects_partial_packed_row() -> None:
    partial = DatumSpec(
        message_log=[],
        packed_cu_seqlens=torch.tensor([0, 4]),
        length=4,
        loss_multiplier=1.0,
        extra_env_info=None,
        idx=0,
    )

    with pytest.raises(ValueError, match="partial Megatron SFT packed fields"):
        rl_collate_fn([partial])


def test_rl_collate_fn_does_not_treat_input_ids_extra_as_packed() -> None:
    datum = DatumSpec(
        message_log=[],
        input_ids=torch.arange(4),
        length=4,
        loss_multiplier=1.0,
        extra_env_info=None,
        idx=0,
    )

    batch = rl_collate_fn([datum])

    assert torch.equal(batch["length"], torch.tensor([4]))


@pytest.mark.parametrize(
    ("field", "value", "error"),
    [
        pytest.param(
            "target_ids",
            torch.arange(3),
            "1D tensors of equal length",
            id="mismatched-tensor-length",
        ),
        pytest.param(
            "packed_cu_seqlens",
            torch.tensor([0, 2, 3]),
            "final value must equal input length",
            id="wrong-final-cu-seqlen",
        ),
        pytest.param(
            "packed_max_seqlen",
            3,
            r"must equal max\(diff\(packed_cu_seqlens\)\)",
            id="wrong-max-seqlen",
        ),
        pytest.param(
            "length",
            3,
            "length must match input_ids length",
            id="wrong-pack-length",
        ),
    ],
)
def test_rl_collate_fn_rejects_invalid_packed_metadata(
    field: str, value: object, error: str
) -> None:
    datum = _packed_datum(0)
    datum[field] = value

    with pytest.raises(ValueError, match=error):
        rl_collate_fn([datum])


def test_rl_collate_fn_rejects_context_parallel_mismatch() -> None:
    datum = _packed_datum(0)
    datum["packed_context_parallel_size"] = 2

    with pytest.raises(ValueError, match="prepared for context_parallel_size=2"):
        rl_collate_fn([datum], megatron_sft_context_parallel_size=1)


def test_rl_collate_fn_preserves_source_order_before_megatron_dp_sharding() -> None:
    data = [_packed_datum(idx) for idx in range(8)]

    batch = rl_collate_fn(data)

    assert list(zip(batch["idx"], batch["input_ids"][:, 0].tolist(), strict=True)) == [
        (0, 0),
        (1, 10),
        (2, 20),
        (3, 30),
        (4, 40),
        (5, 50),
        (6, 60),
        (7, 70),
    ]


@pytest.mark.parametrize("batch_size", [1, 2, 3])
def test_rl_collate_fn_accepts_partial_packed_validation_batch(
    batch_size: int,
) -> None:
    data = [_packed_datum(idx) for idx in range(batch_size)]

    batch = rl_collate_fn(data)

    assert batch["idx"] == list(range(batch_size))


def test_preference_collate_fn():
    """Test that preference_collate_fn correctly processes preference data."""
    # Create mock tokenizer
    mock_tokenizer = MagicMock()
    mock_tokenizer.pad_token_id = 0

    # Create test data with varying sequence lengths
    data_batch = [
        DatumSpec(
            message_log_chosen=[
                {
                    "role": "user",
                    "content": "Hello",
                    "token_ids": torch.tensor([1, 2, 3]),
                },
                {
                    "role": "assistant",
                    "content": "Hi there",
                    "token_ids": torch.tensor([4, 5, 6, 7]),
                },
            ],
            message_log_rejected=[
                {
                    "role": "user",
                    "content": "Hello",
                    "token_ids": torch.tensor([1, 2, 3]),
                },
                {
                    "role": "assistant",
                    "content": "Bye",
                    "token_ids": torch.tensor([8, 9]),
                },
            ],
            length_chosen=7,
            length_rejected=5,
            loss_multiplier=1.0,
            idx=0,
            task_name="test_task",
        ),
        DatumSpec(
            message_log_chosen=[
                {
                    "role": "user",
                    "content": "How are you?",
                    "token_ids": torch.tensor([10, 11, 12]),
                },
                {
                    "role": "assistant",
                    "content": "I'm good",
                    "token_ids": torch.tensor([13, 14, 15]),
                },
            ],
            message_log_rejected=[
                {
                    "role": "user",
                    "content": "How are you?",
                    "token_ids": torch.tensor([10, 11, 12]),
                },
                {
                    "role": "assistant",
                    "content": "Not great",
                    "token_ids": torch.tensor([16, 17, 18, 19]),
                },
            ],
            length_chosen=6,
            length_rejected=7,
            loss_multiplier=0,
            idx=1,
            task_name="test_task",
        ),
    ]

    # Call preference_collate_fn
    train_data = preference_collate_fn(
        data_batch,
        mock_tokenizer,
        make_sequence_length_divisible_by=16,
        add_loss_mask=True,
    )

    # Verify the output structure
    assert isinstance(train_data, BatchedDataDict)
    assert "input_ids" in train_data
    assert "input_lengths" in train_data
    assert "token_mask" in train_data
    assert "sample_mask" in train_data

    # Verify batch size is doubled (chosen + rejected for each example)
    assert train_data["input_ids"].shape[0] == 4  # 2 examples * 2 (chosen + rejected)

    # Verify input_ids shape and padding
    max_length = 16  # max of all sequence lengths, padded to be divisible by 16
    assert train_data["input_ids"].shape == (4, max_length)

    # Verify input_lengths
    expected_lengths = [7, 5, 6, 7]  # chosen1, rejected1, chosen2, rejected2
    assert torch.equal(train_data["input_lengths"], torch.tensor(expected_lengths))

    # Verify token_mask
    assert train_data["token_mask"].shape == (4, max_length)
    # First example chosen (length 7)
    assert torch.all(train_data["token_mask"][0][0:3] == 0)
    assert torch.all(train_data["token_mask"][0][3:7] == 1)
    # First example rejected (length 5)
    assert torch.all(train_data["token_mask"][1][0:3] == 0)
    assert torch.all(train_data["token_mask"][1][3:5] == 1)
    assert torch.all(train_data["token_mask"][1][5:] == 0)

    # Verify sample_mask
    expected_sample_mask = [
        1.0,
        1.0,
        0.0,
        0.0,
    ]  # loss_multiplier repeated for chosen/rejected
    assert torch.equal(train_data["sample_mask"], torch.tensor(expected_sample_mask))

    # Verify message content is preserved
    # First example chosen
    assert torch.equal(train_data["input_ids"][0][0:3], torch.tensor([1, 2, 3]))  # user
    assert torch.equal(
        train_data["input_ids"][0][3:7], torch.tensor([4, 5, 6, 7])
    )  # assistant
    # First example rejected
    assert torch.equal(train_data["input_ids"][1][0:3], torch.tensor([1, 2, 3]))  # user
    assert torch.equal(
        train_data["input_ids"][1][3:5], torch.tensor([8, 9])
    )  # assistant
