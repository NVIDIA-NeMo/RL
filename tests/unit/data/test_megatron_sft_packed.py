# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.

from typing import Any

import numpy as np
import pytest
import torch

from nemo_rl.data.datasets.response_datasets import (
    DATASET_REGISTRY,
    MegatronSFTPackedDataset,
)
from nemo_rl.data.interfaces import TaskDataSpec
from nemo_rl.data.megatron_sft_packed import (
    IGNORE_INDEX,
    NEMOTRON_NANO_V2_TEMPLATE,
    MegatronSFTPackedDatumSpec,
    megatron_sft_packed_preprocessor,
    split_megatron_sft_conversations,
)


class _DummyTokenizer:
    pad_token_id = 99
    unk_token_id = 99
    eos_token_id = 2

    def __init__(self, turn_tokens: dict[tuple[str, str], list[int]]) -> None:
        self.turn_tokens = turn_tokens
        self.calls: list[tuple[list[dict[str, Any]], dict[str, Any]]] = []

    def apply_chat_template(
        self, messages: list[dict[str, Any]], **kwargs: Any
    ) -> list[int] | np.ndarray:
        self.calls.append((messages, kwargs))
        token_ids = [
            token_id
            for message in messages
            for token_id in self.turn_tokens[(message["role"], message["content"])]
        ]
        if kwargs.get("return_tensors") == "np":
            return np.asarray([token_ids])
        return token_ids

    def convert_tokens_to_ids(self, token: str) -> int:
        assert token == "<unk>"
        return self.unk_token_id


class _MegatronTokenizer:
    eod = 2
    pad = 99

    def __init__(self, turn_tokens: dict[tuple[str, str], list[int]]) -> None:
        self.turn_tokens = turn_tokens

    def tokenize_conversation(
        self,
        conversation: list[dict[str, Any]],
        return_target: bool,
        add_generation_prompt: bool,
    ) -> tuple[np.ndarray, np.ndarray]:
        assert return_target is True
        assert add_generation_prompt is False
        tokens = np.asarray(
            [
                token_id
                for message in conversation
                for token_id in self.turn_tokens[(message["role"], message["content"])]
            ]
        )
        return tokens, tokens.copy()


class _MegatronConfig:
    def __init__(
        self,
        tokenizer: _MegatronTokenizer,
        sequence_length: int,
        context_parallel_size: int,
    ) -> None:
        self.tokenizer = tokenizer
        self.sequence_length = sequence_length
        self.context_parallel_size = context_parallel_size
        self.reset_position_ids = False
        self.create_attention_mask = False
        self.reset_attention_mask = False


class _MegatronLowLevelDataset:
    def __init__(self, messages: list[dict[str, str]]) -> None:
        self.messages = messages

    def __getitem__(self, _idx: int) -> list[dict[str, str]]:
        return self.messages


def _preprocess(
    messages: list[dict[str, str]],
    tokenizer: _DummyTokenizer,
    max_seq_length: int,
    **kwargs: Any,
) -> MegatronSFTPackedDatumSpec:
    return megatron_sft_packed_preprocessor(
        {"packed_messages": messages},
        TaskDataSpec(),
        tokenizer,
        max_seq_length,
        idx=7,
        context_parallel_size=kwargs.pop("context_parallel_size", 1),
        **kwargs,
    )


def _dataset_parser() -> MegatronSFTPackedDataset:
    dataset = MegatronSFTPackedDataset.__new__(MegatronSFTPackedDataset)
    dataset.chat_key = "messages"
    dataset.task_name = "megatron_sft_packed"
    return dataset


@pytest.mark.parametrize(
    ("data_config", "message"),
    [
        (
            {"megatron_sft_context_parallel_size": 1},
            "megatron_sft_prompt_format",
        ),
        (
            {"megatron_sft_prompt_format": "identity"},
            "megatron_sft_context_parallel_size",
        ),
        (
            {
                "megatron_sft_prompt_format": "unsupported",
                "megatron_sft_context_parallel_size": 1,
            },
            "unknown SFT prompt format",
        ),
        (
            {
                "megatron_sft_prompt_format": "identity",
                "megatron_sft_context_parallel_size": 1,
                "megatron_sft_assistant_prefix_len": -1,
            },
            "megatron_sft_assistant_prefix_len must be >= 0",
        ),
        (
            {
                "megatron_sft_prompt_format": "identity",
                "megatron_sft_context_parallel_size": 1,
                "megatron_sft_assistant_prefix_len": 1,
            },
            "identity prompt format does not support assistant_prefix_len",
        ),
    ],
)
def test_dataset_processor_rejects_invalid_packed_config_during_setup(
    data_config: dict[str, Any],
    message: str,
) -> None:
    dataset = _dataset_parser()
    dataset.data_config = data_config

    with pytest.raises((KeyError, ValueError, NotImplementedError), match=message):
        dataset.set_processor()


def _megatron_preprocess(
    messages: list[dict[str, str]],
    tokenizer: _MegatronTokenizer,
    max_seq_length: int,
    context_parallel_size: int = 1,
) -> dict[str, torch.Tensor]:
    # Megatron is optional and only available in the mcore test shard.
    from megatron.training.datasets.sft_dataset import SFTDataset

    dataset = SFTDataset.__new__(SFTDataset)
    dataset.dataset = _MegatronLowLevelDataset(messages)
    dataset.indices = np.asarray([0])
    dataset.num_samples = 1
    dataset.config = _MegatronConfig(
        tokenizer,
        sequence_length=max_seq_length,
        context_parallel_size=context_parallel_size,
    )
    return dataset[0]


def test_split_megatron_sft_conversations_starts_each_segment_at_system() -> None:
    messages = [
        {"role": "system", "content": "s1"},
        {"role": "user", "content": "u1"},
        {"role": "assistant", "content": "a1"},
        {"role": "system", "content": "s2"},
        {"role": "user", "content": "u2"},
        {"role": "assistant", "content": "a2"},
    ]

    assert split_megatron_sft_conversations(messages) == [messages[:3], messages[3:]]


def test_dataset_parser_preserves_messages_as_one_packed_row() -> None:
    messages = [
        {"role": "system", "content": "system"},
        {"role": "user", "content": "question"},
        {"role": "assistant", "content": "answer"},
    ]

    parsed = _dataset_parser().format_data({"messages": messages})

    assert parsed == {
        "packed_messages": messages,
        "task_name": "megatron_sft_packed",
    }


@pytest.mark.parametrize(
    ("messages", "error"),
    [
        pytest.param(
            [
                {"role": "user", "content": "question"},
                {"role": "assistant", "content": "answer"},
            ],
            "must start with a system message",
            id="missing-leading-system",
        ),
        pytest.param(
            [
                {"role": "system", "content": "system"},
                {"role": "user", "content": "question"},
            ],
            "must end with an assistant message",
            id="missing-trailing-assistant",
        ),
        pytest.param([], "must start with a system message", id="empty-row"),
    ],
)
def test_dataset_parser_rejects_invalid_packed_rows(
    messages: list[dict[str, str]], error: str
) -> None:
    with pytest.raises(ValueError, match=error):
        _dataset_parser().format_data({"messages": messages})


@pytest.mark.mcore
def test_packed_preprocessor_matches_megatron_without_appending_eod() -> None:
    messages = [
        {"role": "system", "content": "s"},
        {"role": "user", "content": "u"},
        {"role": "assistant", "content": "a"},
    ]
    turn_tokens = {
        ("system", "s"): [10],
        ("user", "u"): [20],
        ("assistant", "a"): [30],
    }

    processed = _preprocess(
        messages,
        _DummyTokenizer(turn_tokens),
        max_seq_length=5,
        prompt_format="identity",
    )
    megatron_processed = _megatron_preprocess(
        messages,
        _MegatronTokenizer(turn_tokens),
        max_seq_length=5,
    )

    assert torch.equal(processed["input_ids"], megatron_processed["tokens"])
    assert torch.equal(processed["target_ids"], megatron_processed["labels"])
    assert torch.equal(processed["token_mask"], megatron_processed["loss_mask"])
    assert torch.equal(processed["position_ids"], megatron_processed["position_ids"])
    assert torch.equal(processed["packed_cu_seqlens"], torch.tensor([0, 5]))
    assert processed["packed_max_seqlen"] == megatron_processed["max_seqlen"].item()


def test_packed_preprocessor_preserves_existing_eod() -> None:
    messages = [
        {"role": "system", "content": "s"},
        {"role": "user", "content": "u"},
        {"role": "assistant", "content": "a"},
    ]
    tokenizer = _DummyTokenizer(
        {("system", "s"): [10], ("user", "u"): [20], ("assistant", "a"): [2]}
    )

    processed = _preprocess(
        messages,
        tokenizer,
        max_seq_length=4,
        prompt_format="identity",
    )

    assert torch.equal(processed["input_ids"], torch.tensor([10, 20, 2, 99]))
    assert torch.equal(processed["target_ids"], torch.tensor([20, 2, 99, 99]))


def test_identity_uses_unk_padding_and_supervises_all_literal_targets() -> None:
    messages = [
        {"role": "system", "content": "s"},
        {"role": "user", "content": "u"},
        {"role": "assistant", "content": "a"},
    ]
    tokenizer = _DummyTokenizer(
        {("system", "s"): [10], ("user", "u"): [20], ("assistant", "a"): [30]}
    )
    tokenizer.pad_token_id = 77

    processed = _preprocess(
        messages,
        tokenizer,
        max_seq_length=5,
        prompt_format="identity",
    )

    assert torch.equal(processed["target_ids"], torch.tensor([20, 30, 99, 99, 99]))
    assert tokenizer.calls[0][1]["add_generation_prompt"] is False


def test_identity_rejects_assistant_prefix_masking() -> None:
    messages = [
        {"role": "system", "content": "s"},
        {"role": "user", "content": "u"},
        {"role": "assistant", "content": "a"},
    ]
    tokenizer = _DummyTokenizer(
        {("system", "s"): [10], ("user", "u"): [20], ("assistant", "a"): [30]}
    )

    with pytest.raises(
        ValueError,
        match="identity prompt format does not support assistant_prefix_len",
    ):
        _preprocess(
            messages,
            tokenizer,
            max_seq_length=5,
            prompt_format="identity",
            assistant_prefix_len=1,
        )


@pytest.mark.parametrize(
    ("messages", "turn_tokens", "expected_input_ids", "expected_target_ids"),
    [
        pytest.param(
            [
                {"role": "system", "content": "sys"},
                {"role": "user", "content": "question"},
                {"role": "assistant", "content": ""},
            ],
            {
                ("system", "sys"): [10],
                ("user", "question"): [20],
                ("assistant", ""): [],
            },
            [10, 20, 99, 99],
            [20, 99, 99, 99],
            id="empty-assistant",
        ),
        pytest.param(
            [
                {"role": "system", "content": "sys"},
                {"role": "user", "content": "question"},
                {"role": "tool", "content": "result"},
                {"role": "assistant", "content": "answer"},
            ],
            {
                ("system", "sys"): [10],
                ("user", "question"): [20],
                ("tool", "result"): [25],
                ("assistant", "answer"): [30],
            },
            [10, 20, 25, 30, 99],
            [20, 25, 30, 99, 99],
            id="tool-turn",
        ),
        pytest.param(
            [
                {"role": "system", "content": "sys"},
                {"role": "assistant", "content": "first"},
                {"role": "assistant", "content": "second"},
            ],
            {
                ("system", "sys"): [10],
                ("assistant", "first"): [30],
                ("assistant", "second"): [40],
            },
            [10, 30, 40, 99, 99],
            [30, 40, 99, 99, 99],
            id="consecutive-assistants",
        ),
    ],
)
def test_identity_accepts_literal_role_streams(
    messages: list[dict[str, str]],
    turn_tokens: dict[tuple[str, str], list[int]],
    expected_input_ids: list[int],
    expected_target_ids: list[int],
) -> None:
    processed = _preprocess(
        messages,
        _DummyTokenizer(turn_tokens),
        max_seq_length=len(expected_input_ids),
        prompt_format="identity",
    )

    assert torch.equal(processed["input_ids"], torch.tensor(expected_input_ids))
    assert torch.equal(processed["target_ids"], torch.tensor(expected_target_ids))


def test_nemotron_preprocessor_uses_expected_tokenizer_contract() -> None:
    messages = [
        {"role": "system", "content": "s"},
        {"role": "user", "content": "u"},
        {"role": "assistant", "content": "a"},
    ]
    tokenizer = _DummyTokenizer(
        {
            ("system", "s"): [10, 11],
            ("user", "u"): [20, 21],
            ("assistant", "a"): [30, 31, 32, 33],
        }
    )

    _preprocess(
        messages,
        tokenizer,
        max_seq_length=12,
        prompt_format="nemotron-nano-v2",
    )

    assert tokenizer.calls[0][1] == {
        "tokenize": True,
        "add_generation_prompt": False,
        "return_assistant_token_mask": False,
        "return_tensors": "np",
        "chat_template": NEMOTRON_NANO_V2_TEMPLATE,
    }


def test_nemotron_preprocessor_masks_prompt_and_assistant_prefix() -> None:
    messages = [
        {"role": "system", "content": "s"},
        {"role": "user", "content": "u"},
        {"role": "assistant", "content": "a"},
    ]
    tokenizer = _DummyTokenizer(
        {
            ("system", "s"): [10, 11],
            ("user", "u"): [20, 21],
            ("assistant", "a"): [30, 31, 32, 33],
        }
    )

    processed = _preprocess(
        messages,
        tokenizer,
        max_seq_length=12,
        prompt_format="nemotron-nano-v2",
    )

    assert torch.equal(
        processed["target_ids"],
        torch.tensor(
            [
                IGNORE_INDEX,
                IGNORE_INDEX,
                IGNORE_INDEX,
                IGNORE_INDEX,
                IGNORE_INDEX,
                IGNORE_INDEX,
                33,
                99,
                99,
                99,
                99,
                99,
            ]
        ),
    )


def test_nemotron_preprocessor_rejects_prefix_longer_than_assistant_turn() -> None:
    messages = [
        {"role": "system", "content": "s"},
        {"role": "user", "content": "u"},
        {"role": "assistant", "content": "a"},
    ]
    tokenizer = _DummyTokenizer(
        {
            ("system", "s"): [10],
            ("user", "u"): [20],
            ("assistant", "a"): [30, 31],
        }
    )

    with pytest.raises(ValueError, match="assistant_prefix_len"):
        _preprocess(
            messages,
            tokenizer,
            max_seq_length=8,
            prompt_format="nemotron-nano-v2",
            assistant_prefix_len=3,
        )


def test_nemotron_preprocessor_masks_tool_output_before_assistant_response() -> None:
    messages = [
        {"role": "system", "content": "s"},
        {"role": "user", "content": "u"},
        {"role": "tool", "content": "tool-output"},
        {"role": "assistant", "content": "answer"},
    ]
    tokenizer = _DummyTokenizer(
        {
            ("system", "s"): [10],
            ("user", "u"): [20],
            ("tool", "tool-output"): [25],
            ("assistant", "answer"): [30, 31, 32, 33],
        }
    )

    processed = _preprocess(
        messages,
        tokenizer,
        max_seq_length=10,
        prompt_format="nemotron-nano-v2",
    )

    assert torch.equal(
        processed["target_ids"],
        torch.tensor(
            [
                IGNORE_INDEX,
                IGNORE_INDEX,
                IGNORE_INDEX,
                IGNORE_INDEX,
                IGNORE_INDEX,
                33,
                99,
                99,
                99,
                99,
            ]
        ),
    )


def test_nemotron_preprocessor_masks_unsupervised_tokens_from_loss() -> None:
    messages = [
        {"role": "system", "content": "s"},
        {"role": "user", "content": "u"},
        {"role": "assistant", "content": "a"},
    ]
    tokenizer = _DummyTokenizer(
        {
            ("system", "s"): [10, 11],
            ("user", "u"): [20, 21],
            ("assistant", "a"): [30, 31, 32, 33],
        }
    )

    processed = _preprocess(
        messages,
        tokenizer,
        max_seq_length=12,
        prompt_format="nemotron-nano-v2",
    )

    assert torch.equal(
        processed["token_mask"],
        torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
    )


def test_nemotron_preprocessor_rejects_empty_assistant_turn() -> None:
    messages = [
        {"role": "system", "content": "s"},
        {"role": "user", "content": "u"},
        {"role": "assistant", "content": ""},
    ]
    tokenizer = _DummyTokenizer(
        {("system", "s"): [10], ("user", "u"): [20], ("assistant", ""): []}
    )

    with pytest.raises(ValueError, match="empty assistant turn"):
        _preprocess(
            messages,
            tokenizer,
            max_seq_length=8,
            prompt_format="nemotron-nano-v2",
        )


def test_packed_preprocessor_cp_pads_each_system_delimited_boundary() -> None:
    messages = [
        {"role": "system", "content": "s1"},
        {"role": "user", "content": "u1"},
        {"role": "assistant", "content": "a1"},
        {"role": "system", "content": "s2"},
        {"role": "user", "content": "u2"},
        {"role": "assistant", "content": "a2"},
    ]
    tokenizer = _DummyTokenizer(
        {
            ("system", "s1"): [10],
            ("user", "u1"): [20],
            ("assistant", "a1"): [2],
            ("system", "s2"): [40],
            ("user", "u2"): [50],
            ("assistant", "a2"): [2],
        }
    )

    processed = _preprocess(
        messages,
        tokenizer,
        max_seq_length=8,
        prompt_format="identity",
        context_parallel_size=2,
    )

    assert torch.equal(
        processed["input_ids"], torch.tensor([10, 20, 2, 99, 40, 50, 2, 99])
    )
    assert torch.equal(
        processed["target_ids"],
        torch.tensor([20, 2, 99, IGNORE_INDEX, 50, 2, 99, 99]),
    )
    assert torch.equal(
        processed["token_mask"],
        torch.tensor([1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0]),
    )
    assert torch.equal(
        processed["position_ids"], torch.tensor([0, 1, 2, 3, 0, 1, 2, 3])
    )
    assert torch.equal(processed["packed_cu_seqlens"], torch.tensor([0, 4, 8]))
    assert processed["packed_max_seqlen"] == 4
    assert processed["packed_context_parallel_size"] == 2


@pytest.mark.parametrize(
    ("context_parallel_size", "expected_cu_seqlens"),
    [
        pytest.param(1, [0, 3, 6, 12], id="cp1"),
        pytest.param(2, [0, 4, 8, 12], id="cp2"),
    ],
)
def test_identity_masks_each_internal_packed_conversation_boundary(
    context_parallel_size: int,
    expected_cu_seqlens: list[int],
) -> None:
    messages = [
        {"role": "system", "content": "s1"},
        {"role": "user", "content": "u1"},
        {"role": "assistant", "content": "a1"},
        {"role": "system", "content": "s2"},
        {"role": "user", "content": "u2"},
        {"role": "assistant", "content": "a2"},
        {"role": "system", "content": "s3"},
        {"role": "user", "content": "u3"},
        {"role": "assistant", "content": "a3"},
    ]
    tokenizer = _DummyTokenizer(
        {
            ("system", "s1"): [10],
            ("user", "u1"): [20],
            ("assistant", "a1"): [30],
            ("system", "s2"): [40],
            ("user", "u2"): [50],
            ("assistant", "a2"): [60],
            ("system", "s3"): [70],
            ("user", "u3"): [80],
            ("assistant", "a3"): [90],
        }
    )

    processed = _preprocess(
        messages,
        tokenizer,
        max_seq_length=12,
        prompt_format="identity",
        context_parallel_size=context_parallel_size,
    )

    assert torch.equal(
        processed["packed_cu_seqlens"], torch.tensor(expected_cu_seqlens)
    )
    internal_boundary_indices = processed["packed_cu_seqlens"][1:-1].long() - 1
    assert torch.equal(
        processed["target_ids"][internal_boundary_indices],
        torch.full((2,), IGNORE_INDEX, dtype=torch.int64),
    )
    assert torch.equal(
        processed["token_mask"][internal_boundary_indices], torch.zeros(2)
    )


def test_packed_preprocessor_right_truncates_to_pack_length_plus_one() -> None:
    messages = [
        {"role": "system", "content": "s"},
        {"role": "user", "content": "u"},
        {"role": "assistant", "content": "a"},
    ]
    tokenizer = _DummyTokenizer(
        {
            ("system", "s"): [10],
            ("user", "u"): [20],
            ("assistant", "a"): [30, 40, 50],
        }
    )

    processed = _preprocess(
        messages,
        tokenizer,
        max_seq_length=4,
        prompt_format="identity",
    )

    assert torch.equal(processed["input_ids"], torch.tensor([10, 20, 30, 40]))
    assert torch.equal(processed["target_ids"], torch.tensor([20, 30, 40, 99]))
    assert torch.equal(processed["position_ids"], torch.arange(4))
    assert torch.equal(processed["packed_cu_seqlens"], torch.tensor([0, 4]))


def test_packed_preprocessor_stops_after_exactly_filling_the_pack() -> None:
    messages = [
        {"role": "system", "content": "s1"},
        {"role": "user", "content": "u1"},
        {"role": "assistant", "content": "a1"},
        {"role": "system", "content": "s2"},
        {"role": "user", "content": "u2"},
        {"role": "assistant", "content": "a2"},
    ]
    tokenizer = _DummyTokenizer(
        {
            ("system", "s1"): [10],
            ("user", "u1"): [20],
            ("assistant", "a1"): [30, 40],
            ("system", "s2"): [50],
            ("user", "u2"): [60],
            ("assistant", "a2"): [70],
        }
    )

    processed = _preprocess(
        messages,
        tokenizer,
        max_seq_length=4,
        prompt_format="identity",
    )

    assert torch.equal(processed["input_ids"], torch.tensor([10, 20, 30, 40]))
    assert torch.equal(processed["packed_cu_seqlens"], torch.tensor([0, 4]))
    assert bool(
        (
            (processed["packed_cu_seqlens"][1:] - processed["packed_cu_seqlens"][:-1])
            > 0
        ).all()
    )


def test_megatron_sft_packed_dataset_is_registered() -> None:
    assert DATASET_REGISTRY["megatron_sft_packed"] is MegatronSFTPackedDataset
