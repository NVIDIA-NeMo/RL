import pytest
import torch
from torch import nn

from nemo_rl.data.collate_fn import rl_collate_fn
from nemo_rl.data.soft_tokens import (
    SOFT_TOKEN_EMBEDDINGS_KEY,
    SOFT_TOKEN_POSITIONS_KEY,
    VLLM_PROMPT_EMBEDS_KEY,
    apply_soft_token_embeddings,
    copy_soft_token_inputs,
    copy_vllm_prompt_embeds,
    get_sample_soft_token_inputs,
    get_sample_vllm_prompt_embeds,
    get_vllm_prompt_embeds_for_sample,
    has_soft_token_inputs,
)
from nemo_rl.distributed.batched_data_dict import BatchedDataDict
from nemo_rl.models.generation.vllm.utils import format_prompt_for_vllm_generation


class DummyEmbeddingModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(16, 4)

    def get_input_embeddings(self):
        return self.embedding


def test_apply_soft_token_embeddings_replaces_selected_positions():
    model = DummyEmbeddingModel()
    input_ids = torch.tensor([[1, 2, 3], [4, 5, 6]])
    soft_token_embeddings = torch.tensor(
        [
            [[10.0, 11.0, 12.0, 13.0], [0.0, 0.0, 0.0, 0.0]],
            [[20.0, 21.0, 22.0, 23.0], [30.0, 31.0, 32.0, 33.0]],
        ]
    )
    soft_token_positions = torch.tensor([[1, -1], [0, 2]])

    model_args = {
        "input_ids": input_ids,
        SOFT_TOKEN_EMBEDDINGS_KEY: soft_token_embeddings,
        SOFT_TOKEN_POSITIONS_KEY: soft_token_positions,
    }
    updated_args = apply_soft_token_embeddings(model, model_args)

    expected = model.embedding(input_ids)
    expected[0, 1] = soft_token_embeddings[0, 0]
    expected[1, 0] = soft_token_embeddings[1, 0]
    expected[1, 2] = soft_token_embeddings[1, 1]

    assert "input_ids" not in updated_args
    torch.testing.assert_close(updated_args["inputs_embeds"], expected)


def test_apply_soft_token_embeddings_rejects_hidden_size_mismatch():
    model = DummyEmbeddingModel()
    model_args = {
        "input_ids": torch.tensor([[1, 2, 3]]),
        SOFT_TOKEN_EMBEDDINGS_KEY: torch.zeros(1, 1, 5),
        SOFT_TOKEN_POSITIONS_KEY: torch.tensor([[0]]),
    }

    with pytest.raises(ValueError, match="hidden size"):
        apply_soft_token_embeddings(model, model_args)


def test_has_soft_token_inputs_requires_paired_fields():
    with pytest.raises(ValueError, match="provided together"):
        has_soft_token_inputs({SOFT_TOKEN_EMBEDDINGS_KEY: torch.zeros(1, 1, 4)})


def test_copy_soft_token_inputs_requires_both_keys():
    destination = {}
    copy_soft_token_inputs(
        {
            SOFT_TOKEN_EMBEDDINGS_KEY: torch.zeros(1, 1, 4),
            SOFT_TOKEN_POSITIONS_KEY: torch.zeros(1, 1, dtype=torch.long),
        },
        destination,
    )

    assert set(destination) == {SOFT_TOKEN_EMBEDDINGS_KEY, SOFT_TOKEN_POSITIONS_KEY}


def test_get_sample_soft_token_inputs_preserves_batch_dimension():
    source = {
        SOFT_TOKEN_EMBEDDINGS_KEY: torch.zeros(3, 2, 4),
        SOFT_TOKEN_POSITIONS_KEY: torch.zeros(3, 2, dtype=torch.long),
    }

    sample = get_sample_soft_token_inputs(source, 1)

    assert sample[SOFT_TOKEN_EMBEDDINGS_KEY].shape == (1, 2, 4)
    assert sample[SOFT_TOKEN_POSITIONS_KEY].shape == (1, 2)


def test_copy_vllm_prompt_embeds_copies_prompt_embeds_only():
    destination = {}
    copy_vllm_prompt_embeds({VLLM_PROMPT_EMBEDS_KEY: torch.zeros(2, 3, 4)}, destination)

    assert set(destination) == {VLLM_PROMPT_EMBEDS_KEY}


def test_get_sample_vllm_prompt_embeds_preserves_batch_dimension():
    source = {VLLM_PROMPT_EMBEDS_KEY: torch.zeros(3, 5, 4)}

    sample = get_sample_vllm_prompt_embeds(source, 1)

    assert sample[VLLM_PROMPT_EMBEDS_KEY].shape == (1, 5, 4)


def test_vllm_prompt_formatter_uses_prompt_embeds():
    prompt_embeds = torch.randn(2, 3, 4)
    data = BatchedDataDict(
        {
            "input_ids": torch.tensor([[1, 2, 0], [3, 4, 5]]),
            "input_lengths": torch.tensor([2, 3], dtype=torch.int32),
            VLLM_PROMPT_EMBEDS_KEY: prompt_embeds,
        }
    )

    prompts = format_prompt_for_vllm_generation(data)

    assert "prompt_token_ids" not in prompts[0]
    torch.testing.assert_close(prompts[0]["prompt_embeds"], prompt_embeds[0, :2])
    torch.testing.assert_close(prompts[1]["prompt_embeds"], prompt_embeds[1, :3])


def test_get_vllm_prompt_embeds_for_sample_rejects_short_prompt_embeds():
    data = {VLLM_PROMPT_EMBEDS_KEY: torch.zeros(1, 2, 4)}

    with pytest.raises(ValueError, match="shorter than input length"):
        get_vllm_prompt_embeds_for_sample(data, sample_idx=0, valid_length=3)


def test_vllm_prompt_formatter_single_sample_uses_prompt_embeds():
    prompt_embeds = torch.randn(2, 3, 4)
    data = BatchedDataDict(
        {
            "input_ids": torch.tensor([[1, 2, 0], [3, 4, 5]]),
            "input_lengths": torch.tensor([2, 3], dtype=torch.int32),
            VLLM_PROMPT_EMBEDS_KEY: prompt_embeds,
        }
    )

    prompt = format_prompt_for_vllm_generation(data, sample_idx=1)

    assert "prompt_token_ids" not in prompt
    torch.testing.assert_close(prompt["prompt_embeds"], prompt_embeds[1, :3])


def test_rl_collate_fn_preserves_soft_token_fields():
    data_batch = [
        {
            "message_log": [
                {
                    "role": "user",
                    "content": "short",
                    "token_ids": torch.tensor([1, 2]),
                }
            ],
            "length": 2,
            "extra_env_info": {},
            "loss_multiplier": 1.0,
            "idx": 0,
            SOFT_TOKEN_EMBEDDINGS_KEY: torch.ones(1, 4),
            SOFT_TOKEN_POSITIONS_KEY: torch.tensor([1]),
            VLLM_PROMPT_EMBEDS_KEY: torch.ones(2, 4),
        },
        {
            "message_log": [
                {
                    "role": "user",
                    "content": "longer",
                    "token_ids": torch.tensor([3, 4, 5]),
                }
            ],
            "length": 3,
            "extra_env_info": {},
            "loss_multiplier": 1.0,
            "idx": 1,
            SOFT_TOKEN_EMBEDDINGS_KEY: torch.full((2, 4), 2.0),
            SOFT_TOKEN_POSITIONS_KEY: torch.tensor([0, 2]),
            VLLM_PROMPT_EMBEDS_KEY: torch.full((3, 4), 3.0),
        },
    ]

    batch = rl_collate_fn(data_batch)

    assert batch[SOFT_TOKEN_EMBEDDINGS_KEY].shape == (2, 2, 4)
    assert batch[SOFT_TOKEN_POSITIONS_KEY].shape == (2, 2)
    assert batch[VLLM_PROMPT_EMBEDS_KEY].shape == (2, 3, 4)
    torch.testing.assert_close(batch[SOFT_TOKEN_EMBEDDINGS_KEY][0, 1], torch.zeros(4))
    torch.testing.assert_close(batch[SOFT_TOKEN_POSITIONS_KEY][0, 1], torch.tensor(-1))
    torch.testing.assert_close(batch[VLLM_PROMPT_EMBEDS_KEY][0, 2], torch.zeros(4))
