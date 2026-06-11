import sys
import types

import pytest
import torch

run_grpo_stub = types.ModuleType("examples.run_grpo")
run_grpo_stub.main = lambda: None
run_grpo_stub.parse_args = lambda: (types.SimpleNamespace(config=None), [])
sys.modules["examples.run_grpo"] = run_grpo_stub

from examples.run_grpo_soft_tokens import (
    _load_datum_tensors,
    _use_vllm_prompt_embeds,
)
from nemo_rl.data.soft_tokens import (
    SOFT_TOKEN_EMBEDDINGS_KEY,
    SOFT_TOKEN_POSITIONS_KEY,
    VLLM_PROMPT_EMBEDS_KEY,
)


def test_load_datum_tensors_builds_placement_from_embeddings_only():
    soft_token_embeddings = torch.arange(12, dtype=torch.float32).view(3, 4)

    loaded_embeddings, soft_token_positions, vllm_prompt_embeds = _load_datum_tensors(
        {SOFT_TOKEN_EMBEDDINGS_KEY: soft_token_embeddings.tolist()},
        prompt_length=5,
        soft_token_config={"hidden_size": 4},
    )

    torch.testing.assert_close(loaded_embeddings, soft_token_embeddings)
    torch.testing.assert_close(soft_token_positions, torch.tensor([2, 3, 4]))
    torch.testing.assert_close(vllm_prompt_embeds[:2], torch.zeros(2, 4))
    torch.testing.assert_close(vllm_prompt_embeds[2:], soft_token_embeddings)


def test_load_datum_tensors_zeroes_soft_tokens_that_do_not_fit_prompt():
    soft_token_embeddings = torch.arange(12, dtype=torch.float32).view(3, 4)

    loaded_embeddings, soft_token_positions, vllm_prompt_embeds = _load_datum_tensors(
        {SOFT_TOKEN_EMBEDDINGS_KEY: soft_token_embeddings.tolist()},
        prompt_length=2,
        soft_token_config={"hidden_size": 4},
    )

    torch.testing.assert_close(
        loaded_embeddings,
        torch.cat([soft_token_embeddings[:2], torch.zeros(1, 4)]),
    )
    torch.testing.assert_close(soft_token_positions, torch.tensor([0, 1, -1]))
    torch.testing.assert_close(vllm_prompt_embeds, soft_token_embeddings[:2])


def test_load_datum_tensors_requires_positions_and_vllm_embeds_together():
    with pytest.raises(ValueError, match="provided together"):
        _load_datum_tensors(
            {
                SOFT_TOKEN_EMBEDDINGS_KEY: [[1.0, 2.0]],
                SOFT_TOKEN_POSITIONS_KEY: [0],
            },
            prompt_length=1,
            soft_token_config={"hidden_size": 2},
        )


def test_load_datum_tensors_preserves_explicit_placement_tensors():
    loaded_embeddings, soft_token_positions, vllm_prompt_embeds = _load_datum_tensors(
        {
            SOFT_TOKEN_EMBEDDINGS_KEY: [[1.0, 2.0]],
            SOFT_TOKEN_POSITIONS_KEY: [0],
            VLLM_PROMPT_EMBEDS_KEY: [[1.0, 2.0]],
        },
        prompt_length=1,
        soft_token_config={"hidden_size": 2},
    )

    torch.testing.assert_close(loaded_embeddings, torch.tensor([[1.0, 2.0]]))
    torch.testing.assert_close(soft_token_positions, torch.tensor([0]))
    torch.testing.assert_close(vllm_prompt_embeds, torch.tensor([[1.0, 2.0]]))


@pytest.mark.parametrize(
    ("vllm_prompt_mode", "expected"),
    [
        ("prompt_embeds", True),
        ("token_ids", False),
    ],
)
def test_use_vllm_prompt_embeds(vllm_prompt_mode, expected):
    assert _use_vllm_prompt_embeds({"vllm_prompt_mode": vllm_prompt_mode}) is expected


def test_use_vllm_prompt_embeds_rejects_unknown_mode():
    with pytest.raises(ValueError, match="vllm_prompt_mode"):
        _use_vllm_prompt_embeds({"vllm_prompt_mode": "unknown"})
