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

"""Checkpoint and math contracts for the DSpark auxiliary heads."""

from __future__ import annotations

import importlib.util
import io
from pathlib import Path
from types import ModuleType

import pytest
import torch
from torch import Tensor, nn
from torch.nn import functional as F


def _load_heads() -> tuple[type[nn.Module], type[nn.Module]]:
    module_path = (
        Path(__file__).resolve().parents[4] / "nemo_rl/models/megatron/draft/dspark.py"
    )
    spec = importlib.util.spec_from_file_location("dspark_head_contract", module_path)
    if spec is None or spec.loader is None:
        pytest.fail("Could not load the DSpark head module", pytrace=False)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    loaded_module = module if isinstance(module, ModuleType) else None
    if loaded_module is None:
        pytest.fail("DSpark head module did not load", pytrace=False)
    return loaded_module.DSparkMarkovHead, loaded_module.DSparkConfidenceHead


DSparkMarkovHead, DSparkConfidenceHead = _load_heads()


def _run_markov_loss(
    *,
    base_logits: Tensor,
    previous_token_ids: Tensor,
    slot_valid: Tensor,
    markov_w1: Tensor,
    markov_w2: Tensor,
) -> tuple[Tensor, Tensor]:
    corrected = base_logits + F.linear(
        F.embedding(previous_token_ids, markov_w1),
        markov_w2,
    )
    corrected = torch.where(slot_valid.unsqueeze(-1), corrected, 0.0)
    coefficients = torch.arange(
        corrected.numel(),
        dtype=corrected.dtype,
        device=corrected.device,
    ).reshape_as(corrected)
    return corrected, (corrected * coefficients).sum()


def test_markov_head_matches_dense_math_and_gradients() -> None:
    torch.manual_seed(123)
    head = DSparkMarkovHead(vocab_size=11, markov_rank=3).double()
    base_logits = torch.randn((2, 4, 11), dtype=torch.float64, requires_grad=True)
    previous_token_ids = torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8]])
    slot_valid = torch.tensor([[True, True, False, True], [True, False, True, True]])

    actual = head(
        base_logits,
        previous_token_ids=previous_token_ids,
        slot_valid=slot_valid,
    )
    coefficients = torch.arange(actual.numel(), dtype=actual.dtype).reshape_as(actual)
    (actual * coefficients).sum().backward()
    actual_gradients = (
        base_logits.grad.detach().clone(),
        head.markov_w1.weight.grad.detach().clone(),
        head.markov_w2.weight.grad.detach().clone(),
    )

    reference_base = base_logits.detach().clone().requires_grad_()
    reference_w1 = head.markov_w1.weight.detach().clone().requires_grad_()
    reference_w2 = head.markov_w2.weight.detach().clone().requires_grad_()
    expected, reference_loss = _run_markov_loss(
        base_logits=reference_base,
        previous_token_ids=previous_token_ids,
        slot_valid=slot_valid,
        markov_w1=reference_w1,
        markov_w2=reference_w2,
    )
    reference_loss.backward()

    torch.testing.assert_close(actual, expected)
    for actual_gradient, expected_gradient in zip(
        actual_gradients,
        (reference_base.grad, reference_w1.grad, reference_w2.grad),
        strict=True,
    ):
        torch.testing.assert_close(actual_gradient, expected_gradient)


def test_markov_head_zeros_invalid_slots_and_their_gradients() -> None:
    torch.manual_seed(456)
    head = DSparkMarkovHead(vocab_size=13, markov_rank=4)
    base_logits = torch.randn((1, 3, 13), requires_grad=True)
    previous_token_ids = torch.tensor([[2, 12, 5]])
    slot_valid = torch.tensor([[True, False, True]])

    corrected = head(
        base_logits,
        previous_token_ids=previous_token_ids,
        slot_valid=slot_valid,
    )
    corrected.sum().backward()

    assert torch.equal(corrected[:, 1], torch.zeros_like(corrected[:, 1]))
    assert torch.equal(base_logits.grad[:, 1], torch.zeros_like(base_logits.grad[:, 1]))
    assert torch.equal(
        head.markov_w1.weight.grad[12],
        torch.zeros_like(head.markov_w1.weight.grad[12]),
    )


def test_markov_head_has_explicit_tp_local_vocab_contract() -> None:
    torch.manual_seed(789)
    head = DSparkMarkovHead(
        vocab_size=17,
        markov_rank=5,
        vocab_start_index=6,
        vocab_end_index=13,
    )
    base_logits = torch.randn((2, 3, 7))
    previous_token_ids = torch.tensor([[0, 8, 16], [2, 4, 6]])
    slot_valid = torch.ones((2, 3), dtype=torch.bool)

    actual = head(
        base_logits,
        previous_token_ids=previous_token_ids,
        slot_valid=slot_valid,
    )
    expected = base_logits + F.linear(
        F.embedding(previous_token_ids, head.markov_w1.weight),
        head.markov_w2.weight,
    )

    assert head.vocab_start_index == 6
    assert head.vocab_end_index == 13
    assert head.local_vocab_size == 7
    assert head.markov_w1.weight.shape == (17, 5)
    assert head.markov_w2.weight.shape == (7, 5)
    torch.testing.assert_close(actual, expected)

    with pytest.raises(ValueError, match="local vocab size"):
        head(
            torch.randn((2, 3, 8)),
            previous_token_ids=previous_token_ids,
            slot_valid=slot_valid,
        )
    with pytest.raises(ValueError, match="vocab shard"):
        DSparkMarkovHead(
            vocab_size=17,
            markov_rank=5,
            vocab_start_index=13,
            vocab_end_index=6,
        )


def test_markov_head_checkpoint_names_and_tp1_shapes_match_public_dspark() -> None:
    heads = nn.ModuleDict(
        {
            "markov_head": DSparkMarkovHead(
                vocab_size=151936,
                markov_rank=256,
                device="meta",
            ),
            "confidence_head": DSparkConfidenceHead(
                hidden_size=4096,
                markov_rank=256,
                with_markov=True,
                device="meta",
            ),
        }
    )
    state = heads.state_dict()

    assert set(state) == {
        "markov_head.markov_w1.weight",
        "markov_head.markov_w2.weight",
        "confidence_head.proj.weight",
        "confidence_head.proj.bias",
    }
    assert state["markov_head.markov_w1.weight"].shape == (151936, 256)
    assert state["markov_head.markov_w2.weight"].shape == (151936, 256)
    assert state["confidence_head.proj.weight"].shape == (1, 4352)
    assert state["confidence_head.proj.bias"].shape == (1,)
    assert not any(
        forbidden in name
        for name, _ in heads.named_parameters()
        for forbidden in ("lm_head", "embed_tokens", "mask")
    )


def test_markov_head_state_dict_round_trip_is_exact() -> None:
    torch.manual_seed(101)
    source = DSparkMarkovHead(vocab_size=19, markov_rank=4).eval()
    checkpoint = io.BytesIO()
    torch.save(source.state_dict(), checkpoint)

    torch.manual_seed(202)
    restored = DSparkMarkovHead(vocab_size=19, markov_rank=4).eval()
    checkpoint.seek(0)
    restored.load_state_dict(torch.load(checkpoint, weights_only=True))

    for name, source_tensor in source.state_dict().items():
        assert torch.equal(source_tensor, restored.state_dict()[name]), name


def test_markov_head_fails_loudly_on_ambiguous_inputs() -> None:
    head = DSparkMarkovHead(vocab_size=11, markov_rank=3)
    base_logits = torch.randn((2, 4, 11))
    previous_token_ids = torch.ones((2, 4), dtype=torch.int64)
    slot_valid = torch.ones((2, 4), dtype=torch.bool)

    with pytest.raises(ValueError, match="leading shape"):
        head(
            base_logits,
            previous_token_ids=previous_token_ids[:, :3],
            slot_valid=slot_valid,
        )
    with pytest.raises(TypeError, match="torch.int64"):
        head(
            base_logits,
            previous_token_ids=previous_token_ids.to(torch.int32),
            slot_valid=slot_valid,
        )
    with pytest.raises(TypeError, match="boolean"):
        head(
            base_logits,
            previous_token_ids=previous_token_ids,
            slot_valid=slot_valid.to(torch.int64),
        )


def test_confidence_head_matches_public_checkpoint_contract() -> None:
    torch.manual_seed(303)
    head = DSparkConfidenceHead(
        hidden_size=8,
        markov_rank=3,
        with_markov=True,
    ).double()
    hidden_states = torch.randn((2, 4, 8), dtype=torch.float64, requires_grad=True)
    markov_embeddings = torch.randn((2, 4, 3), dtype=torch.float64, requires_grad=True)
    slot_valid = torch.tensor([[True, True, False, True], [True, False, True, True]])

    actual = head(
        hidden_states,
        markov_embeddings=markov_embeddings,
        slot_valid=slot_valid,
    )
    expected = F.linear(
        torch.cat((hidden_states, markov_embeddings), dim=-1),
        head.proj.weight,
        head.proj.bias,
    ).squeeze(-1)
    expected = torch.where(slot_valid, expected, 0.0)

    assert set(head.state_dict()) == {"proj.weight", "proj.bias"}
    assert head.proj.weight.shape == (1, 11)
    assert head.proj.bias.shape == (1,)
    assert actual.dtype == torch.float32
    torch.testing.assert_close(actual, expected.float())


def test_confidence_head_without_markov_rejects_unexpected_embeddings() -> None:
    head = DSparkConfidenceHead(
        hidden_size=8,
        markov_rank=3,
        with_markov=False,
    )
    hidden_states = torch.randn((2, 4, 8))
    slot_valid = torch.ones((2, 4), dtype=torch.bool)

    output = head(hidden_states, slot_valid=slot_valid)
    assert output.shape == (2, 4)
    assert head.proj.weight.shape == (1, 8)

    with pytest.raises(ValueError, match="must be omitted"):
        head(
            hidden_states,
            markov_embeddings=torch.randn((2, 4, 3)),
            slot_valid=slot_valid,
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_heads_support_cuda_bfloat16_forward_and_backward() -> None:
    torch.manual_seed(404)
    device = torch.device("cuda")
    markov_head = DSparkMarkovHead(
        vocab_size=23,
        markov_rank=6,
        device=device,
        dtype=torch.bfloat16,
    )
    confidence_head = DSparkConfidenceHead(
        hidden_size=10,
        markov_rank=6,
        with_markov=True,
        device=device,
        dtype=torch.bfloat16,
    )
    base_logits = torch.randn(
        (2, 4, 23),
        device=device,
        dtype=torch.bfloat16,
        requires_grad=True,
    )
    hidden_states = torch.randn(
        (2, 4, 10),
        device=device,
        dtype=torch.bfloat16,
        requires_grad=True,
    )
    previous_token_ids = torch.tensor(
        [[1, 2, 3, 4], [5, 6, 7, 8]],
        device=device,
    )
    slot_valid = torch.tensor(
        [[True, True, False, True], [True, False, True, True]],
        device=device,
    )
    markov_embeddings = markov_head.markov_w1(previous_token_ids)

    corrected_logits = markov_head(
        base_logits,
        previous_token_ids=previous_token_ids,
        slot_valid=slot_valid,
    )
    confidence_logits = confidence_head(
        hidden_states,
        markov_embeddings=markov_embeddings,
        slot_valid=slot_valid,
    )
    (
        corrected_logits.float().square().mean() + confidence_logits.square().mean()
    ).backward()

    assert corrected_logits.dtype == torch.bfloat16
    assert confidence_logits.dtype == torch.float32
    assert torch.isfinite(corrected_logits).all()
    assert torch.isfinite(confidence_logits).all()
    assert torch.equal(
        corrected_logits[~slot_valid],
        torch.zeros_like(corrected_logits[~slot_valid]),
    )
    assert torch.equal(
        confidence_logits[~slot_valid],
        torch.zeros_like(confidence_logits[~slot_valid]),
    )
    for parameter in (*markov_head.parameters(), *confidence_head.parameters()):
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()
