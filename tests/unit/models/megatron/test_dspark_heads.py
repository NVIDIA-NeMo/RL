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
import torch.distributed as dist
import torch.multiprocessing as mp
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

_PUBLIC_DSPARK_REVISION = "737765aa0ff9f5dbda65839a6e010f03b66bc506"
_PUBLIC_DSPARK_CONFIG = {
    "vocab_size": 151936,
    "draft_vocab_size": 32000,
    "markov_rank": 256,
}
_PUBLIC_DSPARK_HEAD_SHAPES = {
    "markov_head.markov_w1.weight": (151936, 256),
    "markov_head.markov_w2.weight": (32000, 256),
    "confidence_head.proj.weight": (1, 4352),
    "confidence_head.proj.bias": (1,),
}


def _run_tp_markov_gradient(
    rank: int,
    world_size: int,
    init_file: str,
) -> None:
    use_cuda = torch.cuda.is_available()
    if use_cuda:
        torch.cuda.set_device(rank)
    device = torch.device("cuda", rank) if use_cuda else torch.device("cpu")
    dist.init_process_group(
        backend="nccl" if use_cuda else "gloo",
        init_method=f"file://{init_file}",
        rank=rank,
        world_size=world_size,
    )
    tp_group = dist.new_group(ranks=list(range(world_size)))
    assert isinstance(tp_group, dist.ProcessGroup)
    try:
        target_vocab_size = 12
        draft_vocab_size = 8
        markov_rank = 3
        local_draft_vocab_size = draft_vocab_size // world_size
        draft_vocab_start = rank * local_draft_vocab_size
        draft_vocab_end = draft_vocab_start + local_draft_vocab_size
        head = DSparkMarkovHead(
            target_vocab_size=target_vocab_size,
            draft_vocab_size=draft_vocab_size,
            markov_rank=markov_rank,
            draft_vocab_start_index=draft_vocab_start,
            draft_vocab_end_index=draft_vocab_end,
            tensor_parallel_group=tp_group,
            device=device,
        ).double()

        global_w1 = (
            torch.arange(
                target_vocab_size * markov_rank,
                dtype=torch.float64,
                device=device,
            ).reshape(target_vocab_size, markov_rank)
            / 17
        )
        global_w2 = (
            torch.arange(
                draft_vocab_size * markov_rank,
                dtype=torch.float64,
                device=device,
            ).reshape(draft_vocab_size, markov_rank)
            / 13
        )
        with torch.no_grad():
            head.markov_w1.weight.copy_(global_w1)
            head.markov_w2.weight.copy_(global_w2[draft_vocab_start:draft_vocab_end])

        previous_token_ids = torch.tensor(
            [[1, 4, 7], [2, 5, 9]],
            device=device,
        )
        slot_valid = torch.ones_like(previous_token_ids, dtype=torch.bool)
        base_logits = torch.zeros(
            (*previous_token_ids.shape, local_draft_vocab_size),
            dtype=torch.float64,
            device=device,
        )
        global_coefficients = torch.arange(
            previous_token_ids.numel() * draft_vocab_size,
            dtype=torch.float64,
            device=device,
        ).reshape(*previous_token_ids.shape, draft_vocab_size)

        actual = head(
            base_logits,
            previous_token_ids=previous_token_ids,
            slot_valid=slot_valid,
        )
        expected_local = F.linear(
            F.embedding(previous_token_ids, global_w1),
            global_w2[draft_vocab_start:draft_vocab_end],
        )
        torch.testing.assert_close(
            actual,
            expected_local,
            rtol=0,
            atol=0,
        )
        (
            actual * global_coefficients[..., draft_vocab_start:draft_vocab_end]
        ).sum().backward()

        reference_w1 = global_w1.clone().requires_grad_()
        reference_embeddings = F.embedding(previous_token_ids, reference_w1)
        reference_loss = sum(
            (
                F.linear(
                    reference_embeddings,
                    global_w2[shard_start : shard_start + local_draft_vocab_size],
                )
                * global_coefficients[
                    ..., shard_start : shard_start + local_draft_vocab_size
                ]
            ).sum()
            for shard_start in range(
                0,
                draft_vocab_size,
                local_draft_vocab_size,
            )
        )
        reference_loss.backward()
        torch.testing.assert_close(
            head.markov_w1.weight.grad,
            reference_w1.grad,
            rtol=0,
            atol=0,
        )
    finally:
        dist.destroy_process_group(tp_group)
        dist.destroy_process_group()


def _run_tp_markov_checkpoint(
    rank: int,
    world_size: int,
    init_file: str,
    checkpoint_dir: str,
) -> None:
    from megatron.core import dist_checkpointing
    from megatron.core.dist_checkpointing.mapping import ShardedTensor

    use_cuda = torch.cuda.is_available()
    if use_cuda:
        torch.cuda.set_device(rank)
    device = torch.device("cuda", rank) if use_cuda else torch.device("cpu")
    dist.init_process_group(
        backend="nccl" if use_cuda else "gloo",
        init_method=f"file://{init_file}",
        rank=rank,
        world_size=world_size,
    )
    tp_group = dist.new_group(ranks=list(range(world_size)))
    assert isinstance(tp_group, dist.ProcessGroup)
    dp_group = None
    for dp_rank in range(world_size):
        group = dist.new_group(ranks=[dp_rank])
        if dp_rank == rank:
            assert isinstance(group, dist.ProcessGroup)
            dp_group = group
    assert dp_group is not None
    try:
        target_vocab_size = 12
        draft_vocab_size = 8
        markov_rank = 3
        local_draft_vocab_size = draft_vocab_size // world_size
        draft_vocab_start = rank * local_draft_vocab_size
        draft_vocab_end = draft_vocab_start + local_draft_vocab_size
        source = DSparkMarkovHead(
            target_vocab_size=target_vocab_size,
            draft_vocab_size=draft_vocab_size,
            markov_rank=markov_rank,
            draft_vocab_start_index=draft_vocab_start,
            draft_vocab_end_index=draft_vocab_end,
            tensor_parallel_group=tp_group,
            device=device,
        ).double()
        with torch.no_grad():
            source.markov_w1.weight.copy_(
                torch.arange(
                    target_vocab_size * markov_rank,
                    dtype=torch.float64,
                    device=device,
                ).reshape(target_vocab_size, markov_rank)
            )
            source.markov_w2.weight.copy_(
                torch.arange(
                    draft_vocab_start * markov_rank,
                    draft_vocab_end * markov_rank,
                    dtype=torch.float64,
                    device=device,
                ).reshape(local_draft_vocab_size, markov_rank)
            )

        metadata = {"dp_cp_group": dp_group}
        sharded_state = source.sharded_state_dict(
            prefix="markov_head.",
            metadata=metadata,
        )
        assert set(sharded_state) == {
            "markov_head.markov_w1.weight",
            "markov_head.markov_w2.weight",
        }
        markov_w2 = sharded_state["markov_head.markov_w2.weight"]
        assert isinstance(markov_w2, ShardedTensor)
        assert markov_w2.key == "markov_head.markov_w2.weight"
        assert markov_w2.local_shape == (local_draft_vocab_size, markov_rank)
        assert markov_w2.global_shape == (draft_vocab_size, markov_rank)
        assert markov_w2.global_offset == (draft_vocab_start, 0)
        assert markov_w2.axis_fragmentations == (world_size, 1)

        Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)
        dist_checkpointing.save({"model": sharded_state}, checkpoint_dir)
        restored = DSparkMarkovHead(
            target_vocab_size=target_vocab_size,
            draft_vocab_size=draft_vocab_size,
            markov_rank=markov_rank,
            draft_vocab_start_index=draft_vocab_start,
            draft_vocab_end_index=draft_vocab_end,
            tensor_parallel_group=tp_group,
            device=device,
        ).double()
        restore_template = restored.sharded_state_dict(
            prefix="markov_head.",
            metadata=metadata,
        )
        loaded = dist_checkpointing.load(
            {"model": restore_template},
            checkpoint_dir,
        )
        incompatible = restored.load_state_dict(
            {
                key.removeprefix("markov_head."): value
                for key, value in loaded["model"].items()
            }
        )
        assert not incompatible.missing_keys
        assert not incompatible.unexpected_keys
        for name, source_tensor in source.state_dict().items():
            torch.testing.assert_close(
                restored.state_dict()[name],
                source_tensor,
                rtol=0,
                atol=0,
            )
    finally:
        dist.destroy_process_group(dp_group)
        dist.destroy_process_group(tp_group)
        dist.destroy_process_group()


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
    head = DSparkMarkovHead(
        target_vocab_size=11,
        draft_vocab_size=7,
        markov_rank=3,
    ).double()
    base_logits = torch.randn((2, 4, 7), dtype=torch.float64, requires_grad=True)
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


@pytest.mark.skipif(not dist.is_available(), reason="torch.distributed is required")
def test_tp2_markov_head_sums_replicated_w1_gradients(tmp_path: Path) -> None:
    mp.spawn(
        _run_tp_markov_gradient,
        args=(2, str(tmp_path / "tp_init")),
        nprocs=2,
        join=True,
    )


@pytest.mark.skipif(not dist.is_available(), reason="torch.distributed is required")
def test_tp2_markov_head_megatron_checkpoint_round_trip(tmp_path: Path) -> None:
    pytest.importorskip("megatron.core.dist_checkpointing")
    mp.spawn(
        _run_tp_markov_checkpoint,
        args=(
            2,
            str(tmp_path / "tp_checkpoint_init"),
            str(tmp_path / "checkpoint"),
        ),
        nprocs=2,
        join=True,
    )


def test_markov_head_zeros_invalid_slots_and_their_gradients() -> None:
    torch.manual_seed(456)
    head = DSparkMarkovHead(
        target_vocab_size=13,
        draft_vocab_size=7,
        markov_rank=4,
    )
    base_logits = torch.randn((1, 3, 7), requires_grad=True)
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
    with pytest.raises(ValueError, match="tensor_parallel_group"):
        DSparkMarkovHead(
            target_vocab_size=17,
            draft_vocab_size=12,
            markov_rank=5,
            draft_vocab_start_index=6,
            draft_vocab_end_index=12,
        )
    with pytest.raises(ValueError, match="draft vocab shard"):
        DSparkMarkovHead(
            target_vocab_size=17,
            draft_vocab_size=12,
            markov_rank=5,
            draft_vocab_start_index=12,
            draft_vocab_end_index=6,
        )


def test_markov_head_loads_pinned_public_dspark_checkpoint_schema() -> None:
    assert _PUBLIC_DSPARK_REVISION == "737765aa0ff9f5dbda65839a6e010f03b66bc506"
    heads = nn.ModuleDict(
        {
            "markov_head": DSparkMarkovHead(
                target_vocab_size=_PUBLIC_DSPARK_CONFIG["vocab_size"],
                draft_vocab_size=_PUBLIC_DSPARK_CONFIG["draft_vocab_size"],
                markov_rank=_PUBLIC_DSPARK_CONFIG["markov_rank"],
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

    assert set(state) == set(_PUBLIC_DSPARK_HEAD_SHAPES)
    assert {
        name: tuple(tensor.shape) for name, tensor in state.items()
    } == _PUBLIC_DSPARK_HEAD_SHAPES
    pinned_checkpoint = {
        name: torch.empty(shape, dtype=torch.bfloat16, device="meta")
        for name, shape in _PUBLIC_DSPARK_HEAD_SHAPES.items()
    }
    incompatible = heads.load_state_dict(pinned_checkpoint, strict=True, assign=True)
    assert not incompatible.missing_keys
    assert not incompatible.unexpected_keys
    assert not any(
        forbidden in name
        for name, _ in heads.named_parameters()
        for forbidden in ("lm_head", "embed_tokens", "mask")
    )


def test_markov_head_state_dict_round_trip_is_exact() -> None:
    torch.manual_seed(101)
    source = DSparkMarkovHead(
        target_vocab_size=19,
        draft_vocab_size=7,
        markov_rank=4,
    ).eval()
    checkpoint = io.BytesIO()
    torch.save(source.state_dict(), checkpoint)

    torch.manual_seed(202)
    restored = DSparkMarkovHead(
        target_vocab_size=19,
        draft_vocab_size=7,
        markov_rank=4,
    ).eval()
    checkpoint.seek(0)
    restored.load_state_dict(torch.load(checkpoint, weights_only=True))

    for name, source_tensor in source.state_dict().items():
        assert torch.equal(source_tensor, restored.state_dict()[name]), name


def test_markov_head_fails_loudly_on_ambiguous_inputs() -> None:
    head = DSparkMarkovHead(
        target_vocab_size=11,
        draft_vocab_size=7,
        markov_rank=3,
    )
    base_logits = torch.randn((2, 4, 7))
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


@pytest.mark.parametrize("with_markov", [False, True])
def test_confidence_head_sanitizes_invalid_nonfinite_features(
    with_markov: bool,
) -> None:
    torch.manual_seed(505)
    head = DSparkConfidenceHead(
        hidden_size=4,
        markov_rank=2,
        with_markov=with_markov,
    ).double()
    hidden_data = torch.randn((1, 4, 4), dtype=torch.float64)
    hidden_data[0, 1, 0] = torch.nan
    hidden_data[0, 2, 1] = torch.inf
    hidden_data[0, 3, 2] = -torch.inf
    hidden_states = hidden_data.requires_grad_()
    markov_data = torch.randn((1, 4, 2), dtype=torch.float64)
    markov_data[0, 1, 0] = torch.inf
    markov_data[0, 2, 1] = torch.nan
    markov_embeddings = markov_data.requires_grad_() if with_markov else None
    slot_valid = torch.tensor([[True, False, False, False]])

    actual = head(
        hidden_states,
        markov_embeddings=markov_embeddings,
        slot_valid=slot_valid,
    )
    actual.sum().backward()

    reference_hidden = hidden_data.detach().clone().requires_grad_()
    reference_features = torch.where(
        slot_valid.unsqueeze(-1),
        reference_hidden,
        torch.zeros_like(reference_hidden),
    )
    reference_markov = None
    if with_markov:
        assert markov_embeddings is not None
        reference_markov = markov_data.detach().clone().requires_grad_()
        safe_reference_markov = torch.where(
            slot_valid.unsqueeze(-1),
            reference_markov,
            torch.zeros_like(reference_markov),
        )
        reference_features = torch.cat(
            (reference_features, safe_reference_markov),
            dim=-1,
        )
    reference_weight = head.proj.weight.detach().clone().requires_grad_()
    reference_bias = head.proj.bias.detach().clone().requires_grad_()
    expected = F.linear(
        reference_features,
        reference_weight,
        reference_bias,
    ).squeeze(-1)
    expected = torch.where(slot_valid, expected, torch.zeros_like(expected)).float()
    expected.sum().backward()

    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    gradient_pairs = [
        (hidden_states.grad, reference_hidden.grad),
        (head.proj.weight.grad, reference_weight.grad),
        (head.proj.bias.grad, reference_bias.grad),
    ]
    if with_markov:
        assert markov_embeddings is not None
        assert reference_markov is not None
        gradient_pairs.append((markov_embeddings.grad, reference_markov.grad))
    for actual_gradient, expected_gradient in gradient_pairs:
        assert actual_gradient is not None
        assert expected_gradient is not None
        assert torch.isfinite(actual_gradient).all()
        torch.testing.assert_close(
            actual_gradient,
            expected_gradient,
            rtol=0,
            atol=0,
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_heads_support_cuda_bfloat16_forward_and_backward() -> None:
    torch.manual_seed(404)
    device = torch.device("cuda")
    markov_head = DSparkMarkovHead(
        target_vocab_size=23,
        draft_vocab_size=17,
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
        (2, 4, 17),
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
