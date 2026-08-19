from functools import partial
from pathlib import Path
from typing import Any

import pytest
import torch
import torch.nn.functional as F

from nemo_rl.algorithms.loss.dspark import dspark_tiled_objective
from nemo_rl.models.megatron.draft.dspark_provider import build_dspark_provider


class _CheckpointBody(torch.nn.Module):
    def __init__(
        self,
        *,
        tp_group: torch.distributed.ProcessGroup,
        device: torch.device,
    ) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.zeros(3, 3, device=device))
        self.tp_group = tp_group

    def sharded_state_dict(
        self,
        prefix: str = "",
        sharded_offsets: tuple[tuple[int, int, int], ...] = (),
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        from megatron.core.transformer.utils import (
            make_sharded_tensors_for_checkpoint,
        )

        dp_cp_group = None if metadata is None else metadata.get("dp_cp_group")
        return make_sharded_tensors_for_checkpoint(
            self.state_dict(prefix="", keep_vars=True),
            prefix,
            {},
            sharded_offsets,
            tp_group=self.tp_group,
            dp_cp_group=dp_cp_group,
        )


def _dense_loss(
    *,
    hidden: torch.Tensor,
    output_weight: torch.Tensor,
    target_logits: torch.Tensor,
    previous_token_ids: torch.Tensor,
    hard_labels: torch.Tensor,
    valid_mask: torch.Tensor,
    slot_bins: torch.Tensor,
    markov_w1: torch.Tensor,
    markov_w2: torch.Tensor,
    confidence_weight: torch.Tensor,
    confidence_bias: torch.Tensor,
    loss_weights: tuple[float, float, float],
) -> tuple[torch.Tensor, tuple[torch.Tensor, ...]]:
    embeddings = F.embedding(previous_token_ids, markov_w1)
    corrected_logits = hidden @ output_weight.T + F.linear(embeddings, markov_w2)
    target_probs = torch.softmax(target_logits.detach().float(), dim=-1)
    draft_probs = torch.softmax(corrected_logits.float(), dim=-1)
    ce_rows = (
        -torch.log_softmax(corrected_logits.float(), dim=-1)
        .gather(-1, hard_labels.unsqueeze(-1))
        .squeeze(-1)
    )
    tv_rows = 0.5 * (target_probs - draft_probs).abs().sum(dim=-1)
    verifier_correct = corrected_logits.detach().argmax(dim=-1).eq(hard_labels).float()
    confidence_logits = (
        F.linear(
            torch.cat((hidden, embeddings), dim=-1),
            confidence_weight,
            confidence_bias,
        )
        .squeeze(-1)
        .float()
    )
    confidence_rows = F.binary_cross_entropy_with_logits(
        confidence_logits,
        verifier_correct,
        reduction="none",
    )
    num_bins = hidden.shape[1]
    numerators = []
    counts = torch.zeros(num_bins, device=hidden.device)
    counts.scatter_add_(0, slot_bins.reshape(-1), valid_mask.reshape(-1).float())
    for rows in (ce_rows, tv_rows, confidence_rows):
        component = torch.zeros(num_bins, device=hidden.device)
        component.scatter_add_(
            0,
            slot_bins.reshape(-1),
            rows.reshape(-1) * valid_mask.reshape(-1).float(),
        )
        numerators.append(component)
    combined = sum(
        weight * component
        for weight, component in zip(loss_weights, numerators, strict=True)
    )
    return combined.sum() / counts.sum(), (*numerators, combined, counts)


def _run_tp2_provider_objective(rank: int, world_size: int) -> None:
    tp_group = torch.distributed.new_group(ranks=list(range(world_size)))
    device = torch.device("cuda")
    vocab_size, hidden_size, markov_rank = 12, 6, 4
    local_vocab_size = vocab_size // world_size
    vocab_start = rank * local_vocab_size
    vocab_end = vocab_start + local_vocab_size
    generator = torch.Generator(device=device).manual_seed(424242)
    hidden_data = torch.randn(
        2, 3, hidden_size, generator=generator, device=device, dtype=torch.bfloat16
    )
    output_weight = torch.randn(
        vocab_size,
        hidden_size,
        generator=generator,
        device=device,
        dtype=torch.bfloat16,
    )
    target_logits = torch.randn(
        2, 3, vocab_size, generator=generator, device=device, dtype=torch.bfloat16
    )
    markov_w1 = torch.randn(
        vocab_size,
        markov_rank,
        generator=generator,
        device=device,
        dtype=torch.bfloat16,
    )
    markov_w2 = torch.randn(
        vocab_size,
        markov_rank,
        generator=generator,
        device=device,
        dtype=torch.bfloat16,
    )
    confidence_weight = torch.randn(
        1,
        hidden_size + markov_rank,
        generator=generator,
        device=device,
        dtype=torch.bfloat16,
    )
    confidence_bias = torch.randn(
        1, generator=generator, device=device, dtype=torch.bfloat16
    )
    previous_token_ids = torch.tensor([[2, 2, 4], [2, 7, 2]], device=device)
    hard_labels = torch.tensor([[1, 9, 2], [5, 0, 10]], device=device)
    valid_mask = torch.tensor([[True, True, False], [True, False, True]], device=device)
    slot_bins = torch.tensor([[0, 1, 2], [0, 1, 2]], device=device)
    loss_weights = (1.5, 0.375, 2.25)

    reference_hidden = hidden_data.clone().requires_grad_()
    reference_w1 = markov_w1.clone().requires_grad_()
    reference_w2 = markov_w2.clone().requires_grad_()
    reference_confidence_weight = confidence_weight.clone().requires_grad_()
    reference_confidence_bias = confidence_bias.clone().requires_grad_()
    expected_loss, expected_stats = _dense_loss(
        hidden=reference_hidden,
        output_weight=output_weight,
        target_logits=target_logits,
        previous_token_ids=previous_token_ids,
        hard_labels=hard_labels,
        valid_mask=valid_mask,
        slot_bins=slot_bins,
        markov_w1=reference_w1,
        markov_w2=reference_w2,
        confidence_weight=reference_confidence_weight,
        confidence_bias=reference_confidence_bias,
        loss_weights=loss_weights,
    )
    expected_loss.backward()

    provider = build_dspark_provider(
        body=torch.nn.Identity(),
        vocab_size=vocab_size,
        hidden_size=hidden_size,
        markov_rank=markov_rank,
        confidence_enabled=True,
        confidence_with_markov=True,
        vocab_start_index=vocab_start,
        vocab_end_index=vocab_end,
        tensor_parallel_group=tp_group,
        device=device,
        dtype=torch.bfloat16,
    )
    with torch.no_grad():
        provider.markov_head.markov_w1.weight.copy_(markov_w1)
        provider.markov_head.markov_w2.weight.copy_(markov_w2[vocab_start:vocab_end])
        assert provider.confidence_head is not None
        provider.confidence_head.proj.weight.copy_(confidence_weight)
        provider.confidence_head.proj.bias.copy_(confidence_bias)
    actual_hidden = hidden_data.clone().requires_grad_()
    local_output_weight = output_weight[vocab_start:vocab_end].clone().requires_grad_()
    local_target_logits = (
        target_logits[..., vocab_start:vocab_end].clone().requires_grad_()
    )
    stats = provider.objective_stats(
        draft_hidden=actual_hidden,
        target_output_weight=local_output_weight,
        target_logits=local_target_logits,
        previous_token_ids=previous_token_ids,
        hard_labels=hard_labels,
        valid_mask=valid_mask,
        slot_bins=slot_bins,
        loss_weights=loss_weights,
        token_chunk_size=2,
        tp_group=tp_group,
    )
    actual_loss = stats.combined.normalized()
    actual_loss.backward()

    for actual, expected in zip(
        (
            stats.ce.numerators,
            stats.tv.numerators,
            stats.confidence.numerators,
            stats.combined.numerators,
            stats.combined.counts,
        ),
        expected_stats,
        strict=True,
    ):
        torch.testing.assert_close(actual, expected, rtol=0.025, atol=0.025)
    torch.testing.assert_close(actual_loss, expected_loss, rtol=0.025, atol=0.025)
    torch.testing.assert_close(
        actual_hidden.grad, reference_hidden.grad, rtol=0.08, atol=0.04
    )
    torch.testing.assert_close(
        provider.markov_head.markov_w1.weight.grad,
        reference_w1.grad,
        rtol=0.08,
        atol=0.04,
    )
    torch.testing.assert_close(
        provider.markov_head.markov_w2.weight.grad,
        reference_w2.grad[vocab_start:vocab_end],
        rtol=0.08,
        atol=0.04,
    )
    assert provider.confidence_head is not None
    torch.testing.assert_close(
        provider.confidence_head.proj.weight.grad,
        reference_confidence_weight.grad,
        rtol=0.08,
        atol=0.04,
    )
    torch.testing.assert_close(
        provider.confidence_head.proj.bias.grad,
        reference_confidence_bias.grad,
        rtol=0.08,
        atol=0.04,
    )
    assert local_output_weight.grad is None
    assert local_target_logits.grad is None


def _run_dp2_raw_additive_stats(rank: int, world_size: int) -> None:
    dp_group = torch.distributed.new_group(ranks=list(range(world_size)))
    device = torch.device("cuda")
    generator = torch.Generator(device=device).manual_seed(1234)
    shape = (1, 3, 8)
    valid_mask = (
        torch.tensor([[True, True, True]], device=device)
        if rank == 0
        else torch.zeros((1, 3), dtype=torch.bool, device=device)
    )
    stats = dspark_tiled_objective(
        target_logits=torch.randn(shape, generator=generator, device=device),
        base_logits=torch.randn(shape, generator=generator, device=device),
        markov_bias=torch.randn(shape, generator=generator, device=device),
        confidence_logits=torch.randn((1, 3), generator=generator, device=device),
        hard_labels=torch.tensor([[1, 3, 5]], device=device),
        valid_mask=valid_mask,
        slot_bins=torch.tensor([[0, 1, 2]], device=device),
        loss_weights=(1.25, 0.5, 2.0),
        token_chunk_size=2,
        tp_group=None,
    )
    reduced = torch.stack((stats.combined.numerators, stats.combined.counts))
    torch.distributed.all_reduce(reduced, group=dp_group)
    if rank == 0:
        torch.testing.assert_close(reduced[0], stats.combined.numerators)
        torch.testing.assert_close(reduced[1], torch.ones(3, device=device))


def _run_tp2_provider_checkpoint(
    rank: int,
    world_size: int,
    checkpoint_dir: str,
) -> None:
    from megatron.core import dist_checkpointing

    tp_group = torch.distributed.new_group(ranks=list(range(world_size)))
    dp_group = None
    for dp_rank in range(world_size):
        group = torch.distributed.new_group(ranks=[dp_rank])
        if dp_rank == rank:
            dp_group = group
    assert isinstance(dp_group, torch.distributed.ProcessGroup)
    device = torch.device("cuda")
    vocab_size, hidden_size, markov_rank = 12, 6, 4
    local_vocab_size = vocab_size // world_size
    vocab_start = rank * local_vocab_size
    vocab_end = vocab_start + local_vocab_size

    def make_provider() -> Any:
        return build_dspark_provider(
            body=_CheckpointBody(tp_group=tp_group, device=device),
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            markov_rank=markov_rank,
            confidence_enabled=True,
            confidence_with_markov=True,
            vocab_start_index=vocab_start,
            vocab_end_index=vocab_end,
            tensor_parallel_group=tp_group,
            device=device,
            dtype=torch.float32,
        )

    source = make_provider()
    with torch.no_grad():
        for parameter_index, parameter in enumerate(source.parameters()):
            values = torch.arange(
                parameter.numel(), device=device, dtype=parameter.dtype
            ).reshape_as(parameter)
            parameter.copy_(values + 1000 * parameter_index)
    metadata = {"dp_cp_group": dp_group}
    sharded_state = source.sharded_state_dict(
        prefix="provider.",
        metadata=metadata,
    )
    assert set(sharded_state) == {
        "provider.body.weight",
        "provider.markov_head.markov_w1.weight",
        "provider.markov_head.markov_w2.weight",
        "provider.confidence_head.proj.weight",
        "provider.confidence_head.proj.bias",
    }
    Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)
    dist_checkpointing.save({"model": sharded_state}, checkpoint_dir)

    restored = make_provider()
    restore_template = restored.sharded_state_dict(
        prefix="provider.",
        metadata=metadata,
    )
    loaded = dist_checkpointing.load({"model": restore_template}, checkpoint_dir)
    incompatible = restored.load_state_dict(
        {
            name.removeprefix("provider."): tensor
            for name, tensor in loaded["model"].items()
        }
    )
    assert not incompatible.missing_keys
    assert not incompatible.unexpected_keys
    for name, source_tensor in source.state_dict().items():
        torch.testing.assert_close(
            restored.state_dict()[name], source_tensor, rtol=0, atol=0
        )


def test_tp2_bf16_output_and_hidden_head_gradients(distributed_test_runner) -> None:
    distributed_test_runner(_run_tp2_provider_objective, world_size=2)


def test_dp2_raw_stats_allow_one_rank_with_zero_slots(distributed_test_runner) -> None:
    distributed_test_runner(_run_dp2_raw_additive_stats, world_size=2)


def test_tp2_real_mcore_checkpoint_round_trip(
    distributed_test_runner,
    tmp_path: Path,
) -> None:
    pytest.importorskip("megatron.core.dist_checkpointing")
    distributed_test_runner(
        partial(_run_tp2_provider_checkpoint, checkpoint_dir=str(tmp_path / "dcp")),
        world_size=2,
    )
