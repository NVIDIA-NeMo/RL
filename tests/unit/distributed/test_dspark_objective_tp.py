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
    actual_loss = stats.combined.normalized(normalization_counts=stats.combined.counts)
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
        draft_hidden=torch.randn((1, 3, 4), generator=generator, device=device),
        target_output_weight=torch.randn((8, 4), generator=generator, device=device),
        markov_w1=torch.randn((8, 2), generator=generator, device=device),
        markov_w2=torch.randn((8, 2), generator=generator, device=device),
        previous_token_ids=torch.tensor([[1, 3, 5]], device=device),
        confidence_logits=torch.randn((1, 3), generator=generator, device=device),
        hard_labels=torch.tensor([[1, 3, 5]], device=device),
        valid_mask=valid_mask,
        slot_bins=torch.tensor([[0, 1, 2]], device=device),
        loss_weights=(1.25, 0.5, 2.0),
        token_chunk_size=2,
        vocab_start_index=0,
        tp_group=None,
    )
    reduced = torch.stack((stats.combined.numerators, stats.combined.counts))
    torch.distributed.all_reduce(reduced, group=dp_group)
    if rank == 0:
        torch.testing.assert_close(reduced[0], stats.combined.numerators)
        torch.testing.assert_close(reduced[1], torch.ones(3, device=device))


def _run_tp2_tv_only_gradient(rank: int, world_size: int) -> None:
    tp_group = torch.distributed.new_group(ranks=list(range(world_size)))
    device = torch.device("cuda")
    vocab_size, hidden_size, markov_rank = 4, 1, 1
    local_vocab_size = vocab_size // world_size
    vocab_start = rank * local_vocab_size
    vocab_end = vocab_start + local_vocab_size
    full_output_weight = torch.tensor(
        [[-2.0], [-0.5], [1.25], [3.0]], device=device, dtype=torch.float64
    )
    full_target_logits = torch.tensor(
        [[[2.0, -1.0, -2.0, 0.5]]], device=device, dtype=torch.float64
    )
    dense_hidden = torch.tensor([[[0.75]]], device=device, dtype=torch.float64)
    with torch.no_grad():
        local_output_weight = full_output_weight[vocab_start:vocab_end]
        local_target_logits = full_target_logits[..., vocab_start:vocab_end]
        local_draft_logits = (dense_hidden @ local_output_weight.T).float()
        local_target_logits_fp32 = local_target_logits.float()
        normalizer_maxima = torch.stack(
            (
                local_target_logits_fp32.amax(dim=-1),
                local_draft_logits.amax(dim=-1),
            ),
            dim=-1,
        )
        torch.distributed.all_reduce(
            normalizer_maxima,
            op=torch.distributed.ReduceOp.MAX,
            group=tp_group,
        )
        normalizer_exp_sums = torch.stack(
            (
                (local_target_logits_fp32 - normalizer_maxima[..., :1])
                .exp()
                .sum(dim=-1),
                (local_draft_logits - normalizer_maxima[..., 1:]).exp().sum(dim=-1),
            ),
            dim=-1,
        )
        torch.distributed.all_reduce(
            normalizer_exp_sums,
            op=torch.distributed.ReduceOp.SUM,
            group=tp_group,
        )
        log_normalizers = normalizer_maxima + normalizer_exp_sums.log()
        target_probs = (local_target_logits_fp32 - log_normalizers[..., :1]).exp()
        draft_probs = (local_draft_logits - log_normalizers[..., 1:]).exp()
        probability_gradient = 0.5 * torch.sign(draft_probs - target_probs)
        probability_expectation = (probability_gradient * draft_probs).sum(
            dim=-1, keepdim=True
        )
        torch.distributed.all_reduce(
            probability_expectation,
            op=torch.distributed.ReduceOp.SUM,
            group=tp_group,
        )
        logits_gradient = draft_probs * (probability_gradient - probability_expectation)
        expected_hidden_gradient = torch.zeros(
            (dense_hidden.numel() // hidden_size, hidden_size),
            device=device,
            dtype=torch.float32,
        )
        expected_hidden_gradient.addmm_(
            logits_gradient.reshape(-1, local_vocab_size),
            local_output_weight.float(),
        )
        torch.distributed.all_reduce(
            expected_hidden_gradient,
            op=torch.distributed.ReduceOp.SUM,
            group=tp_group,
        )

    provider = build_dspark_provider(
        body=_CheckpointBody(tp_group=tp_group, device=device),
        vocab_size=vocab_size,
        hidden_size=hidden_size,
        markov_rank=markov_rank,
        confidence_enabled=False,
        confidence_with_markov=False,
        vocab_start_index=vocab_start,
        vocab_end_index=vocab_end,
        tensor_parallel_group=tp_group,
        device=device,
        dtype=torch.float64,
    )
    with torch.no_grad():
        provider.markov_head.markov_w1.weight.zero_()
        provider.markov_head.markov_w2.weight.zero_()
    actual_hidden = dense_hidden.detach().clone().requires_grad_()
    stats = provider.objective_stats(
        draft_hidden=actual_hidden,
        target_output_weight=full_output_weight[vocab_start:vocab_end],
        target_logits=full_target_logits[..., vocab_start:vocab_end],
        previous_token_ids=torch.zeros((1, 1), device=device, dtype=torch.long),
        hard_labels=torch.zeros((1, 1), device=device, dtype=torch.long),
        valid_mask=torch.ones((1, 1), device=device, dtype=torch.bool),
        slot_bins=torch.zeros((1, 1), device=device, dtype=torch.long),
        loss_weights=(0.0, 1.0, 0.0),
        token_chunk_size=1,
        tp_group=tp_group,
    )
    stats.tv.normalized(normalization_counts=stats.tv.counts).backward()
    torch.testing.assert_close(
        actual_hidden.grad,
        expected_hidden_gradient.reshape_as(actual_hidden).to(
            dtype=actual_hidden.dtype
        ),
        rtol=0,
        atol=0,
    )

    empty_hidden = dense_hidden.detach().clone().requires_grad_()
    empty_stats = provider.objective_stats(
        draft_hidden=empty_hidden,
        target_output_weight=full_output_weight[vocab_start:vocab_end],
        target_logits=full_target_logits[..., vocab_start:vocab_end],
        previous_token_ids=torch.full((1, 1), -1, device=device, dtype=torch.long),
        hard_labels=torch.full((1, 1), -1, device=device, dtype=torch.long),
        valid_mask=torch.zeros((1, 1), device=device, dtype=torch.bool),
        slot_bins=torch.zeros((1, 1), device=device, dtype=torch.long),
        loss_weights=(0.0, 1.0, 0.0),
        token_chunk_size=1,
        tp_group=tp_group,
    )
    empty_stats.tv.normalized(normalization_counts=empty_stats.tv.counts).backward()
    torch.testing.assert_close(empty_hidden.grad, torch.zeros_like(empty_hidden))


def _run_tp2_hard_label_boundaries(rank: int, world_size: int) -> None:
    tp_group = torch.distributed.new_group(ranks=list(range(world_size)))
    device = torch.device("cuda")
    vocab_size = 4
    local_vocab_size = vocab_size // world_size
    vocab_start = rank * local_vocab_size
    provider = build_dspark_provider(
        body=_CheckpointBody(tp_group=tp_group, device=device),
        vocab_size=vocab_size,
        hidden_size=2,
        markov_rank=1,
        confidence_enabled=False,
        confidence_with_markov=False,
        vocab_start_index=vocab_start,
        vocab_end_index=vocab_start + local_vocab_size,
        tensor_parallel_group=tp_group,
        device=device,
        dtype=torch.float64,
    )
    common: dict[str, Any] = {
        "draft_hidden": torch.ones((1, 1, 2), device=device, dtype=torch.float64),
        "target_output_weight": torch.ones(
            (local_vocab_size, 2), device=device, dtype=torch.float64
        ),
        "target_logits": torch.ones(
            (1, 1, local_vocab_size), device=device, dtype=torch.float64
        ),
        "previous_token_ids": torch.zeros((1, 1), device=device, dtype=torch.long),
        "valid_mask": torch.ones((1, 1), device=device, dtype=torch.bool),
        "slot_bins": torch.zeros((1, 1), device=device, dtype=torch.long),
        "loss_weights": (1.0, 0.0, 0.0),
        "token_chunk_size": 1,
        "tp_group": tp_group,
    }
    for bad_label in (-1, vocab_size):
        with pytest.raises(ValueError, match="hard_labels"):
            provider.objective_stats(
                **common,
                hard_labels=torch.full(
                    (1, 1), bad_label, device=device, dtype=torch.long
                ),
            )


def _run_tp2_split_vocab_mapping_and_bounds(rank: int, world_size: int) -> None:
    tp_group = torch.distributed.new_group(ranks=list(range(world_size)))
    device = torch.device("cuda")
    target_vocab_size, draft_vocab_size = 11, 8
    local_draft_vocab_size = draft_vocab_size // world_size
    draft_vocab_start = rank * local_draft_vocab_size
    draft_vocab_end = draft_vocab_start + local_draft_vocab_size
    d2t = torch.tensor([10, 2, 8, 0, 6, 1, 9, 4], device=device)
    generator = torch.Generator(device=device).manual_seed(20260819)
    full_target_weight = torch.randn(
        target_vocab_size, 3, generator=generator, device=device
    )
    mapped_target_weight = full_target_weight.index_select(0, d2t)
    full_target_logits = torch.randn(
        1, 2, target_vocab_size, generator=generator, device=device
    )
    mapped_target_logits = full_target_logits.index_select(-1, d2t)
    provider = build_dspark_provider(
        body=_CheckpointBody(tp_group=tp_group, device=device),
        target_vocab_size=target_vocab_size,
        draft_vocab_size=draft_vocab_size,
        hidden_size=3,
        markov_rank=2,
        confidence_enabled=False,
        confidence_with_markov=False,
        draft_vocab_start_index=draft_vocab_start,
        draft_vocab_end_index=draft_vocab_end,
        tensor_parallel_group=tp_group,
        device=device,
        dtype=torch.float32,
    )
    common: dict[str, Any] = {
        "draft_hidden": torch.randn(1, 2, 3, generator=generator, device=device),
        "target_output_weight": mapped_target_weight[
            draft_vocab_start:draft_vocab_end
        ],
        "target_logits": mapped_target_logits[
            ..., draft_vocab_start:draft_vocab_end
        ],
        "previous_token_ids": torch.tensor([[10, 0]], device=device),
        "hard_labels": torch.tensor([[7, 0]], device=device),
        "valid_mask": torch.ones(1, 2, dtype=torch.bool, device=device),
        "slot_bins": torch.tensor([[0, 1]], device=device),
        "loss_weights": (1.0, 1.0, 0.0),
        "token_chunk_size": 1,
        "tp_group": tp_group,
    }

    stats = provider.objective_stats(**common)
    assert torch.isfinite(stats.combined.numerators).all()
    for field, bad_id in (
        ("previous_token_ids", target_vocab_size),
        ("hard_labels", draft_vocab_size),
    ):
        with pytest.raises(ValueError, match=field):
            provider.objective_stats(
                **{**common, field: torch.tensor([[bad_id, 0]], device=device)}
            )


def _run_dp2_global_normalization_gradient(rank: int, world_size: int) -> None:
    dp_group = torch.distributed.new_group(ranks=list(range(world_size)))
    device = torch.device("cuda")
    local_count = torch.tensor([2.0 if rank == 0 else 1.0], device=device)
    global_count = local_count.clone()
    torch.distributed.all_reduce(global_count, group=dp_group)
    differentiable_numerator = torch.tensor(
        [float(rank + 1)], device=device, requires_grad=True
    )
    from nemo_rl.algorithms.loss.dspark import DSparkLossBins

    bins = DSparkLossBins(differentiable_numerator, local_count)
    bins.normalized(normalization_counts=global_count).backward()
    torch.testing.assert_close(
        differentiable_numerator.grad,
        torch.tensor([1.0 / 3.0], device=device),
    )


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


def test_tp2_tv_only_gradient_uses_global_softmax_jacobian_expectation(
    distributed_test_runner,
) -> None:
    distributed_test_runner(_run_tp2_tv_only_gradient, world_size=2)


def test_tp2_rejects_global_hard_label_boundaries(distributed_test_runner) -> None:
    distributed_test_runner(_run_tp2_hard_label_boundaries, world_size=2)


def test_tp2_split_vocab_mapping_and_bounds(distributed_test_runner) -> None:
    distributed_test_runner(_run_tp2_split_vocab_mapping_and_bounds, world_size=2)


def test_dp2_normalized_gradient_uses_global_counts(distributed_test_runner) -> None:
    distributed_test_runner(_run_dp2_global_normalization_gradient, world_size=2)


def test_tp2_real_mcore_checkpoint_round_trip(
    distributed_test_runner,
    tmp_path: Path,
) -> None:
    pytest.importorskip("megatron.core.dist_checkpointing")
    distributed_test_runner(
        partial(_run_tp2_provider_checkpoint, checkpoint_dir=str(tmp_path / "dcp")),
        world_size=2,
    )
