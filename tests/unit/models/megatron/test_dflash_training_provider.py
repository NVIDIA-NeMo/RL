from __future__ import annotations

from types import SimpleNamespace

import torch
from torch import Tensor

from nemo_rl.algorithms.loss.draft import dflash_projected_vocab_parallel_soft_ce
from nemo_rl.distributed.batched_data_dict import BatchedDataDict
from nemo_rl.models.megatron.draft.hidden_capture import CapturedStates
from nemo_rl.models.megatron.draft.sequence_layout import build_draft_sequence_layout
from nemo_rl.models.megatron.draft.training import (
    DFlashForwardOutput,
    resolve_draft_speculator,
)
from nemo_rl.models.policy.draft_config import DFlashDraftConfig


class _Target(torch.nn.Module):
    def __init__(self, *, hidden_size: int, vocab_size: int) -> None:
        super().__init__()
        self.embedding = SimpleNamespace(
            word_embeddings=torch.nn.Embedding(vocab_size, hidden_size)
        )
        self.output_layer = torch.nn.Linear(hidden_size, vocab_size, bias=False)
        self.share_embeddings_and_output_weights = False


class _Draft(torch.nn.Module):
    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.scale = torch.nn.Parameter(torch.tensor(0.5))
        self.config = SimpleNamespace(hidden_size=hidden_size)
        self.sequence_layout = None
        self.context_parallel_group = None

    def forward(
        self,
        *,
        target_taps: Tensor,
        block_embeddings: Tensor,
        plan: object,
        sequence_layout: object | None = None,
        context_parallel_group: object | None = None,
    ) -> Tensor:
        del target_taps, plan
        self.sequence_layout = sequence_layout
        self.context_parallel_group = context_parallel_group
        return block_embeddings * self.scale


def _provider():
    provider = resolve_draft_speculator(
        DFlashDraftConfig(
            enabled=True,
            gamma=2,
            anchors_per_sample=2,
            mask_token_id=7,
            target_hidden_state_layer_ids=[0, 2],
            num_layers=1,
            seed=13,
            vocab_tile_size=3,
            position_decay=0.5,
            max_cp_boundary_exclusion_fraction=1.0,
        )
    )
    assert provider is not None
    return provider


def test_dflash_provider_prepares_forward_and_raw_position_bins() -> None:
    torch.manual_seed(4)
    batch_size, sequence_length, hidden_size, vocab_size = 2, 8, 4, 8
    provider = _provider()
    target = _Target(hidden_size=hidden_size, vocab_size=vocab_size)
    draft = _Draft(hidden_size)
    data = BatchedDataDict(
        {
            "input_ids": torch.arange(batch_size * sequence_length).reshape(
                batch_size, sequence_length
            ),
            "input_lengths": torch.tensor([8, 7]),
            "draft_sample_ids": torch.tensor([101, 303], dtype=torch.int64),
            "token_mask": torch.ones(batch_size, sequence_length),
            "sample_mask": torch.ones(batch_size),
        }
    )
    captured = CapturedStates(
        hidden_states=torch.randn(sequence_length, batch_size, 2 * hidden_size),
        inputs_embeds=torch.randn(sequence_length, batch_size, hidden_size),
    )

    provider.forward(
        policy_model=target,
        draft_model=draft,
        captured_states=captured,
        input_ids_cp_local=data["input_ids"],
        attention_mask=None,
        data=data,
        optimizer_step=9,
        sequence_layout=None,
        context_parallel_group=None,
        tensor_parallel_group=None,
    )
    output = data["dflash_output"]
    assert isinstance(output, DFlashForwardOutput)
    assert output.hidden.shape == (4, 3, hidden_size)
    assert output.plan.anchor_ids.shape == (4,)

    teacher_logits = torch.randn(
        batch_size,
        sequence_length,
        vocab_size,
        requires_grad=True,
    )
    stats = provider.loss_stats(
        target_logits=teacher_logits,
        data=data,
        prepare_fn=lambda **_: None,
        vocab_parallel_rank=0,
        vocab_parallel_group=None,
        context_parallel_group=None,
    )

    assert stats.numerators.shape == (2,)
    assert torch.equal(stats.weights, torch.tensor([1.0, 0.5]))
    output = data["dflash_output"]
    assert output.sequence_layout is None
    assert output.selected_teacher_logits is None
    expected = dflash_projected_vocab_parallel_soft_ce(
        draft_hidden=output.hidden,
        output_weight=output.output_weight,
        teacher_logits=teacher_logits,
        sample_rows=output.plan.sample_rows,
        label_positions=output.plan.label_positions,
        loss_mask=provider._loss_mask(output.plan, data),
        position_decay=provider.config.position_decay,
        token_chunk_size=provider.config.vocab_tile_size,
        tp_group=None,
    )
    torch.testing.assert_close(stats.numerators, expected.numerators)
    stats.normalized(normalization_counts=stats.counts).backward()
    assert draft.scale.grad is not None
    assert teacher_logits.grad is None
    assert target.output_layer.weight.grad is None


def test_dflash_anchor_identity_is_stable_across_split_order() -> None:
    provider = _provider()
    base = BatchedDataDict(
        {
            "input_ids": torch.ones(2, 8, dtype=torch.int64),
            "input_lengths": torch.tensor([8, 8]),
            "draft_sample_ids": torch.tensor([101, 303], dtype=torch.int64),
        }
    )
    reverse = BatchedDataDict({name: tensor.flip(0) for name, tensor in base.items()})

    forward_plan = provider.prepare_batch(base, optimizer_step=9)
    reverse_plan = provider.prepare_batch(reverse, optimizer_step=9)
    assert forward_plan is not None
    assert reverse_plan is not None

    forward_by_id = {
        sample_id: forward_plan.anchor_positions[index * 2 : (index + 1) * 2]
        for index, sample_id in enumerate(base["draft_sample_ids"].tolist())
    }
    reverse_by_id = {
        sample_id: reverse_plan.anchor_positions[index * 2 : (index + 1) * 2]
        for index, sample_id in enumerate(reverse["draft_sample_ids"].tolist())
    }
    for sample_id in forward_by_id:
        torch.testing.assert_close(forward_by_id[sample_id], reverse_by_id[sample_id])


def test_dflash_provider_consumes_reconstructed_sp_captures_on_cp_owner() -> None:
    torch.manual_seed(5)
    hidden_size, vocab_size = 4, 8
    provider = _provider()
    target = _Target(hidden_size=hidden_size, vocab_size=vocab_size)
    draft = _Draft(hidden_size)
    data = BatchedDataDict(
        {
            "input_ids": torch.arange(16).reshape(2, 8),
            "input_lengths": torch.tensor([7, 5]),
            "draft_sample_ids": torch.tensor([101, 303], dtype=torch.int64),
            "token_mask": torch.ones(2, 8),
            "sample_mask": torch.ones(2),
        }
    )
    layout = build_draft_sequence_layout(
        logical_sample_ids=data["draft_sample_ids"],
        cu_seqlens_q=torch.tensor([0, 7, 12], dtype=torch.int64),
        cu_seqlens_q_padded=torch.tensor([0, 8, 16], dtype=torch.int64),
        cp_rank=1,
        cp_size=2,
        tp_rank=0,
        tp_size=2,
        device=torch.device("cpu"),
    )
    cp_local_length = layout.cp_global_positions.numel()
    captured = CapturedStates(
        hidden_states=torch.randn(cp_local_length, 1, 2 * hidden_size),
        inputs_embeds=torch.randn(cp_local_length, 1, hidden_size),
        output_hidden=torch.randn(cp_local_length, 1, hidden_size),
        sequence_layout=layout,
        sequence_is_reconstructed=True,
    )
    cp_group = object()

    provider.forward(
        policy_model=target,
        draft_model=draft,
        captured_states=captured,
        input_ids_cp_local=torch.ones(1, cp_local_length, dtype=torch.int64),
        attention_mask=None,
        data=data,
        optimizer_step=9,
        sequence_layout=layout,
        context_parallel_group=cp_group,
        tensor_parallel_group=object(),
    )

    output = data["dflash_output"]
    assert isinstance(output, DFlashForwardOutput)
    assert output.sequence_layout is layout
    assert output.plan.owner_cp_ranks.eq(layout.cp_rank).all()
    assert output.hidden.shape == (
        output.plan.sample_rows.numel(),
        3,
        hidden_size,
    )
    assert draft.sequence_layout is layout
    assert draft.context_parallel_group is cp_group
    data["token_mask"][
        output.plan.sample_rows[0],
        output.plan.packed_rope_positions[0, 1],
    ] = 0
    expected_loss_mask = output.plan.loss_mask & data["token_mask"].to(torch.bool)[
        output.plan.sample_rows[:, None],
        output.plan.packed_rope_positions,
    ]
    torch.testing.assert_close(
        provider._loss_mask(output.plan, data, layout),
        expected_loss_mask,
    )

    teacher_logits = torch.randn(
        cp_local_length // layout.tp_size,
        1,
        vocab_size,
        requires_grad=True,
    )
    stats = provider.loss_stats(
        target_logits=teacher_logits,
        data=data,
        prepare_fn=lambda **_: None,
        vocab_parallel_rank=0,
        vocab_parallel_group=None,
        context_parallel_group=None,
    )
    assert stats.counts.sum() > 0
    assert output.selected_teacher_logits is not None
    assert output.selected_teacher_logits.shape == (
        output.plan.sample_rows.numel(),
        provider.config.gamma,
        vocab_size,
    )
    local_counts = provider.normalization_counts(
        data,
        optimizer_step=9,
        sequence_layout=layout,
    )
    torch.testing.assert_close(local_counts, stats.counts)
    stats.normalized(normalization_counts=stats.counts).backward()
    assert draft.scale.grad is not None
    assert teacher_logits.grad is None
