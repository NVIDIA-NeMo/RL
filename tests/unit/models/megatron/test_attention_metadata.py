"""Packing metadata preserves the boundaries used by the model."""

from contextlib import contextmanager

import pytest
import torch

pytestmark = pytest.mark.mcore


@pytest.fixture
def integration():
    # Defer optional Megatron imports until the mcore tests run.
    from nemo_rl.models.megatron import attention, data

    return attention, data


@pytest.mark.parametrize(
    "lengths,factor,multiple,total",
    [
        ([3, 5], 1, 1, None),
        ([0, 7, 3], 4, 1, None),
        ([3, 5], 4, 16, None),
        ([3, 5], 4, 1, 32),
        ([257, 1025], 128, 1, None),
    ],
)
def test_packer_registers_exact_final_boundaries(
    integration, monkeypatch, lengths, factor, multiple, total
):
    attention, data = integration
    seen = []
    monkeypatch.setattr(
        attention,
        "G_REGISTER_CU_SEQLENS",
        lambda tensor, offsets: seen.append((tensor, tuple(offsets))),
    )
    tokens = torch.arange(len(lengths) * max(lengths)).view(len(lengths), -1)
    result = data._pack_sequences_for_megatron(
        tokens,
        torch.tensor(lengths),
        pad_individual_seqs_to_multiple_of=factor,
        pad_packed_seq_to_multiple_of=multiple,
        pad_packed_seq_to=total,
    )
    assert len(seen) == 2
    for tensor, offsets in seen:
        assert tuple(tensor.tolist()) == offsets
    assert any(tensor is result[3] for tensor, _ in seen)
    assert any(tensor is result[4] for tensor, _ in seen)
    # Registration must not change packed tokens, masks, or model metadata.
    monkeypatch.setattr(attention, "G_REGISTER_CU_SEQLENS", None)
    stock = data._pack_sequences_for_megatron(
        tokens,
        torch.tensor(lengths),
        pad_individual_seqs_to_multiple_of=factor,
        pad_packed_seq_to_multiple_of=multiple,
        pad_packed_seq_to=total,
    )
    for index in (0, 1, 3, 4):
        torch.testing.assert_close(result[index], stock[index])


@pytest.mark.parametrize("total", [None, 32])
def test_vlm_registers_actual_padded_prefix(integration, monkeypatch, total):
    attention, data = integration
    seen = []
    monkeypatch.setattr(
        attention,
        "G_REGISTER_CU_SEQLENS",
        lambda tensor, offsets: seen.append((tensor, tuple(offsets))),
    )
    result = data._prepare_vlm_batch_for_megatron(
        torch.arange(16).view(2, 8),
        torch.tensor([3, 5]),
        pad_individual_seqs_to_multiple_of=4,
        pad_full_seq_to=total,
    )
    assert len(seen) == 1
    tensor, offsets = seen[0]
    assert tensor is result[5]
    assert tuple(tensor.tolist()) == offsets
    assert offsets[-1] == result[0].shape[1]
    assert offsets[-1] == (12 if total is None else total)


def test_older_te_needs_no_metadata_api(integration, monkeypatch):
    attention, _ = integration
    monkeypatch.setattr(attention, "G_REGISTER_CU_SEQLENS", None)
    monkeypatch.setattr(attention, "G_ATTENTION_WORKSPACE", None)
    attention.register_cu_seqlens(torch.tensor([0, 3], dtype=torch.int32), [0, 3])
    with attention.attention_backend_workspace():
        pass


def test_workspace_scope_releases_on_failure(integration, monkeypatch):
    attention, _ = integration
    events = []

    @contextmanager
    def scope():
        events.append("enter")
        try:
            yield
        finally:
            events.append("exit")

    monkeypatch.setattr(attention, "G_ATTENTION_WORKSPACE", scope)
    with pytest.raises(ValueError):
        with attention.attention_backend_workspace():
            raise ValueError("training failed")
    assert events == ["enter", "exit"]


def test_prepacked_cpu_boundary_registers_final_tensor(integration, monkeypatch):
    attention, data = integration
    seen = []
    monkeypatch.setattr(
        attention,
        "G_REGISTER_CU_SEQLENS",
        lambda tensor, offsets: seen.append((tensor, tuple(offsets))),
    )
    result = data._prepacked_boundary(
        {"cu_seqlens": torch.tensor([[0, 3, 8]])}, "cu_seqlens", torch.device("cpu")
    )
    assert result.dtype == torch.int32
    assert len(seen) == 1 and seen[0][0] is result and seen[0][1] == (0, 3, 8)
