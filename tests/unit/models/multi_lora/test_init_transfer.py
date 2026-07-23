"""Unit tests for diagnostic-only exact initial-LoRA transfer."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
import torch.nn as nn

from nemo_rl.models.multi_lora.adapter import MultiLinearLoRA
from nemo_rl.models.multi_lora.init_transfer import (
    export_initial_lora,
    import_initial_lora,
)


class PatchedLinearLoRA(nn.Linear):
    """Minimal storage-compatible stand-in for Automodel's runtime class."""

    def __init__(self, in_features: int, out_features: int, rank: int = 3):
        super().__init__(in_features, out_features, bias=False)
        self.weight.requires_grad_(False)
        self.lora_A = nn.Linear(in_features, rank, bias=False)
        self.lora_B = nn.Linear(rank, out_features, bias=False)


class ToySingle(nn.Module):
    def __init__(self):
        super().__init__()
        self.first = PatchedLinearLoRA(5, 7)
        self.second = PatchedLinearLoRA(7, 4)


class ToyMulti(nn.Module):
    def __init__(self):
        super().__init__()
        self.first = MultiLinearLoRA(nn.Linear(5, 7, bias=False), 4, dim=3, alpha=6)
        self.second = MultiLinearLoRA(nn.Linear(7, 4, bias=False), 4, dim=3, alpha=6)


def _single_tensors(model: ToySingle):
    return [
        (model.first.lora_A.weight, model.first.lora_B.weight),
        (model.second.lora_A.weight, model.second.lora_B.weight),
    ]


def _multi_slot_tensors(model: ToyMulti, slot: int):
    return [
        (model.first.lora_A[slot], model.first.lora_B[slot]),
        (model.second.lora_A[slot], model.second.lora_B[slot]),
    ]


@pytest.mark.parametrize("slot", [0, 1, 2, 3])
def test_export_multi_import_single_exact(tmp_path: Path, slot: int):
    torch.manual_seed(123)
    multi = ToyMulti()
    export_initial_lora(multi, tmp_path)

    torch.manual_seed(999)
    single = ToySingle()
    import_initial_lora(single, tmp_path, single_slot=slot)

    for (sa, sb), (ma, mb) in zip(_single_tensors(single), _multi_slot_tensors(multi, slot)):
        assert torch.equal(sa, ma)
        assert torch.equal(sb, mb)


def test_import_all_slots_restores_multi_exact(tmp_path: Path):
    torch.manual_seed(123)
    source = ToyMulti()
    export_initial_lora(source, tmp_path)

    torch.manual_seed(456)
    dest = ToyMulti()
    import_initial_lora(dest, tmp_path, single_slot=None)

    for slot in range(4):
        for (da, db), (sa, sb) in zip(_multi_slot_tensors(dest, slot), _multi_slot_tensors(source, slot)):
            assert torch.equal(da, sa)
            assert torch.equal(db, sb)


def test_rejects_wrong_single_slot_contract(tmp_path: Path):
    multi = ToyMulti()
    export_initial_lora(multi, tmp_path)
    with pytest.raises(RuntimeError, match="required"):
        import_initial_lora(ToySingle(), tmp_path, single_slot=None)


def test_rejects_module_shape_mismatch(tmp_path: Path):
    multi = ToyMulti()
    export_initial_lora(multi, tmp_path)

    class WrongSingle(nn.Module):
        def __init__(self):
            super().__init__()
            self.first = PatchedLinearLoRA(5, 8)
            self.second = PatchedLinearLoRA(7, 4)

    with pytest.raises(RuntimeError, match="out_features mismatch|shape mismatch"):
        import_initial_lora(WrongSingle(), tmp_path, single_slot=0)
