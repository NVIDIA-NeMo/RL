"""Regression tests for storage-contract-based LoRA diagnostic discovery."""

from __future__ import annotations

import torch
import torch.nn as nn

from nemo_rl.models.multi_lora.adapter import MultiLinearLoRA
from nemo_rl.models.multi_lora.diag import iter_lora_tensors


class PatchedLinearLoRA(nn.Linear):
    """Runtime class name differs from LinearLoRA but storage is identical."""

    def __init__(self):
        super().__init__(5, 7, bias=False)
        self.lora_A = nn.Linear(5, 3, bias=False)
        self.lora_B = nn.Linear(3, 7, bias=False)


def test_iter_lora_tensors_detects_patched_single_and_gradients(monkeypatch):
    monkeypatch.setenv("NOUSNET_DIAG_WHO", "single_a")
    model = nn.Sequential(PatchedLinearLoRA())
    model[0].lora_A.weight.grad = torch.ones_like(model[0].lora_A.weight)
    model[0].lora_B.weight.grad = torch.full_like(model[0].lora_B.weight, 2)

    rows = list(iter_lora_tensors(model))
    got = {(who, path, name): tensor for who, path, name, tensor in rows}

    assert set(got) == {
        ("single_a", "0", "lora_A"),
        ("single_a", "0", "lora_B"),
        ("single_a", "0", "lora_A_grad"),
        ("single_a", "0", "lora_B_grad"),
    }
    assert got[("single_a", "0", "lora_A")] is model[0].lora_A.weight
    assert got[("single_a", "0", "lora_B_grad")] is model[0].lora_B.weight.grad


def test_iter_lora_tensors_detects_multi_by_stacked_shape():
    layer = MultiLinearLoRA(nn.Linear(5, 7, bias=False), 2, dim=3, alpha=6)
    model = nn.Sequential(layer)
    layer.lora_A.grad = torch.ones_like(layer.lora_A)
    layer.lora_B.grad = torch.ones_like(layer.lora_B)

    rows = list(iter_lora_tensors(model, ["adapter_a", "adapter_b"]))
    got = {(who, path, name): tensor for who, path, name, tensor in rows}

    assert len(got) == 8
    assert got[("multi_adapter_a", "0", "lora_A")].shape == (3, 5)
    assert got[("multi_adapter_b", "0", "lora_B_grad")].shape == (7, 3)
