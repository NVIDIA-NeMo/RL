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

import torch
from torch import nn

from nemo_rl.utils.refit_debug import (
    RefitDebugStats,
    debug_refit_tensors,
    log_refit_destinations,
    refit_debug_enabled,
    refit_debug_rank,
    select_refit_debug_names,
    tensor_fingerprint,
)


def test_refit_debug_enabled_accepts_only_explicit_truthy_values(monkeypatch):
    monkeypatch.delenv("NRL_REFIT_DEBUG", raising=False)
    assert not refit_debug_enabled()

    for value in ("1", "true", "YES", "on"):
        monkeypatch.setenv("NRL_REFIT_DEBUG", value)
        assert refit_debug_enabled()

    monkeypatch.setenv("NRL_REFIT_DEBUG", "0")
    assert not refit_debug_enabled()


def test_select_refit_debug_names_is_bounded_and_deterministic():
    names = [
        "model.layers.4.mlp.gate.weight",
        "model.layers.0.mlp.gate.weight",
        "model.layers.0.mlp.e_score_correction_bias",
        "model.layers.0.mlp.experts.0.w1.weight",
        "model.layers.0.self_attn.q_proj.weight",
        "model.embed_tokens.weight",
        "lm_head.weight",
    ]

    selected = select_refit_debug_names(reversed(names))

    assert selected == {
        "embedding": "model.embed_tokens.weight",
        "lm_head": "lm_head.weight",
        "attention": "model.layers.0.self_attn.q_proj.weight",
        "router_gate": "model.layers.0.mlp.gate.weight",
        "routing_bias": "model.layers.0.mlp.e_score_correction_bias",
        "expert": "model.layers.0.mlp.experts.0.w1.weight",
    }


def test_tensor_fingerprint_is_stable_bounded_and_non_mutating():
    tensor = torch.arange(64, dtype=torch.bfloat16).reshape(8, 8)
    original = tensor.clone()

    first = tensor_fingerprint(tensor)
    second = tensor_fingerprint(tensor)

    assert first == second
    assert "samples=16" in first
    assert "digest=" in first
    assert "finite=16" in first
    assert torch.equal(tensor, original)


def test_tensor_fingerprint_handles_empty_tensor():
    fingerprint = tensor_fingerprint(torch.empty(0, dtype=torch.float32))
    assert fingerprint == "samples=0 digest=empty finite=0 nonfinite=0"


def test_refit_debug_stats_summarizes_dtype_counts_bytes_and_loaded_names():
    stats = RefitDebugStats()
    stats.observe_metadata("fp32", torch.Size([2, 3]), torch.float32)
    stats.observe_tensor("bf16", torch.ones(4, dtype=torch.bfloat16))
    stats.observe_loaded({"mapped.gate", "mapped.qkv"})

    assert stats.parameter_count == 2
    assert stats.total_bytes == 32
    assert stats.loaded_count == 2
    summary = stats.format()
    assert "parameters=2" in summary
    assert "total_bytes=32" in summary
    assert "torch.bfloat16:count=1,bytes=8" in summary
    assert "torch.float32:count=1,bytes=24" in summary
    assert "loaded=2" in summary


def test_debug_refit_tensors_preserves_identity_and_logs_selected(
    monkeypatch, capsys
):
    monkeypatch.setenv("NRL_REFIT_DEBUG", "1")
    selected = torch.arange(8, dtype=torch.float32)
    unselected = torch.ones(2, dtype=torch.bfloat16)
    stats = RefitDebugStats()

    output = list(
        debug_refit_tensors(
            iter(
                [
                    ("model.layers.0.mlp.gate.weight", selected),
                    ("model.layers.1.input_layernorm.weight", unselected),
                ]
            ),
            phase="policy_payload_ipc",
            selected_names={
                "router_gate": "model.layers.0.mlp.gate.weight",
            },
            rank="policy:0",
            stats=stats,
        )
    )

    assert output[0][1] is selected
    assert output[1][1] is unselected
    assert stats.parameter_count == 2
    captured = capsys.readouterr().out
    assert "[REFIT_DEBUG] phase=policy_payload_ipc rank=policy:0" in captured
    assert "category=router_gate" in captured
    assert "dtype=torch.float32" in captured
    assert "input_layernorm" not in captured


def test_debug_refit_tensors_is_silent_when_disabled(monkeypatch, capsys):
    monkeypatch.delenv("NRL_REFIT_DEBUG", raising=False)
    tensor = torch.ones(1)

    output = list(
        debug_refit_tensors(
            iter([("model.embed_tokens.weight", tensor)]),
            phase="policy_payload_ipc",
            selected_names={"embedding": "model.embed_tokens.weight"},
            rank="policy:0",
            stats=RefitDebugStats(),
        )
    )

    assert output[0][1] is tensor
    assert capsys.readouterr().out == ""


class _TinyDestinationModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Module()
        self.model.embed_tokens = nn.Embedding(4, 2)


def test_log_refit_destinations_distinguishes_exact_and_mapped_names(
    monkeypatch, capsys
):
    monkeypatch.setenv("NRL_REFIT_DEBUG", "1")
    model = _TinyDestinationModel()

    log_refit_destinations(
        model,
        {
            "embedding": "model.embed_tokens.weight",
            "attention": "model.layers.0.self_attn.q_proj.weight",
        },
        rank="vllm:0",
    )

    captured = capsys.readouterr().out
    assert "phase=vllm_destination rank=vllm:0 category=embedding" in captured
    assert "status=exact" in captured
    assert "digest=" in captured
    assert "category=attention" in captured
    assert "status=mapped_or_unresolved" in captured


def test_refit_debug_rank_falls_back_to_environment(monkeypatch):
    monkeypatch.setenv("RANK", "17")
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: False)
    assert refit_debug_rank() == "17"
