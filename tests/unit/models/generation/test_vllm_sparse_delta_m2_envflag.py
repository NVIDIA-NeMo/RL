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

"""Unit tests for M2 NRL_REFIT_SPARSE_APPLY_MODE env-flag dispatch.

Tests cover:
  - _is_plain_linear_name static method (no instance needed)
  - _additive_apply_mode dispatch for all three modes: plan / additive / allowlist
  - VllmSparseDeltaApplier.__init__ reads the env var correctly
  - apply_sparse_delta_via_additive_load is called (spy) vs skipped based on mode

The applier's full _apply_sparse_weight_deltas path requires a live vLLM
model_runner; that is deferred to the E2E integration test in Round 5.
Here we only test the dispatch decision layer, which is pure Python.
"""

import types
from unittest.mock import MagicMock, patch

import pytest
import torch

from nemo_rl.models.generation.vllm.vllm_sparse_delta import VllmSparseDeltaApplier


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_applier(
    monkeypatch: pytest.MonkeyPatch,
    *,
    mode: str = "plan",
    allowlist: str = "",
) -> VllmSparseDeltaApplier:
    """Construct an applier with env vars set and a no-op model_runner stub."""
    monkeypatch.setenv("NRL_REFIT_SPARSE_APPLY_MODE", mode)
    if allowlist:
        monkeypatch.setenv("NRL_REFIT_SPARSE_APPLY_ALLOWLIST", allowlist)
    else:
        monkeypatch.delenv("NRL_REFIT_SPARSE_APPLY_ALLOWLIST", raising=False)

    stub_runner = types.SimpleNamespace()  # __init__ does not call into it
    return VllmSparseDeltaApplier(
        stub_runner,
        torch.device("cuda", 0),
        rank=0,
    )


# ---------------------------------------------------------------------------
# _is_plain_linear_name (static — no instance needed)
# ---------------------------------------------------------------------------


@pytest.mark.vllm
@pytest.mark.parametrize(
    "name,expected",
    [
        # plain-linear (should return True)
        ("model.layers.0.self_attn.o_proj.weight", True),
        ("model.layers.0.mlp.down_proj.weight", True),
        ("model.embed_tokens.weight", True),
        ("lm_head.weight", True),
        # QKV (False)
        ("model.layers.0.self_attn.q_proj.weight", False),
        ("model.layers.0.self_attn.k_proj.weight", False),
        ("model.layers.0.self_attn.v_proj.weight", False),
        # gate / up (False)
        ("model.layers.0.mlp.gate_proj.weight", False),
        ("model.layers.0.mlp.up_proj.weight", False),
        # expert (False)
        (
            "model.layers.0.mlp.experts.0.gate_proj.weight",
            False,
        ),
        (
            "model.layers.0.mlp.experts.3.down_proj.weight",
            False,
        ),
        # mamba (False)
        ("model.layers.0.mixer.in_proj.weight", False),
        ("model.layers.0.mixer.conv1d.weight", False),
    ],
)
def test_is_plain_linear_name(name: str, expected: bool) -> None:
    assert VllmSparseDeltaApplier._is_plain_linear_name(name) is expected


# ---------------------------------------------------------------------------
# _additive_apply_mode dispatch
# ---------------------------------------------------------------------------


@pytest.mark.vllm
def test_additive_apply_mode_plan_always_false(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """mode=plan: additive path never selected regardless of name."""
    applier = _make_applier(monkeypatch, mode="plan")
    assert applier._sparse_apply_mode == "plan"
    # plain-linear name
    assert applier._additive_apply_mode("model.layers.0.mlp.down_proj.weight") is False
    # QKV
    assert applier._additive_apply_mode("model.layers.0.self_attn.q_proj.weight") is False


@pytest.mark.vllm
def test_additive_apply_mode_additive_plain_linear(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """mode=additive: plain-linear → True, QKV → False."""
    applier = _make_applier(monkeypatch, mode="additive")
    assert applier._additive_apply_mode("model.layers.0.mlp.down_proj.weight") is True
    assert applier._additive_apply_mode("model.layers.0.self_attn.q_proj.weight") is False
    assert applier._additive_apply_mode("model.layers.0.mlp.gate_proj.weight") is False


@pytest.mark.vllm
def test_additive_apply_mode_allowlist(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """mode=allowlist: regex match selects additive; non-match uses plan."""
    applier = _make_applier(
        monkeypatch, mode="allowlist", allowlist=r"\.down_proj\."
    )
    assert applier._additive_apply_mode("model.layers.0.mlp.down_proj.weight") is True
    # q_proj doesn't match the pattern → plan path
    assert applier._additive_apply_mode("model.layers.0.self_attn.q_proj.weight") is False


@pytest.mark.vllm
def test_invalid_mode_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    """Invalid NRL_REFIT_SPARSE_APPLY_MODE should raise ValueError at init."""
    monkeypatch.setenv("NRL_REFIT_SPARSE_APPLY_MODE", "bogus")
    monkeypatch.delenv("NRL_REFIT_SPARSE_APPLY_ALLOWLIST", raising=False)
    stub_runner = types.SimpleNamespace()
    with pytest.raises(ValueError, match="NRL_REFIT_SPARSE_APPLY_MODE"):
        VllmSparseDeltaApplier(stub_runner, torch.device("cuda", 0), rank=0)


# ---------------------------------------------------------------------------
# Spy test: apply_sparse_delta_via_additive_load called vs. skipped
# ---------------------------------------------------------------------------


@pytest.mark.vllm
def test_additive_path_selected_for_plain_linear_mode(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """mode=additive: plain-linear → additive selected; QKV → plan selected.

    _apply_sparse_weight_deltas calls apply_sparse_delta_via_additive_load
    via a lazy import inside the function body (not at module scope), so
    patching the module-level name is not possible without importing first.
    Instead we verify the dispatch decision directly through _additive_apply_mode,
    which is what _apply_sparse_weight_deltas gates on before calling the function.
    The actual call is proven separately by test_apply_sparse_delta_via_additive_load_end_to_end
    in test_vllm_sparse_delta_additive.py.
    """
    applier = _make_applier(monkeypatch, mode="additive")

    plain_linear_names = [
        "model.layers.0.mlp.down_proj.weight",
        "model.embed_tokens.weight",
        "lm_head.weight",
        "model.layers.0.self_attn.o_proj.weight",
    ]
    non_plain_names = [
        "model.layers.0.self_attn.q_proj.weight",
        "model.layers.0.self_attn.k_proj.weight",
        "model.layers.0.self_attn.v_proj.weight",
        "model.layers.0.mlp.gate_proj.weight",
        "model.layers.0.mlp.up_proj.weight",
        "model.layers.0.mlp.experts.0.down_proj.weight",
        "model.layers.0.mixer.in_proj.weight",
    ]

    for name in plain_linear_names:
        assert applier._additive_apply_mode(name) is True, (
            f"Expected additive=True for plain-linear name {name!r}"
        )
    for name in non_plain_names:
        assert applier._additive_apply_mode(name) is False, (
            f"Expected additive=False for non-plain-linear name {name!r}"
        )


@pytest.mark.vllm
def test_additive_path_skipped_for_qkv(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """mode=additive + QKV name → additive path NOT selected (falls back to plan)."""
    applier = _make_applier(monkeypatch, mode="additive")
    qkv_name = "model.layers.0.self_attn.q_proj.weight"
    assert applier._additive_apply_mode(qkv_name) is False
