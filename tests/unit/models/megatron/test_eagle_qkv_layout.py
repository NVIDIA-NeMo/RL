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

"""Eagle draft QKV weight layout tests.

Megatron's ``SelfAttention.get_query_key_value_tensors`` interprets
``linear_qkv.weight`` as an *interleaved* per-query-group layout
(``[g0: q.. k v | g1: q.. k v | ...]``). Loading an HF Eagle3 checkpoint
(separate head-major ``q_proj``/``k_proj``/``v_proj``) must reorder into this
layout, and export must reverse it. A naive ``cat([q, k, v])`` is silently
mis-read for GQA models (``num_query_groups < num_attention_heads``), so the
trainer-side draft attention diverges from the HF/vLLM-side attention: draft
training loss decreases while speculative-decoding acceptance degrades.
"""

from types import SimpleNamespace

import pytest
import torch

from nemo_rl.models.megatron.draft.utils import (
    _deinterleave_qkv,
    _EagleModelLayout,
    _interleave_qkv,
    _map_hf_state_to_eagle_state,
)


def _cfg(n_heads, n_kv, head_dim):
    return SimpleNamespace(
        num_attention_heads=n_heads,
        num_query_groups=n_kv,
        kv_channels=head_dim,
        hidden_size=n_heads * head_dim,
    )


def _megatron_split_reference(fused, n_heads, n_kv, head_dim):
    """Independent re-implementation of the q/k/v split Megatron's SelfAttention
    performs on ``linear_qkv`` (megatron/core/transformer/attention.py): view the
    row dim as ``[num_query_groups, (heads_per_group + 2) * head_dim]`` and slice
    q/k/v within each group. Deliberately not expressed via the production helpers.
    """
    r = n_heads // n_kv
    in_dim = fused.shape[1]
    g = fused.reshape(n_kv, (r + 2) * head_dim, in_dim)
    q = g[:, : r * head_dim, :].reshape(n_heads * head_dim, in_dim)
    k = g[:, r * head_dim : (r + 1) * head_dim, :].reshape(n_kv * head_dim, in_dim)
    v = g[:, (r + 1) * head_dim :, :].reshape(n_kv * head_dim, in_dim)
    return q, k, v


def _head_identifiable(n, head_dim, in_dim, base):
    """[n*head_dim, in_dim] where every row of head h holds the value base+h."""
    w = torch.empty(n * head_dim, in_dim, dtype=torch.float32)
    for h in range(n):
        w[h * head_dim : (h + 1) * head_dim] = float(base + h)
    return w


# GQA (the bug-triggering case) and MHA (must keep working).
_CONFIGS = [
    pytest.param(4, 2, 8, 16, id="gqa-4h-2kv"),
    pytest.param(16, 8, 8, 16, id="gqa-16h-8kv"),
    pytest.param(4, 4, 8, 16, id="mha-4h-4kv"),
]


@pytest.mark.mcore
@pytest.mark.parametrize("n_heads,n_kv,head_dim,in_dim", _CONFIGS)
def test_interleave_is_read_correctly_by_megatron_split(
    n_heads, n_kv, head_dim, in_dim
):
    """Interleaved layout must round-trip through Megatron's split contract."""
    cfg = _cfg(n_heads, n_kv, head_dim)
    q = _head_identifiable(n_heads, head_dim, in_dim, base=100.0)
    k = _head_identifiable(n_kv, head_dim, in_dim, base=200.0)
    v = _head_identifiable(n_kv, head_dim, in_dim, base=300.0)

    fused = _interleave_qkv(q, k, v, cfg)
    assert fused.shape == (n_heads * head_dim + 2 * n_kv * head_dim, in_dim)

    rq, rk, rv = _megatron_split_reference(fused, n_heads, n_kv, head_dim)
    assert torch.equal(rq, q), "Megatron would not recover q_proj from loaded qkv"
    assert torch.equal(rk, k), "Megatron would not recover k_proj from loaded qkv"
    assert torch.equal(rv, v), "Megatron would not recover v_proj from loaded qkv"


@pytest.mark.mcore
def test_naive_cat_is_misread_for_gqa():
    """Guards the test's relevance: the old naive cat is wrong for GQA."""
    n_heads, n_kv, head_dim, in_dim = 4, 2, 8, 16
    q = _head_identifiable(n_heads, head_dim, in_dim, base=100.0)
    k = _head_identifiable(n_kv, head_dim, in_dim, base=200.0)
    v = _head_identifiable(n_kv, head_dim, in_dim, base=300.0)

    naive = torch.cat([q, k, v], dim=0)
    rq, rk, rv = _megatron_split_reference(naive, n_heads, n_kv, head_dim)
    # For GQA, Megatron grabs query rows where it expects key/value rows.
    assert not (torch.equal(rq, q) and torch.equal(rk, k) and torch.equal(rv, v))


@pytest.mark.mcore
@pytest.mark.parametrize("n_heads,n_kv,head_dim,in_dim", _CONFIGS)
def test_interleave_deinterleave_roundtrip(n_heads, n_kv, head_dim, in_dim):
    cfg = _cfg(n_heads, n_kv, head_dim)
    q = torch.randn(n_heads * head_dim, in_dim)
    k = torch.randn(n_kv * head_dim, in_dim)
    v = torch.randn(n_kv * head_dim, in_dim)
    rq, rk, rv = _deinterleave_qkv(_interleave_qkv(q, k, v, cfg), cfg)
    assert torch.equal(rq, q)
    assert torch.equal(rk, k)
    assert torch.equal(rv, v)


@pytest.mark.mcore
def test_load_maps_hf_qkv_to_megatron_interleaved_layout():
    """End-to-end load: _map_hf_state_to_eagle_state must store linear_qkv in the
    interleaved layout Megatron reads, for a GQA Eagle3 checkpoint."""
    n_heads, n_kv, head_dim, in_dim = 16, 8, 8, 16  # GQA, eagle3-like
    cfg = _cfg(n_heads, n_kv, head_dim)
    q_dim, kv_dim = n_heads * head_dim, n_kv * head_dim

    qkv_key = "eagle_module.decoder.layers.0.self_attention.linear_qkv.weight"
    model_state = {qkv_key: torch.zeros(q_dim + 2 * kv_dim, in_dim)}  # single TP rank
    layout = _EagleModelLayout.detect(model_state)

    q = _head_identifiable(n_heads, head_dim, in_dim, base=1000.0)
    k = _head_identifiable(n_kv, head_dim, in_dim, base=2000.0)
    v = _head_identifiable(n_kv, head_dim, in_dim, base=3000.0)
    hf_state = {
        "midlayer.self_attn.q_proj.weight": q,
        "midlayer.self_attn.k_proj.weight": k,
        "midlayer.self_attn.v_proj.weight": v,
    }

    mapped = _map_hf_state_to_eagle_state(
        hf_state_dict=hf_state,
        model_state=model_state,
        layout=layout,
        checkpoint_source="unit-test",
        config=cfg,
    )

    loaded_qkv = mapped[qkv_key]
    assert loaded_qkv.shape == (q_dim + 2 * kv_dim, in_dim)
    rq, rk, rv = _megatron_split_reference(loaded_qkv, n_heads, n_kv, head_dim)
    assert torch.equal(rq, q), "loaded qkv mis-routes query heads for GQA"
    assert torch.equal(rk, k), "loaded qkv mis-routes key heads for GQA"
    assert torch.equal(rv, v), "loaded qkv mis-routes value heads for GQA"
