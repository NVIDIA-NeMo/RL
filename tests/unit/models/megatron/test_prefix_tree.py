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
from megatron.core.packed_seq_params import PackedSeqParams
from torch.nn.attention.flex_attention import flex_attention

from nemo_rl.models.megatron import prefix_tree
from nemo_rl.models.megatron.prefix_tree import (
    _tree_attention,
    _tree_decoder,
    _tree_ssm,
    share_packed_prefixes,
)

ROWS = [[1, 2, 3, 4, 6], [1, 2, 3, 5], [1, 2, 9], [7, 8], [1, 2, 3, 4, 6]]


def _pack(rows, pad_to=2):
    lengths = [len(row) for row in rows]
    padded = [-(-n // pad_to) * pad_to for n in lengths]
    cu = torch.tensor([0, *torch.tensor(lengths).cumsum(0).tolist()], dtype=torch.int32)
    cu_padded = torch.tensor(
        [0, *torch.tensor(padded).cumsum(0).tolist()], dtype=torch.int32
    )
    ids = torch.zeros(1, int(cu_padded[-1]), dtype=torch.long)
    for row, start in zip(rows, cu_padded[:-1].tolist()):
        ids[0, start : start + len(row)] = torch.tensor(row)
    params = PackedSeqParams(
        qkv_format="thd",
        cu_seqlens_q=cu_padded,
        cu_seqlens_kv=cu_padded,
        cu_seqlens_q_padded=cu_padded,
        cu_seqlens_kv_padded=cu_padded,
        max_seqlen_q=max(padded),
        max_seqlen_kv=max(padded),
    )
    return ids, params, cu, cu_padded


def test_rows_share_prefixes_in_dfs_order():
    ids, rows, cu, cu_padded = _pack(ROWS)
    params, _ = share_packed_prefixes(ids, rows, cu, cu_padded, None, 4)
    unique_ids = ids[:, params.owner]

    assert torch.equal(unique_ids[0, params.node_of], ids[0])
    assert torch.equal(params.cu_seqlens_q_padded, rows.cu_seqlens_q_padded)
    # 1 2 3 4 6 | 5 | 9 | 7 8 share into 9 DFS tokens plus 3 isolated padding tokens.
    assert unique_ids[0, :9].tolist() == [1, 2, 3, 4, 6, 5, 9, 7, 8]
    assert unique_ids.shape[1] == 12
    starts = cu_padded[:-1].tolist()
    for a, b, shared in [(0, 1, 3), (0, 2, 2), (0, 4, 5), (0, 3, 0)]:
        node_a = params.node_of[starts[a] : starts[a] + 5]
        node_b = params.node_of[starts[b] : starts[b] + len(ROWS[b])]
        common = min(len(ROWS[a]), len(ROWS[b]))
        assert (node_a[:common] == node_b[:common]).tolist() == [
            i < shared for i in range(common)
        ]


def test_router_replay_routes_block_sharing():
    ids, rows, cu, cu_padded = _pack(ROWS[:2])
    routes = torch.zeros(1, ids.shape[1], 2, 1, dtype=torch.int16)
    routes[0, int(cu_padded[1]) + 1] = 5
    params, shared_routes = share_packed_prefixes(ids, rows, cu, cu_padded, routes, 1)
    first = params.node_of[:4].tolist()
    second = params.node_of[int(cu_padded[1]) : int(cu_padded[1]) + 4].tolist()
    assert first[0] == second[0] and first[1] != second[1]
    assert torch.equal(shared_routes[0, params.node_of], routes[0])


def _toy_ssm(zx, packed_seq_params):
    # A causal recurrence that resets at every packed row, like Mamba's seq_idx.
    out = torch.zeros_like(zx)
    bounds = packed_seq_params.cu_seqlens_q_padded.tolist()
    for start, end in zip(bounds[:-1], bounds[1:]):
        state = torch.zeros_like(zx[0])
        for t in range(start, end):
            state = 0.7 * state + zx[t]
            out[t] = state
    return out


def _toy_model(weights, ids, attention, ssm, params):
    def decoder(x, packed_seq_params=None):
        q = (x @ weights["q"]).view(-1, 4, 8)
        k = (x @ weights["k"]).view(-1, 2, 8)
        v = (x @ weights["v"]).view(-1, 2, 8)
        x = (
            x
            + attention(q, k, v, None, packed_seq_params=packed_seq_params)
            @ weights["o"]
        )
        return x + ssm(x.unsqueeze(1), packed_seq_params).squeeze(1)

    x = weights["embed"][ids[0]]
    return _tree_decoder(decoder, False)(x, packed_seq_params=params) @ weights["head"]


def _row_attention(q, k, v, _, packed_seq_params):
    out = torch.zeros_like(q)
    bounds = packed_seq_params.cu_seqlens_q_padded.tolist()
    for start, end in zip(bounds[:-1], bounds[1:]):
        out[start:end] = torch.nn.functional.scaled_dot_product_attention(
            q[start:end].transpose(0, 1),
            k[start:end].repeat_interleave(2, 1).transpose(0, 1),
            v[start:end].repeat_interleave(2, 1).transpose(0, 1),
            is_causal=True,
        ).transpose(0, 1)
    return out.reshape(q.shape[0], -1)


def _dense_flex(q, k, v, block_mask, enable_gqa):
    # FlexAttention has no CPU backward, so evaluate the same mask densely.
    n = q.shape[2]
    positions = torch.arange(n)
    mask = block_mask.mask_mod(0, 0, positions[:, None], positions[None, :])
    return torch.nn.functional.scaled_dot_product_attention(
        q, k, v, attn_mask=mask, enable_gqa=enable_gqa
    )


def test_flex_attention_matches_dense_tree_mask():
    ids, rows, cu, cu_padded = _pack(ROWS)
    params, _ = share_packed_prefixes(ids, rows, cu, cu_padded, None, 4)
    n = params.owner.numel()
    q, k, v = torch.randn(1, 4, n, 8), torch.randn(1, 2, n, 8), torch.randn(1, 2, n, 8)
    torch.testing.assert_close(
        flex_attention(q, k, v, block_mask=params.block_mask, enable_gqa=True),
        _dense_flex(q, k, v, params.block_mask, True),
    )


def test_tree_forward_and_gradients_match_rows(monkeypatch):
    monkeypatch.setattr(prefix_tree, "_compiled_flex_attention", _dense_flex)
    torch.manual_seed(0)
    weights = {
        "embed": torch.randn(10, 32),
        "q": torch.randn(32, 32) / 6,
        "k": torch.randn(32, 16) / 6,
        "v": torch.randn(32, 16) / 6,
        "o": torch.randn(32, 32) / 6,
        "head": torch.randn(32, 10) / 6,
    }
    for weight in weights.values():
        weight.requires_grad_()
    ids, rows, cu, cu_padded = _pack(ROWS)
    real = torch.zeros(ids.shape[1], dtype=torch.bool)
    for start, length in zip(cu_padded[:-1].tolist(), (cu[1:] - cu[:-1]).tolist()):
        real[start : start + length] = True
    probe = torch.randn(ids.shape[1], 10) * real.unsqueeze(1)

    expected = _toy_model(weights, ids, _row_attention, _toy_ssm, rows)
    expected_grads = torch.autograd.grad(
        (expected * probe).sum(), list(weights.values())
    )

    params, _ = share_packed_prefixes(ids, rows, cu, cu_padded, None, 4)
    logits = _toy_model(
        weights, ids, _tree_attention(None), _tree_ssm(_toy_ssm), params
    )
    grads = torch.autograd.grad((logits * probe).sum(), list(weights.values()))

    torch.testing.assert_close(logits[real], expected[real])
    for grad, expected_grad in zip(grads, expected_grads):
        torch.testing.assert_close(grad, expected_grad)


def test_non_tree_params_use_the_original_kernels():
    calls = []
    forward = _tree_attention(lambda *args, **kwargs: calls.append("attention"))
    ssm = _tree_ssm(lambda zx, params: calls.append("ssm"))
    forward(None, None, None, None, packed_seq_params=None)
    ssm(None, None)
    assert calls == ["attention", "ssm"]
