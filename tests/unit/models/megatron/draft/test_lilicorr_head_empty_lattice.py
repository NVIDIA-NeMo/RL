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
"""A rank with no anchors must still run the LiLiCorr head.

Under context parallelism a rank can own zero draft anchors in a microbatch, so the
head sees an empty lattice. It has to return empty scores and still give every
parameter a (zero) gradient, because the draft gradient reduction expects every rank
to contribute.
"""

import torch

from nemo_rl.models.megatron.draft.lilicorr import lilicorr_lattice_nll
from nemo_rl.models.megatron.draft.lilicorr_head import LiLiCorrHead

MODEL_HIDDEN = 64
BLOCK_SIZE = 4
TOPK = 3
VOCAB = 50


def _head() -> LiLiCorrHead:
    torch.manual_seed(0)
    return LiLiCorrHead(
        model_hidden_size=MODEL_HIDDEN,
        hidden_size=32,
        num_layers=2,
        num_heads=4,
        mlp_ratio=2.0,
        block_size=BLOCK_SIZE,
        candidate_topk=TOPK,
        factor_dim=16,
        rms_norm_eps=1e-6,
        vector_eps=1e-4,
        logit_scale=10.0,
    )


def _inputs(n_blocks: int):
    slots = BLOCK_SIZE - 1
    return dict(
        candidate_token_ids=torch.randint(0, VOCAB, (1, n_blocks, slots, TOPK)),
        candidate_log_probs=torch.randn(1, n_blocks, slots, TOPK).log_softmax(-1),
        pass_hidden=torch.randn(1, n_blocks, slots, MODEL_HIDDEN),
        anchor_hidden=torch.randn(1, n_blocks, 1, MODEL_HIDDEN),
        anchor_valid=torch.ones(1, n_blocks, 1, dtype=torch.bool),
    )


def test_empty_lattice_returns_empty_scores_and_zero_grads():
    head = _head()
    embed = torch.nn.Embedding(VOCAB, MODEL_HIDDEN)

    start, pair = head(embed_tokens=embed, **_inputs(0))

    assert start.shape == (1, 0, TOPK)
    assert pair.shape == (1, 0, BLOCK_SIZE - 2, TOPK, TOPK)
    gold = torch.zeros(1, 0, BLOCK_SIZE - 1, dtype=torch.long)
    loss = lilicorr_lattice_nll(head, start, pair, gold)
    assert torch.isfinite(loss) and loss.item() == 0.0
    loss.backward()
    missing = [n for n, p in head.named_parameters() if p.grad is None]
    assert not missing, f"parameters without a gradient on an empty lattice: {missing}"
    assert all(torch.count_nonzero(p.grad) == 0 for p in head.parameters())


def test_non_empty_lattice_is_unchanged_by_the_empty_guard():
    head = _head()
    embed = torch.nn.Embedding(VOCAB, MODEL_HIDDEN)
    inputs = _inputs(3)

    start, pair = head(embed_tokens=embed, **inputs)

    # Reference: the same layers driven straight through nn.MultiheadAttention.
    layer = head.layers[0]
    x = torch.randn(3, (BLOCK_SIZE - 1) * TOPK, 32)
    bias = head._attention_bias(3, device=x.device, dtype=x.dtype, topk=TOPK)
    normed = layer.attn_norm(x)
    ref_attn, _ = layer.attn(normed, normed, normed, attn_mask=bias, need_weights=False)
    ref = x + ref_attn
    ref = ref + layer.mlp(layer.mlp_norm(ref))
    torch.testing.assert_close(layer(x, attention_bias=bias), ref)
    assert start.shape == (1, 3, TOPK) and torch.isfinite(start).all()
    assert torch.isfinite(pair).all()
