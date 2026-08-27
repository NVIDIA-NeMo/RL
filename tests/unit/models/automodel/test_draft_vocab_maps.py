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
"""Reduced-vocab d2t/t2d convention round-trip.

The shared convention across the dspark/dflash block drafters and the eagle3
TTT drafter (matching vLLM's ``arange(draft_vocab) + d2t`` scatter): ``d2t``
stores OFFSETS with ``target_id = draft_idx + d2t[draft_idx]``, and ``t2d``
is a target-vocab boolean membership mask. The derived target->draft inverse
must agree with ``t2d`` exactly.
"""

import torch

TARGET_VOCAB = 100
DRAFT_VOCAB = 30


def _make_vocab_maps(seed: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """A random reduced vocab: sorted unique target ids, offset d2t, bool t2d."""
    g = torch.Generator().manual_seed(seed)
    target_ids = torch.sort(
        torch.randperm(TARGET_VOCAB, generator=g)[:DRAFT_VOCAB]
    ).values
    d2t = target_ids - torch.arange(DRAFT_VOCAB)
    t2d = torch.zeros(TARGET_VOCAB, dtype=torch.bool)
    t2d[target_ids] = True
    return target_ids, d2t, t2d


def _tiny_dspark_model(d2t: torch.Tensor, t2d: torch.Tensor):
    from transformers.models.qwen3.configuration_qwen3 import Qwen3Config

    from nemo_rl.models.automodel.draft.draft_qwen3 import Qwen3DSparkModel

    config = Qwen3Config(
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=16,
        vocab_size=TARGET_VOCAB,
        max_position_embeddings=256,
        rms_norm_eps=1e-6,
        attention_bias=False,
    )
    config._attn_implementation = "sdpa"
    config.block_size = 4
    config.sample_from_anchor = True
    config.mask_token_id = TARGET_VOCAB - 1
    config.target_layer_ids = [-1, 0]
    config.num_anchors = 4
    config.markov_rank = 0
    config.enable_confidence_head = False
    config.draft_vocab_size = DRAFT_VOCAB
    model = Qwen3DSparkModel(config)
    with torch.no_grad():
        model.d2t.copy_(d2t)
        model.t2d.copy_(t2d)
    return model


def _tiny_eagle3_model(d2t: torch.Tensor, t2d: torch.Tensor):
    from transformers.models.llama.configuration_llama import LlamaConfig

    from nemo_rl.models.automodel.draft.eagle3_llama import Eagle3DraftModel

    config = LlamaConfig(
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=16,
        vocab_size=TARGET_VOCAB,
        max_position_embeddings=256,
        rms_norm_eps=1e-6,
        attention_bias=False,
    )
    config.attn_implementation = "eager"
    config.draft_vocab_size = DRAFT_VOCAB
    config.norm_before_residual = True
    config.target_layer_ids = [0, 0, 0]
    config.num_aux_hidden_states = 3
    model = Eagle3DraftModel(config)
    with torch.no_grad():
        model.d2t.copy_(d2t)
        model.t2d.copy_(t2d)
    return model


def test_dspark_d2t_offsets_round_trip_with_t2d():
    target_ids, d2t, t2d = _make_vocab_maps(seed=0)
    model = _tiny_dspark_model(d2t, t2d)

    # Offset semantics: target_id = draft_idx + d2t[draft_idx].
    resolved = model._get_d2t_target_ids()
    assert torch.equal(resolved, target_ids)

    # Derived target->draft inverse: -1 exactly where t2d is False, and the
    # composition inverse[target_ids] is the identity over draft indices.
    inverse = model._get_t2d_index()
    assert torch.equal(inverse >= 0, t2d)
    assert torch.equal(inverse[target_ids], torch.arange(DRAFT_VOCAB))
    assert torch.equal(resolved[inverse[target_ids]], target_ids)


def test_eagle3_d2t_offsets_match_dspark_convention():
    target_ids, d2t, t2d = _make_vocab_maps(seed=1)
    model = _tiny_eagle3_model(d2t, t2d)

    resolved = model.get_d2t_target_ids()
    assert torch.equal(resolved, target_ids)
    # Membership agreement with t2d.
    membership = torch.zeros(TARGET_VOCAB, dtype=torch.bool)
    membership[resolved] = True
    assert torch.equal(membership, t2d)


def test_reduced_vocab_rejects_non_vanilla_markov_heads():
    """Only the vanilla head has a validated split-vocab implementation:
    gated/rnn heads embed and project over one vocabulary, so a reduced
    draft vocab would crash at the first forward — reject at build time."""
    import pytest

    from nemo_rl.models.automodel.draft.markov_head import build_markov_head

    class _Cfg:
        vocab_size = 100
        draft_vocab_size = 32
        markov_rank = 4
        markov_head_type = "gated"
        hidden_size = 8

    with pytest.raises(ValueError, match="reduced"):
        build_markov_head(_Cfg())
    _Cfg.markov_head_type = "rnn"
    with pytest.raises(ValueError, match="reduced"):
        build_markov_head(_Cfg())

    # The vanilla head keeps the split: full-vocab prev-token embedding,
    # draft-vocab bias output.
    _Cfg.markov_head_type = "vanilla"
    head = build_markov_head(_Cfg())
    assert head.markov_w1.num_embeddings == 100
    assert head.markov_w2.out_features == 32

    # Full-vocab checkpoints keep working with every head type.
    _Cfg.draft_vocab_size = None
    _Cfg.markov_head_type = "gated"
    assert build_markov_head(_Cfg()) is not None
