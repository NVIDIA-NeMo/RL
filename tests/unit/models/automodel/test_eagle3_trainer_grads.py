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
"""Gradient flow through the vendored EAGLE3 TTT trainer.

``train_embed_and_head`` must be real: when trainable, both the embedding
table and the lm_head receive gradients from the TTT loss; when frozen,
neither does (while the rest of the drafter still trains).
"""

import pytest
import torch


def _tiny_eagle3_model():
    from transformers.models.qwen3.configuration_qwen3 import Qwen3Config

    from nemo_rl.models.automodel.draft.eagle3_qwen3 import Qwen3Eagle3DraftModel

    config = Qwen3Config(
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=16,
        vocab_size=100,
        max_position_embeddings=256,
        rms_norm_eps=1e-6,
        attention_bias=False,
    )
    config._attn_implementation = "eager"
    config.draft_vocab_size = 40
    config.norm_before_residual = True
    config.target_layer_ids = [0, 0, 0]
    return Qwen3Eagle3DraftModel(config)


def _run_ttt_backward(model):
    torch.manual_seed(0)
    total = 16
    input_ids = torch.randint(0, 100, (1, total))
    loss_mask = torch.zeros(1, total, dtype=torch.bool)
    loss_mask[:, 4:] = True
    terms = model(
        fused_hidden_states=torch.randn(1, total, 3 * 32),
        input_ids=input_ids,
        document_ids=torch.zeros(1, total, dtype=torch.long),
        loss_mask=loss_mask,
        teacher_logits=torch.randn(1, total, 40),
        ttt_steps=2,
    )
    loss = sum(num / den for num, den in zip(terms.loss_nums, terms.loss_dens))
    loss.backward()
    return loss


@pytest.mark.parametrize("trainable", [True, False])
def test_embed_and_head_gradients_follow_trainability(trainable):
    model = _tiny_eagle3_model()
    model.requires_grad_(True)
    model.set_embedding_head_trainable(trainable)

    loss = _run_ttt_backward(model)
    assert torch.isfinite(loss)

    embed_grad = model.embed_tokens.weight.grad
    head_grad = model.lm_head.weight.grad
    if trainable:
        assert embed_grad is not None and embed_grad.abs().sum() > 0, (
            "train_embed_and_head=true must produce embedding gradients"
        )
        assert head_grad is not None and head_grad.abs().sum() > 0
    else:
        assert embed_grad is None, "frozen embeddings must receive no gradient"
        assert head_grad is None, "frozen lm_head must receive no gradient"
    # The drafter trunk always trains.
    fc_grad = model.fc.weight.grad
    assert fc_grad is not None and torch.isfinite(fc_grad).all()


def test_next_token_position_mask_targets_label_positions():
    """Rollout token_mask marks tokens; the TTT forward gates loss at logit
    positions (position t supervises token t+1). The shift must supervise the
    last-prompt-token position (label = FIRST response token, where drafting
    starts) and zero the tail position (no next-token label)."""
    from nemo_rl.models.automodel.draft.integration import next_token_position_mask

    # prompt = positions 0..2, response tokens = positions 3..5 (T = 6).
    token_mask = torch.tensor([[0.0, 0.0, 0.0, 1.0, 1.0, 1.0]])
    shifted = next_token_position_mask(token_mask)
    # Positions 2..4 predict tokens 3..5; position 5 has no label.
    assert torch.equal(shifted, torch.tensor([[0.0, 0.0, 1.0, 1.0, 1.0, 0.0]]))


def test_rope_inv_freq_stays_fp32_after_bf16_cast():
    """The training build casts the drafter to bf16; the RoPE frequency
    buffers must stay fp32 (matching the dspark drafter and vLLM serving's
    fp32 RoPE cache) or positions dephase and acceptance erodes."""
    model = _tiny_eagle3_model()
    model.to(torch.bfloat16)
    assert model.rotary_emb.inv_freq.dtype == torch.float32
    assert model.rotary_emb.original_inv_freq.dtype == torch.float32
    # The rest of the model really did cast.
    assert model.fc.weight.dtype == torch.bfloat16


def test_chunked_kl_matches_unchunked_reference(monkeypatch):
    """Sequence-chunked + checkpointed KL must equal the one-shot computation
    exactly (per-position KL is independent), including gradients."""
    import torch

    from nemo_rl.models.automodel.draft import eagle3_qwen3 as m

    torch.manual_seed(0)
    logits_ref = torch.randn(2, 7, 13, requires_grad=True)
    targets = torch.randn(2, 7, 13)

    reference = m._kl_div_per_position_chunk(logits_ref, targets)
    reference.sum().backward()

    monkeypatch.setattr(m, "_KL_CHUNK_TOKENS", 3)  # force multi-chunk path
    logits_chunked = logits_ref.detach().clone().requires_grad_(True)
    chunked = m._kl_div_per_position(logits_chunked, targets)
    chunked.sum().backward()

    assert torch.allclose(chunked, reference, atol=1e-6)
    assert torch.allclose(logits_chunked.grad, logits_ref.grad, atol=1e-6)

    # No-grad path (eval) takes the plain branch and must agree too.
    with torch.no_grad():
        eval_out = m._kl_div_per_position(logits_ref.detach(), targets)
    assert torch.allclose(eval_out, reference, atol=1e-6)
