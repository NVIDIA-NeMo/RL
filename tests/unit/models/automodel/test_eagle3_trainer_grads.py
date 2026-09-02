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
"""Gradient flow through the EAGLE3 TTT training forward (built on automodel's
LlamaEagle3DraftModel; see nemo_rl/models/automodel/draft/eagle3_llama.py).

``train_embed_and_head`` must be real: when trainable, both the embedding
table and the lm_head receive gradients from the TTT loss; when frozen,
neither does (while the rest of the drafter still trains).
"""

import pytest
import torch

# Skip entire module if nemo_automodel is not available
try:
    import nemo_automodel  # noqa: F401
except ImportError:
    pytest.skip("nemo_automodel not available", allow_module_level=True)

pytestmark = pytest.mark.automodel


def _tiny_eagle3_model(norm_before_residual: bool = True):
    from transformers.models.llama.configuration_llama import LlamaConfig

    from nemo_rl.models.automodel.draft.eagle3_llama import Eagle3DraftModel

    config = LlamaConfig(
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
    config.attn_implementation = "eager"
    config.draft_vocab_size = 40
    config.norm_before_residual = norm_before_residual
    config.target_layer_ids = [0, 0, 0]
    config.num_aux_hidden_states = 3
    # initialize_rms_norm_module defaults to bf16 params regardless of the
    # rest of the model's dtype (see nemo_automodel.components.models.common
    # .utils.initialize_rms_norm_module); force fp32 everywhere for
    # gradient-comparison tests so precision isn't a confound.
    return Eagle3DraftModel(config).float()


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

    embed_grad = model.model.embed_tokens.weight.grad
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
    fc_grad = model.model.fc.weight.grad
    assert fc_grad is not None and torch.isfinite(fc_grad).all()


@pytest.mark.parametrize("norm_before_residual", [True, False])
def test_norm_before_residual_toggle_both_produce_finite_grads(norm_before_residual):
    """automodel's Eagle3LlamaDecoderLayer hardcodes norm_before_residual=False
    with no toggle at all; _Eagle3LlamaDecoderLayerWithResidualToggle restores
    it (RedHatAI's speculators checkpoints declare norm_before_residual=true
    explicitly). Both branches must actually run and train."""
    model = _tiny_eagle3_model(norm_before_residual=norm_before_residual)
    model.requires_grad_(True)
    model.set_embedding_head_trainable(True)
    layer = model.model.layers[0]
    assert layer.norm_before_residual is norm_before_residual

    loss = _run_ttt_backward(model)
    assert torch.isfinite(loss)
    assert model.model.fc.weight.grad is not None


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


def test_rope_stays_precise_after_bf16_cast():
    """The training build casts the drafter to bf16. automodel's
    LlamaRotaryEmbedding recomputes inv_freq from config in fp32 on every
    cache build regardless of the registered buffer's dtype (see its
    _build_cache docstring), so a bf16-cast model's cos/sin cache must match
    a never-cast model's exactly -- unlike the old vendored drafter, there is
    no buffer to pin, but the property this used to guard (fp32-precision
    RoPE surviving a dtype cast, matching vLLM serving's fp32 cache) must
    still hold."""
    torch.manual_seed(0)
    fresh = _tiny_eagle3_model()
    cast = _tiny_eagle3_model()
    cast.load_state_dict(fresh.state_dict())
    cast = cast.to(torch.bfloat16)

    position_ids = torch.arange(8).unsqueeze(0)
    x = torch.randn(1, 8, 32)
    fresh_cos, fresh_sin = fresh.model.layers[0].self_attn.rotary_emb(x, position_ids)
    cast_cos, cast_sin = cast.model.layers[0].self_attn.rotary_emb(
        x.to(torch.bfloat16), position_ids
    )
    assert torch.allclose(cast_cos.float(), fresh_cos.float(), atol=1e-3)
    assert torch.allclose(cast_sin.float(), fresh_sin.float(), atol=1e-3)
    # The rest of the model really did cast.
    assert cast.model.fc.weight.dtype == torch.bfloat16


def test_chunked_kl_matches_unchunked_reference(monkeypatch):
    """Sequence-chunked + checkpointed KL must equal the one-shot computation
    exactly (per-position KL is independent), including gradients."""
    import torch

    from nemo_rl.models.automodel.draft import eagle3_llama as m

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


def _run_ttt_forward_backward(model):
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
    return terms


def test_chunked_lm_head_matches_unchunked_reference(monkeypatch):
    """The TTT forward's per-chunk lm_head must reproduce the single-chunk
    (effectively unchunked) result exactly: loss/accuracy terms and
    gradients, independent of chunk size. Chunking lm_head avoids
    materializing the full [1, T, draft_vocab] logits tensor per TTT step
    (~2.4 GiB at 20k tokens / 64000 draft vocab), but must be a pure
    regrouping of the same per-position computation.
    """
    from nemo_rl.models.automodel.draft import eagle3_llama as m

    model = _tiny_eagle3_model()
    model.requires_grad_(True)
    model.set_embedding_head_trainable(True)

    monkeypatch.setattr(m, "_KL_CHUNK_TOKENS", 1024)  # total=16 tokens: one chunk
    reference = _run_ttt_forward_backward(model)
    reference_grads = {
        name: p.grad.clone()
        for name, p in model.named_parameters()
        if p.grad is not None
    }

    model.zero_grad(set_to_none=True)
    monkeypatch.setattr(m, "_KL_CHUNK_TOKENS", 3)  # force multi-chunk path
    chunked = _run_ttt_forward_backward(model)
    chunked_grads = {
        name: p.grad.clone()
        for name, p in model.named_parameters()
        if p.grad is not None
    }

    for field in (
        "loss_nums",
        "loss_dens",
        "full_acc_nums",
        "full_acc_dens",
        "cond_acc_nums",
        "cond_acc_dens",
    ):
        for ref_val, chunked_val in zip(
            getattr(reference, field), getattr(chunked, field)
        ):
            assert torch.allclose(chunked_val, ref_val, atol=1e-5), field

    assert reference_grads.keys() == chunked_grads.keys()
    for name in reference_grads:
        assert torch.allclose(chunked_grads[name], reference_grads[name], atol=1e-5), (
            name
        )


def _has_flash_attn() -> bool:
    try:
        import flash_attn  # noqa: F401
    except ImportError:
        return False
    return True


@pytest.mark.skipif(
    not torch.cuda.is_available() or not _has_flash_attn(),
    reason="CUDA + flash-attn required (not installed in every test venv)",
)
def test_flash_attention_2_packing_matches_eager_dense_mask():
    """attn_implementation="flash_attention_2" (real training configs) must
    reproduce the eager dense-mask path (unit tests, CPU/no-flash-attn
    fallback) on packed, internally-padded microbatches.

    Every sample occupies a FIXED seq_len-wide slot in the packed row (see
    Eagle3Runtime.compute_loss); real content sits at the front of each
    slot, padding at the tail. The FA2 path declares each fixed slot as one
    varlen "document" (ignoring the real/padding split within it) -- this
    test's samples have deliberately different real lengths within their
    fixed slots to prove that's still correct: causal masking already keeps
    real tokens from ever attending to their own trailing padding (padding
    is chronologically later), so the loss_dens (valid-position counts) and
    gradients must match exactly/closely despite the internal padding.
    """
    from transformers.models.llama.configuration_llama import LlamaConfig

    from nemo_rl.models.automodel.draft.eagle3_llama import Eagle3DraftModel

    device = "cuda"

    def build_model(attn_impl):
        config = LlamaConfig(
            hidden_size=64,
            intermediate_size=128,
            num_hidden_layers=1,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=16,
            vocab_size=200,
            max_position_embeddings=512,
            rms_norm_eps=1e-6,
            attention_bias=False,
        )
        config.attn_implementation = attn_impl
        # See build_eagle3_draft_model: LlamaRotaryEmbedding's cos/sin cache
        # defaults to fp32 without this, which promotes q/k to fp32 through
        # RoPE while the cached V (untouched by RoPE) stays bf16 -- eager
        # attention's attn_probs @ v0 then mismatches dtypes.
        config.torch_dtype = torch.bfloat16
        config.draft_vocab_size = 80
        config.norm_before_residual = True
        config.target_layer_ids = [0, 0, 0]
        config.num_aux_hidden_states = 3
        return Eagle3DraftModel(config).to(device=device, dtype=torch.bfloat16)

    torch.manual_seed(0)
    # Fixed-width slots (seq_len=20) with deliberately different real
    # content lengths (20, 12, 7) -- the last two carry internal padding.
    batch_size, seq_len = 3, 20
    input_lengths = torch.tensor([20, 12, 7], device=device)
    total = batch_size * seq_len

    positions = torch.arange(seq_len, device=device).unsqueeze(0)
    valid = positions < input_lengths.unsqueeze(1)
    document_ids = torch.where(
        valid,
        torch.arange(batch_size, device=device).unsqueeze(1),
        torch.full_like(positions.expand(batch_size, -1), -1),
    )
    position_ids = (1 + positions).expand(batch_size, -1)
    seq_lens = torch.full((1, batch_size), seq_len, dtype=torch.long, device=device)

    input_ids = torch.randint(0, 200, (batch_size, seq_len), device=device)
    loss_mask = valid.clone()
    teacher_logits = torch.randn(
        batch_size, seq_len, 80, device=device, dtype=torch.bfloat16
    )
    fused_hidden = torch.randn(
        batch_size, seq_len, 3 * 64, device=device, dtype=torch.bfloat16
    )

    def run(model):
        torch.manual_seed(123)
        terms = model(
            fused_hidden_states=fused_hidden.reshape(1, total, -1).clone(),
            input_ids=input_ids.reshape(1, total),
            document_ids=document_ids.reshape(1, total),
            loss_mask=loss_mask.reshape(1, total),
            teacher_logits=teacher_logits.reshape(1, total, -1),
            position_ids=position_ids.reshape(1, total),
            ttt_steps=3,
            seq_lens=seq_lens,
        )
        loss = sum(num / den for num, den in zip(terms.loss_nums, terms.loss_dens))
        loss.backward()
        return terms, loss

    model_eager = build_model("eager")
    model_fa2 = build_model("flash_attention_2")
    model_fa2.load_state_dict(model_eager.state_dict())

    terms_eager, loss_eager = run(model_eager)
    terms_fa2, loss_fa2 = run(model_fa2)

    assert torch.isfinite(loss_eager) and torch.isfinite(loss_fa2)
    # Exact: both paths must agree on which positions are valid regardless
    # of attention kernel.
    for a, b in zip(terms_eager.loss_dens, terms_fa2.loss_dens):
        assert torch.equal(a, b)
    for a, b in zip(terms_eager.full_acc_dens, terms_fa2.full_acc_dens):
        assert torch.equal(a, b)
    # Close, not exact: different attention kernels round differently in bf16.
    for a, b in zip(terms_eager.loss_nums, terms_fa2.loss_nums):
        assert torch.allclose(a, b, atol=0.05, rtol=0.02)

    grad_eager = model_eager.model.fc.weight.grad
    grad_fa2 = model_fa2.model.fc.weight.grad
    assert grad_eager is not None and grad_fa2 is not None
    assert torch.allclose(grad_eager, grad_fa2, atol=2e-2)
