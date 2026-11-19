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

import functools
from contextlib import AbstractContextManager, nullcontext
from typing import Optional

import pytest
import torch
from torch import nn
from torch.distributed.fsdp import MixedPrecisionPolicy, fully_shard
from torch.distributed.tensor import DTensor
from transformers import LlamaConfig, LlamaForCausalLM

from nemo_rl.models.automodel.fused_logprobs import FusedLogprobLMHead

BATCH, SEQ, VOCAB = 2, 13, 257


def _tiny_llama(*, tie_word_embeddings: bool, seed: int = 0) -> LlamaForCausalLM:
    config = LlamaConfig(
        vocab_size=VOCAB,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=64,
        tie_word_embeddings=tie_word_embeddings,
    )
    torch.manual_seed(seed)
    return LlamaForCausalLM(config).cuda()


def _batch(seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator(device="cuda").manual_seed(seed)
    input_ids = torch.randint(VOCAB, (BATCH, SEQ), device="cuda", generator=generator)
    token_weights = torch.randn(BATCH, SEQ - 1, device="cuda", generator=generator)
    return input_ids, token_weights


def _use_te_lm_head(model: LlamaForCausalLM, *, bias: bool = False) -> None:
    """Swap in a Transformer Engine Linear head, as AutoModel's linear="te" backend builds."""
    te = pytest.importorskip("transformer_engine.pytorch")
    old = model.lm_head
    model.lm_head = te.Linear(
        old.in_features,
        old.out_features,
        bias=bias,
        params_dtype=old.weight.dtype,
        device="cuda",
    )
    with torch.no_grad():
        model.lm_head.weight.copy_(old.weight)


def _fused_logprobs(
    model: nn.Module,
    head: FusedLogprobLMHead,
    input_ids: torch.Tensor,
    temperature: float,
) -> torch.Tensor:
    with head.capture(input_ids, temperature=temperature) as capture:
        model(input_ids=input_ids, use_cache=False)
    return capture.next_token_logprobs


def _unfused_logprobs(
    model: nn.Module, input_ids: torch.Tensor, temperature: float
) -> torch.Tensor:
    logits = model(input_ids=input_ids, use_cache=False).logits.float()
    if temperature != 1.0:
        logits = logits / temperature
    log_probs = logits[:, :-1].log_softmax(-1)
    return log_probs.gather(-1, input_ids[:, 1:].unsqueeze(-1)).squeeze(-1)


def _grads(model: nn.Module) -> dict[str, torch.Tensor]:
    grads = {}
    for name, param in model.named_parameters():
        assert param.grad is not None, name
        grad = param.grad
        grads[name] = (
            grad.full_tensor() if isinstance(grad, DTensor) else grad
        ).clone()
    model.zero_grad(set_to_none=True)
    return grads


def _assert_grads_close(
    actual: dict[str, torch.Tensor], expected: dict[str, torch.Tensor]
) -> None:
    assert actual.keys() == expected.keys()
    for name in expected:
        torch.testing.assert_close(
            actual[name].float(),
            expected[name].float(),
            rtol=2e-2,
            atol=2e-3,
            msg=lambda msg, name=name: f"{name}: {msg}",
        )


@pytest.fixture
def deterministic_bf16_matmul(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep cuBLAS from choosing a bf16 split-K reduction that depends on the chunk's row count."""
    monkeypatch.setattr(
        torch.backends.cuda.matmul, "allow_bf16_reduced_precision_reduction", False
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
@pytest.mark.usefixtures("deterministic_bf16_matmul")
@pytest.mark.parametrize("tie_word_embeddings", [False, True])
@pytest.mark.parametrize("precision", ["bf16_params", "fp32_params_bf16_autocast"])
@pytest.mark.parametrize("temperature", [1.0, 0.7])
def test_capture_matches_logits_path(
    tie_word_embeddings: bool, precision: str, temperature: float
) -> None:
    """Logprobs and every parameter gradient match log_softmax over the model's logits."""
    model = _tiny_llama(tie_word_embeddings=tie_word_embeddings)
    autocast: Optional[torch.autocast] = None
    if precision == "bf16_params":
        model = model.to(torch.bfloat16)
    else:
        autocast = torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    head = FusedLogprobLMHead(model, chunk_size=5)
    input_ids, token_weights = _batch(seed=1)

    with autocast or nullcontext():
        fused = _fused_logprobs(model, head, input_ids, temperature)
    (fused * token_weights).sum().backward()
    fused_grads = _grads(model)
    with autocast or nullcontext():
        unfused = _unfused_logprobs(model, input_ids, temperature)
    (unfused * token_weights).sum().backward()
    unfused_grads = _grads(model)

    assert fused.dtype == torch.float32 and fused.shape == (BATCH, SEQ - 1)
    torch.testing.assert_close(fused, unfused.detach(), rtol=1e-5, atol=1e-5)
    _assert_grads_close(fused_grads, unfused_grads)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
@pytest.mark.usefixtures("deterministic_bf16_matmul")
def test_te_lm_head_matches_te_forward() -> None:
    """A Transformer Engine LM head gives the same logprobs and gradients as its own forward."""
    model = _tiny_llama(tie_word_embeddings=False).to(torch.bfloat16)
    _use_te_lm_head(model)
    head = FusedLogprobLMHead(model, chunk_size=5)
    input_ids, token_weights = _batch(seed=1)

    fused = _fused_logprobs(model, head, input_ids, 0.7)
    (fused * token_weights).sum().backward()
    fused_grads = _grads(model)
    unfused = _unfused_logprobs(model, input_ids, 0.7)
    (unfused * token_weights).sum().backward()
    unfused_grads = _grads(model)

    torch.testing.assert_close(fused, unfused.detach(), rtol=1e-5, atol=1e-5)
    _assert_grads_close(fused_grads, unfused_grads)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_rejects_te_lm_head_with_bias() -> None:
    model = _tiny_llama(tie_word_embeddings=False)
    _use_te_lm_head(model, bias=True)
    with pytest.raises(ValueError, match="use_bias"):
        FusedLogprobLMHead(model, chunk_size=5)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_head_runs_unchanged_outside_capture() -> None:
    model = _tiny_llama(tie_word_embeddings=False)
    input_ids, _ = _batch(seed=1)
    with torch.no_grad():
        before = model(input_ids=input_ids, use_cache=False).logits
        FusedLogprobLMHead(model, chunk_size=5)
        after = model(input_ids=input_ids, use_cache=False).logits
    torch.testing.assert_close(after, before, rtol=0, atol=0)


class _AdapterLinear(nn.Linear):
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return super().forward(input) * 2


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
@pytest.mark.parametrize(
    "break_model",
    [
        pytest.param(
            lambda m: setattr(m, "lm_head", nn.Linear(64, VOCAB, bias=True)),
            id="bias",
        ),
        pytest.param(
            lambda m: setattr(m, "lm_head", _AdapterLinear(64, VOCAB, bias=False)),
            id="custom_forward",
        ),
        pytest.param(
            lambda m: setattr(m.config, "final_logit_softcapping", 30.0),
            id="softcapping",
        ),
    ],
)
def test_rejects_heads_it_would_compute_wrongly(break_model) -> None:
    model = _tiny_llama(tie_word_embeddings=False)
    break_model(model)
    with pytest.raises(ValueError):
        FusedLogprobLMHead(model, chunk_size=5)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_capture_misuse_fails_loudly() -> None:
    model = _tiny_llama(tie_word_embeddings=False)
    head = FusedLogprobLMHead(model, chunk_size=5)
    input_ids, _ = _batch(seed=1)
    hidden_states = torch.zeros(BATCH, SEQ, 64, device="cuda")

    with pytest.raises(ValueError, match="chunk_size must be positive"):
        FusedLogprobLMHead(model, chunk_size=0)
    with head.capture(input_ids, temperature=1.0) as capture:
        with pytest.raises(RuntimeError, match="did not run"):
            capture.next_token_logprobs
        with pytest.raises(RuntimeError, match="does not nest"):
            with head.capture(input_ids, temperature=1.0):
                pass
        with pytest.raises(ValueError, match="do not align"):
            model.lm_head(hidden_states[:, :-1])
        model.lm_head(hidden_states)
        with pytest.raises(RuntimeError, match="ran twice"):
            model.lm_head(hidden_states)


def _run_fsdp_matches_logits_path(rank: int, world_size: int, layout: str) -> None:
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    model = _tiny_llama(tie_word_embeddings=layout == "tied")
    if layout == "te_lm_head_own_unit":
        _use_te_lm_head(model)
    mp_policy = MixedPrecisionPolicy(
        param_dtype=torch.bfloat16,
        reduce_dtype=torch.float32,
        output_dtype=torch.float32,
    )
    for layer in model.model.layers:
        fully_shard(layer, mp_policy=mp_policy)
    own_unit = layout in ("lm_head_own_unit", "te_lm_head_own_unit")
    if own_unit:
        fully_shard(model.lm_head, mp_policy=mp_policy)
    fully_shard(model, mp_policy=mp_policy)
    head = FusedLogprobLMHead(model, chunk_size=5)
    # Each rank scores different tokens, so a gradient missing the peer
    # contribution differs from the reduced one.
    input_ids, token_weights = _batch(seed=rank + 1)

    # As in AutoModel: custom MoE models shard the LM head on its own and run
    # without autocast; HF models run under bf16 autocast.
    def autocast() -> AbstractContextManager[object]:
        if own_unit:
            return nullcontext()
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)

    with autocast():
        fused = _fused_logprobs(model, head, input_ids, 0.7)
    (fused * token_weights).sum().backward()
    fused_grads = _grads(model)
    with autocast():
        unfused = _unfused_logprobs(model, input_ids, 0.7)
    (unfused * token_weights).sum().backward()
    unfused_grads = _grads(model)

    _assert_grads_close(fused_grads, unfused_grads)


@pytest.mark.parametrize(
    "layout", ["lm_head_own_unit", "te_lm_head_own_unit", "lm_head_in_root", "tied"]
)
def test_fsdp2_gradients_match_logits_path(distributed_test_runner, layout: str):
    """Under FSDP2 DP=2 the LM-head and backbone gradients are reduced as on the logits path."""
    if layout.startswith("te_"):
        pytest.importorskip("transformer_engine.pytorch")
    distributed_test_runner(
        functools.partial(_run_fsdp_matches_logits_path, layout=layout), world_size=2
    )
