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
"""The co-trained LiLiCorr drafter must compute what vLLM serves."""

import pytest
import torch
import torch.nn.functional as F

from nemo_rl.algorithms.loss.draft import projected_streaming_vocab_parallel_soft_ce
from nemo_rl.models.megatron.draft.dflash import DFlashBodyConfig, DFlashGroupedConv
from nemo_rl.models.megatron.draft.hidden_capture import HiddenStateCapture
from nemo_rl.models.megatron.draft.lilicorr_model import vocab_parallel_topk_logprobs
from nemo_rl.models.megatron.draft.utils import _dflash_weight_layout


def _reference_grouped_conv(blocks, delta, base, taps):
    """vLLM 0.33 qwen3_dflash2._grouped_conv (CPU path), on flattened positions."""
    num_blocks, block_size, hidden = blocks.shape
    groups = delta.shape[-1]
    flat = blocks.reshape(-1, groups, hidden // groups)
    flat_delta = delta.reshape(-1, taps, groups)
    coefficients = base.view(1, taps, groups, hidden // groups) + flat_delta.unsqueeze(
        -1
    )
    output = coefficients[:, 0] * flat
    position = torch.arange(flat.shape[0]) % block_size
    for tap in range(1, taps):
        shifted = F.pad(flat[:-tap], (0, 0, 0, 0, tap, 0))
        output = output + coefficients[:, tap] * shifted * (position >= tap).view(
            -1, 1, 1
        )
    return output.reshape(num_blocks, block_size, hidden)


def _random_conv(hidden=32, taps=3, group_size=4):
    torch.manual_seed(0)
    conv = DFlashGroupedConv(hidden, taps, group_size, torch.float32)
    with torch.no_grad():
        conv.base_kernel.normal_()
        conv.kernel_projection.weight.normal_(std=0.1)
    return conv


def test_grouped_conv_matches_vllm_both_sides():
    conv = _random_conv()
    x = torch.randn(5, 8, 32)
    out, kernel = conv.prepare(x)
    coefficients = conv.kernel_projection(x).view(5, 8, 2, conv.taps, conv.num_groups)
    expected_in = _reference_grouped_conv(
        x, coefficients[:, :, 0], conv.base_kernel[0], conv.taps
    )
    torch.testing.assert_close(out, expected_in)
    y = torch.randn(5, 8, 32)
    expected_out = _reference_grouped_conv(
        y, coefficients[:, :, 1], conv.base_kernel[1], conv.taps
    )
    torch.testing.assert_close(conv.finish(y, kernel), expected_out)


def test_grouped_conv_never_reads_across_blocks():
    conv = _random_conv()
    x = torch.randn(4, 8, 32)
    out, _ = conv.prepare(x)
    x2 = x.clone()
    x2[1] += 100.0  # perturb one block only
    out2, _ = conv.prepare(x2)
    torch.testing.assert_close(out[[0, 2, 3]], out2[[0, 2, 3]])


def test_grouped_conv_is_identity_before_load():
    conv = DFlashGroupedConv(32, 3, 4, torch.float32)
    x = torch.randn(2, 8, 32)
    out, kernel = conv.prepare(x)
    torch.testing.assert_close(out, x)
    torch.testing.assert_close(conv.finish(x, kernel), x)


def test_conv_parameters_have_replicated_checkpoint_layouts():
    config = DFlashBodyConfig(hidden_size=64, conv_kernel_size=2, conv_group_size=16)
    shape, axis = _dflash_weight_layout(
        "layers.0.attention_conv.base_kernel", config=config
    )
    assert (shape, axis) == ((2, 2, 64), None)
    shape, axis = _dflash_weight_layout(
        "layers.3.mlp_conv.kernel_projection.weight", config=config
    )
    assert (shape, axis) == ((2 * 2 * 4, 64), None)


def test_conv_config_requires_both_fields():
    with pytest.raises(ValueError, match="set together"):
        DFlashBodyConfig(conv_kernel_size=2)


def test_vocab_parallel_topk_single_rank_matches_log_softmax():
    logits = torch.randn(3, 7, 50)
    log_probs, ids = vocab_parallel_topk_logprobs(
        logits, 4, vocab_start=0, vocab_size=50, tp_group=None
    )
    expected_values, expected_ids = torch.log_softmax(logits, dim=-1).topk(4, dim=-1)
    torch.testing.assert_close(log_probs, expected_values)
    assert torch.equal(ids, expected_ids)


def test_vocab_parallel_topk_never_returns_padded_vocab():
    logits = torch.zeros(2, 3, 16)
    logits[..., 12:] = 10.0  # padded columns carry the largest logits
    log_probs, ids = vocab_parallel_topk_logprobs(
        logits, 3, vocab_start=0, vocab_size=12, tp_group=None
    )
    assert int(ids.max()) < 12
    torch.testing.assert_close(
        log_probs, torch.full_like(log_probs, -torch.log(torch.tensor(12.0)))
    )


def test_vocab_parallel_topk_offsets_ids_to_global_vocab():
    logits = torch.randn(1, 1, 8)
    _, ids = vocab_parallel_topk_logprobs(
        logits, 2, vocab_start=24, vocab_size=64, tp_group=None
    )
    assert int(ids.min()) >= 24


@pytest.mark.parametrize("token_chunk_size", [2, 1024])  # streaming and cached paths
def test_soft_ce_trains_a_drafter_owned_head(token_chunk_size):
    torch.manual_seed(0)
    hidden = torch.randn(2, 3, 8, dtype=torch.float64, requires_grad=True)
    weight = torch.randn(11, 8, dtype=torch.float64, requires_grad=True)
    teacher = torch.randn(2, 3, 11, dtype=torch.float64)
    mask = torch.tensor([[1, 1, 0], [1, 0, 1]], dtype=torch.bool)

    stats = projected_streaming_vocab_parallel_soft_ce(
        student_hidden=hidden,
        output_weight=weight,
        selected_teacher_logits=teacher,
        mask=mask,
        token_chunk_size=token_chunk_size,
        tp_group=None,
        train_output_weight=True,
    )
    stats.numerators.sum().backward()

    ref_hidden = hidden.detach().clone().requires_grad_()
    ref_weight = weight.detach().clone().requires_grad_()
    logits = ref_hidden @ ref_weight.T
    per_token = -(teacher.softmax(-1) * logits.log_softmax(-1)).sum(-1)
    (per_token * mask).sum().backward()

    assert weight.grad is not None, "drafter-owned head received no gradient"
    # The kernels accumulate in float32; compare at float32 precision.
    torch.testing.assert_close(weight.grad, ref_weight.grad, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(hidden.grad, ref_hidden.grad, rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("token_chunk_size", [1, 1024])
def test_soft_ce_never_trains_the_head_unless_asked(token_chunk_size):
    # Even a head that requires grad (a policy head someone forgot to detach) must
    # not learn from the draft loss by default.
    hidden = torch.randn(1, 2, 8, requires_grad=True)
    weight = torch.randn(5, 8, requires_grad=True)
    stats = projected_streaming_vocab_parallel_soft_ce(
        student_hidden=hidden,
        output_weight=weight,
        selected_teacher_logits=torch.randn(1, 2, 5),
        mask=torch.ones(1, 2, dtype=torch.bool),
        token_chunk_size=token_chunk_size,
        tp_group=None,
    )
    stats.numerators.sum().backward()
    assert weight.grad is None
    assert hidden.grad is not None


@pytest.mark.parametrize("positional", [False, True])
def test_capture_takes_decoder_input_when_embedding_is_bypassed(positional):
    capture = object.__new__(HiddenStateCapture)
    capture._captured = {}
    hook = capture._make_decoder_input_hook()
    embeds = torch.randn(4, 1, 8)
    if positional:
        hook(None, (None, None, None, embeds), {})
    else:
        hook(None, (), {"decoder_input": embeds})
    assert capture._captured["embeds"] is not None
    torch.testing.assert_close(capture._captured["embeds"], embeds)


def test_capture_ignores_a_forward_without_decoder_input():
    capture = object.__new__(HiddenStateCapture)
    capture._captured = {}
    capture._make_decoder_input_hook()(None, (torch.zeros(1),), {})
    assert "embeds" not in capture._captured


def test_policy_head_is_found_inside_a_vl_wrapper():
    from types import SimpleNamespace

    from nemo_rl.models.megatron.draft.utils import (
        get_policy_lm_head_weight,
        unwrap_policy_language_model,
    )

    weight = torch.zeros(4, 2)
    language_model = torch.nn.Module()
    language_model.output_layer = SimpleNamespace(weight=weight)
    llava = torch.nn.Module()
    llava.language_model = language_model
    omni = torch.nn.Module()  # NemotronOmniModel layout: .llava_model.language_model
    omni.llava_model = llava
    assert unwrap_policy_language_model(omni) is language_model
    assert get_policy_lm_head_weight(omni) is weight
    assert unwrap_policy_language_model(language_model) is language_model
