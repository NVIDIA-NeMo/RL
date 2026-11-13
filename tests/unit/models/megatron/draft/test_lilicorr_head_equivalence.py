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
"""The native LiLiCorr head must stay bit-identical to ModelOpt's.

The head is a port, not a reimplementation: released drafters are trained against
ModelOpt's parameter layout, and the failure mode of a divergent copy is silent --
weights load by name into a module that computes something slightly different, and the
only symptom is an acceptance length that never becomes useful.

Skipped when modelopt is not installed, so the suite still runs in a minimal image; it
is the image that trains LiLiCorr which must have it.
"""

import pytest
import torch

from nemo_rl.models.megatron.draft.lilicorr_head import LiLiCorrHead as NativeHead

modelopt_lilicorr = pytest.importorskip(
    "modelopt.torch.speculative.plugins.modeling_lilicorr",
    reason="modelopt is not installed",
)

GEOMETRY = dict(
    model_hidden_size=64,
    hidden_size=32,
    num_layers=2,
    num_heads=4,
    mlp_ratio=2.0,
    block_size=4,
    candidate_topk=3,
    factor_dim=8,
    rms_norm_eps=1e-6,
    vector_eps=1e-8,
    logit_scale=1.0,
)


def _build_pair():
    torch.manual_seed(0)
    native = NativeHead(**GEOMETRY)
    reference = modelopt_lilicorr.LiLiCorrHead(**GEOMETRY)
    # Same weights in both, by name: this is also the check that the parameter
    # layouts agree, since load_state_dict is strict by default.
    reference.load_state_dict(native.state_dict())
    return native.eval(), reference.eval()


def test_parameter_layouts_match():
    native, reference = _build_pair()
    assert set(native.state_dict()) == set(reference.state_dict())
    for name, tensor in native.state_dict().items():
        assert tensor.shape == reference.state_dict()[name].shape, name


def test_compose_log_factors_matches():
    native, reference = _build_pair()
    torch.manual_seed(1)
    b, n, slots, k = 2, 3, GEOMETRY["block_size"] - 1, GEOMETRY["candidate_topk"]
    start = torch.randn(b, n, k)
    pair = torch.randn(b, n, slots - 1, k, k)

    got_start, got_pair = native.compose_log_factors(
        start_scores=start, pair_scores=pair
    )
    exp_start, exp_pair = reference.compose_log_factors(
        start_scores=start, pair_scores=pair
    )
    torch.testing.assert_close(got_start, exp_start, rtol=0, atol=0)
    torch.testing.assert_close(got_pair, exp_pair, rtol=0, atol=0)


def _forward_inputs():
    torch.manual_seed(2)
    # slots are the block_size - 1 predicted positions; the anchor is committed.
    b, n, slots, k = 2, 3, GEOMETRY["block_size"] - 1, GEOMETRY["candidate_topk"]
    vocab = 50
    return dict(
        candidate_token_ids=torch.randint(0, vocab, (b, n, slots, k)),
        candidate_log_probs=torch.log_softmax(torch.randn(b, n, slots, k), dim=-1),
        pass_hidden=torch.randn(b, n, slots, GEOMETRY["model_hidden_size"]),
        embed_tokens=torch.nn.Embedding(vocab, GEOMETRY["model_hidden_size"]),
        anchor_hidden=torch.randn(b, n, 1, GEOMETRY["model_hidden_size"]),
    )


def test_forward_matches():
    native, reference = _build_pair()
    inputs = _forward_inputs()
    with torch.no_grad():
        got = native(**inputs)
        exp = reference(**inputs)
    assert len(got) == len(exp)
    for got_tensor, exp_tensor in zip(got, exp):
        torch.testing.assert_close(got_tensor, exp_tensor, rtol=0, atol=0)


def test_the_test_can_fail():
    """Guard the guard: perturbing the reference's weights must break equivalence."""
    native, reference = _build_pair()
    with torch.no_grad():
        for param in reference.parameters():
            param.add_(0.1)
        inputs = _forward_inputs()
        got = native(**inputs)
        exp = reference(**inputs)
    assert not all(torch.equal(a, b_) for a, b_ in zip(got, exp)), (
        "perturbing the weights did not change the output; the comparison proves nothing"
    )
