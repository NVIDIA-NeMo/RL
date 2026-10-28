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

import pytest
import torch

from nemo_rl.algorithms.logits_sampling_utils import apply_top_k_top_p


@pytest.mark.parametrize(
    "device",
    [
        "cpu",
        pytest.param(
            "cuda",
            marks=pytest.mark.skipif(
                not torch.cuda.is_available(), reason="CUDA not available"
            ),
        ),
    ],
)
@pytest.mark.parametrize("top_k", [None, 5])
def test_apply_top_k_top_p_accepts_noncontiguous_logits(
    device: str, top_k: int | None
) -> None:
    """Top-p filtering should support non-contiguous multi-sequence logits."""
    torch.manual_seed(1234)
    full_logits = torch.randn(2, 6, 17, device=device, dtype=torch.float32)
    logits = full_logits[:, :-1, :]

    assert logits.shape == (2, 5, 17)
    assert logits.stride() == (102, 17, 1)
    assert not logits.is_contiguous()

    filtered_logits, keep_mask = apply_top_k_top_p(logits, top_k=top_k, top_p=0.9)
    reference_logits, reference_mask = apply_top_k_top_p(
        logits.contiguous(), top_k=top_k, top_p=0.9
    )

    assert keep_mask is not None
    assert reference_mask is not None
    assert filtered_logits.shape == logits.shape
    assert filtered_logits.dtype == logits.dtype
    assert filtered_logits.device == logits.device
    assert keep_mask.shape == logits.shape
    assert keep_mask.device == logits.device
    torch.testing.assert_close(filtered_logits, reference_logits, rtol=0, atol=0)
    torch.testing.assert_close(keep_mask, reference_mask, rtol=0, atol=0)


_DEVICES = [
    "cpu",
    pytest.param(
        "cuda",
        marks=pytest.mark.skipif(
            not torch.cuda.is_available(), reason="CUDA not available"
        ),
    ),
]


def _tie_inclusive_top_p_support(logits: torch.Tensor, top_p: float) -> torch.Tensor:
    """Reference: keep v iff the mass of tokens with a strictly larger logit is < top_p."""
    probs = torch.softmax(logits.double(), dim=-1)
    larger = logits.unsqueeze(-2) > logits.unsqueeze(-1)  # [..., v, u]: z_u > z_v
    mass_above = (larger.double() * probs.unsqueeze(-2)).sum(dim=-1)
    return mass_above < top_p


@pytest.mark.parametrize("device", _DEVICES)
def test_keep_ties_keeps_the_whole_boundary_tie_group(device: str) -> None:
    logits = torch.zeros(1, 16, device=device)
    _, default_mask = apply_top_k_top_p(logits, top_k=None, top_p=0.5)
    _, tie_mask = apply_top_k_top_p(logits, top_k=None, top_p=0.5, keep_ties=True)
    # The default keeps the 8 tied tokens the ascending sort places last.
    assert int(default_mask.sum()) == 8
    assert bool(tie_mask.all())

    # Tie group straddling the boundary: [3, 1, 1, 1, 1, 0, ...].
    logits = torch.tensor([[3.0, 1.0, 1.0, 1.0, 1.0] + [0.0] * 11], device=device)
    for top_p in (0.5, 0.7, 0.9):
        _, tie_mask = apply_top_k_top_p(logits, top_k=None, top_p=top_p, keep_ties=True)
        expected = _tie_inclusive_top_p_support(logits, top_p)
        assert torch.equal(tie_mask, expected), top_p


@pytest.mark.parametrize("device", _DEVICES)
@pytest.mark.parametrize("top_k", [None, 8])
def test_keep_ties_matches_reference_on_bf16_valued_rows(
    device: str, top_k: int | None
) -> None:
    """bf16 logits tie often; the tie-inclusive set must not depend on the sort."""
    torch.manual_seed(0)
    logits = (torch.randn(512, 64, device=device) * 4).bfloat16().float()
    top_p = 0.9
    _, tie_mask = apply_top_k_top_p(logits, top_k=top_k, top_p=top_p, keep_ties=True)
    if top_k is None:
        expected = _tie_inclusive_top_p_support(logits, top_p)
    else:
        top_k_logits, _ = apply_top_k_top_p(logits, top_k=top_k, top_p=1.0)
        expected = _tie_inclusive_top_p_support(top_k_logits, top_p)
        expected &= torch.isfinite(top_k_logits)
    assert torch.equal(tie_mask, expected)
    # Permuting the vocabulary permutes the support.
    perm = torch.randperm(logits.shape[-1], device=device)
    _, permuted = apply_top_k_top_p(
        logits[:, perm], top_k=top_k, top_p=top_p, keep_ties=True
    )
    assert torch.equal(permuted, tie_mask[:, perm])
    # Non-vacuity: the default support differs on some of these rows.
    _, default_mask = apply_top_k_top_p(logits, top_k=top_k, top_p=top_p)
    assert bool((default_mask != tie_mask).any())


@pytest.mark.parametrize("device", _DEVICES)
def test_keep_ties_is_a_no_op_without_ties(device: str) -> None:
    torch.manual_seed(0)
    # Distinct logits in every row: a random permutation of 0.01 * [0, 1000).
    logits = torch.rand(64, 1000, device=device).argsort(dim=-1).float() * 0.01
    a, mask_a = apply_top_k_top_p(logits, top_k=None, top_p=0.9)
    b, mask_b = apply_top_k_top_p(logits, top_k=None, top_p=0.9, keep_ties=True)
    assert torch.equal(mask_a, mask_b)
    torch.testing.assert_close(a, b, rtol=0, atol=0)


@pytest.mark.parametrize("device", _DEVICES)
def test_keep_ties_masks_gradients_outside_the_support(device: str) -> None:
    logits = torch.zeros(2, 16, device=device, requires_grad=True)
    filtered, keep_mask = apply_top_k_top_p(
        logits, top_k=None, top_p=0.5, keep_ties=True
    )
    torch.log_softmax(filtered, dim=-1)[:, 3].sum().backward()
    assert bool(keep_mask.all())
    assert torch.isfinite(logits.grad).all()
    assert bool((logits.grad != 0).all())


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_keep_ties_contains_every_flashinfer_top_p_draw() -> None:
    """Megatron generation samples top-p with FlashInfer; every draw must have a
    finite training log-prob under the tie-inclusive support."""
    flashinfer = pytest.importorskip("flashinfer")
    torch.manual_seed(0)
    logits = (torch.randn(1024, 32768, device="cuda") * 8).bfloat16().float()
    probs = torch.softmax(logits, dim=-1)
    top_p = 0.9
    _, default_mask = apply_top_k_top_p(logits, top_k=None, top_p=top_p)
    _, tie_mask = apply_top_k_top_p(logits, top_k=None, top_p=top_p, keep_ties=True)
    generator = torch.Generator(device="cuda").manual_seed(0)
    outside_default = outside_tie = 0
    for _ in range(20):
        tokens = flashinfer.sampling.top_p_sampling_from_probs(
            probs, top_p, deterministic=True, generator=generator
        ).long()[:, None]
        outside_default += int((~default_mask.gather(1, tokens)).sum())
        outside_tie += int((~tie_mask.gather(1, tokens)).sum())
    assert outside_tie == 0
    assert outside_default > 0  # non-vacuity: the sort-order support misses draws
