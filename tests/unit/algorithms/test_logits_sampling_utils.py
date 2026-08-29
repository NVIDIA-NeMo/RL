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

from nemo_rl.algorithms.logits_sampling_utils import (
    SamplingMask,
    apply_sampling_mask,
    apply_top_k_top_p,
    validate_sampling_mask_for_active_tokens,
)


def test_apply_sampling_mask_preserves_valid_token_zero_with_padded_zero_slots():
    """Invalid padded slots must not overwrite a valid support entry for ID 0."""
    logits = torch.tensor([[[2.0, 1.0, -1.0, 0.5]]], requires_grad=True)
    target = torch.tensor([[0]])
    sampling_mask = SamplingMask(
        token_ids=torch.tensor([[[0, 0, 0]]], dtype=torch.int32),
        sizes=torch.tensor([[1]], dtype=torch.int32),
    )

    filtered, keep = apply_sampling_mask(logits, target, sampling_mask)

    assert torch.equal(keep, torch.tensor([[[True, False, False, False]]]))
    assert filtered[0, 0, 0] == logits[0, 0, 0]
    assert torch.isneginf(filtered[0, 0, 1:]).all()


def test_apply_sampling_mask_preserves_nonempty_target_and_restricts_gradient():
    """The empty-row fallback must not clear targets from nonempty rows."""
    logits = torch.tensor(
        [[[0.1, 0.7, -0.4, 1.2], [1.0, 0.5, -0.5, 0.0]]],
        requires_grad=True,
    )
    target = torch.tensor([[3, 1]])
    sampling_mask = SamplingMask(
        token_ids=torch.tensor([[[1, 3, 0], [0, 0, 0]]], dtype=torch.int32),
        sizes=torch.tensor([[2, 0]], dtype=torch.int32),
    )

    filtered, keep = apply_sampling_mask(logits, target, sampling_mask)
    expected_keep = torch.tensor(
        [[[False, True, False, True], [False, True, False, False]]]
    )
    assert torch.equal(keep, expected_keep)

    selected = (
        torch.log_softmax(filtered, dim=-1).gather(-1, target.unsqueeze(-1)).squeeze(-1)
    )
    selected.sum().backward()
    assert logits.grad is not None
    expected_nonzero_grad = torch.tensor(
        [[[False, True, False, True], [False, False, False, False]]]
    )
    assert torch.equal(logits.grad != 0, expected_nonzero_grad)


def test_validate_sampling_mask_rejects_missing_active_support_and_target():
    target = torch.tensor([[5, 7]])
    active = torch.tensor([[True, False]])

    with pytest.raises(ValueError, match="nonempty support"):
        validate_sampling_mask_for_active_tokens(
            SamplingMask(
                token_ids=torch.zeros((1, 2, 2), dtype=torch.int32),
                sizes=torch.zeros((1, 2), dtype=torch.int32),
            ),
            target,
            active,
        )

    with pytest.raises(ValueError, match="does not contain"):
        validate_sampling_mask_for_active_tokens(
            SamplingMask(
                token_ids=torch.tensor([[[1, 2], [0, 0]]], dtype=torch.int32),
                sizes=torch.tensor([[2, 0]], dtype=torch.int32),
            ),
            target,
            active,
        )


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
