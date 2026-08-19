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
