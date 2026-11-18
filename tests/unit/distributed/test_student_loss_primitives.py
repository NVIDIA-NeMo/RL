# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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

from nemo_rl.distributed.model_utils import chunked_vocab_parallel_logsumexp


def test_chunked_vocab_logsumexp_tp1_matches_across_chunk_boundaries():
    """TP1 keeps exact forward/gradient math for rows around the fixed cap."""
    torch.manual_seed(7)
    logits = torch.randn(2, 513, 17, dtype=torch.float64, requires_grad=True)
    reference_logits = logits.detach().clone().requires_grad_(True)
    weights = torch.randn(2, 513, dtype=torch.float64)

    actual = chunked_vocab_parallel_logsumexp(
        logits,
        1.7,
        chunk_size=256,
    )
    expected = torch.logsumexp(reference_logits / 1.7, dim=-1)
    assert torch.equal(actual, expected)

    (actual * weights).sum().backward()
    (expected * weights).sum().backward()
    assert torch.equal(logits.grad, reference_logits.grad)


@pytest.mark.parametrize(
    ("temperature", "chunk_size", "match"),
    [(0.0, 256, "temperature"), (1.0, 0, "chunk_size")],
)
def test_chunked_vocab_logsumexp_rejects_invalid_bounds(
    temperature: float,
    chunk_size: int,
    match: str,
):
    logits = torch.zeros(1, 2, 3)
    with pytest.raises(ValueError, match=match):
        chunked_vocab_parallel_logsumexp(
            logits,
            temperature,
            chunk_size=chunk_size,
        )
