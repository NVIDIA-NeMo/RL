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

from nemo_rl.utils.tensor_ops import masked_max, masked_min, pad_and_concat


@pytest.mark.parametrize("trailing_shape", [(), (3,)], ids=["2d", "3d"])
def test_pad_and_concat_pads_seq_dim_only_and_keeps_dtype(trailing_shape):
    """Every chunk is right-padded on dim 1 to target_len, then stacked on dim 0."""
    short = torch.full((2, 3, *trailing_shape), 7, dtype=torch.long)
    full = torch.full((1, 5, *trailing_shape), 9, dtype=torch.long)

    out = pad_and_concat([short, full], target_len=5, pad_value=-1)

    assert out.shape == (3, 5, *trailing_shape)
    assert out.dtype == torch.long
    assert torch.all(out[:2, :3] == 7)
    assert torch.all(out[:2, 3:] == -1)  # only the sequence dim is padded
    assert torch.all(out[2] == 9)  # a chunk already at target_len is untouched


def test_masked_min_max_reduce_the_masked_entries_only() -> None:
    values = torch.tensor([[0.5, 9.0], [2.0, -7.0]])
    mask = torch.tensor([[1.0, 0.0], [1.0, 0.0]])

    assert masked_min(values, mask).item() == 0.5
    assert masked_max(values, mask).item() == 2.0


@pytest.mark.parametrize("shape", [(0, 4), (2, 3)], ids=["empty", "nothing_selected"])
def test_masked_min_max_with_nothing_to_reduce(shape: tuple[int, int]) -> None:
    values = torch.ones(shape, dtype=torch.float64)
    mask = torch.zeros(shape)

    lowest, highest = masked_min(values, mask), masked_max(values, mask)

    assert lowest.item() == float("inf") and highest.item() == float("-inf")
    assert lowest.shape == highest.shape == ()
    assert lowest.dtype == highest.dtype == torch.float64
