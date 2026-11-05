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
import torch


def pad_and_concat(
    tensors: list[torch.Tensor], *, target_len: int, pad_value: float = 0.0
) -> torch.Tensor:
    """Right-pad each tensor along dim 1 to `target_len`, then concatenate along dim 0.

    Args:
        tensors: Tensors with the sequence on dim 1.
        target_len: Sequence length every tensor is padded up to.
        pad_value: Fill value for the padded positions.

    Returns:
        The padded tensors concatenated along dim 0.
    """
    padded: list[torch.Tensor] = []
    for t in tensors:
        padding_needed = target_len - t.shape[1]
        if padding_needed > 0:
            # F.pad's spec runs from the last dim backwards; leave trailing dims alone.
            pad_spec = [0, 0] * (t.dim() - 2) + [0, padding_needed]
            t = torch.nn.functional.pad(t, pad_spec, mode="constant", value=pad_value)
        padded.append(t)
    return torch.cat(padded, dim=0)
