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

"""Preserve RADIO's CPE coordinates without constructing an unused square."""

from functools import wraps
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch


def cropped_cpe_position_embeddings(
    position_embeddings: "torch.Tensor",
    source_rows: int,
    source_columns: int,
    input_dims: tuple[int, int],
) -> "torch.Tensor":
    """Compute the top-left crop of the original square bilinear resize.

    RADIO first resizes its positional grid to ``max(input_dims)`` squared,
    then keeps ``input_dims``. Resizing directly to ``input_dims`` would use
    different coordinates. Explicit ATen scales retain the square's mapping
    while allocating only the crop. Positions still use FP32 interpolation
    followed by the original dtype conversion.
    """
    import torch

    height, width = input_dims
    square_size = max(height, width)
    positions = position_embeddings.reshape(1, source_rows, source_columns, -1).permute(
        0, 3, 1, 2
    )
    # Some interpolation kernels select an identity path when an output axis
    # equals its input axis, even with explicit scales. One extra output cell
    # prevents that shortcut when the square actually uses a different scale.
    output_height = height + (height == source_rows and square_size != source_rows)
    output_width = width + (width == source_columns and square_size != source_columns)
    resized = torch.ops.aten.upsample_bilinear2d.default(
        positions.float(),
        [output_height, output_width],
        False,
        square_size / source_rows,
        square_size / source_columns,
    ).to(position_embeddings.dtype)
    return resized[..., :height, :width].flatten(2).permute(0, 2, 1)


def install_radio_cpe_crop() -> None:
    """Install the bounded resize in this engine worker, without editing files."""
    from vllm.model_executor.models.radio import ViTPatchGenerator

    original = ViTPatchGenerator._get_pos_embeddings
    if getattr(original, "_nemo_rl_cpe_crop", False):
        return

    @wraps(original)
    def get_pos_embeddings(self, batch_size, input_dims):
        if (
            not self.cpe_mode
            or input_dims[0] == input_dims[1]
            or (self.num_rows, self.num_cols) == input_dims
        ):
            return original(self, batch_size, input_dims)
        return cropped_cpe_position_embeddings(
            self.pos_embed, self.num_rows, self.num_cols, input_dims
        )

    get_pos_embeddings._nemo_rl_cpe_crop = True
    ViTPatchGenerator._get_pos_embeddings = get_pos_embeddings
