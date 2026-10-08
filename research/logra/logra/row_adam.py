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

"""Adam-style second moments shared across each sketch row; no first moment."""

import torch
from torch import Tensor


@torch.no_grad()
def row_adam_direction(
    sketch: Tensor,
    second_moment: Tensor,
    *,
    step: int,
    beta2: float,
    epsilon: float,
    inplace: bool = False,
) -> Tensor:
    """Normalize each row by its bias-corrected running RMS.

    The state is indexed by the original weight's output rows, not by the
    changing projection coordinates. Ordinary learning rate controls the update;
    there is no layer-norm restoration or external update-magnitude controller.
    With ``inplace`` the sketch itself is overwritten with the direction, which
    avoids a full-size temporary when the sketch is no longer needed.
    """
    if step < 1 or not 0 <= beta2 < 1 or epsilon <= 0:
        raise ValueError("Invalid RowAdam step, decay, or epsilon")
    if second_moment.shape != (sketch.shape[0], 1):
        raise ValueError("RowAdam requires one second moment per output row")
    if not torch.isfinite(sketch).all():
        raise FloatingPointError("Nonfinite gradient sketch")
    second_moment.mul_(beta2).add_(
        sketch.square().mean(dim=1, keepdim=True), alpha=1 - beta2
    )
    denominator = (second_moment / (1 - beta2**step)).sqrt_().add_(epsilon)
    if inplace:
        return sketch.div_(denominator)
    return sketch / denominator
