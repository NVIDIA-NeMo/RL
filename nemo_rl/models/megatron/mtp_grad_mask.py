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

"""Mask the MTP loss gradient of sequences that in-loss filtering rejects.

The MTP loss mask is fixed before the forward pass, but in-loss sequence filtering
decides which rows to keep only inside the policy loss. Megatron's
``MTPLossGradMask`` lets that decision reach the MTP backward: the holder is
registered for one microbatch's forward and filled once its loss has run.
"""

from typing import Optional

import torch

try:
    from megatron.core.transformer.multi_token_prediction import (
        MTPLossAutoScaler,
        MTPLossGradMask,
    )

    HAVE_MTP_GRAD_MASK = True
except ImportError:
    HAVE_MTP_GRAD_MASK = False

MTP_ROW_IDS_KEY = "mtp_row_ids"


def mtp_row_ids_like(token_mask: torch.Tensor) -> torch.Tensor:
    """Row index + 1 at every position of each row; packing pads it with 0."""
    batch_size = token_mask.shape[0]
    row_ids = torch.arange(
        1, batch_size + 1, device=token_mask.device, dtype=token_mask.dtype
    )
    return row_ids.unsqueeze(-1).expand_as(token_mask).contiguous()


def require_mtp_grad_mask_support() -> None:
    if not HAVE_MTP_GRAD_MASK:
        raise ValueError(
            "loss_fn.seq_logprob_error_in_loss with MTP needs a Megatron-LM that "
            "provides MTPLossGradMask, so MTP trains only on the sequences the loss "
            "keeps. Upgrade Megatron-LM or set mtp_num_layers=0."
        )


class MTPRowGradMask:
    """One microbatch's MTP gradient mask, keyed by row.

    ``row_ids`` is ``mtp_row_ids`` after microbatch processing, so it has the
    packing and CP layout of the MTP loss mask. ``fill`` maps the loss's per-row
    keep mask (in the microbatch's row order) onto that layout.
    """

    def __init__(self, row_ids: torch.Tensor) -> None:
        require_mtp_grad_mask_support()
        self.row_ids = row_ids
        self.holder = MTPLossGradMask()

    def register(self) -> None:
        MTPLossAutoScaler.set_grad_mask_provider(lambda: self.holder)

    @staticmethod
    def unregister() -> None:
        MTPLossAutoScaler.set_grad_mask_provider(None)

    def fill(self, keep_mask: Optional[list[float]]) -> None:
        if keep_mask is None:
            raise RuntimeError(
                "MTP gradient masking needs the in-loss filter's per-row keep mask, "
                "but the loss returned none."
            )
        num_rows = int(self.row_ids.max().item()) if self.row_ids.numel() else 0
        if len(keep_mask) < num_rows:
            raise RuntimeError(
                f"The loss kept {len(keep_mask)} rows but the MTP layout has "
                f"{num_rows}."
            )
        # Index 0 is packing padding, which carries no MTP loss.
        keep = torch.tensor(
            [0.0, *keep_mask], device=self.row_ids.device, dtype=torch.float32
        )
        self.holder.mask = keep[self.row_ids.long()]
