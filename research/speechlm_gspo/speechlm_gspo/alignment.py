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

"""Map audio-expanded hidden states back to the canonical text-token layout."""

import torch


def align_to_text(
    hidden_states: torch.Tensor,
    native_targets: torch.Tensor,
    input_ids: torch.Tensor,
    target_mask: torch.Tensor,
) -> torch.Tensor:
    """Place each native next-token state at its canonical text position.

    Speech expands every audio placeholder and left-pads rows, so position
    ``p`` of ``hidden_states`` predicts ``native_targets[:, p]``. Text position
    ``i`` predicts ``input_ids[:, i + 1]``. Positions not selected by
    ``target_mask[:, 1:]`` (padding, placeholders) and the last position
    stay zero. The native targets of each row must equal its selected text
    targets in order; any other layout raises instead of misaligning logits.
    """
    if hidden_states.shape[:2] != native_targets.shape:
        raise ValueError("Native targets and hidden-state positions differ")
    if input_ids.shape != target_mask.shape or input_ids.shape[0] != len(
        native_targets
    ):
        raise ValueError("Text target mask and native batch dimensions differ")
    native_mask = native_targets != -100
    text_mask = target_mask[:, 1:].bool()
    if not (
        torch.equal(native_mask.sum(1), text_mask.sum(1))
        and torch.equal(native_targets[native_mask], input_ids[:, 1:][text_mask])
    ):
        raise ValueError("Expanded audio targets do not match the text tokens")
    aligned = hidden_states.new_zeros(*input_ids.shape, hidden_states.shape[-1])
    aligned[:, :-1][text_mask] = hidden_states[native_mask]
    return aligned
