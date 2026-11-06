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
from speechlm_gspo.alignment import align_to_text

PAD, AUDIO = 0, 9
# Speech expands row 0 to [1, A, A, 2, 3] and row 1 to [PAD, 4, A, 5, PAD]:
# a row keeps its own right padding and is left-padded to the batch length.
INPUT_IDS = torch.tensor([[1, AUDIO, 2, 3], [4, AUDIO, 5, PAD]])
TARGET_MASK = torch.tensor([[0, 0, 1, 1], [0, 0, 1, 0]], dtype=torch.bool)
EXPANDED_TARGETS = [[-100, -100, -100, 2, 3], [-100, -100, -100, 5, -100]]
NATIVE_TARGETS = torch.tensor([row[1:] for row in EXPANDED_TARGETS])


def _hidden(requires_grad=False):
    """Hidden state at native position ``p`` of row ``b`` is ``10 * b + p``."""
    values = torch.arange(4.0) + torch.tensor([[0.0], [10.0]])
    return values[..., None].repeat(1, 1, 2).requires_grad_(requires_grad)


def test_alignment_places_native_states_at_text_positions():
    aligned = align_to_text(_hidden(), NATIVE_TARGETS, INPUT_IDS, TARGET_MASK)
    # Text position i predicts token i + 1: the placeholder predicts the first
    # response token from the last audio frame, and so on.
    expected = torch.tensor([[0.0, 2.0, 3.0, 0.0], [0.0, 12.0, 0.0, 0.0]])
    torch.testing.assert_close(aligned, expected[..., None].expand(-1, -1, 2))


def test_alignment_gradient_reaches_only_selected_states():
    hidden = _hidden(requires_grad=True)
    align_to_text(hidden, NATIVE_TARGETS, INPUT_IDS, TARGET_MASK).sum().backward()
    selected = (NATIVE_TARGETS != -100).float()
    torch.testing.assert_close(hidden.grad, selected[..., None].expand(-1, -1, 2))


@pytest.mark.parametrize(
    "native_targets",
    [
        torch.tensor([[-100, -100, 2, 4], [-100, -100, -100, 5]]),
        torch.tensor([[-100, -100, -100, 2], [-100, -100, 3, 5]]),
    ],
    ids=["wrong-token", "row-shift"],
)
def test_alignment_rejects_mismatched_targets(native_targets):
    with pytest.raises(ValueError, match="do not match"):
        align_to_text(_hidden(), native_targets, INPUT_IDS, TARGET_MASK)


def test_alignment_matches_speech_target_layout():
    salm = pytest.importorskip("nemo.collections.speechlm2.models.salm")
    embed = torch.nn.Embedding(10, 2)
    audio = [torch.zeros(2, 2), torch.zeros(1, 2)]
    _, target_ids, _ = salm.replace_placeholders_and_build_targets(
        input_ids=INPUT_IDS,
        embeds=embed(INPUT_IDS.masked_fill(INPUT_IDS == AUDIO, PAD)),
        padding_id=PAD,
        placeholder_id=AUDIO,
        replacements=audio,
        target_ids=INPUT_IDS.where(TARGET_MASK, -100),
    )
    torch.testing.assert_close(target_ids, torch.tensor(EXPANDED_TARGETS))
