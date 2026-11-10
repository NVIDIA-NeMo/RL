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
"""A block drafter must be fed the target layers its checkpoint names."""

import json

import pytest

from nemo_rl.models.megatron.draft.utils import resolve_draft_aux_layer_indices


class _Cfg:
    def __init__(self, model_name=None, aux_layer_indices=None):
        self.model_name = model_name
        self.aux_layer_indices = aux_layer_indices


def _write(tmp_path, payload):
    (tmp_path / "config.json").write_text(json.dumps(payload))
    return str(tmp_path)


def test_reads_target_layer_ids_from_the_drafter_checkpoint(tmp_path):
    """The released LiLiCorr for Super 3.5 names [1, 22, 43, 64, 85] over 88 layers.

    Without this the capture falls back to the eagle3 heuristic, which picks three
    different layers -- the drafter is then fed features it was not built to read.
    """
    path = _write(
        tmp_path,
        {
            "num_target_layers": 88,
            "dflash_config": {"target_layer_ids": [1, 22, 43, 64, 85]},
        },
    )
    assert resolve_draft_aux_layer_indices(_Cfg(path)) == (1, 22, 43, 64, 85)


def test_explicit_config_wins_over_the_checkpoint(tmp_path):
    path = _write(
        tmp_path, {"dflash_config": {"target_layer_ids": [1, 22, 43, 64, 85]}}
    )
    assert resolve_draft_aux_layer_indices(_Cfg(path, [3, 7])) == (3, 7)


@pytest.mark.parametrize(
    "payload",
    [{}, {"dflash_config": {}}, {"dflash_config": {"target_layer_ids": []}}],
    ids=["no-dflash-config", "empty-dflash-config", "empty-ids"],
)
def test_returns_none_when_nothing_pins_the_layers(tmp_path, payload):
    """None preserves the eagle3 fallback, so eagle3 runs are untouched."""
    assert resolve_draft_aux_layer_indices(_Cfg(_write(tmp_path, payload))) is None


def test_none_when_no_model_name_and_no_config_file(tmp_path):
    assert resolve_draft_aux_layer_indices(_Cfg(None)) is None
    assert resolve_draft_aux_layer_indices(_Cfg(str(tmp_path))) is None


def test_rejects_ids_the_target_cannot_have(tmp_path):
    """A drafter naming layer 99 of an 88-layer target does not match that target."""
    path = _write(
        tmp_path,
        {"num_target_layers": 88, "dflash_config": {"target_layer_ids": [1, 99]}},
    )
    with pytest.raises(ValueError, match="do not agree"):
        resolve_draft_aux_layer_indices(_Cfg(path))


def test_rejects_negative_ids(tmp_path):
    path = _write(tmp_path, {"dflash_config": {"target_layer_ids": [-1, 4]}})
    with pytest.raises(ValueError, match="negative"):
        resolve_draft_aux_layer_indices(_Cfg(path))
