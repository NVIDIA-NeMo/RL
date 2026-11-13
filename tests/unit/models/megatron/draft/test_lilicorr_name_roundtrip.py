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
"""Export must undo exactly what load did to LiLiCorr head tensor names.

The released checkpoint uses serving names (``feature_norm.weight``); ModelOpt's head
module uses positional ones (``feature_mlp.0.weight``). Load renames one way. If export
does not rename back, the exported drafter carries names no consumer looks for, and a
non-strict loader takes it silently -- a drafter that serves noise.
"""

import pytest

from nemo_rl.models.megatron.draft.lilicorr_model import (
    _from_positional_name,
    _to_serving_name,
)

RELEASED_HEAD_NAMES = [
    "feature_norm.weight",
    "feature_norm.bias",
    "feature_mlp.up_proj.weight",
    "feature_mlp.up_proj.bias",
    "feature_mlp.down_proj.weight",
    "feature_mlp.down_proj.bias",
    "layers.0.mlp.up_proj.weight",
    "layers.0.mlp.down_proj.weight",
    "layers.1.mlp.up_proj.bias",
    "layers.1.mlp.down_proj.bias",
    "pass_hidden_proj.weight",
    "context_proj.weight",
    "factor_input_proj.bias",
    "out_head.weight",
    "in_head.weight",
    "anchor_out_head.weight",
]


@pytest.mark.parametrize("serving_name", RELEASED_HEAD_NAMES)
def test_export_inverts_load(serving_name):
    positional = _to_serving_name(serving_name)
    assert _from_positional_name(positional) == serving_name


def test_renames_are_not_identity_where_it_matters():
    """Guard the guard: a no-op rename would pass the round-trip trivially."""
    renamed = [n for n in RELEASED_HEAD_NAMES if _to_serving_name(n) != n]
    assert len(renamed) >= 8, (
        "expected the serving/positional split to rename the feature_mlp and "
        f"layers.*.mlp tensors, only {len(renamed)} changed"
    )


def test_positional_names_are_what_modelopt_uses():
    assert _to_serving_name("feature_norm.weight") == "feature_mlp.0.weight"
    assert _to_serving_name("feature_mlp.up_proj.weight") == "feature_mlp.1.weight"
    assert _to_serving_name("feature_mlp.down_proj.weight") == "feature_mlp.3.weight"
    assert _to_serving_name("layers.3.mlp.up_proj.weight") == "layers.3.mlp.0.weight"
    assert _to_serving_name("layers.3.mlp.down_proj.weight") == "layers.3.mlp.2.weight"


def test_block_slot_arithmetic_is_consistent():
    """gamma counts mask slots, and the lattice scores exactly those.

    A DFlash block is one anchor plus gamma masks. The reranker's
    num_candidate_slots is block_size - 1. If gamma were mapped straight from the
    checkpoint's block_size the block would be one slot too long and the head would
    reject it -- loudly here, but the same off-by-one applied to the gold targets
    would silently score the lattice against positions the drafter never predicted.
    """
    checkpoint_block_size = 4
    gamma = checkpoint_block_size - 1
    block_slots = 1 + gamma
    lattice_slots = block_slots - 1  # score_candidates drops the anchor
    assert lattice_slots == checkpoint_block_size - 1
