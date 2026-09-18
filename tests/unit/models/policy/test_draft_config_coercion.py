# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
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
from pydantic import ValidationError

from nemo_rl.models.policy.draft_config import (
    DFlashDraftConfig,
    DSparkDraftConfig,
    Eagle3DraftConfig,
    coerce_draft_config,
    draft_refit_enabled,
)


@pytest.mark.parametrize(
    ("method", "extra", "expected"),
    [
        ("eagle3", {}, Eagle3DraftConfig),
        ("dflash", {"gamma": 5}, DFlashDraftConfig),
        ("dspark", {"block_size": 6}, DSparkDraftConfig),
    ],
)
def test_raw_mapping_preserves_speculator_contract(
    method: str, extra: dict[str, int], expected: type
) -> None:
    raw = {"speculator_type": method, "enabled": True, **extra}
    if method != "eagle3":
        raw.update(
            anchors_per_sample=2,
            mask_token_id=1,
            target_hidden_state_layer_ids=[1, 3],
        )
    model = coerce_draft_config(raw)
    assert isinstance(model, expected)
    assert coerce_draft_config(model) is model
    assert draft_refit_enabled(raw)
    assert not draft_refit_enabled({**raw, "enabled": False})


def test_legacy_and_absent_draft_configs() -> None:
    assert coerce_draft_config(None) is None
    assert not draft_refit_enabled(None)
    assert isinstance(coerce_draft_config({}), Eagle3DraftConfig)
    assert not draft_refit_enabled({})


def test_unknown_speculator_is_rejected() -> None:
    with pytest.raises(ValidationError):
        coerce_draft_config({"speculator_type": "unknown", "enabled": True})
