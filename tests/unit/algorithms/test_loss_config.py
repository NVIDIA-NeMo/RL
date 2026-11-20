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
from pydantic import TypeAdapter, ValidationError

from nemo_rl.algorithms.loss import (
    ClippedPGLossConfig,
    ClippedPGLossFn,
    PolicyGradientLossConfig,
    build_policy_gradient_loss,
)

LOSS_CONFIG = TypeAdapter(PolicyGradientLossConfig)
LEGACY_SETTINGS = {"use_cispo": True, "ratio_clip_max": 0.3, "extension_option": 7}


@pytest.mark.parametrize("tag", [{}, {"kind": "legacy_pg"}])
def test_legacy_tag_is_optional_and_never_serialized(tag):
    parsed = LOSS_CONFIG.validate_python({**tag, **LEGACY_SETTINGS})
    untagged = ClippedPGLossConfig(**LEGACY_SETTINGS)

    assert isinstance(parsed, ClippedPGLossConfig)
    assert parsed.kind == "legacy_pg"
    # Dumps feed checkpoint fingerprints, so the tag must not appear in them.
    assert "kind" not in parsed.model_dump()
    assert parsed.model_dump() == untagged.model_dump()
    assert parsed.model_dump(exclude_unset=True) == LEGACY_SETTINGS
    assert LOSS_CONFIG.validate_json(parsed.model_dump_json()) == parsed


def test_unknown_kind_is_rejected():
    with pytest.raises(ValidationError, match="legacy_pg"):
        LOSS_CONFIG.validate_python({"kind": "unknown"})


def test_factory_builds_the_clipped_pg_loss_with_trainer_settings():
    cfg = ClippedPGLossConfig(
        force_on_policy_ratio=True, seq_logprob_error_in_loss=True
    )

    loss_fn = build_policy_gradient_loss(
        cfg, use_fused_linear_logprobs=True, seq_logprob_error_threshold=0.5
    )

    assert isinstance(loss_fn, ClippedPGLossFn)
    assert loss_fn.use_fused_linear_logprobs
    assert loss_fn.seq_logprob_error_threshold == 0.5
    assert loss_fn.opd_full is None
