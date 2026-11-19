# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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


@pytest.mark.mcore
def test_register_draft_grad_norm_group_is_idempotent_and_preserves_existing():
    from megatron.core.optimizer import optimizer as mcore_opt

    from nemo_rl.models.megatron.draft.utils import (
        DRAFT_GRAD_NORM_GROUP,
        register_draft_grad_norm_group,
    )

    original = mcore_opt.SEPARATE_GRAD_NORM_GROUPS
    try:
        register_draft_grad_norm_group()
        after_first = mcore_opt.SEPARATE_GRAD_NORM_GROUPS
        assert DRAFT_GRAD_NORM_GROUP in after_first
        assert "mtp" in after_first  # not overwritten
        register_draft_grad_norm_group()
        assert mcore_opt.SEPARATE_GRAD_NORM_GROUPS == after_first  # no-op
    finally:
        mcore_opt.SEPARATE_GRAD_NORM_GROUPS = original


@pytest.mark.mcore
def test_draft_model_detached():
    import torch.nn as nn

    from nemo_rl.models.megatron.draft.utils import draft_model_detached

    chunk = nn.Module()
    chunk.decoder = nn.Linear(2, 2)
    draft = nn.Module()
    draft.eagle_module = nn.Linear(2, 2)
    chunk.draft_model = draft

    with draft_model_detached([chunk]):
        assert not hasattr(chunk, "draft_model")
        assert all(
            not name.startswith("draft_model.") for name, _ in chunk.named_parameters()
        )
    assert chunk.draft_model is draft
    assert any(name.startswith("draft_model.") for name, _ in chunk.named_parameters())

    with pytest.raises(RuntimeError, match="boom"):
        with draft_model_detached([chunk]):
            raise RuntimeError("boom")
    assert chunk.draft_model is draft

    # No draft attached: a plain no-op.
    with draft_model_detached([nn.Module()]):
        pass
