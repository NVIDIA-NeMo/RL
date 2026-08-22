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

from nemo_rl.models.policy.workers.megatron_policy_worker import (
    draft_local_update_outcome,
)


def test_zero_anchor_model_is_structural_owner_without_draft_payload() -> None:
    draft_model = object()

    local_owner, local_success = draft_local_update_outcome(
        draft_model=draft_model,
        update_successful=True,
    )

    assert local_owner is True
    assert local_success is True


def test_clip_grad_zero_does_not_require_draft_grad_norm_for_success() -> None:
    local_owner, local_success = draft_local_update_outcome(
        draft_model=object(),
        update_successful=True,
    )

    assert local_owner is True
    assert local_success is True


def test_optimizer_failure_marks_structural_owner_unsuccessful() -> None:
    local_owner, local_success = draft_local_update_outcome(
        draft_model=object(),
        update_successful=False,
    )

    assert local_owner is True
    assert local_success is False
