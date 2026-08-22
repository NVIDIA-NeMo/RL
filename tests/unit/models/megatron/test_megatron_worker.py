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

from nemo_rl.algorithms.draft_update_schedule import DraftUpdateDecision
from nemo_rl.models.policy.workers.megatron_policy_worker import draft_execution_inputs


def _decision(*, update_requested: bool) -> DraftUpdateDecision:
    return DraftUpdateDecision(
        global_step=2,
        decision_id=3,
        update_requested=update_requested,
        draft_refit_requested=update_requested,
        reason="always" if update_requested else "none",
        observed_acceptance=None,
    )


def test_skip_omits_every_draft_execution_input() -> None:
    draft_model = object()
    draft_provider = object()

    inputs = draft_execution_inputs(
        _decision(update_requested=False), draft_model, draft_provider
    )

    assert inputs == {
        "run_draft": False,
        "enable_hidden_capture": False,
        "draft_model": None,
        "draft_provider": None,
    }


def test_requested_update_preserves_every_draft_execution_input() -> None:
    draft_model = object()
    draft_provider = object()

    inputs = draft_execution_inputs(
        _decision(update_requested=True), draft_model, draft_provider
    )

    assert inputs == {
        "run_draft": True,
        "enable_hidden_capture": True,
        "draft_model": draft_model,
        "draft_provider": draft_provider,
    }
