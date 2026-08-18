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

"""Single-branch rollout tree: one child per turn, no forking, no compaction."""

from nemo_rl.environments.central_agent_helpers.rollout_tree.base import BaseRolloutTree


class LinearRolloutTree(BaseRolloutTree):
    """A chain of TurnNodes. Every turn extends the one branch.

    Neither branching nor compaction happens, so the branch only ever grows and
    the token contiguity that ``_postprocess_nemo_gym_to_nemo_rl_result`` asserts
    holds: turn k's ``prompt_token_ids`` starts with every token seen so far. A
    compacting subclass breaks that assumption and needs a postprocessor that
    emits explicit per-token train masks instead (tracked as a follow-up).
    """

    def branch_and_compact_context(self) -> None:
        self.start_new_turn(kind="turn")
