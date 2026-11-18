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

from typing import TYPE_CHECKING

from nemo_rl.algorithms.loss.loss_functions import ClippedPGLossConfig, ClippedPGLossFn

if TYPE_CHECKING:
    # Import-time only: nemo_rl.algorithms.opd imports the data plane.
    from nemo_rl.algorithms.opd import OnPolicyDistillationFullConfig

# The ``loss_fn`` block of the policy-gradient trainers. It has one member
# today; objective-specific configs join it as a union discriminated on
# ``kind``, each carrying only the settings its objective supports.
PolicyGradientLossConfig = ClippedPGLossConfig


def build_policy_gradient_loss(
    cfg: PolicyGradientLossConfig,
    *,
    use_fused_linear_logprobs: bool = False,
    opd_full: "OnPolicyDistillationFullConfig | None" = None,
    seq_logprob_error_threshold: float | None = None,
) -> ClippedPGLossFn:
    """Build the loss implementation selected by a policy-gradient loss config.

    The trainers construct their actor loss here, so a new config variant is
    dispatched in one place instead of at every trainer.
    """
    return ClippedPGLossFn(
        cfg,
        use_fused_linear_logprobs=use_fused_linear_logprobs,
        opd_full=opd_full,
        seq_logprob_error_threshold=seq_logprob_error_threshold,
    )
