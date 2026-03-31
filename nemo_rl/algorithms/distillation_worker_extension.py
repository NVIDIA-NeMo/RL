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

"""Worker extension for distillation that adds EMA state dict methods.

Uses NeMo RL's worker_extension_cls_fqn pattern to add distillation-specific
methods without modifying the base worker classes.
"""

import ray
import torch

from nemo_rl.models.policy.utils import get_runtime_env_for_policy_worker
from nemo_rl.models.policy.workers.dtensor_policy_worker_v2 import (
    DTensorPolicyWorkerV2Impl,
)


class DistillationWorkerV2Impl(DTensorPolicyWorkerV2Impl):
    """DTensor V2 worker extension with EMA state dict communication methods."""

    def get_model_state_dict(self) -> dict[str, torch.Tensor]:
        """Get the model's state dict for EMA teacher updates."""
        return self.model.state_dict()

    def load_model_state_dict(self, state_dict: dict[str, torch.Tensor]) -> None:
        """Load a state dict into the model for EMA teacher initialization."""
        self.model.load_state_dict(state_dict, strict=True)

    def update_model_with_ema(
        self, student_state_dict: dict[str, torch.Tensor], ema_decay: float
    ) -> None:
        """Update model parameters using EMA from student state dict.

        Formula: teacher_param = ema_decay * teacher_param + (1 - ema_decay) * student_param
        """
        with torch.no_grad():
            teacher_state_dict = self.model.state_dict()
            for name, teacher_param in teacher_state_dict.items():
                if name in student_state_dict:
                    student_param = student_state_dict[name]
                    if student_param.device != teacher_param.device:
                        student_param = student_param.to(teacher_param.device)
                    teacher_param.mul_(ema_decay).add_(
                        student_param, alpha=(1.0 - ema_decay)
                    )


# Ray actors can't be subclassed, so we need a separate @ray.remote decorated
# class that inherits from the Impl. This is the standard NeMo RL pattern
# (see DTensorPolicyWorkerV2 and the template_project example).
@ray.remote(
    runtime_env=get_runtime_env_for_policy_worker("dtensor_policy_worker_v2")
)  # pragma: no cover
class DistillationWorkerV2(DistillationWorkerV2Impl):
    pass
