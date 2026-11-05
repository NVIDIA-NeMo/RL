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


"""Install policy timing after normal worker initialization has assigned its GPU."""

from typing import Any

import ray

from nemo_rl.models.policy.utils import get_runtime_env_for_policy_worker
from nemo_rl.models.policy.workers import automodel_policy_worker
from nemo_rl.models.policy.workers.automodel_policy_worker import (
    AutomodelPolicyWorkerImpl,
)
from validation.pr3.audit_workers import timed_gpu_call


@ray.remote(runtime_env=get_runtime_env_for_policy_worker("automodel_policy_worker"))
class AuditedPolicyWorker(AutomodelPolicyWorkerImpl):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        original = automodel_policy_worker.automodel_forward_backward

        def backward(*call_args: Any, **call_kwargs: Any) -> Any:
            if call_kwargs.get("forward_only"):
                return original(*call_args, **call_kwargs)
            return timed_gpu_call("training_gpu", original, *call_args, **call_kwargs)

        automodel_policy_worker.automodel_forward_backward = backward
