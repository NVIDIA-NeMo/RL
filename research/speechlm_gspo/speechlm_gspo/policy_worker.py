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

import ray

from nemo_rl.models.policy.utils import get_runtime_env_for_policy_worker
from nemo_rl.models.policy.workers.automodel_policy_worker import (
    AutomodelPolicyWorkerImpl,
)
from speechlm_gspo.model import register_speechlm

# Ray imports this module in every worker process before constructing the
# actor, so the generic Automodel setup can resolve the native model by name.
register_speechlm()


@ray.remote(
    runtime_env=get_runtime_env_for_policy_worker("speechlm_policy_worker")
)  # pragma: no cover
class SpeechLMPolicyWorker(AutomodelPolicyWorkerImpl):
    """Generic Automodel policy worker in the SpeechLM policy environment."""
