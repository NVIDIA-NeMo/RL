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

from typing import Any

import torch

from nemo_rl.models.policy.workers.dtensor_policy_worker_v2 import DTensorPolicyWorkerV2


class DTensorPolicyWorkerV2Extension(DTensorPolicyWorkerV2):
    """Example worker extension that adds a custom method callable via run_all_workers_single_data."""

    def get_worker_info(self) -> dict[str, Any]:
        """Return per-worker info. Used to demonstrate run_all_workers_single_data."""
        rank = torch.distributed.get_rank()
        device = getattr(self, "device", None)
        return {
            "rank": rank,
            "device": str(device) if device is not None else "unknown",
        }
