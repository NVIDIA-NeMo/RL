# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Profile the first training dispatch on rank0, without any artificial gate."""

import os
from pathlib import Path
from typing import Any

import ray
import torch

from nemo_rl.models.policy.utils import get_runtime_env_for_policy_worker
from nemo_rl.models.policy.workers import automodel_policy_worker as module
from nemo_rl.models.policy.workers.automodel_policy_worker import (
    AutomodelPolicyWorkerImpl,
)


@ray.remote(runtime_env=get_runtime_env_for_policy_worker("automodel_policy_worker"))
class ProfilePolicyWorker(AutomodelPolicyWorkerImpl):  # pragma: no cover
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.profiled = False
        self.profile_step = 0
        original = module.automodel_forward_backward

        def backward(*values: Any, **named: Any) -> Any:
            if (
                named.get("forward_only")
                or torch.distributed.get_rank() != 0
                or self.profiled
                or self.profile_step
                != int(os.environ.get("PR3_KL_PROFILE_ITERATION", "1"))
            ):
                return original(*values, **named)
            self.profiled = True
            with torch.profiler.profile(
                activities=[
                    torch.profiler.ProfilerActivity.CPU,
                    torch.profiler.ProfilerActivity.CUDA,
                ]
            ) as profiler:
                torch.cuda.synchronize()
                with torch.profiler.record_function("pr3_training_gpu"):
                    result = original(*values, **named)
                    torch.cuda.synchronize()
            directory = Path(os.environ["PR3_AUDIT_DIR"])
            directory.mkdir(parents=True, exist_ok=True)
            profiler.export_chrome_trace(
                str(directory / f"training_gpu-{os.getpid()}.trace.json")
            )
            return result

        module.automodel_forward_backward = backward

    def begin_train_step(self, *args: Any, **kwargs: Any) -> None:
        super().begin_train_step(*args, **kwargs)
        self.profile_step += 1
