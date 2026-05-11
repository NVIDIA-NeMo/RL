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
import os
from typing import List, Optional

from psutil import Process
from pydantic import BaseModel, Field

# Commented out: kept for reference. The Ray plasma snapshot it powered was
# disabled in get_snapshot_str() to avoid 60s gRPC DEADLINE_EXCEEDED stalls
# under heavy plasma load. Re-import and restore the call if you want the
# Plasma summary back for debugging — see the comment in get_snapshot_str().
# from ray.scripts.scripts import memory_summary


class MemoryTrackerDataPoint(BaseModel):
    stage: str
    memory_used_before_stage_gb: float
    variables_before_stage: List[str]

    memory_used_after_stage_gb: Optional[float] = None
    variables_after_stage: Optional[List[str]] = None

    @property
    def mem_used_diff_gb(self) -> float:
        return self.memory_used_after_stage_gb - self.memory_used_before_stage_gb

    @property
    def new_variables(self) -> List[str]:
        return [
            v
            for v in self.variables_after_stage
            if v not in self.variables_before_stage
        ]

    def get_snapshot_str(self) -> str:
        # Ray's FormatGlobalMemoryInfo gRPC has a hardcoded 60s deadline and
        # has been observed to time out (DEADLINE_EXCEEDED) under heavy plasma
        # load between training stages, parking the driver for up to 60s and
        # deflating performance/tokens_per_sec for that step. The Plasma
        # snapshot itself is read-only telemetry that never touches actor
        # state, training tensors, or any metric, and it duplicates info
        # already visible via `ray status` and the wandb GPU/RAM panels —
        # so we skip it entirely. The local CPU tracker (driver RSS + new
        # variables) is unaffected.
        # To restore: uncomment the line below and the `memory_summary` import
        # at the top of this file, and remove the placeholder assignment.
        # ray_memory_summary = memory_summary(stats_only=True, num_entries=5)
        ray_memory_summary = "<skipped: Ray plasma snapshot disabled to avoid monitor timeouts>"
        return f"""💭 Driver CPU memory tracker for {self.stage}:
- Mem usage before                  {self.memory_used_before_stage_gb:>7.2f} GB
- Mem usage after                   {self.memory_used_after_stage_gb:>7.2f} GB
- Mem usage diff (after - before)   {self.mem_used_diff_gb:>+7.2f} GB
- New variables: {self.new_variables}

⚡️ Ray memory snapshot:
{ray_memory_summary}"""


class MemoryTracker(BaseModel):
    data_points: List[MemoryTrackerDataPoint] = Field(default_factory=list)

    def model_post_init(self, context):
        self._process = Process(os.getpid())
        return super().model_post_init(context)

    def snapshot_start_of_stage(
        self, new_stage: str, all_current_variables: List[str]
    ) -> None:
        mem_info = self._process.memory_info()
        current_mem_used_gb: float = mem_info.rss / (1024**3)

        if self.data_points:
            last_data_point = self.data_points[-1]
            last_data_point.memory_used_after_stage_gb = current_mem_used_gb
            last_data_point.variables_after_stage = all_current_variables

            print(last_data_point.get_snapshot_str())

        self.data_points.append(
            MemoryTrackerDataPoint(
                stage=new_stage,
                memory_used_before_stage_gb=current_mem_used_gb,
                variables_before_stage=all_current_variables,
            )
        )
