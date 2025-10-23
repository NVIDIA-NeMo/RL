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

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class JobDependencies:
    """Job dependency configuration for GitLab pipeline generation.

    This class defines dependencies between stages and job groups for organizing
    test execution in CI/CD pipelines.

    Attributes:
        stages: Dict mapping stage names to their dependencies
                Format: {"stage_name": {"depends_on": ["other_stage"], "needs": ["other_stage"]}}
        job_groups: Dict mapping job group names to their metadata
                    Format: {"job_group_name": {"stage": "stage_name", "depends_on": ["other_job_group"], "needs": ["job1", "job2"]}}

    Example:
        # Default pipeline structure (training -> validation)
        JobDependencies(
            stages={
                "training": {"depends_on": [], "needs": []},
                "validation": {"depends_on": ["training"], "needs": []}
            },
            job_groups={
                "job_1": {"stage": "validation", "depends_on": [], "needs": []},
                "job_2": {"stage": "validation", "depends_on": ["job_1"], "needs": ["job_1"]}
            }
        )
    """

    stages: Dict[str, Dict[str, List[str]]] = field(
        default_factory=lambda: {
            "training": {"depends_on": [], "needs": []},
            "validation": {"depends_on": ["training"], "needs": []},
        }
    )
    job_groups: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    def get_stage_dependencies(self, stage_name: str) -> List[str]:
        """Get dependencies for a specific stage.

        Args:
            stage_name: Name of the stage

        Returns:
            List of stage names this stage depends on
        """
        return self.stages.get(stage_name, {}).get("depends_on", [])

    def get_job_group_dependencies(self, job_group_name: str) -> List[str]:
        """Get dependencies for a specific job group.

        Args:
            job_group_name: Name of the job group

        Returns:
            List of job group names this job group depends on
        """
        return self.job_groups.get(job_group_name, {}).get("depends_on", [])

    def get_job_group_stage(self, job_group_name: str) -> Optional[str]:
        """Get the stage for a specific job group.

        Args:
            job_group_name: Name of the job group

        Returns:
            Stage name or None if not defined
        """
        return self.job_groups.get(job_group_name, {}).get("stage")

    def get_stage_needs(self, stage_name: str) -> List[str]:
        """Get the 'needs' list for a specific stage.

        Args:
            stage_name: Name of the stage

        Returns:
            List of job/stage names this stage needs
        """
        return self.stages.get(stage_name, {}).get("needs", [])

    def get_job_group_needs(self, job_group_name: str) -> List[str]:
        """Get the 'needs' list for a specific job group.

        Args:
            job_group_name: Name of the job group

        Returns:
            List of job names this job group needs
        """
        return self.job_groups.get(job_group_name, {}).get("needs", [])
