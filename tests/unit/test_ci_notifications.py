# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

from pathlib import Path

import yaml

WORKFLOW = Path(__file__).resolve().parents[2] / ".github/workflows/cicd-main.yml"
ACTION = "NVIDIA-NeMo/FW-CI-templates/.github/actions/notify-ci-failure@631c404d00a9e60afc591cd071d35d2e18f82fc6"


def test_nightly_notification_reports_all_direct_job_results() -> None:
    jobs = yaml.safe_load(WORKFLOW.read_text())["jobs"]
    notify = jobs["notify-nightly-failure"]
    assert set(notify["needs"]) == set(jobs) - {"notify-nightly-failure"}
    assert notify["environment"] == {"name": "main", "deployment": False}
    assert notify["runs-on"] == "ubuntu-latest"
    assert notify["permissions"] == {}
    (step,) = notify["steps"]
    assert step["uses"] == ACTION
    assert step["with"] == {
        "needs-json": "${{ toJSON(needs) }}",
        "webhook": "${{ secrets.SLACK_GITHUB_CI_WEBHOOK }}",
    }
    # No failed-run checkout, hand-built payload, or unchecked curl remains.
    assert "run" not in step
    assert "env" not in step


def test_nightly_notification_ignores_cancelled_and_non_scheduled_runs() -> None:
    jobs = yaml.safe_load(WORKFLOW.read_text())["jobs"]
    condition = jobs["notify-nightly-failure"]["if"]
    assert "always() && !cancelled() && failure()" in condition
    assert "github.repository == 'NVIDIA-NeMo/RL'" in condition
    assert "github.event_name == 'schedule'" in condition
    assert "github.ref_name == github.event.repository.default_branch" in condition
    # Setup output/aggregate-gate state cannot hide a failed dependency.
    assert "needs." not in condition
    assert "workflow_dispatch" not in condition
    assert "push" not in condition
