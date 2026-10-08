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

import json
from pathlib import Path

import pytest

from examples.nemo_gym.freeze_osworld_rlvr_band import freeze_band


def _write_source(root: Path, *, rate: float = 0.125) -> Path:
    task_id = "task_001"
    asset = root / "live" / "uploads" / "input.txt"
    asset.parent.mkdir(parents=True)
    asset.write_text("portable asset\n")

    function_root = root / "live" / "tmp_funcs" / task_id
    for role in ("getters", "metrics"):
        role_root = function_root / role
        role_root.mkdir(parents=True)
        (role_root / "custom.py").write_text(
            f"def {role}_custom(value=None):\n    return value\n"
        )

    task = {
        "id": task_id,
        "snapshot": "os",
        "instruction": "Edit the uploaded file.",
        "config": [
            {
                "type": "upload_file",
                "parameters": {
                    "files": [
                        {
                            "local_path": str(asset),
                            "path": "/home/user/input.txt",
                        }
                    ]
                },
            }
        ],
        "evaluator": {
            "func": "metrics_custom",
            "result": {"type": "getters_custom"},
        },
        "metadata": {
            "calib_reps": 8,
            "calib_pass_rate": rate,
            "new_functions_paths": [
                str(function_root / "getters" / "custom.py"),
                str(function_root / "metrics" / "custom.py"),
            ],
        },
    }
    tasks = root / "live" / "tasks"
    tasks.mkdir()
    (tasks / f"{task_id}.json").write_text(json.dumps(task))
    manifest = {
        "task_id": task_id,
        "calib_reps": 8,
        "calib_pass_rate": rate,
    }
    (root / "live" / "manifest.jsonl").write_text(json.dumps(manifest) + "\n")
    return root / "live"


def test_freeze_band_copies_and_rewrites_complete_gym_snapshot(tmp_path):
    source = _write_source(tmp_path)
    output = tmp_path / "frozen"

    freeze_band(source, output, num_repeats=2)

    snapshot = json.loads((output / "SNAPSHOT.json").read_text())
    assert snapshot["unique_tasks"] == 1
    assert snapshot["gym_rows"] == 2
    assert snapshot["upload_assets"] == 1
    assert (output / "SHA256SUMS").is_file()

    task = json.loads((output / "tasks" / "task_001.json").read_text())
    frozen_asset = Path(task["config"][0]["parameters"]["files"][0]["local_path"])
    assert frozen_asset.is_relative_to(output)
    assert frozen_asset.read_text() == "portable asset\n"
    assert all(
        Path(path).is_relative_to(output / "tmp_funcs" / "task_001")
        for path in task["metadata"]["new_functions_paths"]
    )

    rows = [
        json.loads(line) for line in (output / "train.jsonl").read_text().splitlines()
    ]
    assert len(rows) == 2
    assert rows[0]["agent_ref"]["name"] == "nemotron_osworld"
    assert rows[0]["verifier_metadata"]["id"] == "task_001"
    assert rows[1]["context_compaction_group_id"].endswith("repeat:1")


def test_freeze_band_rejects_tasks_outside_offline_dapo_band(tmp_path):
    source = _write_source(tmp_path, rate=1.0)
    output = tmp_path / "frozen"

    with pytest.raises(ValueError, match="offline DAPO filter failed"):
        freeze_band(source, output, num_repeats=1)

    assert not output.exists()
