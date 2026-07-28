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

from pathlib import Path


def test_hybridep_launcher_preserves_project_pythonpath() -> None:
    project_root = Path(__file__).resolve().parents[3]
    launcher = (
        project_root
        / "scripts"
        / "experiments"
        / "oci-hsg"
        / "hybridep"
        / "submit_grpo.sh"
    )
    source = launcher.read_text()

    project_path = 'PYTHONPATH="${PROJECT_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"'
    overlay_path = 'PYTHONPATH="${DEEPEP_OVERLAY}:${PYTHONPATH}"'
    export_path = "export PYTHONPATH"

    assert project_path in source
    assert overlay_path in source
    assert export_path in source
    assert source.index(project_path) < source.index(overlay_path)
    assert source.index(overlay_path) < source.index(export_path)
