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

"""Exercise research test runners without installing packages or running GPUs."""

import os
import shutil
import subprocess
from pathlib import Path

import pytest


@pytest.mark.parametrize("kind", ["unit", "functional"])
@pytest.mark.parametrize("fail_sync", [False, True])
def test_research_runners_sync_before_tests(
    tmp_path: Path, kind: str, fail_sync: bool
) -> None:
    root = Path(__file__).resolve().parents[2]
    runner = (
        "tests/unit/L0_Unit_Tests_Other.sh"
        if kind == "unit"
        else "tests/functional/L1_Functional_Tests_Other_1.sh"
    )
    for name in (runner, "tests/unit/run_unit_shard_common.sh"):
        destination = tmp_path / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(root / name, destination)
    # This core test is invoked directly by bash rather than through uv.
    frozen_test = tmp_path / "tests/functional/test_frozen_env.sh"
    frozen_test.parent.mkdir(parents=True, exist_ok=True)
    frozen_test.write_text("exit 0\n")
    projects = [tmp_path / "research" / name for name in ("first", "second project")]
    for project in projects:
        tests = project / "tests" / kind
        tests.mkdir(parents=True)
        if kind == "functional":
            (tests / "smoke.sh").touch()
    mock_bin = tmp_path / "bin"
    mock_bin.mkdir()
    uv = mock_bin / "uv"
    uv.write_text(
        "#!/bin/bash\n"
        'printf "%s|%s\\n" "$PWD" "$*" >> "$CALL_LOG"\n'
        'if [[ "$1" == sync && "$FAIL_SYNC" == 1 ]]; then exit 17; fi\n'
    )
    uv.chmod(0o755)
    log = tmp_path / "calls.log"
    result = subprocess.run(
        ["bash", str(tmp_path / runner)],
        env={
            **os.environ,
            "PATH": f"{mock_bin}:{os.environ['PATH']}",
            "CALL_LOG": str(log),
            "FAIL_SYNC": str(int(fail_sync)),
            "FAST": "0",
        },
        capture_output=True,
        text=True,
    )
    research_calls = [
        line for line in log.read_text().splitlines() if "/research/" in line
    ]
    expected = []
    for project in projects:
        expected.append(f"{project}|sync --locked --inexact --group test")
        if fail_sync:
            break
        test_command = (
            "pytest tests/unit" if kind == "unit" else "bash tests/functional/smoke.sh"
        )
        expected.append(f"{project}|run --no-sync {test_command}")
    assert result.returncode == (17 if fail_sync else 0), result.stderr
    assert research_calls == expected
