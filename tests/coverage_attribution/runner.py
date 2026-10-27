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
"""CI payload: wrap one pytest invocation, keeping its contexts and exit status.

Installed location in the proposed patch: tests/coverage_attribution/runner.py.
The parent CI still gets a combined .coverage; attribution gets isolated raw DBs.
"""

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path


def main() -> int:
    root = Path(__file__).resolve().parents[2]
    parent = Path(
        os.environ.get("NRL_TEST_CONTEXT_DIR", str(root / "tests/test-contexts"))
    )
    parent.mkdir(parents=True, exist_ok=True)
    output = Path(tempfile.mkdtemp(prefix="pytest-", dir=parent))
    env = dict(os.environ)
    env["COVERAGE_FILE"] = str(output / ".coverage")
    env["PYTHONPATH"] = os.pathsep.join(
        [str(Path(__file__).parent), str(root), env.get("PYTHONPATH", "")]
    )
    # Explicit root is needed when pytest runs from repo/tests, as in run_unit.sh.
    command = [
        sys.executable,
        "-m",
        "pytest",
        *sys.argv[1:],
        "--rootdir=" + str(root),
        "-p",
        "nrl_context_plugin",
        "--cov=nemo_rl",
        "--cov-branch",
        "--cov-context=test",
        "--nrl-context-meta=" + str(output / "tests.json"),
    ]
    status = subprocess.run(command, env=env).returncode
    result = {
        "pytest_exit_status": status,
        "command": command,
        "raw_coverage": str(output / ".coverage"),
    }
    (output / "invocation.json").write_text(json.dumps(result, indent=2) + "\n")
    if not (output / ".coverage").is_file() or not (output / "tests.json").is_file():
        print("Context capture incomplete: " + str(output), file=sys.stderr)
        return status or 2
    # Keep the original CI aggregate artifact while retaining each isolated DB.
    combined = subprocess.run(
        [
            sys.executable,
            "-m",
            "coverage",
            "combine",
            "--append",
            "--keep",
            str(output / ".coverage"),
        ],
        env=os.environ,
    ).returncode
    return status or combined


if __name__ == "__main__":
    raise SystemExit(main())
