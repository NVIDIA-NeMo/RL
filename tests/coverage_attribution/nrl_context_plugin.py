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
"""Record pytest identities, phases and outcomes alongside pytest-cov contexts."""

import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import pytest


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption("--nrl-context-meta", help="Write test attribution metadata here")


def pytest_configure(config: pytest.Config) -> None:
    global G_STATE
    root = Path(config.rootpath)
    G_STATE = {
        "schema": 1,
        "started_at": datetime.now(timezone.utc).isoformat(),
        "repo": str(root),
        "python": platform.python_version(),
        "argv": list(config.invocation_params.args),
        "confcutdir": str(config.option.confcutdir or ""),
        "remote_process_attribution": "not validated; driver contexts only",
        "source_hashes": {
            str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted((root / "nemo_rl").rglob("*.py"))
        },
        "packages": {
            d.metadata["Name"]: d.version for d in importlib.metadata.distributions()
        },
        "tests": {},
        "collection_errors": [],
    }
    G_STATE["commit"] = subprocess.check_output(
        ["git", "-C", str(root), "rev-parse", "HEAD"], text=True
    ).strip()
    G_STATE["working_tree_status"] = subprocess.check_output(
        ["git", "-C", str(root), "status", "--short"], text=True
    )


def pytest_collection_finish(session: pytest.Session) -> None:
    for item in session.items:
        G_STATE["tests"][item.nodeid] = {
            "nodeid": item.nodeid,
            "file": str(item.path.relative_to(session.config.rootpath)),
            "line": item.location[1] + 1,
            "markers": sorted({m.name for m in item.iter_markers()}),
            "parameterized": hasattr(item, "callspec"),
            "phases": {},
        }


def pytest_collectreport(report: pytest.CollectReport) -> None:
    if report.failed:
        G_STATE["collection_errors"].append(
            {"nodeid": report.nodeid, "error": str(report.longrepr)[:4000]}
        )


def pytest_runtest_logreport(report: pytest.TestReport) -> None:
    item = G_STATE["tests"].setdefault(
        report.nodeid, {"nodeid": report.nodeid, "phases": {}}
    )
    item["phases"][report.when] = {
        "outcome": report.outcome,
        "duration_seconds": report.duration,
        "wasxfail": str(getattr(report, "wasxfail", "")),
    }


def pytest_sessionfinish(session: pytest.Session, exitstatus: int) -> None:
    path = session.config.getoption("--nrl-context-meta")
    if path:
        G_STATE["exit_status"] = int(exitstatus)
        G_STATE["finished_at"] = datetime.now(timezone.utc).isoformat()
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_suffix(".tmp")
        tmp.write_text(json.dumps(G_STATE, indent=2) + "\n")
        os.replace(tmp, p)
