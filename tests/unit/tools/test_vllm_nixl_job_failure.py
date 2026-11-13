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

"""Exercise the distributed fail-stop code without running or cancelling jobs."""

import hashlib
import importlib.util
import json
import logging
import subprocess
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock

import pytest

BUNDLE = Path(__file__).resolve().parents[3] / "tools/vllm_patches/nixl-lease-v0.30"


@pytest.fixture(scope="module")
def job_module(tmp_path_factory: pytest.TempPathFactory) -> ModuleType:
    """Load the exact job-failure module distributed in the patch."""
    section = next(
        part
        for part in (BUNDLE / "lease-verification.patch")
        .read_text()
        .split("diff --git ")
        if part.startswith(
            "a/vllm/distributed/kv_transfer/kv_connector/v1/nixl/job_failure.py "
        )
    )
    source = "".join(
        line[1:]
        for line in section.splitlines(keepends=True)
        if line.startswith("+") and not line.startswith("+++")
    )
    record = next(
        row
        for row in json.loads((BUNDLE / "manifest.json").read_text())["files"]
        if row["path"].endswith("/job_failure.py")
    )
    assert hashlib.sha256(source.encode()).hexdigest() == record["patched_sha256"]
    path = tmp_path_factory.mktemp("bundled-job-failure") / "job_failure.py"
    path.write_text(source)
    spec = importlib.util.spec_from_file_location("bundled_nixl_job_failure", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def policy(job_module: ModuleType, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("SLURM_JOB_ID", "123456")
    monkeypatch.setenv("SLURM_JOB_NODELIST", "node-[1-2]")
    monkeypatch.setenv("SLURMD_NODENAME", "node-2")
    monkeypatch.setattr(job_module.socket, "gethostname", lambda: "node-2")
    monkeypatch.setattr(job_module.shutil, "which", lambda name: "/mock/scancel")
    return job_module.KVFailurePolicy.create(
        "stop_job", load_failure_policy="fail", lease_verification=True
    )


@pytest.mark.parametrize("job_id", ["", "123;456", "--all", "１２３"])
def test_invalid_job_cannot_start(job_module, monkeypatch, job_id):
    monkeypatch.setenv("SLURM_JOB_ID", job_id)
    with pytest.raises(ValueError, match="SLURM_JOB_ID"):
        job_module.KVFailurePolicy.create(
            "stop_job", load_failure_policy="fail", lease_verification=True
        )


def test_missing_slurm_client_cannot_start(job_module, monkeypatch):
    monkeypatch.setenv("SLURM_JOB_ID", "123456")
    monkeypatch.setattr(job_module.shutil, "which", lambda name: None)
    with pytest.raises(ValueError, match="scancel"):
        job_module.KVFailurePolicy.create(
            "stop_job", load_failure_policy="fail", lease_verification=True
        )


@pytest.mark.parametrize(
    ("load_policy", "verify"), [("recompute", True), ("fail", False)]
)
def test_fail_stop_rejects_recovery_or_missing_verification(
    job_module, load_policy, verify
):
    with pytest.raises(ValueError, match="stop_job requires"):
        job_module.KVFailurePolicy.create(
            "stop_job", load_failure_policy=load_policy, lease_verification=verify
        )


def test_unknown_action_is_rejected(job_module):
    with pytest.raises(ValueError, match="kv_failure_action"):
        job_module.KVFailurePolicy.create(
            "ignore", load_failure_policy="fail", lease_verification=True
        )


def test_request_diagnostics_do_not_cancel(job_module, monkeypatch):
    cancel = Mock()
    monkeypatch.delenv("SLURM_JOB_ID", raising=False)
    monkeypatch.setattr(job_module.subprocess, "run", cancel)
    subject = job_module.KVFailurePolicy.create(
        "fail_request", load_failure_policy="recompute", lease_verification=True
    )
    subject.on_failure(reason="expired", context={})
    cancel.assert_not_called()


def test_node_report_precedes_job_cancellation(job_module, policy, monkeypatch, caplog):
    def cancel(args, **kwargs):
        assert args == ["/mock/scancel", "123456"]
        assert kwargs["check"] and kwargs["timeout"] == 10
        message = caplog.records[-1].getMessage()
        assert "please report to AGA support team" in message
        report = json.loads(message[message.index("{") :])
        assert report["slurm_job_id"] == "123456"
        assert report["local_node"] == "node-2"
        assert report["remote_nodes_by_tp_rank"] == {"0": "node-1"}
        assert report["request_id"] == "request-1" and report["tp_rank"] == 1
        assert report["time_utc"].endswith("+00:00")

    monkeypatch.setattr(job_module.subprocess, "run", cancel)
    with (
        caplog.at_level(logging.CRITICAL),
        pytest.raises(RuntimeError, match="requested"),
    ):
        policy.on_failure(
            reason="lease nonce mismatch",
            context={
                "request_id": "request-1",
                "tp_rank": 1,
                "remote_nodes_by_tp_rank": {0: "node-1"},
            },
        )


@pytest.mark.parametrize(
    "error",
    [
        FileNotFoundError(),
        subprocess.CalledProcessError(1, "scancel"),
        subprocess.TimeoutExpired("scancel", 10),
    ],
)
def test_cancellation_failure_is_fatal(job_module, policy, monkeypatch, error):
    monkeypatch.setattr(job_module.subprocess, "run", Mock(side_effect=error))
    with pytest.raises(RuntimeError, match="terminate this allocation manually"):
        policy.on_failure(reason="transfer failed", context={})


@pytest.mark.parametrize("failed", [True, False])
def test_native_verifier_stops_before_request_recovery(
    job_module, policy, monkeypatch, failed
):
    """Execute the bundled worker methods with a failed asynchronous lease check."""
    section = (BUNDLE / "lease-verification.patch").read_text().split("diff --git ")[1]
    methods = []
    for name in ["_verify_read_leases", "_stop_on_kv_failure"]:
        added = section.split(f"+    def {name}(", 1)[1].split("+    def ", 1)[0]
        lines = (f"+    def {name}(" + added).splitlines(keepends=True)
        methods.append("".join(line[1:] for line in lines if line.startswith("+")))
    namespace = {"logger": logging.getLogger("test"), "ReqId": str}
    exec(
        "from __future__ import annotations\nclass Worker:\n" + "".join(methods),
        namespace,
    )
    worker = namespace["Worker"]()
    worker._kv_failure_policy = policy
    worker.engine_id = "decode"
    worker.tp_rank = 1
    worker._remote_node_names = {"prefill": {0: "node-1"}}
    worker._recving_metadata = {
        "request-1": SimpleNamespace(
            remote=SimpleNamespace(
                engine_id="prefill", host="10.0.0.1", request_id="P-1"
            )
        )
    }
    worker._recv_failures = set()
    worker._lease_verifier = Mock()
    worker._lease_verifier.step.return_value = (
        (set(), {"request-1": "nonce mismatch"}) if failed else ({"request-1"}, {})
    )
    worker.xfer_stats = Mock()
    worker._send_pending_recv_notifs = Mock()
    cancel = Mock()
    monkeypatch.setattr(job_module.subprocess, "run", cancel)
    if not failed:
        assert worker._verify_read_leases({"request-1"}) == {"request-1"}
        cancel.assert_not_called()
        return
    with pytest.raises(RuntimeError, match="allocation cancellation requested"):
        worker._verify_read_leases({"request-1"})
    cancel.assert_called_once()
    worker._send_pending_recv_notifs.assert_not_called()


def test_real_fake_client_receives_only_current_job(job_module, monkeypatch, tmp_path):
    """Run a local stand-in binary, never the cluster's Slurm client."""
    receipt = tmp_path / "cancelled-job"
    client = tmp_path / "scancel"
    client.write_text(
        f"#!{sys.executable}\nimport pathlib, sys\n"
        f"pathlib.Path({str(receipt)!r}).write_text(repr(sys.argv[1:]))\n"
    )
    client.chmod(0o755)
    monkeypatch.setenv("SLURM_JOB_ID", "123456")
    monkeypatch.setattr(job_module.shutil, "which", lambda name: str(client))
    subject = job_module.KVFailurePolicy.create(
        "stop_job", load_failure_policy="fail", lease_verification=True
    )
    with pytest.raises(RuntimeError, match="allocation cancellation requested"):
        subject.on_failure(reason="transfer failed", context={})
    assert receipt.read_text() == "['123456']"
