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

import importlib.util
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
_spec = importlib.util.spec_from_file_location(
    "review_identity", REPO_ROOT / "skills/review-pr-team/scripts/review_identity.py"
)
ri = importlib.util.module_from_spec(_spec)
sys.modules["review_identity"] = ri  # its dataclass looks itself up here
_spec.loader.exec_module(ri)

PR = {"author": {"login": "alice"}, "headRefOid": "a" * 40, "files": [{"path": "x.py"}]}


def _fake_gh(user_login):
    """A `gh` stand-in: `api user` returns the login, or fails like an app token does when it's None."""

    def gh(*args):
        if args[:2] == ("api", "user"):
            if user_login is None:
                raise subprocess.CalledProcessError(1, ["gh", *args], stderr="HTTP 403")
            return user_login + "\n"
        raise AssertionError(f"unexpected gh call {args}")

    return gh


def _run(monkeypatch, capsys, user_login):
    monkeypatch.setattr(ri, "_gh", _fake_gh(user_login))
    monkeypatch.setattr(ri, "_pr", lambda pr: PR)
    real_detect = ri.detect
    monkeypatch.setattr(
        ri, "detect", lambda: real_detect(env={}, home=Path("/nonexistent"))
    )
    code = ri.main(["preflight", "--pr", "1"])
    return code, capsys.readouterr().out


def test_denied_user_lookup_is_an_automated_expert_review(monkeypatch, capsys):
    code, out = _run(monkeypatch, capsys, None)
    assert code == 0
    assert "Review kind: expert (automated reviewer)" in out
    assert "OK (automated reviewer)" in out
    assert "WARNING" not in out


def test_human_without_session_log_still_gets_the_warning(monkeypatch, capsys):
    code, out = _run(monkeypatch, capsys, "bob")
    assert code == 0
    assert "Review kind: expert (bob)" in out
    assert "WARNING: no session log found" in out
