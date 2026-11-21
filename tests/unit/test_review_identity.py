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
import json
import sys
from pathlib import Path

import pytest

_spec = importlib.util.spec_from_file_location(
    "review_identity",
    Path(__file__).resolve().parents[2]
    / "skills/review-pr-team/scripts/review_identity.py",
)
rid = importlib.util.module_from_spec(_spec)
# Dataclasses with postponed annotations look the module up in sys.modules.
sys.modules["review_identity"] = rid
_spec.loader.exec_module(rid)


def _write(path: Path, rows: list[dict]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    return path


def test_detect_claude_code(tmp_path: Path) -> None:
    _write(
        tmp_path / ".claude/projects/-repo/sid1.jsonl",
        [
            {"type": "assistant", "message": {"model": "claude-sonnet-5-5"}},
            {"type": "assistant", "message": {"model": "claude-opus-5-5"}},
        ],
    )
    env = {
        "CLAUDECODE": "1",
        "CLAUDE_CODE_SESSION_ID": "sid1",
        "CLAUDE_EFFORT": "high",
        "CLAUDE_CODE_EXECPATH": "/x/versions/2.1.330",
    }
    ident = rid.detect(env, tmp_path)
    assert (ident.harness, ident.version, ident.model, ident.effort) == (
        "claude-code",
        "2.1.330",
        "claude-opus-5-5",
        "high",
    )
    assert rid.not_allowed_reason(ident) is None


def test_detect_codex(tmp_path: Path) -> None:
    _write(
        tmp_path / ".codex/sessions/2026/11/20/rollout-2026-11-20T17-00-46-cid.jsonl",
        [
            {
                "type": "turn_context",
                "payload": {"model": "gpt-6-astra", "effort": "high"},
            }
        ],
    )
    env = {"CODEX_SESSION_ID": "cid", "CODEX_VERSION": "0.167.2"}
    ident = rid.detect(env, tmp_path)
    assert (ident.harness, ident.version, ident.model, ident.effort) == (
        "codex",
        "0.167.2",
        "gpt-6-astra",
        "high",
    )


def test_unknown_harness(tmp_path: Path) -> None:
    ident = rid.detect({}, tmp_path)
    assert ident.harness == "unknown"
    assert "harness" in rid.not_allowed_reason(ident)


@pytest.mark.parametrize(
    ("harness", "model", "effort", "reason"),
    [
        ("claude-code", "claude-sonnet-5-5", "high", "model"),
        ("claude-code", "claude-opus-5-5", "low", "effort"),
        ("codex", "gpt-5.6-sol", "high", "model"),
        ("codex", "gpt-6.1-sol", "minimal", "effort"),
        ("codex", "gpt-6.1-sol", "medium", None),
        ("claude-code", "claude-fable-5-1", "max", None),
    ],
)
def test_allow_list(harness: str, model: str, effort: str, reason: str | None) -> None:
    got = rid.not_allowed_reason(rid.Identity(harness, "v", model, effort, None))
    assert (got is None) if reason is None else (reason in got)


def test_same_session_claude_edit_and_commit(tmp_path: Path) -> None:
    transcript = _write(
        tmp_path / "t.jsonl",
        [
            {
                "type": "assistant",
                "cwd": "/repo/sub",
                "message": {
                    "content": [
                        {
                            "type": "tool_use",
                            "name": "Edit",
                            "input": {"file_path": "/repo/nemo_rl/a.py"},
                        },
                        {
                            "type": "tool_use",
                            "name": "Bash",
                            "input": {"command": "git commit -s -m x"},
                        },
                        {
                            "type": "tool_use",
                            "name": "Edit",
                            "input": {"file_path": "/repo/other.py"},
                        },
                    ]
                },
            }
        ],
    )
    got = rid.same_session_evidence(transcript, ["nemo_rl/a.py"], "/repo")
    assert got == ["edited nemo_rl/a.py", "ran git commit/push in this repo"]


def test_same_session_codex_patch(tmp_path: Path) -> None:
    transcript = _write(
        tmp_path / "t.jsonl",
        [
            {
                "type": "response_item",
                "payload": {
                    "type": "custom_tool_call",
                    "name": "exec",
                    "input": "tools.apply_patch('*** Begin Patch\\n*** Update File: /repo/nemo_rl/a.py\\n')",
                },
            }
        ],
    )
    assert rid.same_session_evidence(transcript, ["nemo_rl/a.py"], "/repo") == [
        "edited nemo_rl/a.py"
    ]


def test_fresh_session_has_no_evidence(tmp_path: Path) -> None:
    transcript = _write(
        tmp_path / "t.jsonl",
        [
            {
                "type": "assistant",
                "cwd": "/repo",
                "message": {
                    "content": [
                        {
                            "type": "tool_use",
                            "name": "Read",
                            "input": {"file_path": "/repo/nemo_rl/a.py"},
                        }
                    ]
                },
            }
        ],
    )
    assert rid.same_session_evidence(transcript, ["nemo_rl/a.py"], "/repo") == []
    assert rid.same_session_evidence(None, ["nemo_rl/a.py"], "/repo") == []


def _claude_calls(
    tmp_path: Path,
    cwd: str,
    calls: list[tuple[str, dict]],
    failed: tuple[int, ...] = (),
) -> Path:
    rows = [
        {
            "type": "assistant",
            "cwd": cwd,
            "message": {
                "content": [
                    {"type": "tool_use", "id": f"t{i}", "name": name, "input": args}
                    for i, (name, args) in enumerate(calls)
                ]
            },
        },
        {
            "type": "user",
            "message": {
                "content": [
                    {"type": "tool_result", "tool_use_id": f"t{i}", "is_error": True}
                    for i in failed
                ]
            },
        },
    ]
    return _write(tmp_path / "t.jsonl", rows)


@pytest.mark.parametrize(
    ("cwd", "command", "expected"),
    [
        ("/tmp", "printf x > /repo/nemo_rl/a.py", ["edited nemo_rl/a.py"]),
        ("/repo/nemo_rl", "echo x >> a.py", ["edited nemo_rl/a.py"]),
        ("/tmp", "cd /repo && sed -i 's/a/b/' nemo_rl/a.py", ["edited nemo_rl/a.py"]),
        ("/repo", "echo x | tee nemo_rl/a.py", ["edited nemo_rl/a.py"]),
        ("/repo", "cp /tmp/x nemo_rl/a.py", ["edited nemo_rl/a.py"]),
        ("/repo", "cat nemo_rl/a.py > /tmp/copy", []),
        ("/repo", "printf x > other.py", []),
        ("/repo", "printf '%s\\n' 'git commit'", []),
        ("/repo-copy", "git commit -m x", []),
        ("/repo", "cd /repo-copy && git commit -m x", []),
        (
            "/tmp",
            "cd /repo/sub && git commit -m x",
            ["ran git commit/push in this repo"],
        ),
        ("/tmp", "git -C /repo push", ["ran git commit/push in this repo"]),
        (
            "/repo",
            "GIT_EDITOR=true git -c a=b commit",
            ["ran git commit/push in this repo"],
        ),
    ],
)
def test_same_session_claude_bash(
    tmp_path: Path, cwd: str, command: str, expected: list[str]
) -> None:
    transcript = _claude_calls(tmp_path, cwd, [("Bash", {"command": command})])
    assert rid.same_session_evidence(transcript, ["nemo_rl/a.py"], "/repo") == expected


def test_same_session_skips_failed_calls(tmp_path: Path) -> None:
    transcript = _claude_calls(
        tmp_path,
        "/repo",
        [
            ("Bash", {"command": "printf x > nemo_rl/a.py"}),
            ("Edit", {"file_path": "/repo/nemo_rl/a.py"}),
        ],
        failed=(0, 1),
    )
    assert rid.same_session_evidence(transcript, ["nemo_rl/a.py"], "/repo") == []


def test_same_session_edit_paths_are_repo_aware(tmp_path: Path) -> None:
    unrelated = _claude_calls(
        tmp_path, "/repo", [("Edit", {"file_path": "/unrelated/nemo_rl/a.py"})]
    )
    assert rid.same_session_evidence(unrelated, ["nemo_rl/a.py"], "/repo") == []
    relative = _claude_calls(
        tmp_path, "/repo/nemo_rl", [("Write", {"file_path": "a.py"})]
    )
    assert rid.same_session_evidence(relative, ["nemo_rl/a.py"], "/repo") == [
        "edited nemo_rl/a.py"
    ]


def test_same_session_codex_relative_patch_and_quoted_git(tmp_path: Path) -> None:
    def codex(code: str) -> Path:
        return _write(
            tmp_path / "t.jsonl",
            [
                {"type": "turn_context", "payload": {"cwd": "/repo"}},
                {
                    "type": "response_item",
                    "payload": {
                        "type": "custom_tool_call",
                        "name": "exec",
                        "input": code,
                    },
                },
            ],
        )

    relative = codex(
        "tools.apply_patch('*** Begin Patch\\n*** Update File: nemo_rl/a.py\\n')"
    )
    assert rid.same_session_evidence(relative, ["nemo_rl/a.py"], "/repo") == [
        "edited nemo_rl/a.py"
    ]
    doc_patch = codex(
        "tools.apply_patch('*** Begin Patch\\n*** Update File: /repo/docs/x.md\\n"
        "+Run git commit in /repo\\n*** End Patch')"
    )
    assert rid.same_session_evidence(doc_patch, ["nemo_rl/a.py"], "/repo") == []
    commit = codex('tools.exec_command({cmd: "git commit -m x", workdir: "/repo/sub"})')
    assert rid.same_session_evidence(commit, ["nemo_rl/a.py"], "/repo") == [
        "ran git commit/push in this repo"
    ]


def test_footer_cli_requires_reviewed_head(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    reviewed, live = "a" * 40, "b" * 40
    monkeypatch.setattr(
        rid,
        "_pr",
        lambda pr: {"author": {"login": "me"}, "headRefOid": live, "files": []},
    )
    monkeypatch.setattr(rid, "_gh", lambda *args: "me\n")
    monkeypatch.setattr(
        rid, "detect", lambda: rid.Identity("codex", "v", "gpt-6-astra", "high", None)
    )
    with pytest.raises(SystemExit):
        rid.main(["footer", "--pr", "1"])
    with pytest.raises(SystemExit):
        rid.main(["footer", "--pr", "1", "--head", reviewed])
    assert "nemo-rl-review" not in capsys.readouterr().out
    assert rid.main(["footer", "--pr", "1", "--head", live]) == 0
    out = capsys.readouterr().out
    assert "kind=self harness=codex/v model=gpt-6-astra effort=high" in out
    assert f"head={live} -->" in out


def test_same_session_reads_subagent_transcripts(tmp_path: Path) -> None:
    project = tmp_path / ".claude/projects/-repo"
    _write(
        project / "sid1.jsonl",
        [
            {
                "type": "assistant",
                "cwd": "/repo",
                "message": {
                    "content": [
                        {
                            "type": "tool_use",
                            "id": "a1",
                            "name": "Agent",
                            "input": {"prompt": "edit"},
                        }
                    ]
                },
            }
        ],
    )
    _write(
        project / "sid1/subagents/agent-1.jsonl",
        [
            {
                "type": "assistant",
                "cwd": "/repo",
                "message": {
                    "content": [
                        {
                            "type": "tool_use",
                            "id": "s1",
                            "name": "Edit",
                            "input": {"file_path": "/repo/nemo_rl/a.py"},
                        }
                    ]
                },
            }
        ],
    )
    ident = rid.detect({"CLAUDECODE": "1", "CLAUDE_CODE_SESSION_ID": "sid1"}, tmp_path)
    assert rid.same_session_evidence(ident.transcript, ["nemo_rl/a.py"], "/repo") == [
        "edited nemo_rl/a.py"
    ]


def test_missing_gh_stops_with_install_hint(monkeypatch: pytest.MonkeyPatch) -> None:
    def no_gh(*args, **kwargs):
        raise FileNotFoundError("gh")

    monkeypatch.setattr(rid.subprocess, "run", no_gh)
    with pytest.raises(SystemExit, match="needs the GitHub CLI `gh`"):
        rid.main(["preflight", "--pr", "1"])


def test_footer_marker() -> None:
    ident = rid.Identity("claude-code", "2.1.330", "claude-opus-5-5", "medium", None)
    out = rid.footer(ident=ident, kind="self", head="67e151b702a9ff81")
    visible, marker = out.splitlines()
    assert visible.startswith("_Self-review: Claude Code 2.1.330 · claude-opus-5-5")
    assert marker == (
        "<!-- nemo-rl-review: kind=self harness=claude-code/2.1.330 model=claude-opus-5-5 "
        f"effort=medium skill_rev={rid.skill_revision()} head=67e151b702a9ff81 -->"
    )
