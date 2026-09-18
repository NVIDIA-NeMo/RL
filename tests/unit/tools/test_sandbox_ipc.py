# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Patch contract plus an opt-in actual sandbox worker/Pipe regression."""

import importlib.util
import multiprocessing as mp
import os
from pathlib import Path
import subprocess

import pytest

PATCH = (
    Path(__file__).resolve().parents[3]
    / "tools/super_rl/patches/sandbox_exception_ipc.patch"
)


def test_patch_is_one_boolean_protocol_change(tmp_path):
    lines = PATCH.read_text().splitlines()[3:]
    before = "\n".join(line[1:] for line in lines if line.startswith((" ", "-"))) + "\n"
    source = tmp_path / "main.py"
    source.write_text(before)
    subprocess.run(
        ["patch", "--batch", "--forward", "--fuzz=0", str(source), str(PATCH)],
        check=True,
        capture_output=True,
    )
    assert source.read_text() == before.replace(
        '"has_error": res.error_before_exec or res.error_in_exec,',
        '"has_error": bool(res.error_before_exec or res.error_in_exec),',
    )


@pytest.mark.skipif(
    not os.environ.get("NRL_SANDBOX_MODULE"),
    reason="Requires actual patched sandbox module and native sandbox dependencies",
)
def test_native_worker_preserves_state_after_exceptions():
    """Run inside the sandbox image, not a newer Gym/client interpreter."""
    path = Path(os.environ["NRL_SANDBOX_MODULE"])
    spec = importlib.util.spec_from_file_location("sandbox_ipc_regression", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    context = mp.get_context("fork")
    parent, child = context.Pipe()
    process = context.Process(target=module.shell_worker, args=(child,))
    process.start()
    child.close()
    cases = [
        ("state", "x = 41\nprint(x)", False),
        (
            "json",
            'import requests\nraise requests.exceptions.JSONDecodeError("test", "x", 0)',
            True,
        ),
        ("value", 'raise ValueError("expected")', True),
        ("syntax", "if :", True),
        ("state_after_errors", "x += 1\nprint(x)", False),
    ]
    try:
        for name, code, expected in cases:
            parent.send({"cmd": "exec", "id": name, "code": code})
            assert parent.poll(30), name
            result = parent.recv()
            assert result["status"] == "ok", result
            assert type(result["has_error"]) is bool
            assert result["has_error"] is expected
            if name == "state_after_errors":
                assert result["stdout"].strip() == "42"
        parent.send({"cmd": "shutdown"})
        process.join(10)
        assert process.exitcode == 0
    finally:
        if process.is_alive():
            process.terminate()
            process.join(10)
        parent.close()
