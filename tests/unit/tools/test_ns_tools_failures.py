# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Test failure propagation in the actual patched ns_tools verify method."""

import ast
import asyncio
import os
from pathlib import Path
import subprocess
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

ROOT = Path(__file__).resolve().parents[3]
GYM = ROOT / "3rdparty/Gym-workspace/Gym"
BASE = "749432dc5de23b8eeb3d044c80350a7c0ae9a03f"  # pragma: allowlist secret
RELATIVE = "resources_servers/ns_tools/app.py"


@pytest.fixture(scope="module")
def verify(tmp_path_factory):
    source = subprocess.run(
        ["git", "-C", str(GYM), "show", f"{BASE}:{RELATIVE}"],
        capture_output=True,
        check=False,
    )
    if source.returncode:
        message = "Initialize the pinned Gym submodule to test its failure overlay"
        if os.environ.get("NRL_REQUIRE_PINNED_GYM"):
            pytest.fail(message)  # CI opt-in: no pin means no signal
        pytest.skip(message)
    stage = tmp_path_factory.mktemp("ns-tools-failures")
    path = stage / RELATIVE
    path.parent.mkdir(parents=True)
    path.write_bytes(source.stdout)
    subprocess.run(
        ["patch", "--batch", "--fuzz=0", "-p1", "-d", str(stage), "-i",
         str(ROOT / "tools/super_rl/patches/gym_ns_tools_failures.patch")],
        check=True,
        capture_output=True,
    )
    # Isolate the method to avoid importing the optional NeMo-Skills tool stack
    # in CPU unit CI. Native tests must additionally check its Pydantic response.
    cls = next(node for node in ast.parse(path.read_text()).body
               if isinstance(node, ast.ClassDef) and node.name == "NSToolsResourcesServer")
    method = next(node for node in cls.body if isinstance(node, ast.AsyncFunctionDef) and node.name == "verify")
    namespace = {"Request": object, "NSToolsVerifyRequest": object,
                 "NSToolsVerifyResponse": dict, "SESSION_ID_KEY": "session_id"}
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(path), "exec"), namespace)
    return namespace["verify"]


@pytest.mark.parametrize("delegated", [
    {"reward": 1.0},
    {"reward": 0.0},
    {"reward": 0.0, "_ng_failure_class": "judge_failed", "_ng_failure_judge_error": "missing verdict"},
    {"reward": 0.0, "_ng_failure_class": "verifier_unavailable", "_ng_failure_detail": "HTTP 503"},
])
def test_delegated_failure_is_visible_at_training_boundary(verify, delegated):
    client = SimpleNamespace(post=AsyncMock(return_value=SimpleNamespace(json=AsyncMock(return_value=delegated))))
    cleanup = AsyncMock()
    server = SimpleNamespace(
        config=SimpleNamespace(verbose_tool_logging=False, default_verifier="math_with_judge",
                               verifiers={"math_with_judge": SimpleNamespace(name="math_with_judge")}),
        server_client=client,
        tool_manager=SimpleNamespace(cleanup_request=cleanup),
        _aggregate_timing_metrics=lambda session: {"num_tool_calls": 2},
    )
    body = SimpleNamespace(verifier_type=None, model_dump=lambda: {"response": {"output": []}})
    actual = asyncio.run(verify(server, SimpleNamespace(session={"session_id": "row-1"}), body))
    assert actual["reward"] == delegated["reward"]
    assert actual["delegated_response"] == delegated
    assert actual["num_tool_calls"] == 2
    assert actual["response"] == {"output": []}
    assert {key: value for key, value in actual.items() if key.startswith("_ng_failure_")} == {
        key: value for key, value in delegated.items() if key.startswith("_ng_failure_")
    }
    cleanup.assert_awaited_once_with("row-1")
