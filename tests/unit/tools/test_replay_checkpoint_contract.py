# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import ast
from pathlib import Path

import pytest

from nemo_rl.utils.replay_checkpoint import validate_replay_restore


@pytest.mark.parametrize("load", [None, True])
def test_ray_resume_requires_explicit_discard(load):
    with pytest.raises(ValueError, match="load_replay_buffer=false"):
        validate_replay_restore(
            checkpoint_path="/checkpoints/step_18",
            ray_reference_transport=True,
            load_replay_buffer=load,
        )


@pytest.mark.parametrize("load", [None, False, True])
def test_fresh_ray_or_inline_resume_unchanged(load):
    validate_replay_restore(
        checkpoint_path=None, ray_reference_transport=True, load_replay_buffer=load
    )
    validate_replay_restore(
        checkpoint_path="/checkpoints/step_18",
        ray_reference_transport=False,
        load_replay_buffer=load,
    )


def test_discard_keeps_checkpoint_source():
    validate_replay_restore(
        checkpoint_path="/checkpoints/step_18",
        ray_reference_transport=True,
        load_replay_buffer=False,
    )


def test_guard_is_wired_before_state_and_payload_loading():
    tree = ast.parse(
        (Path(__file__).resolve().parents[3] / "nemo_rl/algorithms/grpo.py").read_text()
    )
    functions = {
        node.name: node for node in tree.body if isinstance(node, ast.FunctionDef)
    }
    for name, load_name in [
        ("setup", "load_training_info"),
        ("_maybe_restore_async_replay_buffer_checkpoint", "load_from_path"),
    ]:
        calls = [
            node for node in ast.walk(functions[name]) if isinstance(node, ast.Call)
        ]
        guard = next(
            node
            for node in calls
            if isinstance(node.func, ast.Name)
            and node.func.id == "validate_replay_restore"
        )
        loads = [node for node in calls if load_name in ast.unparse(node.func)]
        assert loads and guard.lineno < min(node.lineno for node in loads)


def test_setup_guard_applies_only_to_async_grpo():
    """Sync GRPO has no replay buffer, so a Ray-transport resume must not be blocked."""
    tree = ast.parse(
        (Path(__file__).resolve().parents[3] / "nemo_rl/algorithms/grpo.py").read_text()
    )
    setup = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "setup"
    )
    guards = [
        node
        for node in ast.walk(setup)
        if isinstance(node, ast.If)
        and any(
            isinstance(call, ast.Call)
            and isinstance(call.func, ast.Name)
            and call.func.id == "validate_replay_restore"
            for call in ast.walk(node)
        )
    ]
    assert guards, "validate_replay_restore is not conditional in setup()"
    assert all("async_grpo.enabled" in ast.unparse(node.test) for node in guards)
