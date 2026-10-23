# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Bounded exporter probes; not a production adapter or a live harness runner.

Optional Harbor and historical fork orchestration are not installed. Load the
named source definitions unchanged, retaining source filenames/line numbers,
and substitute only their external runtime/configuration surfaces in tests.
"""

import __future__

import ast
import hashlib
import sys
from pathlib import Path
from types import ModuleType
from typing import Any
from uuid import uuid4

import nemo_gym

from nemo_rl.experience.rollout_reassembler import (
    ActionOutputFlags,
    RolloutReassembler,
    RolloutSelection,
)
from tests.unit.experience.common_output_comparison import selected_response_ids

GYM_ROOT = Path(nemo_gym.__file__).resolve().parent.parent
SOURCE_PROVENANCE: dict[str, dict] = {}


def load_source_definitions(
    path: Path,
    names: dict[str, set[str] | None],
    dependencies: dict[str, Any],
) -> ModuleType:
    """Execute exact definitions without importing unrelated launch/install code.

    A method subset gets an object shell instead of its orchestration superclass.
    No method body is rewritten. Callers must disclose injected dependencies.
    """
    raw = path.read_bytes()
    tree = ast.parse(raw, filename=str(path))
    selected = []
    for node in tree.body:
        if (
            isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef))
            and node.name in names
        ):
            methods = names[node.name]
            if methods is not None:
                assert isinstance(node, ast.ClassDef)
                node.bases = [ast.Name(id="object", ctx=ast.Load())]
                node.keywords = []
                node.decorator_list = []
                node.body = [
                    child
                    for child in node.body
                    if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef))
                    and child.name in methods
                ]
                assert {child.name for child in node.body} == methods
            selected.append(node)
    assert {node.name for node in selected} == set(names)
    module = ModuleType(f"_exporter_probe_{uuid4().hex}")
    module.__dict__.update(dependencies)
    sys.modules[module.__name__] = module
    code = compile(
        ast.fix_missing_locations(ast.Module(body=selected, type_ignores=[])),
        str(path),
        "exec",
        flags=__future__.annotations.compiler_flag,
    )
    exec(code, module.__dict__)
    SOURCE_PROVENANCE[str(path)] = {
        "sha256": hashlib.sha256(raw).hexdigest(),
        "definitions": {
            name: sorted(methods) if methods is not None else "whole definition"
            for name, methods in names.items()
        },
    }
    return module


def selected_chain_plan(records: list, response: dict) -> tuple[tuple[str, ...], list]:
    """Feed the existing comparison join into the actual RL segment planner."""
    selected = selected_response_ids(records, response)
    terminal = next(record for record in records if record.response_id == selected[-1])
    owner = terminal.staging_key.split("/")[0]
    selection = RolloutSelection(
        selected, tuple(ActionOutputFlags(False, False) for _ in selected)
    )
    receipt = {
        "rollout_id": owner,
        "manifest": [record.model_dump(mode="json") for record in records],
        "terminal_model_call_id": terminal.model_call_id,
        "terminal_selection": "declared",
    }
    plan = RolloutReassembler._plan_selected_calls([owner], [receipt], [selection])[0]
    return selected, plan
