# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Exercise the worker entrypoint without importing GPU-only backends."""

import ast
from pathlib import Path
from types import SimpleNamespace


def test_resident_reference_writes_reference_column_using_only_its_own_model():
    source = Path("nemo_rl/data_plane/worker_mixin.py").read_text()
    tree = ast.parse(source)
    method = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef)
        and node.name == "get_reference_policy_logprobs_presharded"
    )
    method.decorator_list = []
    module = ast.Module(
        body=[
            ast.ImportFrom(
                module="__future__", names=[ast.alias(name="annotations")], level=0
            ),
            method,
        ],
        type_ignores=[],
    )
    ast.fix_missing_locations(module)
    scope = {}
    exec(compile(module, "nemo_rl/data_plane/worker_mixin.py", "exec"), scope)
    writes = []

    def forbidden(*args, **kwargs):
        raise AssertionError(
            "A separate reference must not access a second reference model"
        )

    worker = SimpleNamespace(
        _fetch=lambda meta: {"input_ids": [1, 2]},
        _attach_or_repack_pack_metadata=lambda data, meta: data,
        get_logprobs=lambda **kwargs: {"logprobs": [0.0, -0.5]},
        get_reference_policy_logprobs=forbidden,
        _write_back_result_field=lambda *args, **kwargs: writes.append((args, kwargs)),
    )
    meta = object()
    scope["get_reference_policy_logprobs_presharded"](
        worker, meta, use_policy_model=True
    )
    assert writes == [
        (
            (meta, {"logprobs": [0.0, -0.5]}),
            {"result_key": "logprobs", "tq_field": "reference_policy_logprobs"},
        )
    ]
