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
"""CPU ownership regression before optional native inference construction."""

import ast
from collections.abc import Callable
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch


@pytest.fixture
def initialize_model() -> Callable[[SimpleNamespace, dict], None]:
    """Execute the actual initialization prefix before native context allocation."""
    source = (
        Path(__file__).resolve().parents[4]
        / "nemo_rl/models/generation/megatron/megatron_worker.py"
    )
    tree = ast.parse(source.read_text())
    cls = next(
        n
        for n in tree.body
        if isinstance(n, ast.ClassDef) and n.name == "MegatronGenerationMixin"
    )
    method = next(
        n
        for n in cls.body
        if isinstance(n, ast.FunctionDef) and n.name == "_initialize_inference_engine"
    )
    body = []
    for statement in method.body:
        if isinstance(statement, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == "pg_collection"
            for target in statement.targets
        ):
            break
        # Optional Megatron imports supply the subsequent engine construction.
        if not isinstance(statement, (ast.Import, ast.ImportFrom)):
            body.append(statement)
    else:
        raise AssertionError("native context boundary missing")
    method.body = body
    namespace = {}
    exec(
        compile(ast.Module(body=[method], type_ignores=[]), str(source), "exec"),
        namespace,
    )
    return namespace[method.name]


@pytest.mark.parametrize("colocated,dedicated", [(False, False), (True, True)])
@pytest.mark.parametrize("media", [False, True])
def test_owned_inference_freezes_without_changing_learner(
    initialize_model: Callable[[SimpleNamespace, dict], None],
    colocated: bool,
    dedicated: bool,
    media: bool,
) -> None:
    learner = torch.nn.Linear(4, 3)
    engine = torch.nn.Linear(4, 3)
    # Shared CUDA IPC binding preserves separate Parameter objects and .data aliases.
    engine.weight.data = learner.weight.data
    pointers = [p.data_ptr() for p in engine.parameters()]
    parameters = list(engine.parameters())
    worker = SimpleNamespace(
        _inference_engine_initialized=False,
        is_generation_colocated=colocated,
        inference_model=engine if dedicated else None,
        _get_megatron_inference_wrapper_cls=lambda: None,
        _inference_model_and_media_parts=lambda _: (engine, engine if media else None),
        _gen_model=lambda: engine,
    )
    initialize_model(worker, {})
    assert all(not parameter.requires_grad for parameter in engine.parameters())
    assert all(
        current is original
        for current, original in zip(engine.parameters(), parameters)
    )
    assert pointers == [p.data_ptr() for p in engine.parameters()]
    learner(torch.ones(2, 4)).sum().backward()
    assert all(parameter.grad is not None for parameter in learner.parameters())
    assert all(parameter.requires_grad for parameter in learner.parameters())


def test_shared_colocated_learner_preserves_flags_and_backward(
    initialize_model: Callable[[SimpleNamespace, dict], None],
) -> None:
    model = torch.nn.Linear(4, 3)
    model.bias.requires_grad_(False)
    worker = SimpleNamespace(
        _inference_engine_initialized=False,
        is_generation_colocated=True,
        inference_model=None,
        _get_megatron_inference_wrapper_cls=lambda: None,
        _inference_model_and_media_parts=lambda _: (model, None),
        _gen_model=lambda: model,
    )
    initialize_model(worker, {})
    model(torch.ones(2, 4)).sum().backward()
    assert model.weight.requires_grad and model.weight.grad is not None
    assert not model.bias.requires_grad and model.bias.grad is None


def test_initialized_engine_does_not_revisit_ownership(
    initialize_model: Callable[[SimpleNamespace, dict], None],
) -> None:
    initialize_model(SimpleNamespace(_inference_engine_initialized=True), {})
