# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy at http://www.apache.org/licenses/LICENSE-2.0
"""Parity and publication-boundary checks using vLLM's real flat container."""

from __future__ import annotations

import ast
import importlib.util
import math
import os
import random
import sys
import threading
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

from nemo_rl.models.generation.vllm.packed_logprobs import PackedSelectedLogprobs


@pytest.fixture(scope="module")
def extraction() -> ModuleType:
    path = (
        Path(__file__).resolve().parents[4] / "nemo_rl/models/generation/vllm/utils.py"
    )
    tree = ast.parse(path.read_text())
    names = {
        "_flat_logprob_arrays",
        "_extract_selected_token_logprobs",
        "extract_published_generation",
        "attach_token_information_to_chat_response_choices",
    }
    functions = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name in names
    ]
    constants = [
        node
        for node in tree.body
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id == "VLLM_LOGPROB_FLOOR"
            for target in node.targets
        )
    ]
    future = ast.ImportFrom(
        module="__future__", names=[ast.alias(name="annotations")], level=0
    )
    module = ModuleType("_flat_capture_production_functions")
    module.PackedSelectedLogprobs = PackedSelectedLogprobs
    exec(
        compile(
            ast.fix_missing_locations(
                ast.Module(body=[future, *constants, *functions], type_ignores=[])
            ),
            str(path),
            "exec",
        ),
        module.__dict__,
    )
    return module


@pytest.fixture(scope="module")
def real_vllm_logprobs() -> ModuleType:
    # The file is stdlib-only. Read the installed source directly so these
    # CPU tests need neither GPU initialization nor vLLM's package imports.
    override = os.environ.get("NRL_TEST_VLLM_LOGPROBS_SOURCE")
    if override:
        path = Path(override)
    else:
        package = importlib.util.find_spec("vllm")
        if package is None or package.origin is None:
            pytest.skip("vLLM logprobs source is unavailable")
        path = Path(package.origin).parent / "logprobs.py"
    spec = importlib.util.spec_from_file_location(
        "_real_vllm_logprobs_for_capture", path
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def make_flat(module: ModuleType, positions: list[list[tuple[int, float]]]):
    flat = module.FlatLogprobs()
    for row in positions:
        flat.append_fast(
            [token for token, _ in row],
            [value for _, value in row],
            iter(range(len(row))),
            [None] * len(row),
        )
    return flat


def assert_identical(actual: list[float], expected: list[float]) -> None:
    assert len(actual) == len(expected)
    for value, reference in zip(actual, expected, strict=True):
        if math.isnan(reference):
            assert math.isnan(value)
        else:
            assert value == reference


@pytest.mark.parametrize("seed", range(20))
def test_real_flat_matches_dict_semantics(extraction, real_vllm_logprobs, seed):
    rng = random.Random(seed)
    ids, positions = [], []
    for _ in range(128):
        token = rng.randrange(32)
        ids.append(token)
        row = [
            (rng.randrange(32), rng.uniform(-15000, 0)) for _ in range(rng.randrange(6))
        ]
        row.insert(0, (token, rng.uniform(-15000, 0)))
        # Duplicate selected IDs must follow FlatLogprobs' last-write wins.
        if rng.choice([False, True]):
            row.append((token, rng.uniform(-15000, 0)))
        positions.append(row)
    flat = make_flat(real_vllm_logprobs, positions)
    reference = [
        max(float(flat[index][token].logprob), -9999.0)
        for index, token in enumerate(ids)
    ]
    assert_identical(extraction._extract_selected_token_logprobs(ids, flat), reference)
    for start in [0, 1, 57, 128]:
        tokens, values = extraction.extract_published_generation(ids, flat, start=start)
        assert tokens == ids[start:]
        assert_identical(values, reference[start:])
    legacy = [flat[index] for index in range(len(flat))]
    assert_identical(
        extraction._extract_selected_token_logprobs(ids, legacy), reference
    )


@pytest.mark.parametrize(
    "value", [float("-inf"), -10000.0, -9999.0, -0.25, 0.0, float("inf"), float("nan")]
)
def test_floor_and_nonfinite_parity(extraction, real_vllm_logprobs, value):
    flat = make_flat(real_vllm_logprobs, [[(9, value)]])
    assert_identical(
        extraction._extract_selected_token_logprobs([9], flat),
        [max(float(flat[0][9].logprob), -9999.0)],
    )


def test_no_position_dicts_or_logprob_objects_on_hot_paths(
    extraction, real_vllm_logprobs, monkeypatch
):
    count = 131072
    flat = real_vllm_logprobs.FlatLogprobs(
        start_indices=list(range(count)),
        end_indices=list(range(1, count + 1)),
        token_ids=[7] * count,
        logprobs=[-0.25] * count,
        ranks=[1] * count,
        decoded_tokens=[None] * count,
    )
    ids = [7] * count

    def forbidden(*args, **kwargs):
        raise AssertionError("hot capture path reconstructed per-token logprob objects")

    monkeypatch.setattr(real_vllm_logprobs.FlatLogprobs, "__getitem__", forbidden)
    monkeypatch.setattr(real_vllm_logprobs.FlatLogprobs, "__iter__", forbidden)
    monkeypatch.setattr(real_vllm_logprobs.Logprob, "__init__", forbidden)
    assert (
        extraction._extract_selected_token_logprobs(ids, flat, stop=count)
        == [-0.25] * count
    )
    assert extraction.extract_published_generation(ids, flat, start=65536) == (
        ids[65536:],
        [-0.25] * 65536,
    )
    output = SimpleNamespace(index=0, token_ids=ids, logprobs=flat)
    response = SimpleNamespace(
        choices=[SimpleNamespace(index=0, message=SimpleNamespace())]
    )
    final = SimpleNamespace(prompt_token_ids=[1, 2], outputs=[output])
    extraction.attach_token_information_to_chat_response_choices(response, final)
    assert response.choices[0].message.generation_log_probs == [-0.25] * count


def test_unfinished_append_does_not_publish_partial_logprobs(
    extraction, real_vllm_logprobs
):
    flat = make_flat(real_vllm_logprobs, [[(7, -0.1)]])
    token_ids = [7, 8]
    half_published = threading.Event()
    finish = threading.Event()

    def append():
        flat.start_indices.append(1)
        flat.token_ids.append(8)
        half_published.set()
        assert finish.wait(3)
        flat.logprobs.append(-0.2)
        flat.ranks.append(1)
        flat.decoded_tokens.append(None)
        flat.end_indices.append(2)

    thread = threading.Thread(target=append, daemon=True)
    thread.start()
    try:
        assert half_published.wait(3)
        assert extraction.extract_published_generation(token_ids, flat, start=0) == (
            [7],
            [-0.1],
        )
        assert extraction.extract_published_generation(token_ids, flat, start=1) == (
            [],
            [],
        )
    finally:
        finish.set()
        thread.join(3)
    assert not thread.is_alive()
    assert extraction.extract_published_generation(token_ids, flat, start=0) == (
        [7, 8],
        [-0.1, -0.2],
    )


def test_frozen_bound_ignores_later_engine_appends(extraction, real_vllm_logprobs):
    flat = make_flat(real_vllm_logprobs, [[(7, -0.1)], [(8, -0.2)]])
    frozen_ids = [7, 8]
    flat.append({9: real_vllm_logprobs.Logprob(-0.3)})
    assert extraction._extract_selected_token_logprobs(frozen_ids, flat, stop=2) == [
        -0.1,
        -0.2,
    ]
    with pytest.raises(ValueError, match="equal lengths"):
        extraction._extract_selected_token_logprobs(frozen_ids, flat)


@pytest.mark.parametrize("flat_format", [False, True])
def test_missing_selected_token_fails_with_context(
    extraction, real_vllm_logprobs, flat_format
):
    details = make_flat(real_vllm_logprobs, [[(5, -0.1)]])
    if not flat_format:
        details = [details[0]]
    with pytest.raises(RuntimeError, match="choice_idx=3, token_id=7"):
        extraction._extract_selected_token_logprobs(
            [7], details, where="choice_idx=3, "
        )


@pytest.mark.parametrize("begin,end", [(-1, 1), (2, 1), (0, 3)])
def test_invalid_flat_bounds_fail_closed(extraction, real_vllm_logprobs, begin, end):
    flat = make_flat(real_vllm_logprobs, [[(7, -0.1)]])
    flat.start_indices[0], flat.end_indices[0] = begin, end
    with pytest.raises(RuntimeError, match="invalid published bounds"):
        extraction._extract_selected_token_logprobs([7], flat)


def test_empty_cut_and_legacy_no_logprobs(extraction):
    assert extraction._extract_selected_token_logprobs([], None, stop=0) == []
    assert extraction.extract_published_generation([], None, start=0) == ([], [])
    with pytest.raises(RuntimeError, match="did not include logprobs"):
        extraction.extract_published_generation([7], None, start=0)


def test_published_token_count_mismatch_fails(extraction, real_vllm_logprobs):
    flat = make_flat(real_vllm_logprobs, [[(7, -0.1)], [(8, -0.2)]])
    with pytest.raises(RuntimeError, match="mismatched generation token IDs"):
        extraction.extract_published_generation([7], flat, start=0)
