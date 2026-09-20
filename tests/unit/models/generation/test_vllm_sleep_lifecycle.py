# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Run production lifecycle methods without CUDA/Ray imports."""
import ast
import asyncio
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

ROOT = Path(__file__).resolve().parents[4] / 'nemo_rl/models/generation/vllm'


def method(filename, name, namespace):
    tree = ast.parse((ROOT / filename).read_text())
    node = next(n for cls in tree.body if isinstance(cls, ast.ClassDef)
                for n in cls.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
                and n.name == name)
    node.decorator_list = []
    exec(compile(ast.Module(body=[node], type_ignores=[]), filename, 'exec'), namespace)
    return namespace[name]


@pytest.mark.parametrize('async_engine', [False, True])
@pytest.mark.parametrize('level', [None, 1, 2])
def test_sleep_forwards_level_and_preserves_default(async_engine, level):
    cfg = {'async_engine': async_engine}
    if level is not None:
        cfg['sleep_level'] = level
    llm = SimpleNamespace(sleep=AsyncMock() if async_engine else Mock(),
                          reset_prefix_cache=AsyncMock(),
                          llm_engine=SimpleNamespace(reset_prefix_cache=Mock()))
    worker = SimpleNamespace(cfg={'vllm_cfg': cfg}, llm=llm)
    ns = {'gc': SimpleNamespace(collect=Mock()),
          'torch': SimpleNamespace(cuda=SimpleNamespace(empty_cache=Mock()))}
    if async_engine:
        asyncio.run(method('vllm_worker_async.py', 'sleep_async', ns)(worker))
        llm.sleep.assert_awaited_once_with(level=1 if level is None else level)
    else:
        method('vllm_worker.py', 'sleep', ns)(worker)
        llm.sleep.assert_called_once_with(level=1 if level is None else level)


@pytest.mark.parametrize('name', ['finish_generation', 'prepare_for_generation'])
def test_transition_failure_propagates_original_cause(name):
    failure = RuntimeError('worker killed during CPU weight backup')
    worker_group = SimpleNamespace(run_all_workers_single_data=Mock(return_value=['ref']))
    generation = SimpleNamespace(cfg={'colocated': {'enabled': True},
                                      'vllm_cfg': {'async_engine': False}},
                                 worker_group=worker_group)
    ns = {'Any': object, 'ray': SimpleNamespace(get=Mock(side_effect=failure))}
    fn = method('vllm_generation.py', name, ns)
    with pytest.raises(RuntimeError, match='sleep/wake transition failed') as result:
        fn(generation)
    assert result.value.__cause__ is failure
