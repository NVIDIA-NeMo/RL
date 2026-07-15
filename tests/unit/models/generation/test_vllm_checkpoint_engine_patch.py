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

"""Tests for the vLLM worker patch used by NIXL checkpoint-engine refit."""

import sys
from contextlib import contextmanager
from types import ModuleType
from unittest.mock import MagicMock

import pytest

from nemo_rl.models.generation.vllm import checkpoint_engine_patch, patches


@pytest.mark.parametrize(
    "config",
    [None, {}, {"enabled": False}, {"enabled": True, "backend": "custom"}],
)
def test_nixl_preinit_patch_ignores_disabled_configs(monkeypatch, config):
    monkeypatch.setattr(
        checkpoint_engine_patch,
        "_get_vllm_file",
        lambda _path: pytest.fail("disabled config must not inspect vLLM files"),
    )

    checkpoint_engine_patch.patch_vllm_worker_nixl_preinit(config)


def test_nixl_preinit_patch_injects_extracted_checkpoint_engine_hook(monkeypatch):
    old_snippet = (
        "        with set_current_vllm_config(self.vllm_config):\n"
        "            # To make vLLM config available during worker initialization\n"
        "            self.worker = worker_class(**kwargs)"
    )
    writes = []

    @contextmanager
    def locked_patch(file_path):
        assert file_path == "/tmp/worker_base.py"
        yield old_snippet, writes.append

    monkeypatch.setattr(
        checkpoint_engine_patch,
        "_get_vllm_file",
        lambda _path: "/tmp/worker_base.py",
    )
    monkeypatch.setattr(checkpoint_engine_patch, "_locked_file_patch", locked_patch)

    from nemo_rl.utils.checkpoint_engines import nixl

    monkeypatch.setattr(
        nixl,
        "resolve_nixl_backend_kwargs",
        lambda _kwargs: ("UCX", {"foo": "bar"}),
    )

    checkpoint_engine_patch.patch_vllm_worker_nixl_preinit(
        {
            "enabled": True,
            "backend": "nixl",
            "engine_kwargs": {"nixl": {}},
        }
    )

    assert len(writes) == 1
    assert (
        "from nemo_rl.models.generation.vllm.checkpoint_engine import "
        "maybe_preinit_nixl_for_vllm_worker"
    ) in writes[0]
    assert "backend_name='UCX'" in writes[0]
    assert "backend_init_params={'foo': 'bar'}" in writes[0]


def test_nixl_preinit_patch_leaves_unknown_vllm_layout_unchanged(monkeypatch):
    writes = []

    @contextmanager
    def locked_patch(_file_path):
        yield "unexpected vLLM source", writes.append

    monkeypatch.setattr(
        checkpoint_engine_patch,
        "_get_vllm_file",
        lambda _path: "/tmp/worker_base.py",
    )
    monkeypatch.setattr(checkpoint_engine_patch, "_locked_file_patch", locked_patch)

    checkpoint_engine_patch.patch_vllm_worker_nixl_preinit(
        {
            "enabled": True,
            "backend": "nixl",
            "engine_kwargs": {"nixl": {}},
        }
    )

    assert writes == []


def test_apply_vllm_patches_dispatches_checkpoint_engine_patch(monkeypatch):
    logger_module = ModuleType("vllm.logger")
    logger_module.init_logger = MagicMock(return_value=MagicMock())
    monkeypatch.setitem(sys.modules, "vllm", ModuleType("vllm"))
    monkeypatch.setitem(sys.modules, "vllm.logger", logger_module)

    monkeypatch.setattr(patches, "_patch_vllm_init_workers_ray", MagicMock())
    monkeypatch.setattr(patches, "_patch_vllm_llama_eagle3_own_lm_head", MagicMock())
    monkeypatch.setattr(
        patches, "_patch_vllm_hermes_tool_parser_thread_safety", MagicMock()
    )
    checkpoint_patch = MagicMock()
    monkeypatch.setattr(
        checkpoint_engine_patch,
        "patch_vllm_worker_nixl_preinit",
        checkpoint_patch,
    )
    config = {"enabled": True, "backend": "nixl"}

    patches._apply_vllm_patches(
        "/usr/bin/python",
        extra_env_vars=["UCX_TLS"],
        checkpoint_engine_config=config,
    )

    patches._patch_vllm_init_workers_ray.assert_called_once_with(
        "/usr/bin/python", ["UCX_TLS"]
    )
    checkpoint_patch.assert_called_once_with(config)
