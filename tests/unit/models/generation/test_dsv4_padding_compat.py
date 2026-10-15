# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
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

"""CPU-only checks for the opt-in external DSV4 wrapper compatibility patch."""

import ast
import importlib
import importlib.util
import logging
import sys
import types
from pathlib import Path
from unittest.mock import Mock

import pytest

# Like serve_vllm_on_ray.py, load patches without importing the training stack.
_path = (
    Path(__file__).resolve().parents[4] / "nemo_rl/models/generation/vllm/patches.py"
)
_spec = importlib.util.spec_from_file_location("external_dsv4_patches", _path)
patches = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(patches)


@pytest.fixture
def dsv4_source(tmp_path, monkeypatch):
    # Extract the pinned old snippet; assertions below independently exercise
    # its runtime call signature, forwarding, idempotency and failure paths.
    tree = ast.parse(Path(patches.__file__).read_text())
    fn = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "_patch_vllm_dsv4_topk_padding"
    )
    old = next(
        ast.literal_eval(node.value)
        for node in fn.body
        if isinstance(node, ast.Assign)
        and any(isinstance(t, ast.Name) and t.id == "old_snippet" for t in node.targets)
    )
    source = tmp_path / "_custom_ops.py"
    source.write_text("import torch\n\n" + old)
    monkeypatch.setattr(patches, "_get_vllm_file", lambda _: str(source))

    vllm = types.ModuleType("vllm")
    vllm.__path__ = [str(tmp_path)]
    monkeypatch.setitem(sys.modules, "vllm", vllm)
    monkeypatch.delitem(sys.modules, "vllm._custom_ops", raising=False)
    # Ensure the imported temporary module is cleaned up by monkeypatch too.
    monkeypatch.setitem(sys.modules, "vllm._custom_ops", None)
    del sys.modules["vllm._custom_ops"]
    kernel = Mock()
    kernel.default._schema.arguments = [None] * 9 + [
        types.SimpleNamespace(name="is_padding", type="Optional[Tensor]")
    ]
    torch = types.ModuleType("torch")
    torch.Tensor = type("Tensor", (), {})
    torch.ops = types.SimpleNamespace(
        _moe_C=types.SimpleNamespace(topk_softplus_sqrt=kernel)
    )
    monkeypatch.setitem(sys.modules, "torch", torch)
    return source, kernel


def test_dsv4_padding_forwarding_and_idempotency(dsv4_source):
    source, kernel = dsv4_source
    logger = logging.getLogger(__name__)
    patches._patch_vllm_dsv4_topk_padding(logger)
    patched = source.read_text()
    ops = importlib.import_module("vllm._custom_ops")
    args = tuple(object() for _ in range(9))
    ops.topk_hash_softplus_sqrt(*args)
    kernel.assert_called_once_with(*args, None)
    mask = object()
    ops.topk_hash_softplus_sqrt(*args, is_padding=mask)
    kernel.assert_called_with(*args, mask)
    patches._patch_vllm_dsv4_topk_padding(logger)
    assert source.read_text() == patched
    assert patched.count("    is_padding: torch.Tensor | None = None,") == 1


def test_dsv4_padding_rejects_unknown_source(dsv4_source):
    source, _ = dsv4_source
    unknown = source.read_text().replace(
        "routed_scaling_factor: float = 1.0", "routed_scaling_factor: float = 2.0"
    )
    source.write_text(unknown)
    with pytest.raises(RuntimeError, match="unknown DSV4 topk wrapper"):
        patches._patch_vllm_dsv4_topk_padding(logging.getLogger(__name__))
    assert source.read_text() == unknown


def test_dsv4_padding_rejects_already_imported_old_wrapper(dsv4_source):
    source, _ = dsv4_source
    original = source.read_text()
    importlib.import_module("vllm._custom_ops")
    with pytest.raises(RuntimeError, match="before importing"):
        patches._patch_vllm_dsv4_topk_padding(logging.getLogger(__name__))
    assert source.read_text() == original


@pytest.mark.parametrize("bad_schema", ["arity", "name", "type"])
def test_dsv4_padding_rejects_incompatible_kernel(dsv4_source, bad_schema):
    _, kernel = dsv4_source
    args = kernel.default._schema.arguments
    if bad_schema == "arity":
        args.pop()
    elif bad_schema == "name":
        args[-1].name = "different_argument"
    else:
        args[-1].type = "Tensor"
    with pytest.raises(RuntimeError, match="Unsupported DSV4 wrapper/kernel pair"):
        patches._patch_vllm_dsv4_topk_padding(logging.getLogger(__name__))


@pytest.mark.parametrize("enabled", [False, True])
def test_dsv4_padding_external_opt_in(monkeypatch, enabled):
    monkeypatch.setenv("NRL_DSV4_TOPK_PADDING_COMPAT", str(int(enabled)))
    vllm = types.ModuleType("vllm")
    vllm.__path__ = []
    logger = types.ModuleType("vllm.logger")
    logger.init_logger = logging.getLogger
    monkeypatch.setitem(sys.modules, "vllm", vllm)
    monkeypatch.setitem(sys.modules, "vllm.logger", logger)
    for name in (
        "_patch_vllm_tool_parser_namespace_tool",
        "_patch_vllm_ray_executor_v2_tcpstore_port",
        "_patch_vllm_shm_broadcast_bind_retry",
    ):
        monkeypatch.setattr(patches, name, lambda _: None)
    patch = Mock()
    monkeypatch.setattr(patches, "_patch_vllm_dsv4_topk_padding", patch)
    patches._apply_external_vllm_patches()
    assert patch.call_count == int(enabled)
