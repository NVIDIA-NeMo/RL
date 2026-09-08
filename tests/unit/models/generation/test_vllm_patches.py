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

"""Guards for the vLLM source patches that had no coverage.

The two port patches ship their own suites. These cover the remaining two:

* ``_patch_vllm_tool_parser_namespace_tool`` is the most load-bearing patch in
  the repo -- it is the only thing that makes vLLM 0.25.1 importable against
  the pinned ``openai==2.6.1``. If upstream reorders that import block the
  patch logs a warning and returns, and every engine then dies on
  ``import vllm.tool_parsers``. So the anchor needs pinning.
* the ``VLLM_RAY_EXTRA_ENV_VARS_TO_COPY`` merge replaced the old
  ``ADDITIONAL_ENV_VARS`` file patch and is what now carries
  ``RAY_ENABLE_UV_RUN_RUNTIME_ENV`` and every user ``extra_env_vars`` to the
  Ray workers. Being additive rather than clobbering is the whole point of the
  rewrite, and it is pure string handling, so it is cheap to pin.
"""

import ast
import hashlib
import logging
import os
import sys
import types
from pathlib import Path

import pytest
import torch

from nemo_rl.models.generation.vllm import patches
from tests.unit.models.generation.vllm_patch_source_utils import (
    write_unpatched_copy,
)

_TOOL_PARSER_SOURCE = "tool_parsers/utils.py"
_PATCH_FN = "_patch_vllm_tool_parser_namespace_tool"
_MARKER = "except ImportError:  # openai < 2.25.0 predates namespace tools"
_RADIO_SOURCE = "model_executor/models/radio.py"
_RADIO_PATCH_FN = "_patch_vllm_radio_layerscale_loader"
_RADIO_MARKER = "initializer_factor = self.config.initializer_factor"
_ROUTED_EXPERTS_MARKER = "def get_routed_experts_layer_indices"


def test_external_vllm_patch_allowlist(monkeypatch):
    vllm_module = types.ModuleType("vllm")
    vllm_module.__path__ = []
    logger_module = types.ModuleType("vllm.logger")
    logger_module.init_logger = logging.getLogger
    monkeypatch.setitem(sys.modules, "vllm", vllm_module)
    monkeypatch.setitem(sys.modules, "vllm.logger", logger_module)

    calls = []
    allowed_patches = (
        "_patch_vllm_tool_parser_namespace_tool",
        "_patch_vllm_ray_executor_v2_tcpstore_port",
        "_patch_vllm_shm_broadcast_bind_retry",
    )
    for patch_name in allowed_patches:
        monkeypatch.setattr(
            patches,
            patch_name,
            lambda _logger, name=patch_name: calls.append(name),
        )

    patches._apply_external_vllm_patches()

    assert calls == list(allowed_patches)


@pytest.fixture
def patched_tool_parser_source(tmp_path, monkeypatch):
    """The installed tool_parsers/utils.py, unpatched then patched in tmp."""
    copied = write_unpatched_copy(_TOOL_PARSER_SOURCE, _PATCH_FN, tmp_path / "utils.py")
    monkeypatch.setattr(patches, "_get_vllm_file", lambda _relative: str(copied))
    patches._patch_vllm_tool_parser_namespace_tool(logging.getLogger(__name__))
    return copied


@pytest.fixture
def patched_radio_source(tmp_path, monkeypatch):
    """The installed vLLM RADIO loader, unpatched then patched in tmp."""
    copied = write_unpatched_copy(_RADIO_SOURCE, _RADIO_PATCH_FN, tmp_path / "radio.py")
    monkeypatch.setattr(patches, "_get_vllm_file", lambda _relative: str(copied))
    patches._patch_vllm_radio_layerscale_loader(logging.getLogger(__name__))
    return copied


@pytest.mark.vllm
def test_namespace_tool_patch_anchor_still_matches_installed_vllm(
    patched_tool_parser_source,
):
    """A source edit becomes a silent no-op if upstream reorders the import."""
    content = patched_tool_parser_source.read_text()
    assert _MARKER in content, (
        "the NamespaceTool compat patch did not apply to the installed vLLM; "
        "its anchor import block has probably changed upstream. Every vLLM "
        "engine will fail to import tool_parsers against the pinned openai."
    )
    ast.parse(content)  # the edit must leave valid Python


@pytest.mark.vllm
def test_namespace_tool_patch_is_idempotent(patched_tool_parser_source, monkeypatch):
    """Every worker on a node runs the patch against the same file."""
    before = patched_tool_parser_source.read_text()
    monkeypatch.setattr(
        patches, "_get_vllm_file", lambda _relative: str(patched_tool_parser_source)
    )
    patches._patch_vllm_tool_parser_namespace_tool(logging.getLogger(__name__))
    assert patched_tool_parser_source.read_text() == before


@pytest.mark.vllm
def test_namespace_tool_stub_never_matches(patched_tool_parser_source):
    """The stub must be a plain class, so isinstance() is always False.

    All upstream uses are ``isinstance(tool, NamespaceTool)`` guarding a
    namespace-tools branch, so degrading to "no namespace tools" is correct for
    a client that cannot construct them -- but only if nothing can be an
    instance of the stub.
    """
    namespace: dict = {}
    tree = ast.parse(patched_tool_parser_source.read_text())
    stub = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.ClassDef) and node.name == "NamespaceTool"
    )
    exec(compile(ast.Module(body=[stub], type_ignores=[]), "<stub>", "exec"), namespace)
    stub_cls = namespace["NamespaceTool"]
    for value in ({}, "tool", 0, None, object()):
        assert not isinstance(value, stub_cls)


@pytest.mark.vllm
def test_radio_layerscale_patch_anchor_still_matches_installed_vllm(
    patched_radio_source,
):
    """Pin the vLLM 0.25.1 RADIO loader shape used by the source patch."""
    content = patched_radio_source.read_text()
    assert _RADIO_MARKER in content
    assert "Skip layer-scale entries that vLLM doesn't use" not in content
    ast.parse(content)


@pytest.mark.vllm
def test_radio_layerscale_patch_loads_explicit_and_initializes_folded_weights(
    patched_radio_source,
):
    content = patched_radio_source.read_text()
    assert 'vllm_key = f"model.encoder.layers.{layer_idx}.{suffix}"' in content
    assert 'name.endswith((".ls1", ".ls2"))' in content
    assert "param.data.fill_(initializer_factor)" in content
    assert "loaded_params.add(name)" in content


@pytest.mark.vllm
def test_radio_layerscale_patch_is_idempotent(patched_radio_source, monkeypatch):
    before = patched_radio_source.read_text()
    monkeypatch.setattr(
        patches, "_get_vllm_file", lambda _relative: str(patched_radio_source)
    )

    patches._patch_vllm_radio_layerscale_loader(logging.getLogger(__name__))

    assert patched_radio_source.read_text() == before


def test_radio_layerscale_patch_warns_on_unknown_source(monkeypatch, tmp_path, caplog):
    radio_source = tmp_path / "radio.py"
    radio_source.write_text("class RadioModel:\n    pass\n")
    monkeypatch.setattr(patches, "_get_vllm_file", lambda _relative: str(radio_source))

    with caplog.at_level(logging.WARNING):
        patches._patch_vllm_radio_layerscale_loader(logging.getLogger(__name__))

    assert radio_source.read_text() == "class RadioModel:\n    pass\n"
    assert "vLLM 0.25.1 source shape was not found" in caplog.text


@pytest.fixture
def compact_routed_experts_source(tmp_path, monkeypatch):
    """Copy and compact one of the two supported vLLM capturer sources."""
    installed_source = Path(
        patches._get_vllm_file(patches._ROUTED_EXPERTS_CAPTURER_SOURCE)
    )
    copied_source = tmp_path / "routed_experts_capturer.py"
    copied_source.write_text(installed_source.read_text())
    monkeypatch.setattr(patches, "_get_vllm_file", lambda _relative: str(copied_source))

    before_sha256 = hashlib.sha256(copied_source.read_bytes()).hexdigest()
    assert before_sha256 in {
        patches._ROUTED_EXPERTS_CAPTURER_V0251_SHA256,
        patches._ROUTED_EXPERTS_CAPTURER_COMPACT_SHA256,
    }
    assert patches._patch_vllm_routed_experts_compact_layers(
        logging.getLogger(__name__)
    )
    return copied_source


@pytest.mark.vllm
def test_routed_experts_compact_source_patch_matches_pinned_result(
    compact_routed_experts_source,
):
    content = compact_routed_experts_source.read_text()
    assert hashlib.sha256(content.encode()).hexdigest() == (
        patches._ROUTED_EXPERTS_CAPTURER_COMPACT_SHA256
    )
    assert _ROUTED_EXPERTS_MARKER in content
    assert "routed_layers=%d" in content
    assert "self.layer_id_to_capture_index" in content
    ast.parse(content)


@pytest.mark.vllm
def test_routed_experts_compact_source_patch_is_idempotent(
    compact_routed_experts_source, monkeypatch
):
    before = compact_routed_experts_source.read_text()
    monkeypatch.setattr(
        patches,
        "_get_vllm_file",
        lambda _relative: str(compact_routed_experts_source),
    )

    assert patches._patch_vllm_routed_experts_compact_layers(
        logging.getLogger(__name__)
    )
    assert compact_routed_experts_source.read_text() == before


@pytest.mark.vllm
def test_routed_experts_compact_source_selects_only_hybrid_moe_layers(
    compact_routed_experts_source,
):
    tree = ast.parse(compact_routed_experts_source.read_text())
    resolver = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef)
        and node.name == "get_routed_experts_layer_indices"
    )
    namespace = {}
    exec(
        compile(ast.Module(body=[resolver], type_ignores=[]), "<resolver>", "exec"),
        namespace,
    )
    resolve = namespace["get_routed_experts_layer_indices"]

    hybrid = types.SimpleNamespace(
        num_hidden_layers=6,
        layers_block_type=["mamba", "moe", "attention", "moe", "mamba", "moe"],
        mtp_layers_block_type=["moe"],
    )
    fallback = types.SimpleNamespace(num_hidden_layers=4)
    assert resolve(hybrid) == (1, 3, 5)
    assert resolve(fallback) == (0, 1, 2, 3)


def test_routed_experts_compact_source_patch_rejects_unknown_source(
    monkeypatch, tmp_path, caplog
):
    unknown_source = tmp_path / "routed_experts_capturer.py"
    unknown_source.write_text("class RoutedExpertsCapturer:\n    pass\n")
    monkeypatch.setattr(
        patches, "_get_vllm_file", lambda _relative: str(unknown_source)
    )

    with caplog.at_level(logging.ERROR):
        assert not patches._patch_vllm_routed_experts_compact_layers(
            logging.getLogger(__name__)
        )

    assert unknown_source.read_text() == "class RoutedExpertsCapturer:\n    pass\n"
    assert "Refusing to patch unknown vLLM routed-experts capturer" in caplog.text


@pytest.mark.parametrize(
    "existing,extra,expected",
    [
        (None, None, "RAY_ENABLE_UV_RUN_RUNTIME_ENV"),
        ("", ["MY_VAR"], "MY_VAR,RAY_ENABLE_UV_RUN_RUNTIME_ENV"),
        # A value the caller already set must survive, not be clobbered.
        ("PRESET", ["MY_VAR"], "MY_VAR,PRESET,RAY_ENABLE_UV_RUN_RUNTIME_ENV"),
        # Duplicates collapse and surrounding whitespace is stripped.
        (
            " PRESET , MY_VAR ",
            ["MY_VAR"],
            "MY_VAR,PRESET,RAY_ENABLE_UV_RUN_RUNTIME_ENV",
        ),
    ],
)
def test_ray_extra_env_vars_merge_is_additive(
    monkeypatch, tmp_path, existing, extra, expected
):
    """vLLM 0.25 replaced the ADDITIONAL_ENV_VARS source patch with this hook.

    It must add to whatever the caller already set rather than overwrite it --
    otherwise user ``extra_env_vars`` silently stop reaching the Ray workers.
    """
    ray_executor = tmp_path / "ray_executor.py"
    ray_executor.write_text("self._init_workers_ray(placement_group)\n")
    monkeypatch.setattr(patches, "_get_vllm_file", lambda _r: str(ray_executor))

    if existing is None:
        monkeypatch.delenv("VLLM_RAY_EXTRA_ENV_VARS_TO_COPY", raising=False)
    else:
        monkeypatch.setenv("VLLM_RAY_EXTRA_ENV_VARS_TO_COPY", existing)

    patches._patch_vllm_init_workers_ray("py", extra)

    assert os.environ["VLLM_RAY_EXTRA_ENV_VARS_TO_COPY"] == expected


def test_init_workers_ray_reports_a_missing_anchor(monkeypatch, tmp_path):
    """A reshaped call site must not be reported as a successful patch."""
    ray_executor = tmp_path / "ray_executor.py"
    ray_executor.write_text("self._init_workers_ray_renamed(placement_group)\n")
    monkeypatch.setattr(patches, "_get_vllm_file", lambda _r: str(ray_executor))
    monkeypatch.delenv("VLLM_RAY_EXTRA_ENV_VARS_TO_COPY", raising=False)

    assert patches._patch_vllm_init_workers_ray("py", None) is False
    # The env merge still has to happen; it is independent of the file patch.
    assert os.environ["VLLM_RAY_EXTRA_ENV_VARS_TO_COPY"] == (
        "RAY_ENABLE_UV_RUN_RUNTIME_ENV"
    )


def test_init_workers_ray_reports_success_and_is_idempotent(monkeypatch, tmp_path):
    """Patching twice against the same file still reports success."""
    ray_executor = tmp_path / "ray_executor.py"
    ray_executor.write_text("self._init_workers_ray(placement_group)\n")
    monkeypatch.setattr(patches, "_get_vllm_file", lambda _r: str(ray_executor))

    assert patches._patch_vllm_init_workers_ray("py-exec", None) is True
    once = ray_executor.read_text()
    assert 'runtime_env={"py_executable": "py-exec"}' in once

    assert patches._patch_vllm_init_workers_ray("py-exec", None) is True
    assert ray_executor.read_text() == once


@pytest.fixture
def fake_vllm_unquantized_moe_modules(monkeypatch):
    """Install a tiny fake vLLM MoE module tree for the runtime monkeypatch."""
    package_names = [
        "vllm",
        "vllm.model_executor",
        "vllm.model_executor.layers",
        "vllm.model_executor.layers.fused_moe",
        "vllm.model_executor.layers.fused_moe.oracle",
    ]
    for name in package_names:
        module = types.ModuleType(name)
        module.__path__ = []
        monkeypatch.setitem(sys.modules, name, module)

    method_module = types.ModuleType(
        "vllm.model_executor.layers.fused_moe.unquantized_fused_moe_method"
    )
    oracle_module = types.ModuleType(
        "vllm.model_executor.layers.fused_moe.oracle.unquantized"
    )
    monkeypatch.setitem(sys.modules, method_module.__name__, method_module)
    monkeypatch.setitem(sys.modules, oracle_module.__name__, oracle_module)

    class FakeUnquantizedMoeBackend:
        FLASHINFER_TRTLLM = "flashinfer_trtllm"
        TRITON = "triton"

    class FakeExperts:
        pass

    class FakeKernel:
        def __init__(self):
            self.apply_kwargs = None
            self.apply_monolithic_args = None
            self.apply_monolithic_kwargs = None

        def apply(self, **kwargs):
            self.apply_kwargs = kwargs
            return kwargs["w1"].shape, kwargs["w2"].shape

        def apply_monolithic(self, *args, **kwargs):
            self.apply_monolithic_args = args
            self.apply_monolithic_kwargs = kwargs
            return args[1].shape, args[2].shape

    kernels = []

    def make_unquantized_moe_kernel(**_kwargs):
        kernel = FakeKernel()
        kernels.append(kernel)
        return kernel

    def convert_to_unquantized_kernel_format(
        _unquantized_backend,
        *,
        moe_config,
        w13_weight,
        w2_weight,
    ):
        moe_config.converted = True
        return w13_weight.reshape(4, 2).clone(), w2_weight.reshape(2, 4).clone()

    class FakeUnquantizedFusedMoEMethod:
        def __init__(self, backend):
            self.unquantized_backend = backend
            self.moe_kernel = None
            self.moe_quant_config = None
            self.moe = types.SimpleNamespace(name="moe")
            self.experts_cls = FakeExperts
            self.is_monolithic = True
            self.original_setup_called = False
            self.original_forward_native_called = False
            self.original_apply_monolithic_called = False

        def _setup_kernel(self, layer, w13, w2):
            self.original_setup_called = True

        def forward_native(
            self,
            layer,
            x,
            topk_weights,
            topk_ids,
            shared_experts,
            shared_experts_input,
        ):
            self.original_forward_native_called = True
            return "original-forward-native"

        def apply_monolithic(self, layer, x, router_logits, input_ids=None):
            self.original_apply_monolithic_called = True
            return "original-apply-monolithic"

        def get_fused_moe_quant_config(self, layer):
            return types.SimpleNamespace(name="quant")

    class FakeLayer(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.moe_config = types.SimpleNamespace(converted=False)
            self.w13_weight = torch.nn.Parameter(
                torch.arange(8, dtype=torch.float32).reshape(2, 4),
                requires_grad=False,
            )
            self.w2_weight = torch.nn.Parameter(
                torch.arange(8, 16, dtype=torch.float32).reshape(4, 2),
                requires_grad=False,
            )
            self.activation = "silu"
            self.apply_router_weight_on_input = False
            self.global_num_experts = 2
            self.expert_map = None
            self.num_expert_group = 1
            self.topk_group = 1
            self.e_score_correction_bias = None
            self.routed_scaling_factor = 1.0

        def _expert_routing_tables(self):
            return ("route",)

    oracle_module.UnquantizedMoeBackend = FakeUnquantizedMoeBackend
    oracle_module.convert_to_unquantized_kernel_format = (
        convert_to_unquantized_kernel_format
    )
    oracle_module.make_unquantized_moe_kernel = make_unquantized_moe_kernel
    method_module.UnquantizedFusedMoEMethod = FakeUnquantizedFusedMoEMethod

    return types.SimpleNamespace(
        backend=FakeUnquantizedMoeBackend,
        method_cls=FakeUnquantizedFusedMoEMethod,
        layer_cls=FakeLayer,
        oracle_module=oracle_module,
        kernels=kernels,
    )


def test_flashinfer_trtllm_refit_patch_preserves_kernel_weight_storage(
    fake_vllm_unquantized_moe_modules,
):
    fake_vllm = fake_vllm_unquantized_moe_modules
    patches._apply_vllm_flashinfer_trtllm_refit_buffer_runtime_patch(
        logging.getLogger(__name__)
    )

    method = fake_vllm.method_cls(fake_vllm.backend.FLASHINFER_TRTLLM)
    layer = fake_vllm.layer_cls()
    method._setup_kernel(layer, layer.w13_weight, layer.w2_weight)

    w13_kernel = getattr(layer, patches.G_FLASHINFER_TRTLLM_W13_KERNEL_ATTR)
    w2_kernel = getattr(layer, patches.G_FLASHINFER_TRTLLM_W2_KERNEL_ATTR)
    first_kernel = method.moe_kernel
    assert layer.w13_weight.shape == torch.Size([2, 4])
    assert layer.w2_weight.shape == torch.Size([4, 2])
    assert w13_kernel.shape == torch.Size([4, 2])
    assert w2_kernel.shape == torch.Size([2, 4])
    assert layer.w13_weight.data_ptr() == w13_kernel.data_ptr()
    assert layer.w2_weight.data_ptr() == w2_kernel.data_ptr()

    w13_kernel_id = id(w13_kernel)
    w2_kernel_id = id(w2_kernel)
    w13_kernel_ptr = w13_kernel.data_ptr()
    w2_kernel_ptr = w2_kernel.data_ptr()

    with torch.no_grad():
        layer.w13_weight.copy_(torch.arange(100, 108, dtype=torch.float32).view(2, 4))
        layer.w2_weight.copy_(torch.arange(200, 208, dtype=torch.float32).view(4, 2))

    method._setup_kernel(layer, layer.w13_weight, layer.w2_weight)

    assert method.moe_kernel is first_kernel
    assert getattr(layer, patches.G_FLASHINFER_TRTLLM_W13_KERNEL_ATTR) is w13_kernel
    assert getattr(layer, patches.G_FLASHINFER_TRTLLM_W2_KERNEL_ATTR) is w2_kernel
    assert id(w13_kernel) == w13_kernel_id
    assert id(w2_kernel) == w2_kernel_id
    assert w13_kernel.data_ptr() == w13_kernel_ptr
    assert w2_kernel.data_ptr() == w2_kernel_ptr
    assert torch.equal(
        w13_kernel, torch.arange(100, 108, dtype=torch.float32).view(4, 2)
    )
    assert torch.equal(
        w2_kernel, torch.arange(200, 208, dtype=torch.float32).view(2, 4)
    )

    x = torch.empty(1)
    topk_weights = torch.empty(1)
    topk_ids = torch.empty(1, dtype=torch.int64)
    assert method.forward_native(layer, x, topk_weights, topk_ids, None, None) == (
        torch.Size([4, 2]),
        torch.Size([2, 4]),
    )
    assert fake_vllm.kernels[-1].apply_kwargs["w1"] is w13_kernel
    assert fake_vllm.kernels[-1].apply_kwargs["w2"] is w2_kernel

    assert method.apply_monolithic(layer, x, torch.empty(1)) == (
        torch.Size([4, 2]),
        torch.Size([2, 4]),
    )
    assert fake_vllm.kernels[-1].apply_monolithic_args[1] is w13_kernel
    assert fake_vllm.kernels[-1].apply_monolithic_args[2] is w2_kernel


def test_flashinfer_trtllm_refit_patch_leaves_other_backends_on_original_paths(
    fake_vllm_unquantized_moe_modules,
):
    fake_vllm = fake_vllm_unquantized_moe_modules
    patches._apply_vllm_flashinfer_trtllm_refit_buffer_runtime_patch(
        logging.getLogger(__name__)
    )

    method = fake_vllm.method_cls(fake_vllm.backend.TRITON)
    layer = fake_vllm.layer_cls()

    method._setup_kernel(layer, layer.w13_weight, layer.w2_weight)
    assert method.original_setup_called
    assert method.forward_native(layer, None, None, None, None, None) == (
        "original-forward-native"
    )
    assert method.original_forward_native_called
    assert method.apply_monolithic(layer, None, None) == "original-apply-monolithic"
    assert method.original_apply_monolithic_called
    assert not hasattr(layer, patches.G_FLASHINFER_TRTLLM_W13_KERNEL_ATTR)
    assert not hasattr(layer, patches.G_FLASHINFER_TRTLLM_W2_KERNEL_ATTR)


def test_flashinfer_trtllm_refit_patch_rejects_padded_non_gated_layout_early(
    fake_vllm_unquantized_moe_modules,
):
    fake_vllm = fake_vllm_unquantized_moe_modules

    def convert_should_not_run(*_args, **_kwargs):
        raise AssertionError("conversion should not run for padded layouts")

    fake_vllm.oracle_module.convert_to_unquantized_kernel_format = (
        convert_should_not_run
    )
    patches._apply_vllm_flashinfer_trtllm_refit_buffer_runtime_patch(
        logging.getLogger(__name__)
    )

    method = fake_vllm.method_cls(fake_vllm.backend.FLASHINFER_TRTLLM)
    layer = fake_vllm.layer_cls()
    layer.moe_config.is_act_and_mul = False
    layer.moe_config.intermediate_size_per_partition = 672

    with pytest.raises(
        ValueError,
        match="intermediate_size_per_partition=672.*not divisible by 128",
    ):
        method._setup_kernel(layer, layer.w13_weight, layer.w2_weight)


def test_flashinfer_trtllm_refit_patch_rejects_expanded_kernel_layout(
    fake_vllm_unquantized_moe_modules,
):
    fake_vllm = fake_vllm_unquantized_moe_modules

    def convert_to_expanded_kernel_format(
        _unquantized_backend,
        *,
        moe_config,
        w13_weight,
        w2_weight,
    ):
        return torch.empty(9), w2_weight.reshape(2, 4).clone()

    fake_vllm.oracle_module.convert_to_unquantized_kernel_format = (
        convert_to_expanded_kernel_format
    )
    patches._apply_vllm_flashinfer_trtllm_refit_buffer_runtime_patch(
        logging.getLogger(__name__)
    )

    method = fake_vllm.method_cls(fake_vllm.backend.FLASHINFER_TRTLLM)
    layer = fake_vllm.layer_cls()

    with pytest.raises(ValueError, match="preserve the number of elements"):
        method._setup_kernel(layer, layer.w13_weight, layer.w2_weight)


@pytest.fixture
def fake_vllm_process_entrypoint_modules(monkeypatch):
    """Install fake vLLM process-entrypoint modules for propagation tests."""
    package_names = [
        "vllm",
        "vllm.v1",
        "vllm.v1.engine",
        "vllm.v1.executor",
    ]
    for name in package_names:
        module = types.ModuleType(name)
        module.__path__ = []
        monkeypatch.setitem(sys.modules, name, module)

    engine_core_module = types.ModuleType("vllm.v1.engine.core")
    ray_executor_v2_module = types.ModuleType("vllm.v1.executor.ray_executor_v2")
    ray_executor_module = types.ModuleType("vllm.v1.executor.ray_executor")
    ray_utils_module = types.ModuleType("vllm.v1.executor.ray_utils")
    logger_module = types.ModuleType("vllm.logger")
    monkeypatch.setitem(sys.modules, engine_core_module.__name__, engine_core_module)
    monkeypatch.setitem(
        sys.modules, ray_executor_v2_module.__name__, ray_executor_v2_module
    )
    monkeypatch.setitem(sys.modules, ray_executor_module.__name__, ray_executor_module)
    monkeypatch.setitem(sys.modules, ray_utils_module.__name__, ray_utils_module)
    monkeypatch.setitem(sys.modules, logger_module.__name__, logger_module)

    calls = []

    class FakeEngineCoreProc:
        @staticmethod
        def run_engine_core(*args, **kwargs):
            calls.append(("engine_core", args, kwargs))
            return "engine-core-result"

    class FakeRayWorkerProc:
        def initialize_worker(self, *args, **kwargs):
            calls.append(("ray_worker_initialize", args, kwargs))
            return "ray-worker-result"

    class FakeWorkerHandle:
        def __init__(self, name):
            self.name = name
            self.execute_method = types.SimpleNamespace(remote=self.remote)

        def remote(self, func):
            calls.append(("ray_execute_method", self.name, func))
            func(None)
            return f"{self.name}-future"

    class FakeRay:
        @staticmethod
        def get(futures):
            calls.append(("ray_get", tuple(futures)))
            return futures

    class FakeRayDistributedExecutor:
        def __init__(self):
            self.workers = [FakeWorkerHandle("w0"), FakeWorkerHandle("w1")]

        def collective_rpc(self, *args, **kwargs):
            calls.append(("ray_collective_rpc", args, kwargs))
            return "ray-collective-result"

    engine_core_module.EngineCoreProc = FakeEngineCoreProc
    ray_executor_v2_module.RayWorkerProc = FakeRayWorkerProc
    ray_executor_module.RayDistributedExecutor = FakeRayDistributedExecutor
    ray_utils_module.ray = FakeRay
    logger_module.init_logger = logging.getLogger

    return types.SimpleNamespace(
        calls=calls,
        engine_core_cls=FakeEngineCoreProc,
        ray_worker_cls=FakeRayWorkerProc,
        ray_executor_cls=FakeRayDistributedExecutor,
    )


def test_flashinfer_trtllm_refit_collective_rpc_patch_is_idempotent(
    fake_vllm_process_entrypoint_modules,
    monkeypatch,
):
    fake_vllm = fake_vllm_process_entrypoint_modules
    runtime_calls = []

    def apply_runtime_patch(logger=None):
        runtime_calls.append(logger)

    monkeypatch.setattr(
        patches,
        "_apply_vllm_flashinfer_trtllm_refit_buffer_runtime_patch",
        apply_runtime_patch,
    )

    original_engine_core = fake_vllm.engine_core_cls.run_engine_core
    original_ray_worker = fake_vllm.ray_worker_cls.initialize_worker
    original_ray_collective = fake_vllm.ray_executor_cls.collective_rpc

    patches._patch_vllm_flashinfer_trtllm_refit_from_collective_rpc(
        logging.getLogger(__name__)
    )
    first_engine_core = fake_vllm.engine_core_cls.run_engine_core
    first_ray_worker = fake_vllm.ray_worker_cls.initialize_worker
    first_ray_collective = fake_vllm.ray_executor_cls.collective_rpc

    patches._patch_vllm_flashinfer_trtllm_refit_from_collective_rpc(
        logging.getLogger(__name__)
    )

    assert fake_vllm.engine_core_cls.run_engine_core is first_engine_core
    assert fake_vllm.ray_worker_cls.initialize_worker is first_ray_worker
    assert fake_vllm.ray_executor_cls.collective_rpc is first_ray_collective
    assert (
        getattr(
            fake_vllm.engine_core_cls,
            patches.G_FLASHINFER_TRTLLM_ENGINE_CORE_ORIGINAL_ATTR,
        )
        is original_engine_core
    )
    assert (
        getattr(
            fake_vllm.ray_worker_cls,
            patches.G_FLASHINFER_TRTLLM_RAY_WORKER_ORIGINAL_ATTR,
        )
        is original_ray_worker
    )
    assert (
        getattr(
            fake_vllm.ray_executor_cls,
            patches.G_FLASHINFER_TRTLLM_RAY_EXECUTOR_ORIGINAL_ATTR,
        )
        is original_ray_collective
    )

    assert (
        fake_vllm.engine_core_cls.run_engine_core("engine", vllm_config="cfg")
        == "engine-core-result"
    )
    assert (
        fake_vllm.ray_worker_cls().initialize_worker("ray-worker")
        == "ray-worker-result"
    )

    ray_executor = fake_vllm.ray_executor_cls()
    assert ray_executor.collective_rpc("ray-call") == "ray-collective-result"
    assert ray_executor.collective_rpc("ray-call-2") == "ray-collective-result"

    assert len(runtime_calls) == 4
    assert fake_vllm.calls == [
        ("engine_core", ("engine",), {"vllm_config": "cfg"}),
        ("ray_worker_initialize", ("ray-worker",), {}),
        (
            "ray_execute_method",
            "w0",
            patches._apply_vllm_flashinfer_trtllm_refit_buffer_runtime_patch_on_worker,
        ),
        (
            "ray_execute_method",
            "w1",
            patches._apply_vllm_flashinfer_trtllm_refit_buffer_runtime_patch_on_worker,
        ),
        ("ray_get", ("w0-future", "w1-future")),
        ("ray_collective_rpc", ("ray-call",), {}),
        ("ray_collective_rpc", ("ray-call-2",), {}),
    ]
