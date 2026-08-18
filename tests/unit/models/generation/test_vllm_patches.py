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
import importlib.util
import logging
import os
from pathlib import Path

import numpy as np
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
_VIDEO_SOURCE = "transformers_utils/processors/nano_nemotron_vl.py"
_VIDEO_PATCH_FN = "_patch_vllm_nemotron_video_preprocessing"
_VIDEO_MARKER = "# patched by nemo_rl: match policy-side video resize"


class _Logger:
    def __init__(self):
        self.info_messages = []
        self.warning_messages = []

    def info(self, message, *args):
        self.info_messages.append(message % args if args else message)

    def warning(self, message, *args):
        self.warning_messages.append(message % args if args else message)


def _load_module(path: Path):
    spec = importlib.util.spec_from_file_location("patched_nano_nemotron_vl", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


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


@pytest.fixture
def patched_video_source(tmp_path, monkeypatch):
    """The installed Nemotron video processor, unpatched then patched in tmp."""
    copied = write_unpatched_copy(
        _VIDEO_SOURCE, _VIDEO_PATCH_FN, tmp_path / "nano_nemotron_vl.py"
    )
    monkeypatch.setattr(patches, "_get_vllm_file", lambda _relative: str(copied))
    monkeypatch.setattr(patches, "version", lambda _package: "0.25.1")
    patches._patch_vllm_nemotron_video_preprocessing(
        logging.getLogger(__name__), required=True
    )
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


@pytest.mark.vllm
def test_nemotron_video_patch_anchor_still_matches_installed_vllm(
    patched_video_source,
):
    """Pin the stock vLLM 0.25.1 Nemotron video preprocessing source shape."""
    content = patched_video_source.read_text()
    assert _VIDEO_MARKER in content
    ast.parse(content)


def test_radio_layerscale_patch_warns_on_unknown_source(monkeypatch, tmp_path, caplog):
    radio_source = tmp_path / "radio.py"
    radio_source.write_text("class RadioModel:\n    pass\n")
    monkeypatch.setattr(patches, "_get_vllm_file", lambda _relative: str(radio_source))

    with caplog.at_level(logging.WARNING):
        patches._patch_vllm_radio_layerscale_loader(logging.getLogger(__name__))

    assert radio_source.read_text() == "class RadioModel:\n    pass\n"
    assert "vLLM 0.25.1 source shape was not found" in caplog.text


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


def _stock_nemotron_video_preprocessing_source() -> str:
    return """import numpy as np
import torch

def video_to_pixel_values(
    video,
    *,
    size,
    norm_mean=None,
    norm_std=None,
    dtype=torch.float32,
):
    tensor = torch.from_numpy(video)
    return _bicubic_resize_and_normalize(tensor, size, norm_mean, norm_std, dtype)
"""


def test_nemotron_video_preprocessing_patch_matches_policy_resize(
    tmp_path, monkeypatch
):
    processor = tmp_path / "nano_nemotron_vl.py"
    processor.write_text(_stock_nemotron_video_preprocessing_source())
    monkeypatch.setattr(patches, "_get_vllm_file", lambda _: str(processor))
    monkeypatch.setattr(patches, "version", lambda _package: "0.25.1")
    logger = _Logger()

    patches._patch_vllm_nemotron_video_preprocessing(logger, required=True)
    patched = processor.read_text()

    assert "# patched by nemo_rl: match policy-side video resize" in patched
    assert "np.ascontiguousarray(video)" in patched
    assert not logger.warning_messages

    module = _load_module(processor)
    video = np.arange(2 * 3 * 5 * 3, dtype=np.uint8).reshape(2, 3, 5, 3)
    norm_mean = torch.tensor([0.5, 0.4, 0.3]).view(3, 1, 1)
    norm_std = torch.tensor([0.2, 0.3, 0.4]).view(3, 1, 1)
    actual = module.video_to_pixel_values(
        video,
        size=(4, 6),
        norm_mean=norm_mean,
        norm_std=norm_std,
        dtype=torch.float32,
    )

    expected = torch.from_numpy(video).permute(0, 3, 1, 2)
    expected = torch.nn.functional.interpolate(
        expected,
        size=(4, 6),
        mode="bicubic",
        align_corners=False,
        antialias=True,
    )
    expected = (expected / 255.0 - norm_mean) / norm_std
    torch.testing.assert_close(actual, expected.contiguous(), rtol=0, atol=0)

    patches._patch_vllm_nemotron_video_preprocessing(logger, required=True)
    assert processor.read_text() == patched
    assert logger.info_messages[-1] == (
        "vLLM Nemotron video preprocessing patch already applied."
    )


def test_nemotron_video_preprocessing_patch_fails_loudly_when_required(
    tmp_path, monkeypatch
):
    processor = tmp_path / "nano_nemotron_vl.py"
    processor.write_text("# newer vLLM source\n")
    monkeypatch.setattr(patches, "_get_vllm_file", lambda _: str(processor))
    monkeypatch.setattr(patches, "version", lambda _package: "0.25.1")

    with pytest.raises(RuntimeError, match="expected vLLM 0.25.1 source shape"):
        patches._patch_vllm_nemotron_video_preprocessing(_Logger(), required=True)

    assert processor.read_text() == "# newer vLLM source\n"


def test_nemotron_video_preprocessing_patch_requires_exact_vllm_version(monkeypatch):
    monkeypatch.setattr(patches, "version", lambda _package: "0.25.2")

    with pytest.raises(RuntimeError, match="requires vLLM 0.25.1"):
        patches._patch_vllm_nemotron_video_preprocessing(_Logger(), required=True)


def test_nemotron_video_preprocessing_patch_warns_when_not_required(monkeypatch):
    monkeypatch.setattr(patches, "version", lambda _package: "0.25.2")
    logger = _Logger()

    patches._patch_vllm_nemotron_video_preprocessing(logger, required=False)

    assert logger.warning_messages == [
        "The Nemotron video preprocessing patch requires vLLM 0.25.1; found 0.25.2."
    ]
