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
"""Unit tests for the ``examples/converters/convert_megatron_to_hf.py`` CLI surface.

The script imports ``nemo_rl.models.megatron.community_import`` at module
load time, which transitively imports ``megatron.bridge``. These tests
are therefore marked ``mcore`` and only run when the mcore extra is
installed (``--mcore-only``).
"""

import importlib.util
import os
import sys
from types import ModuleType
from unittest import mock

import pytest

pytestmark = pytest.mark.mcore

_CONVERTER_PATH = os.path.normpath(
    os.path.join(
        os.path.dirname(__file__),
        "..",
        "..",
        "..",
        "examples",
        "converters",
        "convert_megatron_to_hf.py",
    )
)


def _load_module() -> ModuleType:
    """Load ``convert_megatron_to_hf.py`` as a module via ``importlib``."""
    spec = importlib.util.spec_from_file_location(
        "convert_megatron_to_hf_under_test", _CONVERTER_PATH
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def converter_module() -> ModuleType:
    return _load_module()


def test_parse_args_all_flags(converter_module: ModuleType) -> None:
    """All long-form flags populate the expected fields."""
    argv = [
        "convert_megatron_to_hf.py",
        "--config",
        "/tmp/cfg.yaml",
        "--hf-model-name",
        "Qwen/Qwen2-0.5B",
        "--megatron-ckpt-path",
        "/tmp/megatron",
        "--hf-ckpt-path",
        "/tmp/hf",
    ]
    with mock.patch.object(sys, "argv", argv):
        args = converter_module.parse_args()

    assert args.config == "/tmp/cfg.yaml"
    assert args.hf_model_name == "Qwen/Qwen2-0.5B"
    assert args.megatron_ckpt_path == "/tmp/megatron"
    assert args.hf_ckpt_path == "/tmp/hf"
    # --no-strict defaults to False (i.e. strict=True at the API call site).
    assert args.no_strict is False


def test_parse_args_no_strict_flag_is_a_boolean_toggle(
    converter_module: ModuleType,
) -> None:
    """``--no-strict`` is a flag (no value); presence flips the bool."""
    argv = [
        "convert_megatron_to_hf.py",
        "--config",
        "/tmp/cfg.yaml",
        "--megatron-ckpt-path",
        "/tmp/megatron",
        "--hf-ckpt-path",
        "/tmp/hf",
        "--no-strict",
    ]
    with mock.patch.object(sys, "argv", argv):
        args = converter_module.parse_args()
    assert args.no_strict is True


def test_parse_args_defaults_are_none(converter_module: ModuleType) -> None:
    """Unset flags default to ``None`` so misuse fails loudly downstream."""
    with mock.patch.object(sys, "argv", ["convert_megatron_to_hf.py"]):
        args = converter_module.parse_args()

    assert args.config is None
    assert args.hf_model_name is None
    assert args.megatron_ckpt_path is None
    assert args.hf_ckpt_path is None
    assert args.no_strict is False


def test_parse_args_unknown_flag_exits(converter_module: ModuleType) -> None:
    """Unknown flags must trigger an argparse SystemExit (exit code 2)."""
    argv = ["convert_megatron_to_hf.py", "--bogus-flag", "1"]
    with mock.patch.object(sys, "argv", argv):
        with pytest.raises(SystemExit) as excinfo:
            converter_module.parse_args()
    assert excinfo.value.code == 2


def test_main_invokes_export_with_strict_flag(
    converter_module: ModuleType, tmp_path
) -> None:
    """``main()`` should pass ``strict=not args.no_strict`` to the
    underlying ``export_model_from_megatron`` call.

    We patch the call site so this test stays fast and does not actually
    materialize a Megatron model. This is the first place a regression
    in the ``--no-strict`` wiring would show up.
    """
    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text(
        "policy:\n  model_name: dummy/Model\n  tokenizer:\n    name: dummy/Tokenizer\n"
    )

    argv = [
        "convert_megatron_to_hf.py",
        "--config",
        str(cfg_path),
        "--megatron-ckpt-path",
        str(tmp_path / "megatron"),
        "--hf-ckpt-path",
        str(tmp_path / "hf"),
        "--no-strict",
    ]

    # Patch on the script module — this is where the symbol is bound.
    with (
        mock.patch.object(sys, "argv", argv),
        mock.patch.object(
            converter_module, "export_model_from_megatron"
        ) as mock_export,
    ):
        converter_module.main()

    mock_export.assert_called_once()
    kwargs = mock_export.call_args.kwargs
    assert kwargs["hf_model_name"] == "dummy/Model"
    assert kwargs["hf_tokenizer_path"] == "dummy/Tokenizer"
    assert kwargs["input_path"] == str(tmp_path / "megatron")
    assert kwargs["output_path"] == str(tmp_path / "hf")
    assert kwargs["strict"] is False  # --no-strict was passed
    # Empty/None ``hf_overrides`` in the config must collapse to an empty dict.
    assert kwargs["hf_overrides"] == {}


def test_main_uses_hf_model_name_override(
    converter_module: ModuleType, tmp_path
) -> None:
    """When ``--hf-model-name`` is supplied it must override the value in
    the config file."""
    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text(
        "policy:\n"
        "  model_name: from/Config\n"
        "  tokenizer:\n"
        "    name: from/ConfigTokenizer\n"
    )

    argv = [
        "convert_megatron_to_hf.py",
        "--config",
        str(cfg_path),
        "--hf-model-name",
        "from/CLI",
        "--megatron-ckpt-path",
        str(tmp_path / "megatron"),
        "--hf-ckpt-path",
        str(tmp_path / "hf"),
    ]

    with (
        mock.patch.object(sys, "argv", argv),
        mock.patch.object(
            converter_module, "export_model_from_megatron"
        ) as mock_export,
    ):
        converter_module.main()

    kwargs = mock_export.call_args.kwargs
    assert kwargs["hf_model_name"] == "from/CLI"
    # Tokenizer is never overridden — it always comes from config.
    assert kwargs["hf_tokenizer_path"] == "from/ConfigTokenizer"
    assert kwargs["strict"] is True  # --no-strict not passed
