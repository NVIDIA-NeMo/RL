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
"""Unit tests for the ``examples/converters/convert_dcp_to_hf.py`` CLI surface.

These tests exercise ``parse_args()`` and ``main()``'s argument-handling
logic. The underlying ``convert_dcp_to_hf`` library function is patched
out, so the tests stay fast (no model loading, no Ray, no GPU). The
real end-to-end conversion is covered by
``tests/functional/test_converter_roundtrip.py``.
"""

import importlib.util
import os
import sys
from types import ModuleType
from unittest import mock

import pytest

_CONVERTER_PATH = os.path.normpath(
    os.path.join(
        os.path.dirname(__file__),
        "..",
        "..",
        "..",
        "examples",
        "converters",
        "convert_dcp_to_hf.py",
    )
)


def _load_module() -> ModuleType:
    """Load ``convert_dcp_to_hf.py`` as a module via ``importlib``.

    The converter scripts live under ``examples/`` and are not packaged,
    so we load them by file path. Each test gets a freshly imported
    module to avoid leaking ``sys.argv`` patches across tests.
    """
    spec = importlib.util.spec_from_file_location(
        "convert_dcp_to_hf_under_test", _CONVERTER_PATH
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
        "convert_dcp_to_hf.py",
        "--config",
        "/tmp/cfg.yaml",
        "--dcp-ckpt-path",
        "/tmp/dcp",
        "--hf-ckpt-path",
        "/tmp/hf",
    ]
    with mock.patch.object(sys, "argv", argv):
        args = converter_module.parse_args()

    assert args.config == "/tmp/cfg.yaml"
    assert args.dcp_ckpt_path == "/tmp/dcp"
    assert args.hf_ckpt_path == "/tmp/hf"


def test_parse_args_defaults_are_none(converter_module: ModuleType) -> None:
    """Unset flags must default to ``None`` so ``main()`` can fail loudly
    instead of silently using an unintended path."""
    with mock.patch.object(sys, "argv", ["convert_dcp_to_hf.py"]):
        args = converter_module.parse_args()

    assert args.config is None
    assert args.dcp_ckpt_path is None
    assert args.hf_ckpt_path is None


def test_parse_args_unknown_flag_exits(converter_module: ModuleType) -> None:
    """Unknown flags must trigger an argparse SystemExit (exit code 2)."""
    argv = ["convert_dcp_to_hf.py", "--definitely-not-a-flag", "x"]
    with mock.patch.object(sys, "argv", argv):
        with pytest.raises(SystemExit) as excinfo:
            converter_module.parse_args()
    assert excinfo.value.code == 2


def test_main_propagates_missing_config(converter_module: ModuleType, tmp_path) -> None:
    """``main()`` should fail with FileNotFoundError when --config points
    to a non-existent path, rather than silently swallowing the error."""
    missing = tmp_path / "does_not_exist.yaml"
    argv = [
        "convert_dcp_to_hf.py",
        "--config",
        str(missing),
        "--dcp-ckpt-path",
        str(tmp_path / "dcp"),
        "--hf-ckpt-path",
        str(tmp_path / "hf"),
    ]
    with mock.patch.object(sys, "argv", argv):
        with pytest.raises(FileNotFoundError):
            converter_module.main()


def _write_minimal_config(path, *, hf_overrides=None) -> None:
    """Write a minimal config.yaml accepted by ``convert_dcp_to_hf.main()``."""
    cfg = {
        "policy": {
            "model_name": "dummy/Model",
            "tokenizer": {"name": "dummy/Tokenizer"},
        }
    }
    if hf_overrides is not None:
        cfg["policy"]["hf_overrides"] = hf_overrides
    import yaml  # local import to keep top-level imports minimal

    path.write_text(yaml.safe_dump(cfg))


def test_main_uses_local_tokenizer_when_directory_present(
    converter_module: ModuleType, tmp_path
) -> None:
    """When ``<dcp_ckpt_path>/../tokenizer`` exists, ``main()`` must pass
    that path (not the config tokenizer name) to ``convert_dcp_to_hf``.

    This is the branch users hit after a real training run, where the
    train loop saves the tokenizer next to the DCP weights.
    """
    cfg_path = tmp_path / "cfg.yaml"
    _write_minimal_config(cfg_path)

    dcp_dir = tmp_path / "weights"
    dcp_dir.mkdir()
    tokenizer_dir = tmp_path / "tokenizer"
    tokenizer_dir.mkdir()

    argv = [
        "convert_dcp_to_hf.py",
        "--config",
        str(cfg_path),
        "--dcp-ckpt-path",
        str(dcp_dir),
        "--hf-ckpt-path",
        str(tmp_path / "hf"),
    ]
    with (
        mock.patch.object(sys, "argv", argv),
        mock.patch.object(converter_module, "convert_dcp_to_hf") as mock_convert,
    ):
        converter_module.main()

    mock_convert.assert_called_once()
    kwargs = mock_convert.call_args.kwargs
    assert kwargs["dcp_ckpt_path"] == str(dcp_dir)
    assert kwargs["hf_ckpt_path"] == str(tmp_path / "hf")
    assert kwargs["model_name_or_path"] == "dummy/Model"
    # Local tokenizer dir takes precedence over the config tokenizer name.
    assert os.path.samefile(kwargs["tokenizer_name_or_path"], tokenizer_dir)


def test_main_falls_back_to_config_tokenizer_without_local_dir(
    converter_module: ModuleType, tmp_path
) -> None:
    """When no local tokenizer dir exists alongside the DCP ckpt, the
    fallback must use the tokenizer name from the config file."""
    cfg_path = tmp_path / "cfg.yaml"
    _write_minimal_config(cfg_path)

    dcp_dir = tmp_path / "weights"
    dcp_dir.mkdir()
    # NOTE: deliberately do NOT create tmp_path/tokenizer

    argv = [
        "convert_dcp_to_hf.py",
        "--config",
        str(cfg_path),
        "--dcp-ckpt-path",
        str(dcp_dir),
        "--hf-ckpt-path",
        str(tmp_path / "hf"),
    ]
    with (
        mock.patch.object(sys, "argv", argv),
        mock.patch.object(converter_module, "convert_dcp_to_hf") as mock_convert,
    ):
        converter_module.main()

    kwargs = mock_convert.call_args.kwargs
    assert kwargs["tokenizer_name_or_path"] == "dummy/Tokenizer"


def test_main_passes_hf_overrides_when_present(
    converter_module: ModuleType, tmp_path
) -> None:
    """``hf_overrides`` from the config must be forwarded verbatim."""
    cfg_path = tmp_path / "cfg.yaml"
    _write_minimal_config(cfg_path, hf_overrides={"some_key": "some_value"})

    dcp_dir = tmp_path / "weights"
    dcp_dir.mkdir()

    argv = [
        "convert_dcp_to_hf.py",
        "--config",
        str(cfg_path),
        "--dcp-ckpt-path",
        str(dcp_dir),
        "--hf-ckpt-path",
        str(tmp_path / "hf"),
    ]
    with (
        mock.patch.object(sys, "argv", argv),
        mock.patch.object(converter_module, "convert_dcp_to_hf") as mock_convert,
    ):
        converter_module.main()

    kwargs = mock_convert.call_args.kwargs
    assert kwargs["hf_overrides"] == {"some_key": "some_value"}


def test_main_collapses_null_hf_overrides_to_empty_dict(
    converter_module: ModuleType, tmp_path
) -> None:
    """A YAML ``hf_overrides: null`` (often emitted by humans) must be
    treated as ``{}`` rather than passed through as ``None``."""
    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text(
        "policy:\n"
        "  model_name: dummy/Model\n"
        "  tokenizer:\n"
        "    name: dummy/Tokenizer\n"
        "  hf_overrides: null\n"
    )

    dcp_dir = tmp_path / "weights"
    dcp_dir.mkdir()

    argv = [
        "convert_dcp_to_hf.py",
        "--config",
        str(cfg_path),
        "--dcp-ckpt-path",
        str(dcp_dir),
        "--hf-ckpt-path",
        str(tmp_path / "hf"),
    ]
    with (
        mock.patch.object(sys, "argv", argv),
        mock.patch.object(converter_module, "convert_dcp_to_hf") as mock_convert,
    ):
        converter_module.main()

    kwargs = mock_convert.call_args.kwargs
    assert kwargs["hf_overrides"] == {}
