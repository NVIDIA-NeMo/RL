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
"""Unit tests for the ``examples/converters/convert_lora_to_hf.py`` CLI surface.

The script's top-level imports are stdlib + ``yaml``; the heavy
``megatron.*`` imports are deferred inside ``_build_megatron_model_with_lora``.
These tests therefore run without the mcore extra and validate the
argparse contract plus ``main()`` dispatch logic.
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
        "convert_lora_to_hf.py",
    )
)

_REQUIRED_ARGS = [
    "--base-ckpt",
    "/tmp/base",
    "--adapter-ckpt",
    "/tmp/adapter",
    "--hf-model-name",
    "zai-org/GLM-5",
    "--hf-ckpt-path",
    "/tmp/out",
]


def _load_module() -> ModuleType:
    """Load ``convert_lora_to_hf.py`` as a module via ``importlib``."""
    spec = importlib.util.spec_from_file_location(
        "convert_lora_to_hf_under_test", _CONVERTER_PATH
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def converter_module() -> ModuleType:
    return _load_module()


def test_parse_args_required_args_present(converter_module: ModuleType) -> None:
    argv = ["convert_lora_to_hf.py", *_REQUIRED_ARGS]
    with mock.patch.object(sys, "argv", argv):
        args = converter_module.parse_args()

    assert args.base_ckpt == "/tmp/base"
    assert args.adapter_ckpt == "/tmp/adapter"
    assert args.hf_model_name == "zai-org/GLM-5"
    assert args.hf_ckpt_path == "/tmp/out"
    assert args.adapter_only is False


def test_parse_args_adapter_only_toggle(converter_module: ModuleType) -> None:
    """``--adapter-only`` flips the boolean to True."""
    argv = ["convert_lora_to_hf.py", *_REQUIRED_ARGS, "--adapter-only"]
    with mock.patch.object(sys, "argv", argv):
        args = converter_module.parse_args()
    assert args.adapter_only is True


@pytest.mark.parametrize(
    "missing_flag",
    ["--base-ckpt", "--adapter-ckpt", "--hf-model-name", "--hf-ckpt-path"],
)
def test_parse_args_required_flag_missing_exits(
    converter_module: ModuleType, missing_flag: str
) -> None:
    """Each ``required=True`` argparse flag must enforce its requirement."""
    argv = ["convert_lora_to_hf.py"]
    skip_next = False
    for token in _REQUIRED_ARGS:
        if skip_next:
            skip_next = False
            continue
        if token == missing_flag:
            skip_next = True  # also drop the value following the flag
            continue
        argv.append(token)
    with mock.patch.object(sys, "argv", argv):
        with pytest.raises(SystemExit) as excinfo:
            converter_module.parse_args()
    assert excinfo.value.code == 2


def test_main_dispatches_to_merge_by_default(converter_module: ModuleType) -> None:
    """Without ``--adapter-only`` ``main()`` calls ``merge_lora_to_hf``."""
    argv = ["convert_lora_to_hf.py", *_REQUIRED_ARGS]
    with (
        mock.patch.object(sys, "argv", argv),
        mock.patch.object(converter_module, "merge_lora_to_hf") as mock_merge,
        mock.patch.object(converter_module, "export_lora_adapter_to_hf") as mock_export,
    ):
        converter_module.main()

    mock_merge.assert_called_once_with(
        base_ckpt="/tmp/base",
        adapter_ckpt="/tmp/adapter",
        hf_model_name="zai-org/GLM-5",
        hf_ckpt_path="/tmp/out",
    )
    mock_export.assert_not_called()


def test_main_dispatches_to_adapter_only_when_flag_set(
    converter_module: ModuleType,
) -> None:
    """With ``--adapter-only`` ``main()`` calls ``export_lora_adapter_to_hf``."""
    argv = ["convert_lora_to_hf.py", *_REQUIRED_ARGS, "--adapter-only"]
    with (
        mock.patch.object(sys, "argv", argv),
        mock.patch.object(converter_module, "merge_lora_to_hf") as mock_merge,
        mock.patch.object(converter_module, "export_lora_adapter_to_hf") as mock_export,
    ):
        converter_module.main()

    mock_export.assert_called_once_with(
        base_ckpt="/tmp/base",
        adapter_ckpt="/tmp/adapter",
        hf_model_name="zai-org/GLM-5",
        hf_ckpt_path="/tmp/out",
    )
    mock_merge.assert_not_called()


def test_merge_lora_to_hf_rejects_existing_output(
    converter_module: ModuleType, tmp_path
) -> None:
    """Both export functions must refuse to overwrite an existing output dir."""
    existing = tmp_path / "already_there"
    existing.mkdir()
    with pytest.raises(FileExistsError):
        converter_module.merge_lora_to_hf(
            base_ckpt=str(tmp_path / "base"),
            adapter_ckpt=str(tmp_path / "adapter"),
            hf_model_name="zai-org/GLM-5",
            hf_ckpt_path=str(existing),
        )


def test_export_lora_adapter_to_hf_rejects_existing_output(
    converter_module: ModuleType, tmp_path
) -> None:
    existing = tmp_path / "already_there"
    existing.mkdir()
    with pytest.raises(FileExistsError):
        converter_module.export_lora_adapter_to_hf(
            base_ckpt=str(tmp_path / "base"),
            adapter_ckpt=str(tmp_path / "adapter"),
            hf_model_name="zai-org/GLM-5",
            hf_ckpt_path=str(existing),
        )
