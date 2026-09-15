# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json

import pytest

from tools.super_rl.build_mcore_helpers import (
    build,
    render_makefile,
    runtime_identity,
    verify,
)

MAKEFILE = "CPPFLAGS += $(shell python3 -m pybind11 --includes)\nLIBEXT = $(shell python3-config --extension-suffix)\n"


def test_uses_exact_python_without_python_config():
    result = render_makefile(MAKEFILE, "/worker/bin/python")
    assert result.startswith("PYTHON := /worker/bin/python\n")
    assert "python3-config" not in result
    assert 'sysconfig.get_config_var("EXT_SUFFIX")' in result
    assert result.count('"$(PYTHON)"') == 2


@pytest.mark.parametrize("source", ["unrecognized", MAKEFILE * 2])
def test_unknown_makefile_fails_closed(source):
    with pytest.raises(ValueError):
        render_makefile(source, "/worker/bin/python")


def test_cannot_write_inside_source(tmp_path):
    (tmp_path / "Makefile").write_text(MAKEFILE)
    with pytest.raises(ValueError):
        build(tmp_path, tmp_path / "overlay")
    assert not (tmp_path / "overlay").exists()


def test_refuses_to_overwrite_existing_overlay(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    (source / "Makefile").write_text(MAKEFILE)
    output = tmp_path / "existing"
    output.mkdir()
    with pytest.raises(FileExistsError):
        build(source, output)


def test_wrong_architecture_rejected_before_native_import(tmp_path):
    (tmp_path / "nrl-build.json").write_text(
        json.dumps({"runtime": runtime_identity() | {"architecture": "wrong"}})
    )
    with pytest.raises(RuntimeError, match="different worker"):
        verify(tmp_path)


def test_writable_package_is_not_readonly_validation(tmp_path):
    (tmp_path / "nrl-build.json").write_text(
        json.dumps({"runtime": runtime_identity()})
    )
    with pytest.raises(RuntimeError, match="read-only"):
        verify(tmp_path)
