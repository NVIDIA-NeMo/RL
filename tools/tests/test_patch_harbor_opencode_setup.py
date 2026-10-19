# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import importlib.util
import os
import subprocess
from pathlib import Path

import pytest

SPEC = importlib.util.spec_from_file_location(
    "patch_harbor_opencode_setup",
    Path(__file__).parents[1] / "patch_harbor_opencode_setup.py",
)
assert SPEC is not None and SPEC.loader is not None
PATCH = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(PATCH)


@pytest.mark.parametrize("curl_present,apt_exit", [(True, 17), (False, 0), (False, 17)])
def test_native_curl_prerequisite_shell_behavior(
    tmp_path: Path, curl_present: bool, apt_exit: int
) -> None:
    binaries = tmp_path / "bin"
    binaries.mkdir()
    if curl_present:
        curl = binaries / "curl"
        curl.write_text("#!/bin/sh\nexit 0\n")
        curl.chmod(0o755)
    apt = binaries / "apt-get"
    apt.write_text(
        '#!/bin/sh\nprintf "%s\\n" "$*" >> "$APT_TEST_LOG"\nexit "$APT_TEST_EXIT"\n'
    )
    apt.chmod(0o755)
    log = tmp_path / "apt.log"
    command = PATCH.CONDITIONAL.removeprefix('command="').removesuffix('",')
    result = subprocess.run(
        ["/bin/bash", "-c", command],
        env={
            **os.environ,
            "PATH": str(binaries),
            "APT_TEST_LOG": str(log),
            "APT_TEST_EXIT": str(apt_exit),
        },
        check=False,
    )
    if curl_present:
        assert result.returncode == 0
        assert not log.exists()
    else:
        assert result.returncode == apt_exit
        assert log.read_text().splitlines() == (
            ["update", "install -y curl"] if apt_exit == 0 else ["update"]
        )


def test_only_native_curl_command_changes_and_patch_is_idempotent(
    tmp_path: Path,
) -> None:
    source = tmp_path / "opencode.py"
    original = "before\n" + PATCH.ORIGINAL + "\nNVM/npm/native agent code unchanged\n"
    source.write_text(original)
    assert PATCH.patch_installer(source) is True
    assert source.read_text() == original.replace(PATCH.ORIGINAL, PATCH.CONDITIONAL)
    assert (
        source.with_name("opencode.py.before-curl-prerequisite-check").read_text()
        == original
    )
    assert PATCH.patch_installer(source) is False


def test_unknown_installer_is_not_modified(tmp_path: Path) -> None:
    source = tmp_path / "opencode.py"
    original = "changed upstream installer\n"
    source.write_text(original)
    with pytest.raises(ValueError, match="Unrecognized Harbor"):
        PATCH.patch_installer(source)
    assert source.read_text() == original
