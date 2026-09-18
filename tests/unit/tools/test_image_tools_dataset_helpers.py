# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Check isolated helper staging and the bundled Python ABI shim without GPUs."""

import os
from pathlib import Path
import subprocess
import sys
import sysconfig

import pytest


def test_stage_copies_only_build_inputs_and_refuses_reuse(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    for name in ("helpers.cpp", "Makefile", "__init__.py", "utils.py"):
        (source / name).write_text(name)
    (source / "stale.so").write_text("not a build input")
    target = tmp_path / "runtime"
    script = Path("tools/image_tools_dataset_helpers.sh").resolve()
    subprocess.run(["bash", str(script), "stage", str(source), str(target)], check=True)
    assert sorted(p.name for p in target.iterdir()) == [
        "Makefile",
        "__init__.py",
        "helpers.cpp",
        "utils.py",
    ]
    assert (target / "helpers.cpp").read_text() == "helpers.cpp"
    with pytest.raises(subprocess.CalledProcessError):
        subprocess.run(
            ["bash", str(script), "stage", str(source), str(target)], check=True
        )


def test_python_config_uses_selected_interpreter() -> None:
    shim = Path("tools/image_tools_build_bin/python3-config").resolve()
    result = subprocess.check_output(
        ["bash", str(shim), "--extension-suffix"],
        env={**os.environ, "IMAGE_TOOLS_BUILD_PYTHON": sys.executable},
        text=True,
    )
    assert result.strip() == sysconfig.get_config_var("EXT_SUFFIX")


def test_launcher_builds_before_ray_and_preserves_readonly_source() -> None:
    launcher = Path("tools/launch_image_tools_train_hsg.sh").read_text()
    assert "$PROJECT_ROOT:/opt/nemo-rl:ro" in launcher
    assert "$RUN_DIR/runtime/mcore-datasets:$IMAGE_TOOLS_DATASET_DIR" in launcher
    assert 'image_tools_dataset_helpers.sh build "$IMAGE_TOOLS_DATASET_DIR"' in launcher
    assert launcher.index('image_tools_dataset_helpers.sh" stage') < launcher.index(
        'job_id=$("${submit[@]}")'
    )
