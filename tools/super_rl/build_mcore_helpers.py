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

"""Prebuild a read-only Megatron datasets overlay using the exact worker Python."""

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import sysconfig


def render_makefile(source: str, interpreter: str) -> str:
    """Change only the two Python probes in the pinned datasets Makefile."""
    if any(char in interpreter for char in "\n\r$#'\"\\"):
        raise ValueError("Interpreter path contains unsupported Make/shell characters")
    replacements = {
        "$(shell python3 -m pybind11 --includes)": '$(shell "$(PYTHON)" -m pybind11 --includes)',
        "$(shell python3-config --extension-suffix)": '$(shell "$(PYTHON)" -c \'import sysconfig; print(sysconfig.get_config_var("EXT_SUFFIX"))\')',
    }
    for before, after in replacements.items():
        if source.count(before) != 1:
            raise ValueError(f"Unsupported Makefile; expected exactly one {before}")
        source = source.replace(before, after)
    return f"PYTHON := {interpreter}\n" + source


def file_hash(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def runtime_identity() -> dict[str, str]:
    suffix = sysconfig.get_config_var("EXT_SUFFIX")
    if not isinstance(suffix, str) or not suffix:
        raise RuntimeError("Worker Python has no extension ABI suffix")
    return {
        "interpreter": sys.executable,
        "python": sys.version,
        "architecture": platform.machine(),
        "extension_suffix": suffix,
    }


def check_functions(binary: Path) -> None:
    # NumPy belongs to the native worker environment, not the preparation driver.
    import numpy as np

    spec = importlib.util.spec_from_file_location("helpers_cpp", binary)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load helper extension: {binary}")
    helper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helper)
    for function, dtype in [
        (helper.build_sample_idx_int32, np.int32),
        (helper.build_sample_idx_int64, np.int64),
    ]:
        actual = function(
            np.array([8], dtype=np.int32),
            np.array([0], dtype=np.int32),
            3,
            1,
            8,
            True,
            1,
        )
        np.testing.assert_array_equal(
            actual, np.array([[0, 0], [0, 3], [0, 6]], dtype=dtype)
        )
        if actual.dtype != dtype:
            raise RuntimeError("Unexpected sample index dtype")


def build(source: Path, output: Path) -> None:
    """Build a fresh package; leave failures unsealed for inspection."""
    identity = runtime_identity()
    makefile = render_makefile((source / "Makefile").read_text(), sys.executable)
    if output.resolve().is_relative_to(source.resolve()):
        raise ValueError("Output must be outside the immutable source package")
    shutil.copytree(
        source, output, ignore=shutil.ignore_patterns("__pycache__", "helpers_cpp*.so")
    )
    (output / "Makefile").write_text(makefile)
    subprocess.run(["make", "-B", "-C", str(output)], check=True)
    subprocess.run(["make", "-q", "-C", str(output)], check=True)
    check_functions(output / ("helpers_cpp" + identity["extension_suffix"]))
    files = {
        str(path.relative_to(output)): file_hash(path)
        for path in sorted(output.rglob("*"))
        if path.is_file()
    }
    with (output / "nrl-build.json").open("x") as stream:
        json.dump(
            {
                "runtime": identity,
                "source": str(source),
                "source_makefile_sha256": file_hash(source / "Makefile"),
                "files": files,
            },
            stream,
            indent=2,
        )
        stream.write("\n")


def verify(package: Path) -> None:
    """Exercise actual compile_helpers with the overlay mounted read-only."""
    receipt = json.loads((package / "nrl-build.json").read_text())
    if receipt["runtime"] != runtime_identity():
        raise RuntimeError(
            "Overlay was built for a different worker Python/ABI/architecture"
        )
    if not os.statvfs(package).f_flag & os.ST_RDONLY:
        raise RuntimeError("Verify requires a genuinely read-only package mount")
    for relative, expected in receipt["files"].items():
        if file_hash(package / relative) != expected:
            raise RuntimeError(f"Overlay checksum mismatch: {relative}")
    # Import the real installed package only in the native verification path.
    from megatron.core.datasets import helpers_cpp, utils

    if (
        Path(utils.__file__).parent.resolve() != package.resolve()
        or Path(helpers_cpp.__file__).parent.resolve() != package.resolve()
    ):
        raise RuntimeError("Megatron imports do not resolve to the mounted overlay")
    utils.compile_helpers()
    check_functions(Path(helpers_cpp.__file__))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="action", required=True)
    builder = commands.add_parser("build")
    builder.add_argument("--source", type=Path, required=True)
    builder.add_argument("--output", type=Path, required=True)
    verifier = commands.add_parser("verify")
    verifier.add_argument("--package", type=Path, required=True)
    args = parser.parse_args()
    sys.dont_write_bytecode = True
    if args.action == "build":
        build(args.source, args.output)
    else:
        verify(args.package)
    print(
        json.dumps(
            {"action": args.action, "complete": True, "runtime": runtime_identity()}
        )
    )


if __name__ == "__main__":
    main()
