# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Stage and verify pure-Python test wheels without modifying any environment."""

import argparse
import hashlib
from pathlib import Path
import tomllib
import urllib.request


PACKAGES = {"pytest", "iniconfig", "packaging", "pluggy", "pygments"}


def wheel_paths(root: Path, *, download: bool = False) -> list[Path]:
    """Return hash-verified wheel paths selected from the existing source lock."""
    lock = tomllib.loads((root / "uv.lock").read_text())
    packages = [p for p in lock["package"] if p["name"] in PACKAGES]
    if len(packages) != len(PACKAGES):
        raise ValueError("Expected exactly one locked version of each test dependency")
    directory = root / "cluster_inputs/callback_test_wheels"
    if download:
        directory.mkdir(parents=True, exist_ok=True)
    paths = []
    for package in packages:
        wheels = [
            w for w in package["wheels"] if w["url"].endswith("-py3-none-any.whl")
        ]
        if len(wheels) != 1:
            raise ValueError(f"Expected one universal wheel for {package['name']}")
        wheel = wheels[0]
        path = directory / wheel["url"].rsplit("/", 1)[1]
        if download and not path.exists():
            with urllib.request.urlopen(wheel["url"], timeout=60) as response:
                data = response.read()
            if "sha256:" + hashlib.sha256(data).hexdigest() != wheel["hash"]:
                raise ValueError(f"Downloaded wheel hash mismatch: {path.name}")
            path.write_bytes(data)
        if "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest() != wheel["hash"]:
            raise ValueError(f"Test wheel hash mismatch: {path.name}")
        paths.append(path)
    return paths


def main() -> None:
    """Download on the Mac, or verify offline and emit a test-only PYTHONPATH."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--download", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    print(":".join(map(str, wheel_paths(root, download=args.download))))


if __name__ == "__main__":
    main()
