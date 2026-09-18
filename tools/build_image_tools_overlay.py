#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Build an additive, locked Python overlay; never modify a bundled environment."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tomllib


ROOTS = (
    "openai==2.6.1",
    "anthropic==0.109.2",
    "nemo-lens[sdk] @ git+https://github.com/NVIDIA-NeMo/Lens.git@b85578fc2b736a1804705e537001b5f45e9c715d",
)
PROTECTED = {
    "torch",
    "torchvision",
    "torchaudio",
    "triton",
    "vllm",
    "ray",
    "transformer-engine",
    "transformer-engine-torch",
    "numpy",
}


def lock_constraints(lock):
    """Constrain resolver choices to unambiguous versions in the source lockfile."""
    versions = {}
    for package in lock["package"]:
        if "registry" in package["source"]:
            versions.setdefault(package["name"], set()).add(package["version"])
    return (
        "\n".join(
            f"{name}=={next(iter(values))}"
            for name, values in sorted(versions.items())
            if len(values) == 1
        )
        + "\n"
    )


def check_resolution(text):
    """Reject any resolution that would shadow the existing GPU/framework stack."""
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith(("#", "--")):
            continue
        name = (
            stripped.split("==")[0]
            .split(" @ ")[0]
            .split("[")[0]
            .strip()
            .lower()
            .replace("_", "-")
        )
        if name in PROTECTED or name.startswith("nvidia-"):
            raise ValueError(
                f"Overlay must not replace GPU/framework dependency: {name}"
            )


def main():
    """Resolve/install only the small dependency closure into a new directory."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--lock", required=True, type=Path)
    args = parser.parse_args()
    if not os.environ.get("SLURM_JOB_ID"):
        parser.error("Build inside a Slurm allocation")
    args.output.mkdir(parents=True, exist_ok=False)
    raw_lock = args.lock.read_bytes()
    (args.output / "requirements.in").write_text("\n".join(ROOTS) + "\n")
    (args.output / "constraints.txt").write_text(
        lock_constraints(tomllib.loads(raw_lock.decode()))
    )
    env = os.environ.copy()
    env["UV_CACHE_DIR"] = str(args.output / "uv-cache")
    env["UV_NO_PROGRESS"] = "1"
    subprocess.run(
        [
            "uv",
            "pip",
            "compile",
            str(args.output / "requirements.in"),
            "--constraint",
            str(args.output / "constraints.txt"),
            "--python",
            sys.executable,
            "--output-file",
            str(args.output / "requirements.lock"),
        ],
        check=True,
        env=env,
        timeout=600,
    )
    requirements = (args.output / "requirements.lock").read_text()
    check_resolution(requirements)
    subprocess.run(
        [
            "uv",
            "pip",
            "install",
            "--python",
            sys.executable,
            "--no-deps",
            "--target",
            str(args.output / "packages"),
            "--requirement",
            str(args.output / "requirements.lock"),
        ],
        check=True,
        env=env,
        timeout=600,
    )
    report = {
        "source_lock_sha256": hashlib.sha256(raw_lock).hexdigest(),
        "requirements_sha256": hashlib.sha256(requirements.encode()).hexdigest(),
        "python": sys.version,
        "roots": ROOTS,
        "note": "Built only; runtime qualification is separate. Bundled environments unchanged.",
    }
    (args.output / "READY.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
