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
"""Stage the fixed Gym commit plus reviewed overlays into a fresh runtime tree."""

import argparse
import hashlib
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import tarfile
import tempfile

# Immutable Git object ID, not a credential.
GYM_BASE = "749432dc5de23b8eeb3d044c80350a7c0ae9a03f"  # pragma: allowlist secret
ASSETS = Path(__file__).resolve().parent


def _current_umask() -> int:
    umask = os.umask(0)
    os.umask(umask)
    return umask


def stage(source: Path, output: Path) -> None:
    """Archive a fixed Git object; never copy a dirty worktree or overwrite output.

    Everything is assembled in a sibling staging directory and renamed into
    place only after every patch and overlay applied, so a failure leaves no
    partially patched tree behind and a rerun is not blocked by one.
    """
    subprocess.run(
        ["git", "-C", str(source), "cat-file", "-e", f"{GYM_BASE}^{{commit}}"],
        check=True,
    )
    if output.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(prefix=f".{output.name}.staging-", dir=output.parent)
    )
    try:
        # mkdtemp is private by default; the staged tree is shared runtime code.
        staging.chmod(0o777 & ~_current_umask())
        archive = subprocess.check_output(
            ["git", "-C", str(source), "archive", GYM_BASE]
        )
        with tarfile.open(fileobj=io.BytesIO(archive)) as stream:
            stream.extractall(staging, filter="data")
        patches = sorted((ASSETS / "patches").glob("gym_*.patch"))
        for patch in patches:
            # --forward turns an already-applied hunk into a failure. Without it
            # GNU patch assumes -R and silently reverses the hunk, which is what a
            # drifted Gym pin would otherwise do here.
            command = [
                "patch",
                "--batch",
                "--forward",
                "--fuzz=0",
                "-p1",
                "-d",
                str(staging),
                "-i",
                str(patch),
            ]
            subprocess.run([*command, "--dry-run"], check=True)
            subprocess.run(command, check=True)
        for helper in sorted((ASSETS / "gym_overlays").glob("*.py")):
            destination = staging / "nemo_gym" / helper.name
            if destination.exists():
                raise FileExistsError(destination)
            shutil.copyfile(helper, destination)
        assets = [*patches, *sorted((ASSETS / "gym_overlays").glob("*.py"))]
        manifest = {
            "base_commit": GYM_BASE,
            "overlays": {
                str(path.relative_to(ASSETS)): hashlib.sha256(
                    path.read_bytes()
                ).hexdigest()
                for path in assets
            },
        }
        (staging / "super-rl-overlay-manifest.json").write_text(
            json.dumps(manifest, indent=2) + "\n"
        )
        os.rename(staging, output)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    print(json.dumps(manifest))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    stage(args.source.resolve(), args.output.resolve())


if __name__ == "__main__":
    main()
