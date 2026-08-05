#!/usr/bin/env python3
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
"""Apply this directory's *.patch files to every installed copy of SGLang.

SGLang is installed here from a PyPI wheel and its `srt/` tree is pure Python,
so fixes that touch only `.py` files can be applied to the installed package
rather than requiring a forked wheel. See README.md for what is applied and why.

Every sglang install gets its own materialised copy of each touched file:
`uv` populates venvs with symlinks (and sometimes hardlinks) into
`/root/.cache/uv`, and GNU patch refuses to operate on a symlink at all
("not a regular file"), so the links are broken first. The cache archive is
patched too, so a later `uv sync` that relinks a venv still lands on patched
code.

Exits non-zero unless every discovered install ends up carrying every patch.
A silent no-op is not possible.
"""

import argparse
import glob
import os
import re
import shutil
import subprocess
import sys
import tempfile

# Patches are generated against the sglang git tree (`a/python/sglang/...`),
# while the wheel installs `sglang/...` — so two leading components are stripped
# and patches are applied from the directory that contains the package.
STRIP = 2

SEARCH_GLOBS = [
    "/opt/ray_venvs/*/lib/python*/site-packages/sglang",
    "/opt/nemo_rl_venv/lib/python*/site-packages/sglang",
    "/opt/venv*/lib/python*/site-packages/sglang",
    "/usr/local/lib/python*/*-packages/sglang",
    "/root/.cache/uv/archive-*/*/sglang",
    "/root/.cache/uv/builds-*/*/lib/python*/site-packages/sglang",
]


def targets_of(patch_path):
    """Paths a patch touches, relative to the package parent dir."""
    out = []
    with open(patch_path, encoding="utf-8", errors="replace") as fh:
        for line in fh:
            m = re.match(r"^\+\+\+ b/(\S+)", line)
            if m:
                out.append("/".join(m.group(1).split("/")[STRIP - 1 :]))
    return out


def discover(all_targets):
    """Package-parent dirs that look like a real sglang install."""
    found = []
    for pattern in SEARCH_GLOBS:
        for d in glob.glob(pattern):
            if not os.path.isdir(d):
                continue
            parent = os.path.dirname(d)
            if all(os.path.exists(os.path.join(parent, t)) for t in all_targets):
                found.append(parent)
    # Dedupe on the literal path, not realpath: venvs that symlink into a shared
    # cache must each be materialised separately, not collapsed into one entry.
    return sorted(set(found))


def materialise(parent, targets):
    """Give this install private, regular-file copies of the patch targets."""
    for rel in targets:
        path = os.path.join(parent, rel)
        real = os.path.realpath(path)
        if os.path.islink(path) or os.stat(path).st_nlink > 1:
            fd, tmp = tempfile.mkstemp(dir=os.path.dirname(path))
            os.close(fd)
            shutil.copyfile(real, tmp)
            shutil.copymode(real, tmp)
            os.replace(tmp, path)


def run_patch(parent, patch_path, dry_run=False, reverse=False):
    cmd = ["patch", f"-p{STRIP}", "--batch", "-d", parent, "-i", patch_path]
    if dry_run:
        cmd.append("--dry-run")
    cmd.append("--reverse" if reverse else "--forward")
    proc = subprocess.run(cmd, capture_output=True, text=True)
    return proc.returncode, (proc.stdout + proc.stderr).strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--patch-dir", default=os.path.dirname(os.path.abspath(__file__)))
    args = ap.parse_args()

    patches = sorted(glob.glob(os.path.join(args.patch_dir, "*.patch")))
    if not patches:
        print(f"no patches in {args.patch_dir}, nothing to do")
        return 0

    all_targets = sorted({t for p in patches for t in targets_of(p)})
    installs = discover(all_targets)
    if not installs:
        print(f"FAIL: no sglang install contains all of {all_targets}", file=sys.stderr)
        return 2

    print(f"{len(patches)} patch(es), {len(installs)} sglang install(s)")
    for parent in installs:
        if not os.access(parent, os.W_OK):
            print(f"FAIL: not writable: {parent}", file=sys.stderr)
            return 3
        for patch_path in patches:
            name = os.path.basename(patch_path)
            targets = targets_of(patch_path)
            materialise(parent, targets)

            # Already applied? A reverse dry-run succeeds only if it is present.
            if run_patch(parent, patch_path, dry_run=True, reverse=True)[0] == 0:
                print(f"  [already applied] {name} -> {parent}")
                continue

            rc, out = run_patch(parent, patch_path, dry_run=True)
            if rc != 0:
                print(
                    f"FAIL: {name} does not apply to {parent}\n{out}", file=sys.stderr
                )
                return 4
            rc, out = run_patch(parent, patch_path)
            if rc != 0:
                print(
                    f"FAIL: {name} failed after a clean dry-run\n{out}", file=sys.stderr
                )
                return 5
            if run_patch(parent, patch_path, dry_run=True, reverse=True)[0] != 0:
                print(
                    f"FAIL: {name} not detectable after applying to {parent}",
                    file=sys.stderr,
                )
                return 6
            print(f"  [applied] {name} -> {parent}")

    print(f"OK: {len(patches)} patch(es) present in {len(installs)} install(s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
