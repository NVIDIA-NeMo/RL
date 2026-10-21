#!/usr/bin/env python3
# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Apply only the pinned hybrid-SWA KV offload fix inside the task container."""

from __future__ import annotations

import argparse
import ast
import difflib
import hashlib
import json
import os
import stat
import tempfile
from datetime import datetime, timezone
from pathlib import Path

SITE_PACKAGES = Path("/opt/flint/envs/sglang/lib/python3.13/site-packages")
PINS = {
    "sglang/srt/mem_cache/kv_cache_configurator.py": (
        "0a3810ce07fe9d73f2c091c595c81a6b9db130cbe3230439a79cfc7eac195c78",
        "c67d197b06f407964b5b209f140535bb08665d2fe9c502c89e3f2b68fc1ad28d",
    ),
    "sglang/srt/mem_cache/swa_memory_pool.py": (
        "d0d8c9ad1a7652978195bfea112611cd281012c6ff9c12dfd05f445c13534724",
        "75a2f5d385c1c1ec886f0f553e19178b8bed498ce688fa99c88651231a8e8f45",
    ),
}


def digest(content):
    return hashlib.sha256(content).hexdigest()


def atomic_write(path, content, mode=0o600):
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        os.chmod(temporary, mode)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def prepare(workspace, site_packages=SITE_PACKAGES):
    manifest = json.loads(
        (workspace / "sglang-overlay/hybrid-kv-offload.json").read_text()
    )
    changes = manifest["changes"]
    if len(changes) != len(PINS) or {
        change["relative_path"] for change in changes
    } != set(PINS):
        raise RuntimeError("Unexpected SGLang overlay target list")
    plans, patch = [], []
    # Check every source, substitution, candidate, and target before changing any
    # runtime file. A repeat accepts only the exact known candidate or baseline.
    for change in changes:
        relative = Path(change["relative_path"])
        baseline_hash, candidate_hash = PINS[str(relative)]
        if (change["baseline_sha256"], change["candidate_sha256"]) != (
            baseline_hash,
            candidate_hash,
        ):
            raise RuntimeError("Overlay manifest differs from reviewed source pins")
        baseline_path = (
            workspace
            / "evidence/runtime-sglang-source"
            / relative.relative_to("sglang")
        )
        baseline = baseline_path.read_bytes()
        if digest(baseline) != baseline_hash:
            raise RuntimeError(f"Captured baseline hash differs: {relative}")
        text = baseline.decode()
        if text.count(change["old"]) != 1:
            raise RuntimeError(f"Expected one exact substitution: {relative}")
        candidate = text.replace(change["old"], change["new"]).encode()
        if digest(candidate) != candidate_hash:
            raise RuntimeError(f"Candidate hash differs: {relative}")
        ast.parse(candidate, filename=str(relative))
        target = site_packages / relative
        if target.is_symlink() or target.resolve() != target or not target.is_file():
            raise RuntimeError(
                f"Refusing redirected or nonregular runtime source: {target}"
            )
        before = digest(target.read_bytes())
        if before not in (baseline_hash, candidate_hash):
            raise RuntimeError(
                f"Refusing unknown runtime source hash {before}: {target}"
            )
        plans.append(
            {
                "target": target,
                "relative_path": str(relative),
                "candidate": candidate,
                "baseline_sha256": baseline_hash,
                "candidate_sha256": candidate_hash,
                "before_sha256": before,
                "mode": stat.S_IMODE(target.stat().st_mode),
            }
        )
        patch.extend(
            difflib.unified_diff(
                text.splitlines(True),
                candidate.decode().splitlines(True),
                fromfile=f"a/{relative}",
                tofile=f"b/{relative}",
            )
        )
    expected_patch = "".join(patch).encode()
    if (
        workspace / "sglang-overlay/hybrid-kv-offload.patch"
    ).read_bytes() != expected_patch:
        raise RuntimeError("Review patch differs from verified source substitutions")
    return plans, expected_patch


def install(workspace, *, check_only=False, site_packages=SITE_PACKAGES):
    plans, patch = prepare(workspace, site_packages)
    records = []
    for plan in plans:
        target = plan["target"]
        # Recheck immediately before replacing each file; do not overwrite a
        # concurrent edit made after the complete preflight.
        if (
            target.is_symlink()
            or target.resolve() != target
            or digest(target.read_bytes()) != plan["before_sha256"]
        ):
            raise RuntimeError(f"Runtime source changed during preflight: {target}")
        status = (
            "already_applied"
            if plan["before_sha256"] == plan["candidate_sha256"]
            else "baseline_verified"
        )
        if not check_only and status == "baseline_verified":
            atomic_write(target, plan["candidate"], plan["mode"])
            status = "applied"
        after = digest(target.read_bytes())
        if not check_only and after != plan["candidate_sha256"]:
            raise RuntimeError(
                f"Runtime source did not match candidate after installation: {target}"
            )
        records.append(
            {
                key: str(value) if isinstance(value, Path) else value
                for key, value in plan.items()
                if key not in ("candidate", "mode")
            }
        )
        records[-1].update(status=status, after_sha256=after)
    receipt = {
        "schema_version": 1,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "status": "verified" if check_only else "applied_or_already_applied",
        "check_only": check_only,
        "scope": "Propagate memory-saver enablement into both hybrid SWA leaf KV pools; no model weights or arithmetic changed",
        "files": records,
        "patch_sha256": digest(patch),
    }
    if not check_only:
        atomic_write(workspace / "evidence/sglang-hybrid-kv-offload.patch", patch)
        atomic_write(
            workspace / "evidence/sglang-runtime-overlay.json",
            (json.dumps(receipt, indent=2) + "\n").encode(),
        )
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    workspace = Path(__file__).resolve().parents[1]
    print(json.dumps(install(workspace, check_only=args.check_only)), flush=True)


if __name__ == "__main__":
    main()
