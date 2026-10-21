#!/usr/bin/env python3
# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
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
"""Apply reviewed container fixes while keeping the earlier run read-only."""

from __future__ import annotations

import argparse
import ast
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import socket
import stat


ROUTER_ARGS_TARGET = Path("/opt/flint/envs/sglang/lib/python3.13/site-packages/sglang_router/router_args.py")
ROUTER_ARGS_BASELINE_SHA256 = "189b46ce5ed15067c7e97605d402e15767d64f5b2810b6fb2db77db22cbf8016"
ROUTER_ARGS_CANDIDATE_SHA256 = "09cd8984a1f693cc2efe40b963befd6d5bc578e972272e994a046f183807f3e8"
ROUTER_ARGS_SUFFIX = (
    b"\n\n# Compatibility for NeMo RL's managed-router configuration field.\n"
    b"RouterArgs.router_policy = property(lambda self: self.policy, lambda self, value: setattr(self, 'policy', value))\n"
)


def load_module(path, name):
    specification = importlib.util.spec_from_file_location(name, path)
    if specification is None or specification.loader is None:
        raise RuntimeError(f"Cannot load reviewed installer: {path}")
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


def prepare_automodel(base, module):
    baseline = (base / "evidence/runtime-source/flint_layers.py").read_bytes()
    candidate = (base / "automodel-overlay" / module.RELATIVE_SOURCE).read_bytes()
    if module.digest(baseline) != module.BASELINE_SHA256:
        raise RuntimeError("Automodel baseline differs from its reviewed pin")
    if module.digest(candidate) != module.CANDIDATE_SHA256:
        raise RuntimeError("Automodel overlay differs from its reviewed pin")
    target = module.TARGET
    ast.parse(candidate, filename=str(target))
    if target.is_symlink() or target.resolve() != target or not target.is_file():
        raise RuntimeError("Refusing a redirected Automodel runtime path")
    before = module.digest(target.read_bytes())
    if before not in (module.BASELINE_SHA256, module.CANDIDATE_SHA256):
        raise RuntimeError(f"Unknown Automodel runtime source hash: {before}")
    return {
        "target": target,
        "relative_path": str(module.RELATIVE_SOURCE),
        "candidate": candidate,
        "baseline_sha256": module.BASELINE_SHA256,
        "candidate_sha256": module.CANDIDATE_SHA256,
        "before_sha256": before,
        "mode": stat.S_IMODE(target.stat().st_mode),
    }


def prepare_router_args():
    """Build the reviewed append in memory from the writable container target."""
    target = ROUTER_ARGS_TARGET
    if target.is_symlink() or target.resolve() != target or not target.is_file():
        raise RuntimeError("Refusing a redirected router runtime path")
    current = target.read_bytes()
    before = hashlib.sha256(current).hexdigest()
    if before == ROUTER_ARGS_BASELINE_SHA256:
        baseline = current
    elif before == ROUTER_ARGS_CANDIDATE_SHA256:
        if not current.endswith(ROUTER_ARGS_SUFFIX):
            raise RuntimeError("Router candidate lacks its reviewed suffix")
        baseline = current[:-len(ROUTER_ARGS_SUFFIX)]
    else:
        raise RuntimeError(f"Unknown router runtime source hash: {before}")
    if hashlib.sha256(baseline).hexdigest() != ROUTER_ARGS_BASELINE_SHA256:
        raise RuntimeError("Router baseline differs from its reviewed pin")
    candidate = baseline + ROUTER_ARGS_SUFFIX
    if hashlib.sha256(candidate).hexdigest() != ROUTER_ARGS_CANDIDATE_SHA256:
        raise RuntimeError("Router candidate differs from its reviewed pin")
    ast.parse(candidate, filename=str(target))
    return {
        "target": target,
        "relative_path": "sglang_router/router_args.py",
        "candidate": candidate,
        "baseline_sha256": ROUTER_ARGS_BASELINE_SHA256,
        "candidate_sha256": ROUTER_ARGS_CANDIDATE_SHA256,
        "before_sha256": before,
        "mode": stat.S_IMODE(target.stat().st_mode),
    }


def install(base, receipt_path):
    automodel = load_module(base / "tools/apply_automodel_overlay.py", "flint_auto_overlay")
    sglang = load_module(base / "tools/apply_sglang_overlay.py", "flint_sg_overlay")
    # Validate all four targets before replacing any runtime file. The SGLang
    # helper reads the already captured baseline; it makes no new source copies.
    plans = [prepare_automodel(base, automodel)]
    sglang_plans, patch = sglang.prepare(base)
    plans.extend(sglang_plans)
    plans.append(prepare_router_args())
    # Recheck the entire set together before the first mutation, including
    # changes to an earlier target while a later target was being prepared.
    for plan in plans:
        target = plan["target"]
        if (
            target.is_symlink()
            or target.resolve() != target
            or sglang.digest(target.read_bytes()) != plan["before_sha256"]
        ):
            raise RuntimeError(f"Runtime target changed after preflight: {target}")
    records = []
    for plan in plans:
        target = plan["target"]
        status = "already_applied"
        if plan["before_sha256"] == plan["baseline_sha256"]:
            sglang.atomic_write(target, plan["candidate"], plan["mode"])
            status = "applied"
        after = sglang.digest(target.read_bytes())
        if after != plan["candidate_sha256"]:
            raise RuntimeError(f"Runtime overlay verification failed: {target}")
        record = {
            key: str(value) if isinstance(value, Path) else value
            for key, value in plan.items()
            if key not in ("candidate", "mode")
        }
        record.update(status=status, after_sha256=after)
        records.append(record)
    receipt = {
        "schema_version": 1,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "hostname": socket.gethostname(),
        "status": "pass",
        "base_workspace": str(base),
        "files": records,
        "sglang_patch_sha256": sglang.digest(patch),
        "scope": "Container runtime targets only; earlier run and model sources are read-only",
    }
    sglang.atomic_write(receipt_path, (json.dumps(receipt, indent=2) + "\n").encode())
    print(json.dumps({"status": "pass", "hostname": socket.gethostname(), "targets": len(records)}))
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, default=Path("/flint-work"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    if not output.is_relative_to(Path("/flint-run/evidence")):
        raise RuntimeError("Runtime receipts must be written under the new run evidence directory")
    install(args.base, output)


if __name__ == "__main__":
    main()
