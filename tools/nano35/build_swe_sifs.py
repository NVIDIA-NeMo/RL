#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "swebench @ git+https://github.com/HeyyyyyyG/SWE-bench.git@d546b5e4be7e5fd5eaec6ee67b7588d85306a422",
# ]
# ///
# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Build and gold-check ARM64 Verified images, then export the matching SWE SIFs."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import shutil
import subprocess
import tempfile
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path
from typing import Any

PROFILE = Path(__file__).with_name("swe_verified_403.json")
HARNESS_COMMIT = "d546b5e4be7e5fd5eaec6ee67b7588d85306a422"
IMAGE_TAG = "nano35-" + HARNESS_COMMIT[:12]


def sha256(path: Path) -> str:
    """Hash large container files without reading the whole file into memory."""
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_data(path: Path) -> list[dict[str, Any]]:
    """Require the complete pinned raw input emitted by prepare_swe_data.py."""
    profile = json.loads(PROFILE.read_text())
    if sha256(path) != profile["raw_sha256"]:
        raise ValueError("Raw data checksum differs; run prepare_swe_data.py first")
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    if [row["instance_id"] for row in rows] != profile["instance_ids"]:
        raise ValueError("Expected all 403 recorded instances in order")
    return rows


def build(*, data: Path, sif_dir: Path, work_dir: Path, workers: int) -> None:
    """Export only native ARM images that pass the pinned harness's gold tests."""
    if platform.machine() not in ("aarch64", "arm64"):
        raise RuntimeError("Use a native ARM64 build host")
    for command in ("docker", "apptainer"):
        if shutil.which(command) is None:
            raise RuntimeError(f"Install {command} on the ARM64 build host first")

    # Build-only dependencies are intentionally separate from training actors.
    from swebench.harness.docker_build import build_instance_images
    from swebench.harness.run_evaluation import run_instance
    from swebench.harness.test_spec.test_spec import make_test_spec

    import docker

    rows = read_data(data)
    data_hash = sha256(data)
    sif_dir.mkdir(parents=True, exist_ok=True)
    work_dir.mkdir(parents=True, exist_ok=True)
    os.chdir(work_dir)
    run_id = "nano35-gold-" + uuid.uuid4().hex
    client = docker.from_env()
    if client.info()["Architecture"] not in ("aarch64", "arm64"):
        raise RuntimeError("The Docker daemon must also run on ARM64")
    specs = []
    for row in rows:
        spec = make_test_spec(
            row,
            namespace=None,
            base_image_tag=IMAGE_TAG,
            env_image_tag=IMAGE_TAG,
            instance_image_tag=IMAGE_TAG,
        )
        if spec.arch != "arm64":
            print(
                f"Building {spec.instance_id} natively; upstream's x86 fallback is disabled"
            )
        specs.append(replace(spec, arch="arm64"))

    records: dict[str, dict[str, Any]] = {}
    pending = []
    for spec in specs:
        target = sif_dir / f"swe-bench.eval.arm64.{spec.instance_id}.sif"
        receipt = target.with_suffix(".json")
        if target.exists() or receipt.exists():
            if not target.is_file() or not receipt.is_file():
                raise ValueError(f"Incomplete existing SIF/receipt pair: {target}")
            record = json.loads(receipt.read_text())
            if (
                record["instance_id"] != spec.instance_id
                or record["status"] != "passed"
                or record["harness_commit"] != HARNESS_COMMIT
                or record["data_sha256"] != data_hash
                or record["gold_resolved"] is not True
                or record["sif_sha256"] != sha256(target)
            ):
                raise ValueError(
                    f"Existing SIF does not match its build receipt: {target}"
                )
            records[spec.instance_id] = record
        else:
            pending.append(spec)

    # The upstream builder can return partial success. Check every requested
    # Docker image below instead of treating its aggregate message as success.
    if pending:
        build_instance_images(client, pending, max_workers=workers, tag=IMAGE_TAG)
    by_id = {row["instance_id"]: row for row in rows}

    def verify_and_export(spec: Any) -> tuple[str, dict[str, Any]]:
        instance_id = spec.instance_id
        try:
            image = client.images.get(spec.instance_image_key)
            if image.attrs["Architecture"] != "arm64":
                raise ValueError("Docker image architecture is not arm64")
            result = run_instance(
                test_spec=spec,
                pred={
                    "instance_id": instance_id,
                    "model_name_or_path": "nano35-gold",
                    "model_patch": by_id[instance_id]["patch"],
                },
                rm_image=False,
                force_rebuild=False,
                client=client,
                run_id=run_id,
                timeout=1200,
            )
            if result is None or result[1][instance_id]["resolved"] is not True:
                raise ValueError(
                    "Gold patch did not pass FAIL_TO_PASS/PASS_TO_PASS tests"
                )
            target = sif_dir / f"swe-bench.eval.arm64.{instance_id}.sif"
            # The archive stays on build scratch. The SIF temporary is on the
            # destination filesystem for an atomic, no-overwrite final link.
            with tempfile.TemporaryDirectory(dir=work_dir) as archive_temp:
                archive = Path(archive_temp) / "image.tar"
                subprocess.run(
                    [
                        "docker",
                        "save",
                        "--output",
                        str(archive),
                        spec.instance_image_key,
                    ],
                    check=True,
                )
                with tempfile.TemporaryDirectory(
                    prefix=".nano35-", dir=sif_dir
                ) as sif_temp:
                    candidate = Path(sif_temp) / "image.sif"
                    subprocess.run(
                        [
                            "apptainer",
                            "build",
                            "--disable-cache",
                            str(candidate),
                            f"docker-archive://{archive}",
                        ],
                        check=True,
                    )
                    subprocess.run(
                        [
                            "apptainer",
                            "exec",
                            "--cleanenv",
                            str(candidate),
                            "/bin/bash",
                            "-lc",
                            "test -d /testbed && test $(uname -m) = aarch64",
                        ],
                        check=True,
                    )
                    record = {
                        "instance_id": instance_id,
                        "status": "passed",
                        "harness_commit": HARNESS_COMMIT,
                        "data_sha256": data_hash,
                        "docker_image_id": image.id,
                        "gold_resolved": True,
                        "sif_sha256": sha256(candidate),
                        "size_bytes": candidate.stat().st_size,
                    }
                    os.link(candidate, target)
            with target.with_suffix(".json").open("x") as receipt:
                receipt.write(json.dumps(record, indent=2) + "\n")
            return instance_id, record
        except Exception as exc:
            # Retain other completed images and report each failed instance;
            # any failure makes the whole command fail after the report is saved.
            return instance_id, {"status": "failed", "error": str(exc)}

    with ThreadPoolExecutor(max_workers=workers) as pool:
        for instance_id, record in pool.map(verify_and_export, pending):
            records[instance_id] = record
            print(instance_id, record["status"], flush=True)
    failed = [key for key, value in records.items() if value["status"] != "passed"]
    report = {"status": "failed" if failed else "passed", "instances": records}
    (work_dir / "sif-build-report.json").write_text(json.dumps(report, indent=2) + "\n")
    client.close()
    if failed:
        raise RuntimeError(
            f"{len(failed)} instances failed; inspect {work_dir}/sif-build-report.json"
        )
    print(
        f"All {len(records)} SIFs built and gold-checked. Run GPU/SWE preflight next."
    )


def main() -> None:
    """Build once, reuse matching receipts, and fail without dropping any rows."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--sif-dir", required=True, type=Path)
    parser.add_argument("--work-dir", required=True, type=Path)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--plan-only", action="store_true")
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be positive")
    rows = read_data(args.data)
    if args.plan_only:
        print(
            json.dumps(
                {
                    "instances": len(rows),
                    "architecture": "arm64",
                    "harness_commit": HARNESS_COMMIT,
                    "gold_verification": "required",
                    "sif_dir": str(args.sif_dir),
                },
                indent=2,
            )
        )
        return
    build(
        data=args.data.resolve(),
        sif_dir=args.sif_dir.resolve(),
        work_dir=args.work_dir.resolve(),
        workers=args.workers,
    )


if __name__ == "__main__":
    main()
