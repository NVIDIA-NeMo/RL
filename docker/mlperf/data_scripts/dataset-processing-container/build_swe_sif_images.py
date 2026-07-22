# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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

"""Convert the per-instance SWE images published by the ARM build steps into the ``.sif`` layout the Nemotron-3-Ultra SWE recipe expects under ``${SIF_DIR}``.

Pass ``--swe-gym-ids``/``--swe-gym-ids-file`` and/or ``--rebench-report``

Usage:
    python build_swe_sif_images.py --registry "$REGISTRY" --sif-dir "$SIF_DIR" \
        --swe-gym-ids-file swe-gym-arm-build/swe_gym_instance_ids.txt \
        --rebench-report swe-rebench-v2-arm-build/eval_report.json
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


def _apptainer_build(
    sif_path: Path, work_path: Path, docker_ref: str, skip_existing: bool
) -> tuple[str, str]:
    """Build one .sif. Returns (status, instance) where status is built/skipped/failed."""
    instance = sif_path.stem
    if skip_existing and sif_path.exists() and sif_path.stat().st_size > 0:
        return ("skipped", instance)
    sif_path.parent.mkdir(parents=True, exist_ok=True)
    work_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = sif_path.with_suffix(".tmp")

    def _check_error(proc) -> bool:
        if proc.returncode == 0:
            return False
        # Missing-in-registry (failed/unpushed instance) or a real build error.
        err_msg = (
            proc.stderr.strip().splitlines()[-1] if proc.stderr.strip() else "error"
        )
        print(
            f"  FAILED {instance}: {err_msg}",
            flush=True,
        )
        return True

    commands = [
        [
            "apptainer",
            "-d",
            "build",
            "--disable-cache",
            "--mksquashfs-args",
            "-processors 2 -no-xattrs",
            str(work_path),
            docker_ref,
        ],
        ["cp", str(work_path), str(tmp_path)],
        ["mv", str(tmp_path), str(sif_path)],
        ["rm", str(work_path)],
    ]
    for cmd in commands:
        proc = subprocess.run(cmd, capture_output=True, text=True)
        if _check_error(proc):
            return ("failed", instance)
    print(f"  SUCCESS {instance}")
    return ("built", instance)


def _swe_gym_jobs(
    registry: str, sif_dir: Path, work_dir: Path, instance_ids: list[str]
) -> list[tuple[Path, Path, str]]:
    jobs = []
    for iid in instance_ids:
        jobs.append(
            (
                sif_dir / "swegym" / f"sweb.eval.arm64.{iid}.sif",
                work_dir / "swegym" / f"sweb.eval.arm64.{iid}.sif",
                f"docker://{registry}/swe-gym:sweb.eval.arm64.{iid}",
            )
        )
    return jobs

def _r2e_gym_jobs(
    registry: str, sif_dir: Path, work_dir: Path, instance_ids_file: Path
) -> list[tuple[Path, Path, str]]:
    jobs = []
    for iid in instance_ids_file.read_text().splitlines():
        jobs.append(
            (
                sif_dir / "r2egym" / f"{iid}.sif",
                work_dir / "r2egym" / f"{iid}.sif",
                f"docker://{registry}/r2e-gym:{iid}",
            )
        )
    return jobs

def _swe_rebench_jobs(
    registry: str, sif_dir: Path, work_dir: Path, report_file: Path
) -> list[tuple[Path, Path, str]]:
    report = json.loads(report_file.read_text())
    # New schema: {total, completed, ok, ..., items: [...]}; old schema: a bare list.
    items = report.get("items", []) if isinstance(report, dict) else report
    jobs = []
    for r in items:
        if not isinstance(r, dict):
            continue
        if not r.get("passed_match"):  # respect the verify gate
            continue
        # passed but the registry push failed -> not pullable
        if r.get("uploaded") is False:
            continue
        iid = r["instance_id"]
        jobs.append(
            (
                sif_dir / "swerebench" / f"{iid}.sif",
                work_dir / "swerebench" / f"{iid}.sif",
                f"docker://{registry}/swerebenchv2:{iid}",
            )
        )
    return jobs


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Build SWE .sif images for the Ultra SWE recipe."
    )
    ap.add_argument(
        "--registry",
        default=os.environ.get("REGISTRY"),
        help="Registry endpoint (default: $REGISTRY).",
    )
    ap.add_argument(
        "--work-dir",
        type=Path,
        default=os.environ.get("WORK_DIR"),
        help="Work directory in which to initially write the SIF files, ideally local, not NFS, Lustre or the like",
    )
    ap.add_argument(
        "--sif-dir",
        type=Path,
        default=os.environ.get("SIF_DIR"),
        help="Output dir (default: $SIF_DIR).",
    )
    ap.add_argument(
        "--swe-gym-ids",
        default=None,
        help="Comma-separated SWE-Gym instance IDs; build SWE-Gym images when given.",
    )
    ap.add_argument(
        "--swe-gym-ids-file",
        type=Path,
        default=None,
        help="swe_gym_instance_ids.txt; build SWE-Gym images when given.",
    )
    ap.add_argument(
        "--r2e-gym-ids-file",
        type=Path,
        default=None,
        help="r2e_gym_instance_ids.txt; build R2E-Gym images when given.",
    )
    ap.add_argument(
        "--rebench-report",
        type=Path,
        default=None,
        help="eval_report.json; build SWE-rebench-V2 images when given.",
    )
    ap.add_argument(
        "--max-workers",
        type=int,
        default=int(os.environ.get("MAX_WORKERS", "1")),
        help="Concurrent apptainer builds (default 1).",
    )
    ap.add_argument(
        "--no-skip-existing",
        action="store_true",
        help="Rebuild even if the .sif already exists.",
    )
    args = ap.parse_args()
    if args.swe_gym_ids and args.swe_gym_ids_file:
        ap.error("provide only one of --swe-gym-ids or --swe-gym-ids-file")

    swe_gym_instance_ids: list[str] = []
    if args.swe_gym_ids:
        swe_gym_instance_ids = [
            iid.strip() for iid in args.swe_gym_ids.split(",") if iid.strip()
        ]
    elif args.swe_gym_ids_file:
        swe_gym_instance_ids = [
            line.strip()
            for line in args.swe_gym_ids_file.read_text().splitlines()
            if line.strip() and not line.strip().startswith("#")
        ]

    if not args.registry:
        ap.error("--registry (or $REGISTRY) is required")
    if not args.sif_dir:
        ap.error("--sif-dir (or $SIF_DIR) is required")
    if not args.work_dir:
        ap.error("--work-dir (or $WORK_DIR) is required")

    if not swe_gym_instance_ids and not args.rebench_report and not args.r2e_gym_ids_file:
        ap.error("provide --swe-gym-ids, --swe-gym-ids-file, and/or --rebench-report, or --r2e-gym-ids-file")

    jobs: list[tuple[Path, Path, str]] = []
    if swe_gym_instance_ids:
        jobs += _swe_gym_jobs(
            registry=args.registry,
            sif_dir=args.sif_dir,
            work_dir=args.work_dir,
            instance_ids=swe_gym_instance_ids,
        )
    if args.rebench_report:
        jobs += _swe_rebench_jobs(
            args.registry, args.sif_dir, args.work_dir, args.rebench_report
        )
    if args.r2e_gym_ids_file:
        jobs += _r2e_gym_jobs(
                args.registry, args.sif_dir, args.work_dir, args.r2e_gym_ids_file
        )

    print(
        f"Converting {len(jobs)} image(s) into {args.sif_dir} (workers={args.max_workers})",
        flush=True,
    )
    counts = {"built": 0, "skipped": 0, "failed": 0}
    failed: list[str] = []
    with ThreadPoolExecutor(max_workers=args.max_workers) as ex:
        futs = [
            ex.submit(
                _apptainer_build, sif_path, work_path, ref, not args.no_skip_existing
            )
            for sif_path, work_path, ref in jobs
        ]
        for fut in as_completed(futs):
            status, instance = fut.result()
            counts[status] += 1
            if status == "failed":
                failed.append(instance)

    print(
        f"\nDone: {counts['built']} built, {counts['skipped']} skipped, {counts['failed']} failed.",
        flush=True,
    )
    if failed:
        missing_log = args.sif_dir / "missing_instances.txt"
        missing_log.write_text("\n".join(sorted(failed)) + "\n")
        print(f"Failed/missing instances written to {missing_log}", flush=True)


if __name__ == "__main__":
    main()
