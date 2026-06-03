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

Pass ``--swe-gym-ids`` and/or ``--rebench-report``

Usage:
    python build_swe_sif_images.py --registry "$REGISTRY" --sif-dir "$SIF_DIR" \
        --swe-gym-ids swe-gym-arm-build/swe_gym_instance_ids.txt \
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
    sif_path: Path, docker_ref: str, skip_existing: bool
) -> tuple[str, str]:
    """Build one .sif. Returns (status, instance) where status is built/skipped/failed."""
    instance = sif_path.stem
    if skip_existing and sif_path.exists() and sif_path.stat().st_size > 0:
        return ("skipped", instance)
    sif_path.parent.mkdir(parents=True, exist_ok=True)
    proc = subprocess.run(
        ["apptainer", "build", str(sif_path), docker_ref],
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        # Missing-in-registry (failed/unpushed instance) or a real build error.
        print(
            f"  FAILED {instance}: {proc.stderr.strip().splitlines()[-1] if proc.stderr.strip() else 'see apptainer output'}",
            flush=True,
        )
        return ("failed", instance)
    return ("built", instance)


def _swe_gym_jobs(
    registry: str, sif_dir: Path, ids_file: Path
) -> list[tuple[Path, str]]:
    jobs = []
    for line in ids_file.read_text().splitlines():
        iid = line.strip()
        if not iid or iid.startswith("#"):
            continue
        jobs.append(
            (
                sif_dir / "swegym" / f"sweb.eval.arm64.{iid}.sif",
                f"docker://{registry}/swe-gym:sweb.eval.arm64.{iid}",
            )
        )
    return jobs


def _swe_rebench_jobs(
    registry: str, sif_dir: Path, report_file: Path
) -> list[tuple[Path, str]]:
    report = json.loads(report_file.read_text())
    # New schema: {total, completed, ok, ..., items: [...]}; old schema: a bare list.
    items = report.get("items", []) if isinstance(report, dict) else report
    jobs = []
    for r in items:
        if not isinstance(r, dict):
            continue
        if not r.get("passed_match"):  # respect the verify gate
            continue
        if r.get("uploaded") is False:  # passed but the registry push failed -> not pullable
            continue
        iid = r["instance_id"]
        jobs.append(
            (
                sif_dir / "swerebench" / f"{iid}.sif",
                f"docker://{registry}/swerebenchv2/{iid}:latest",
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
        "--sif-dir",
        type=Path,
        default=os.environ.get("SIF_DIR"),
        help="Output dir (default: $SIF_DIR).",
    )
    ap.add_argument(
        "--swe-gym-ids",
        type=Path,
        default=None,
        help="swe_gym_instance_ids.txt; build SWE-Gym images when given.",
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
        default=1,
        help="Concurrent apptainer builds (default 1).",
    )
    ap.add_argument(
        "--no-skip-existing",
        action="store_true",
        help="Rebuild even if the .sif already exists.",
    )
    args = ap.parse_args()

    if not args.registry:
        ap.error("--registry (or $REGISTRY) is required")
    if not args.sif_dir:
        ap.error("--sif-dir (or $SIF_DIR) is required")

    if not args.swe_gym_ids and not args.rebench_report:
        ap.error("provide --swe-gym-ids and/or --rebench-report")

    jobs: list[tuple[Path, str]] = []
    if args.swe_gym_ids:
        jobs += _swe_gym_jobs(args.registry, args.sif_dir, args.swe_gym_ids)
    if args.rebench_report:
        jobs += _swe_rebench_jobs(args.registry, args.sif_dir, args.rebench_report)

    print(
        f"Converting {len(jobs)} image(s) into {args.sif_dir} (workers={args.max_workers})",
        flush=True,
    )
    counts = {"built": 0, "skipped": 0, "failed": 0}
    failed: list[str] = []
    with ThreadPoolExecutor(max_workers=args.max_workers) as ex:
        futs = [
            ex.submit(_apptainer_build, sif_path, ref, not args.no_skip_existing)
            for sif_path, ref in jobs
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
