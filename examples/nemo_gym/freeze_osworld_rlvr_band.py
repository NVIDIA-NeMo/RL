# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Freeze a live RLVR OSWorld band into an immutable, portable Gym dataset.

The source ``band_live`` export is refreshed in place. This command validates
the offline DAPO band (0 < pass@8 < 8), copies tasks, evaluator functions, and
all upload assets into one snapshot, rewrites upload paths, and atomically
publishes the result.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import tempfile
from collections.abc import Iterable
from pathlib import Path
from typing import Any


def _read_manifest(path: Path) -> tuple[bytes, list[dict[str, Any]]]:
    raw = path.read_bytes()
    rows = [
        json.loads(line)
        for line in raw.decode("utf-8").splitlines()
        if line.strip()
    ]
    if not rows:
        raise ValueError(f"empty manifest: {path}")
    return raw, rows


def _validate_band_row(row: dict[str, Any], seen: set[str]) -> str:
    task_id = row.get("task_id")
    if not isinstance(task_id, str) or not task_id:
        raise ValueError(f"manifest row has invalid task_id: {row!r}")
    if task_id in seen:
        raise ValueError(f"duplicate task_id in manifest: {task_id}")
    seen.add(task_id)

    reps = row.get("calib_reps")
    rate = row.get("calib_pass_rate")
    if reps != 8 or not isinstance(rate, (int, float)):
        raise ValueError(f"{task_id}: expected numeric pass@8 calibration")
    passes = float(rate) * reps
    if not (0.0 < passes < reps) or abs(passes - round(passes)) > 1e-6:
        raise ValueError(
            f"{task_id}: offline DAPO filter failed "
            f"(calib_pass_rate={rate}, calib_reps={reps})"
        )
    return task_id


def _upload_entries(task: dict[str, Any]) -> Iterable[dict[str, Any]]:
    for step in task.get("config", []):
        if not isinstance(step, dict) or step.get("type") != "upload_file":
            continue
        parameters = step.get("parameters", {})
        if not isinstance(parameters, dict):
            continue
        files = parameters.get("files", [])
        if not isinstance(files, list):
            continue
        for entry in files:
            if isinstance(entry, dict) and "local_path" in entry:
                yield entry


def _unreadable_upload_assets(task: dict[str, Any]) -> list[str]:
    failures: list[str] = []
    for entry in _upload_entries(task):
        value = entry.get("local_path")
        if not isinstance(value, str) or not value:
            failures.append(repr(value))
            continue
        path = Path(value).expanduser()
        try:
            if not path.is_file():
                raise FileNotFoundError(path)
            with path.open("rb") as source:
                source.read(1)
        except OSError as error:
            failures.append(f"{path}: {type(error).__name__}")
    return failures


def _copy_task_assets(
    task: dict[str, Any],
    *,
    task_id: str,
    stage: Path,
    final: Path,
) -> int:
    copied: dict[Path, Path] = {}
    for entry in _upload_entries(task):
        source_value = entry.get("local_path")
        if not isinstance(source_value, str) or not source_value:
            raise ValueError(f"{task_id}: invalid upload_file.local_path")
        source = Path(source_value).expanduser().resolve()
        if not source.is_file():
            raise FileNotFoundError(f"{task_id}: upload asset not found: {source}")
        destination = copied.get(source)
        if destination is None:
            digest = hashlib.sha256(source.read_bytes()).hexdigest()[:12]
            destination = (
                Path("assets")
                / task_id
                / f"{len(copied):03d}-{digest}-{source.name}"
            )
            target = stage / destination
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
            copied[source] = destination
        entry["local_path"] = str(final / destination)
    return len(copied)


def _rewrite_function_provenance(
    task: dict[str, Any], *, task_id: str, final: Path
) -> None:
    metadata = task.get("metadata")
    if not isinstance(metadata, dict):
        return
    paths = metadata.get("new_functions_paths")
    if not isinstance(paths, list):
        return
    rewritten: list[str] = []
    for value in paths:
        if not isinstance(value, str):
            continue
        path = Path(value)
        role = next(
            (candidate for candidate in ("getters", "metrics") if candidate in path.parts),
            None,
        )
        if role is not None:
            rewritten.append(
                str(final / "tmp_funcs" / task_id / role / path.name)
            )
    metadata["new_functions_paths"] = rewritten


def _gym_row(task: dict[str, Any], repeat_index: int) -> dict[str, Any]:
    task_id = task["id"]
    return {
        "responses_create_params": {
            "input": [{"role": "user", "content": task["instruction"]}],
            "max_output_tokens": 16384,
            "temperature": 0.6,
            "top_p": 1.0,
        },
        "verifier_metadata": task,
        "agent_ref": {
            "type": "responses_api_agents",
            "name": "nemotron_osworld",
        },
        "context_compaction_task_id": task_id,
        "context_compaction_group_id": (
            f"rlvr-band:{task_id}:repeat:{repeat_index}"
        ),
    }


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> int:
    count = 0
    with path.open("w", encoding="utf-8") as output:
        for row in rows:
            output.write(
                json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n"
            )
            count += 1
    return count


def _write_checksums(root: Path) -> str:
    records: list[str] = []
    for path in sorted(candidate for candidate in root.rglob("*") if candidate.is_file()):
        if path.name == "SHA256SUMS":
            continue
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        records.append(f"{digest}  {path.relative_to(root)}")
    content = "\n".join(records) + "\n"
    (root / "SHA256SUMS").write_text(content, encoding="utf-8")
    return hashlib.sha256(content.encode()).hexdigest()


def freeze_band(
    source: Path,
    output: Path,
    *,
    num_repeats: int,
    quarantine_unreadable_assets: bool = False,
) -> None:
    source = source.resolve()
    output = output.resolve()
    if num_repeats < 1:
        raise ValueError("num_repeats must be positive")
    if output.exists():
        raise FileExistsError(f"refusing to replace existing snapshot: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)

    manifest_path = source / "manifest.jsonl"
    manifest_before, manifest_rows = _read_manifest(manifest_path)
    stage = Path(
        tempfile.mkdtemp(prefix=f".{output.name}.tmp-", dir=output.parent)
    )
    try:
        (stage / "tasks").mkdir()
        (stage / "tmp_funcs").mkdir()
        (stage / "assets").mkdir()
        seen: set[str] = set()
        tasks: list[dict[str, Any]] = []
        included_manifest_rows: list[dict[str, Any]] = []
        quarantined_rows: list[dict[str, Any]] = []
        asset_count = 0
        for manifest_row in manifest_rows:
            task_id = _validate_band_row(manifest_row, seen)
            task_path = source / "tasks" / f"{task_id}.json"
            task = json.loads(task_path.read_text(encoding="utf-8"))
            if task.get("id") != task_id:
                raise ValueError(f"{task_id}: task JSON id does not match filename")
            metadata = task.get("metadata", {})
            if metadata.get("calib_reps") != manifest_row["calib_reps"]:
                raise ValueError(f"{task_id}: task/manifest calib_reps mismatch")
            if abs(
                float(metadata.get("calib_pass_rate", -1))
                - float(manifest_row["calib_pass_rate"])
            ) > 1e-9:
                raise ValueError(f"{task_id}: task/manifest pass rate mismatch")

            asset_failures = _unreadable_upload_assets(task)
            if asset_failures:
                if not quarantine_unreadable_assets:
                    raise OSError(
                        f"{task_id}: unreadable upload assets: "
                        + "; ".join(asset_failures)
                    )
                quarantined_rows.append(
                    {
                        "task_id": task_id,
                        "reason": "unreadable_upload_assets",
                        "failures": asset_failures,
                    }
                )
                continue

            function_source = source / "tmp_funcs" / task_id
            function_paths = metadata.get("new_functions_paths", [])
            if not function_source.is_dir() and function_paths:
                raise FileNotFoundError(
                    f"{task_id}: evaluator function directory not found"
                )
            if function_source.is_dir():
                shutil.copytree(
                    function_source,
                    stage / "tmp_funcs" / task_id,
                    copy_function=shutil.copy2,
                )
            asset_count += _copy_task_assets(
                task,
                task_id=task_id,
                stage=stage,
                final=output,
            )
            _rewrite_function_provenance(task, task_id=task_id, final=output)
            (stage / "tasks" / f"{task_id}.json").write_text(
                json.dumps(task, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            tasks.append(task)
            included_manifest_rows.append(manifest_row)

        if manifest_path.read_bytes() != manifest_before:
            raise RuntimeError(
                "band_live manifest changed while freezing; rerun for a coherent snapshot"
            )

        _write_jsonl(stage / "manifest.jsonl", included_manifest_rows)
        if quarantined_rows:
            _write_jsonl(stage / "quarantine.jsonl", quarantined_rows)
        row_count = _write_jsonl(
            stage / "train.jsonl",
            (
                _gym_row(task, repeat_index)
                for repeat_index in range(num_repeats)
                for task in tasks
            ),
        )
        metadata = {
            "schema_version": 1,
            "source": str(source),
            "source_manifest_sha256": hashlib.sha256(manifest_before).hexdigest(),
            "filter": "calib_reps == 8 and 0 < passes < 8",
            "unique_tasks": len(tasks),
            "source_tasks": len(manifest_rows),
            "quarantined_tasks": len(quarantined_rows),
            "num_repeats": num_repeats,
            "gym_rows": row_count,
            "upload_assets": asset_count,
        }
        (stage / "SNAPSHOT.json").write_text(
            json.dumps(metadata, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        _write_checksums(stage)
        os.rename(stage, output)
    except BaseException:
        shutil.rmtree(stage, ignore_errors=True)
        raise

    print(
        f"Frozen {len(tasks)} tasks / {row_count} Gym rows / "
        f"{asset_count} upload assets at {output}; "
        f"quarantined {len(quarantined_rows)} tasks"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--num-repeats", type=int, default=1)
    parser.add_argument(
        "--quarantine-unreadable-assets",
        action="store_true",
        help="Exclude and record tasks whose upload assets cannot be copied.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    freeze_band(
        args.source,
        args.output,
        num_repeats=args.num_repeats,
        quarantine_unreadable_assets=args.quarantine_unreadable_assets,
    )


if __name__ == "__main__":
    main()
